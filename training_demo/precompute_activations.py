import os
import json
from transformers import BatchEncoding
import torch as t
import gc
from tqdm import trange
import sys

from dictionary_learning.utils import (
    hf_dataset_to_generator,
    get_model_tokenizer_submodule,
)
from dictionary_learning.pytorch_buffer import collect_activations
from dictionary_learning.activault_s3_buffer import (
    shuffle_megabatch_tokens,
    ActivaultS3ActivationBuffer,
)
from training_demo.config import EnvironmentConfig


def tokenized_batch(
    tokenizer,
    text_generator,
    batch_size,
    ctx_len,
    add_special_tokens,
):
    """
    Return a batch of tokenized inputs.
    """

    batch = []
    while len(batch) < batch_size:
        try:
            sample_text = next(text_generator)
        except StopIteration:
            raise StopIteration("End of data stream reached")

        sample_tok_L = tokenizer(
            sample_text,
            return_tensors="pt",
            max_length=ctx_len,
            truncation=True,
            padding=False,
            add_special_tokens=add_special_tokens,
        )

        # Only select texts longer or equal than ctx_len
        if sample_tok_L.input_ids.shape[1] < ctx_len:
            print("Skipping this sample since it does not contain enough tokens to fill context.")
            continue
        else:
            batch.append(sample_tok_L)
        # TODO make sure that no padding occurs across the whole repo!

    ids_BL = t.cat([s.input_ids for s in batch], dim=0)
    mask_BL = t.cat([s.attention_mask for s in batch], dim=0)
    encoding_BL = BatchEncoding(
        {
            "input_ids": ids_BL,
            "attention_mask": mask_BL,  # Mask is expected to be all true
        }
    )

    return encoding_BL


def tokenized_batch_generator(dataset_name, split, tokenizer, batch_size, ctx_len, add_special_tokens=True, max_batches=10000):
    text_generator = hf_dataset_to_generator(dataset_name, split, streaming=True)

    for _ in range(max_batches):
        yield tokenized_batch(tokenizer, text_generator, batch_size, ctx_len, add_special_tokens=True)


def generate_metadata(save_dir, num_files):
    """Generate metadata.json file compatible with S3RCache."""
    first_states_path = os.path.join(
        save_dir, f"states_{0:05d}_of_{num_files:05d}.pkl"
    )
    first_input_ids_path = os.path.join(
        save_dir, f"input_ids_{0:05d}_of_{num_files:05d}.pkl"
    )

    # Load first files to get tensor shapes and dtype
    with open(first_states_path, "rb") as f:
        first_states = t.load(f)
    with open(first_input_ids_path, "rb") as f:
        first_input_ids = t.load(f)

    states_shape = list(first_states.shape)
    input_ids_shape = list(first_input_ids.shape)
    dtype_str = str(first_states.dtype)

    # Get file sizes in bytes
    states_bytes_per_file = os.path.getsize(first_states_path)
    input_ids_bytes_per_file = os.path.getsize(first_input_ids_path)

    metadata = {
        "shape": states_shape,
        "input_ids_shape": input_ids_shape,
        "dtype": dtype_str,
        "states_bytes_per_file": states_bytes_per_file,
        "input_ids_bytes_per_file": input_ids_bytes_per_file,
    }

    # Save metadata.json
    metadata_path = os.path.join(save_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {metadata_path}")
    return metadata


def precompute_activations(
    model,
    submodule,
    tokenizer,
    text_generator,
    num_total_tokens,
    num_tokens_per_file,
    llm_batch_size,
    ctx_len,
    submodule_dim,
    add_special_tokens,
    save_dir,
    device,
    dtype,
) -> None:

    os.makedirs(save_dir, exist_ok=True)

    assert (
        num_total_tokens % num_tokens_per_file == 0
    ), "Num_total_tokens must be a multiple of tokens_per_file"
    assert (
        num_tokens_per_file % llm_batch_size * ctx_len == 0
    ), "Num_tokens_per_file must be a multiple of llm_batch_size"

    num_files = num_total_tokens // num_tokens_per_file
    num_batches = num_tokens_per_file // (llm_batch_size * ctx_len)


    for file_idx in range(num_files):
        file_acts_nBLD = t.zeros(num_batches, llm_batch_size, ctx_len, submodule_dim, device=device, dtype=dtype)
        file_inputs_nBL = t.zeros(num_batches, llm_batch_size, ctx_len, device=device, dtype=dtype)
        for batch_idx in trange(num_batches, desc=f"Processing file {file_idx}/{num_files}"
        ):
            # Collect input token batch
            encoding_BL = tokenized_batch(
                tokenizer=tokenizer,
                text_generator=text_generator,
                batch_size=llm_batch_size,
                ctx_len=ctx_len,
                add_special_tokens=add_special_tokens,
            )
            file_inputs_nBL[batch_idx] = encoding_BL.input_ids

            # Collect LLM activations
            encoding_BL = encoding_BL.to(device)
            acts_BLD = collect_activations(model, submodule, encoding_BL)
            file_acts_nBLD[batch_idx] = acts_BLD

        # Save states and input_ids as separate files
        states_name = f"states_{file_idx:05d}_of_{num_files:05d}.pkl"
        input_ids_name = f"input_ids_{file_idx:05d}_of_{num_files:05d}.pkl"
        
        states_path = os.path.join(save_dir, states_name)
        input_ids_path = os.path.join(save_dir, input_ids_name)
        
        # Save states file
        with open(states_path, "wb") as f:
            t.save(file_acts_nBLD.flatten(0,1).cpu(), f)
            
        # Save input_ids file
        with open(input_ids_path, "wb") as f:
            t.save(file_inputs_nBL.flatten(0,1).cpu(), f)

        del file_inputs_nBL, file_acts_nBLD
        gc.collect()
        t.cuda.empty_cache()

    # Generate metadata.json after all files are saved
    generate_metadata(save_dir, num_files)

    print(f"LLM activations successfully precomputed!")


class LocalCache:
    """Lightweight cache that loads precomputed activation files sequentially. Only loads states for faster performance."""

    def __init__(self, save_dir, seed=42):
        self.save_dir = save_dir
        self.seed = seed

        # Load metadata
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Find all states files (only load states for performance)
        self.states_file_paths = []
        
        for filename in os.listdir(save_dir):
            if filename.startswith("states_") and filename.endswith(".pkl"):
                self.states_file_paths.append(os.path.join(save_dir, filename))

        self.states_file_paths.sort()  # Ensure consistent order
        self.current_file_idx = 0

    def __iter__(self):
        self.current_file_idx = 0
        return self

    def __next__(self):
        if self.current_file_idx >= len(self.states_file_paths):
            raise StopIteration

        states_path = self.states_file_paths[self.current_file_idx]
        self.current_file_idx += 1
        
        with open(states_path, "rb") as f:
            states = t.load(f)
            
        return {"states": states}

    def finalize(self):
        """Cleanup method for compatibility with ActivaultS3ActivationBuffer."""
        pass


if __name__ == "__main__":
    env_config = EnvironmentConfig()

    model, tokenizer, submodule = get_model_tokenizer_submodule(env_config)

    text_generator = hf_dataset_to_generator(
        dataset_name=env_config.dataset_name,
        split=env_config.dataset_split,
        streaming=True,
    )

    precompute_activations(
        model,
        submodule,
        tokenizer,
        text_generator,
        num_total_tokens=env_config.num_total_tokens,
        num_tokens_per_file=env_config.num_tokens_per_file,
        llm_batch_size=env_config.llm_batch_size,
        ctx_len=env_config.ctx_len,
        submodule_dim=env_config.submodule_dim,
        add_special_tokens=env_config.add_special_tokens,
        save_dir=env_config.precomputed_act_save_dir,
        device=env_config.device,
        dtype=env_config.dtype
    )

    # Example usage of ActivaultS3ActivationBuffer with LocalCache
    print("\nExample usage of ActivaultS3ActivationBuffer with LocalCache:")

    # Create LocalCache from precomputed activations
    local_cache = LocalCache(
        save_dir=env_config.precomputed_act_save_dir,
        seed=env_config.seed,
    )

    # Create ActivaultS3ActivationBuffer using LocalCache
    activation_buffer = ActivaultS3ActivationBuffer(
        cache=local_cache,
        batch_size=env_config.sae_batch_size,
        device=env_config.device,
        io="out",
    )

    # Iterate through activation batches
    print(f"Loading activation batches from {env_config.precomputed_act_save_dir}")
    for i, batch in enumerate(activation_buffer):
        print(f"Batch {i}: shape {batch.shape}, device {batch.device}")
        if i >= 2:  # Show first 3 batches as example
            break

    # Clean up
    activation_buffer.close()

    t.cuda.synchronize()
    print(f"last line!")
    sys.exit(0)  # Somehow not properly exiting the file automatically if using Huggingface Fineweb datatset. Works fine with other datasets.
