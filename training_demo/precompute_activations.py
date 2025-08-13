import os
import json
from datasets import load_dataset
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
from training_demo.config import LLMConfig, DataConfig


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


def generate_metadata(save_dir, num_files):
    """Generate metadata.json file compatible with S3RCache."""
    first_file_path = os.path.join(
        save_dir, f"activations_{0:05d}_of_{num_files:05d}.pkl"
    )

    # Load first file to get tensor shapes and dtype
    with open(first_file_path, "rb") as f:
        first_data = t.load(f)

    states_shape = list(first_data["states"].shape)
    input_ids_shape = list(first_data["input_ids"].shape)
    dtype_str = str(first_data["states"].dtype)

    # Get file size in bytes
    bytes_per_file = os.path.getsize(first_file_path)

    metadata = {
        "shape": states_shape,
        "input_ids_shape": input_ids_shape,
        "dtype": dtype_str,
        "bytes_per_file": bytes_per_file,
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

        save_dict = {
            "input_ids": file_inputs_nBL.flatten(0,1).cpu(), 
            "states": file_acts_nBLD.flatten(0,1).cpu()
        }

        save_name = f"activations_{file_idx:05d}_of_{num_files:05d}.pkl"
        save_path = os.path.join(save_dir, save_name)
        with open(save_path, "wb") as f:
            t.save(save_dict, f)

        del file_inputs_nBL, file_acts_nBLD
        gc.collect()
        t.cuda.empty_cache()

    # Generate metadata.json after all files are saved
    generate_metadata(save_dir, num_files)

    print(f"LLM activations successfully precomputed!")


class LocalCache:
    """Lightweight cache that loads precomputed activation files sequentially."""

    def __init__(self, save_dir, shuffle=True, seed=42, return_ids=True):
        self.save_dir = save_dir
        self.shuffle = shuffle
        self.seed = seed
        self.return_ids = return_ids

        # Load metadata
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Find all activation files
        self.file_paths = []
        for filename in os.listdir(save_dir):
            if filename.startswith("activations_") and filename.endswith((".pkl", "")):
                if not filename.endswith(".pkl"):
                    # Handle files without extension
                    self.file_paths.append(os.path.join(save_dir, filename))
                else:
                    self.file_paths.append(os.path.join(save_dir, filename))

        self.file_paths.sort()  # Ensure consistent order, don't shuffle file order
        self.current_file_idx = 0

    def __iter__(self):
        self.current_file_idx = 0
        return self

    def __next__(self):
        if self.current_file_idx >= len(self.file_paths):
            raise StopIteration

        file_path = self.file_paths[self.current_file_idx]
        self.current_file_idx += 1

        with open(file_path, "rb") as f:
            data = t.load(f)

        # Apply shuffling to activations within the file
        if self.shuffle and not self.return_ids:
            # Only shuffle states if not returning input_ids
            data["states"] = shuffle_megabatch_tokens(data["states"], self.seed)
        elif self.shuffle and self.return_ids:
            # Shuffle both states and input_ids together
            states = data["states"]
            shuffled_states = shuffle_megabatch_tokens(states, self.seed)
            data["states"] = shuffled_states

        if not self.return_ids:
            # Return only states for backward compatibility
            return data["states"]

        return data  # Returns {"states": tensor, "input_ids": tensor}

    def finalize(self):
        """Cleanup method for compatibility with ActivaultS3ActivationBuffer."""
        pass


if __name__ == "__main__":
    llm_config = LLMConfig()
    data_config = DataConfig()

    model, tokenizer, submodule = get_model_tokenizer_submodule(llm_config)

    text_generator = hf_dataset_to_generator(
        dataset_name=data_config.dataset_name,
        split=data_config.dataset_split,
        streaming=True,
    )

    precompute_activations(
        model,
        submodule,
        tokenizer,
        text_generator,
        num_total_tokens=data_config.num_total_tokens,
        num_tokens_per_file=data_config.num_tokens_per_file,
        llm_batch_size=llm_config.llm_batch_size,
        ctx_len=data_config.ctx_len,
        submodule_dim=llm_config.submodule_dim,
        add_special_tokens=data_config.add_special_tokens,
        save_dir=data_config.precomputed_act_save_dir,
        device=llm_config.device,
        dtype=llm_config.dtype
    )

    # Example usage of ActivaultS3ActivationBuffer with LocalCache
    print("\nExample usage of ActivaultS3ActivationBuffer with LocalCache:")

    # Create LocalCache from precomputed activations
    local_cache = LocalCache(
        save_dir=data_config.precomputed_act_save_dir,
        shuffle=True,
        seed=data_config.seed,
        return_ids=True,
    )

    # Create ActivaultS3ActivationBuffer using LocalCache
    activation_buffer = ActivaultS3ActivationBuffer(
        cache=local_cache,
        batch_size=llm_config.sae_batch_size,
        device=llm_config.device,
        io="out",
    )

    # Iterate through activation batches
    print(f"Loading activation batches from {data_config.precomputed_act_save_dir}")
    for i, batch in enumerate(activation_buffer):
        print(f"Batch {i}: shape {batch.shape}, device {batch.device}")
        if i >= 2:  # Show first 3 batches as example
            break

    # Clean up
    activation_buffer.close()

    t.cuda.synchronize()
    print(f"last line!")
    sys.exit(0)  # Somehow not properly exiting the file automatically
