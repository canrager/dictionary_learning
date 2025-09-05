from training_demo.config import BaseConfig
import dictionary_learning.utils as utils
from training_demo.precompute_activations import tokenized_batch_generator
from nnsight import LanguageModel

env_config = BaseConfig()

model, tokenizer, submodule = utils.get_model_tokenizer_submodule(
    env_config, do_truncate_model=False, load_with_nnsight=True
)

token_generator = tokenized_batch_generator(
    dataset_name=env_config.dataset_name,
    split=env_config.dataset_split,
    tokenizer=tokenizer,
    batch_size=env_config.sae_batch_size,
    ctx_len=env_config.ctx_len,
)

encodings_BL = next(token_generator)
print(f"Input shape: {encodings_BL.input_ids.shape}")

model = LanguageModel(
    env_config.llm_name, device_map="cuda", torch_dtype=env_config.dtype, dispatch=True
)

print(model.device)  # meta

encodings_BL = encodings_BL.to(model.device)

with model.trace(
    encodings_BL
):  # Throws error: NotImplementedError: Cannot copy out of meta tensor; no data!
    logits = model.lm_head.output.save()

print(f"logits shape: {logits.shape}")
