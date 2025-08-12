from training_demo.config import LLMConfig, DataConfig, architecture_configs
from training_demo.config_defaults import get_trainer_configs
from training_demo.precompute_activations import LocalCache
from dictionary_learning.activault_s3_buffer import ActivaultS3ActivationBuffer
from dictionary_learning.training import trainSAE
import torch as t

# load configs
llm_config = LLMConfig()
data_config = DataConfig()
trainer_configs = get_trainer_configs(architecture_configs)

# Load activation buffer
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


# run training


print(f"len trainer configs: {len(trainer_configs)}")
assert len(trainer_configs) > 0
save_dir = f"{data_config.trained_sae_save_dir}/{llm_config.submodule_name}"

# actually run the sweep
trainSAE(
    data=activation_buffer,
    trainer_configs=trainer_configs,
    use_wandb=data_config.use_wandb,
    steps=data_config.steps,
    save_steps=data_config.save_steps,
    save_dir=data_config.trained_sae_save_dir,
    log_steps=data_config.log_steps,
    wandb_project=data_config.wandb_project_name,
    normalize_activations=llm_config.normalize_input_acts,
    verbose=False,
    autocast_dtype=t.bfloat16,
    backup_steps=10,
)
