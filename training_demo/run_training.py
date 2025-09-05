from training_demo.config import (
    BaseConfig,
    StandardTrainerConfig,
    TopKTrainerConfig,
    BatchTopKTrainerConfig,
    get_trainer_configs,
)
from training_demo.precompute_activations import LocalCache
from dictionary_learning.activault_s3_buffer import ActivaultS3ActivationBuffer
from dictionary_learning.training import trainSAE
import torch as t


# Load Configs
env_config = BaseConfig()
trainer_config_list = get_trainer_configs(
    [StandardTrainerConfig(), TopKTrainerConfig(), BatchTopKTrainerConfig()]
)
print(f"Found {len(trainer_config_list)} trainer configs in total.")

# Create activation buffer that loads precomputed activations from local
local_cache = LocalCache(
    save_dir=env_config.precomputed_act_save_dir,
    seed=env_config.seed,
)

activation_buffer = ActivaultS3ActivationBuffer(
    cache=local_cache,
    batch_size=env_config.sae_batch_size,
    device=env_config.device,
    io="out",
)

# actually run the sweep
trainSAE(
    data=activation_buffer,
    trainer_configs=trainer_config_list,
    use_wandb=env_config.use_wandb,
    steps=env_config.steps,
    save_steps=env_config.save_steps,
    save_dir=env_config.trained_sae_save_dir,
    log_steps=env_config.log_steps,
    wandb_project=env_config.wandb_project_name,
    normalize_activations=env_config.normalize_input_acts,
    verbose=False,
    autocast_dtype=env_config.dtype,
    backup_steps=env_config.backup_steps,
)
