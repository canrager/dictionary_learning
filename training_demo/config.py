'''
Central file for configurations of all SAE training run hyperparameters.
'''

from typing import Optional, Type, Any, List, Dict
import torch as t

from dictionary_learning.utils import unroll_config


from dictionary_learning.trainers.standard import (
    StandardTrainerAprilUpdate,
)
from dictionary_learning.trainers.splinterp import SplinterpTrainer
from dictionary_learning.dictionary import (
    AutoEncoder,
)
from dictionary_learning.trainers.top_k import (
    TopKTrainer,
    AutoEncoderTopK,
)


##########
# Central config class for global experiment parameters
##########

class EnvironmentConfig:
    def __init__(self) -> None:
        # Text dataset
        self.dataset_name: str = "HuggingFaceFW/fineweb"
        self.dataset_split: str = "train"
        self.num_total_tokens: int = 50_000_000 # <30mins on A6000
        self.num_tokens_per_file: int = 2_000_000  # Roughly 10GB at ctx_len
        self.ctx_len: int = 50
        self.add_special_tokens: bool = False

        # LLM Activations: SAE Input data
        self.llm_name: str = "google/gemma-2-2b"
        self.llm_batch_size: int = 500
        self.dtype: t.dtype = t.bfloat16
        self.layer: int = 12
        self.submodule_name: str = f"resid_post_layer_{self.layer}"
        self.submodule_dim: int = 2304
        self.normalize_input_acts: bool = True
        self.precomputed_act_save_dir = f"precomputed_activations_{self.dataset_name.split("/")[-1]}_{self.num_total_tokens}_tokens"

        # Activation Buffer
        self.sae_batch_size: int = 100
        self.device: str = "cuda:0"
        self.seed: int = 42
        self.steps = int(self.num_total_tokens / self.sae_batch_size)

        # Saving trained sae artifacts
        self.trained_sae_save_dir = "trained_sae_splinterp_sweep"
        self.save_steps = self.relative_log_steps_to_absolute(t.logspace(-3, 0, 7))

        # Wandb logging
        self.use_wandb: bool = False 
        self.wandb_project_name: str = "splinterp_sae_sweep"
        self.log_steps: int = 100

    def relative_log_steps_to_absolute(self, relative_log_steps: t.Tensor):
        relative_log_steps = relative_log_steps.tolist()
        relative_log_steps = [0.0] + relative_log_steps[:-1]
        absolute_steps = [int(self.steps * rel) for rel in relative_log_steps]
        absolute_steps.sort()
        return absolute_steps


##########
# Training configurations for individual SAE architectures
#
# Simply pass a list to sweep over a hyperparameter.
# Hyperparameter combinations will be automatically factored out 
# into individual TrainerConfigs with get_trainer_configs below.
##########

class BaseTrainerConfig:
    """
    Base Configuration shared by all SAE architectures.
    DO NOT EDIT THIS CONFIG, but individual SAE configs below.
    """

    def __init__(self) -> None:
        # Inherit env_configs, if necessary renaming to adhere to trainSAE 
        env_config = EnvironmentConfig()
        self.activation_dim: int = env_config.submodule_dim
        self.layer: str = env_config.layer
        self.lm_name: str = env_config.llm_name
        self.submodule_name: str = env_config.submodule_name
        self.device: str = env_config.device
        self.steps: int = env_config.steps

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert all hyperparameter attributes to a dictionary.
        """
        param_dict = {}

        # hacky way to adhere to the format expected by SAE trainers
        skipped_arguments = ["architecture_name", "to_dict", "wandb_project_name"]

        for attr_name in dir(self):
            # Skip private attributes and methods
            if attr_name.startswith("_") or attr_name in skipped_arguments:
                continue
            param_dict[attr_name] = getattr(self, attr_name)

        return param_dict


class StandardTrainerConfig(BaseTrainerConfig):
    def __init__(self) -> None:
        super().__init__()

        self.architecture_name: str = "standard"
        self.trainer: Type[Any] = StandardTrainerAprilUpdate
        self.dict_class: Type[Any] = AutoEncoder
        self.wandb_project_name = (
            f"StandardTrainer-{self.lm_name}-{self.submodule_name}"
        )

        self.dict_size: int | List[int] = 10_000
        self.lr: float | List[float] = [3e-4]
        self.warmup_steps: int | List[int] = 10
        self.l1_penalty: float | List[float] = [0.015, 0.06]
        self.sparsity_warmup_steps: Optional[int] = 10
        self.seed: int | List[int] = [0]


class SplinterpTrainerConfig(BaseTrainerConfig):
    def __init__(self) -> None:
        super().__init__()

        self.architecture_name: str = "splinterp"
        self.trainer: Type[Any] = SplinterpTrainer
        self.dict_class: Type[Any] = AutoEncoder
        self.wandb_project_name = (
            f"SplinterpTrainer-{self.lm_name}-{self.submodule_name}"
        )

        self.dict_size: int | List[int] = 10_000
        self.lr: float | List[float] = [3e-4]
        self.warmup_steps: int | List[int] = 10
        self.l1_penalty: float | List[float] = [0.015, 0.06]
        self.sparsity_warmup_steps: Optional[int] = 10
        self.seed: int | List[int] = [0]

        self.mu_enc : float = 0.001
        self.nu_enc : float = 0.001
        self.mu_dec : float = 0.001
        self.nu_dec : float = 0.001
        self.alpha_w : float = 1e-6
        self.beta_b : float = 1e-6
        self.decoder_reg : float = 1e-5
        self.prev_decoder_bias: Optional[t.Tensor] = None # Tensor values have difficulties with exporting to json!


class TopKTrainerConfig(BaseTrainerConfig):
    def __init__(self) -> None:
        super().__init__()

        self.architecture_name: str = "top_k"
        self.trainer: Type[Any] = TopKTrainer
        self.dict_class: Type[Any] = AutoEncoderTopK
        self.wandb_project_name = (
            f"TopKTrainer-{self.lm_name}-{self.submodule_name}"
        )

        self.dict_size: int | List[int] = 10_000
        self.lr: float = [3e-4]
        self.warmup_steps: int | List[int] = 10
        self.k: int | List[int] = [80, 160]
        self.auxk_alpha: float = 1 / 32
        self.threshold_beta: float = 0.999
        self.threshold_start_step: int = 1000  # when to begin tracking the average threshold
        self.seed: int | List[int] = [0]


# Unroll hyperparameter sweeps into the list of individual configs expected by trainSAE
def get_trainer_configs(configs_with_lists: List[Any]) -> list[dict]:
    """For each architecture, generate every combination of hyperparameters lists as a single trainer config."""

    trainer_config_dicts = []
    for cfg in configs_with_lists:
        unrolled_configs = unroll_config(cfg)
        unrolled_config_dicts = [c.to_dict() for c in unrolled_configs]
        trainer_config_dicts.extend(unrolled_config_dicts)

    return trainer_config_dicts


if __name__ == "__main__":
    # Example usage
    
    TRAINER_CONFIG_LIST = get_trainer_configs(
        [StandardTrainerConfig(), TopKTrainerConfig()]
    )
    print(f"Found {len(TRAINER_CONFIG_LIST)} in total.")
