"""
Central file for configurations of all SAE training run hyperparameters.
"""

from typing import Optional, Type, Any, List, Dict
import torch as t
from datetime import datetime

from dictionary_learning.utils import unroll_config


from dictionary_learning.trainers.standard import (
    StandardTrainerAprilUpdate,
)
from dictionary_learning.dictionary import (
    AutoEncoder,
)
from dictionary_learning.trainers.top_k import (
    TopKTrainer,
    AutoEncoderTopK,
)
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE, BatchTopKTrainer


##########
# Central config class for global experiment parameters
##########


class BaseConfig:
    def __init__(self) -> None:
        # Text dataset
        # self.dataset_name: str = "HuggingFaceFW/fineweb"
        self.dataset_name: str = "monology/pile-uncopyrighted"
        self.dataset_split: str = "train"
        self.num_total_tokens: int = 100_000_000  # <30mins on A6000
        self.num_tokens_per_file: int = 2_000_000  # Roughly 10GB at ctx_len
        self.ctx_len: int = 500
        self.add_special_tokens: bool = False

        # LLM Activations: SAE Input data
        # self.llm_name: str = "google/gemma-2-2b"
        self.llm_name: str = "meta-llama/Llama-3.1-8B"
        self.llm_batch_size: int = 100
        self.dtype: t.dtype = t.bfloat16
        self.layer: int = 16
        self.submodule_name: str = f"resid_post_layer_{self.layer}"
        self.submodule_dim: int = 4096
        self.normalize_input_acts: bool = False
        # self.precomputed_act_save_dir = f"precomputed_activations_{self.dataset_name.split("/")[-1]}_{self.num_total_tokens}_tokens"
        self.precomputed_act_save_dir = f"precomputed_activations_llama_fineweb_100_000_000_tokens"
        self.hf_cache_dir: str = "/home/can/models"

        # Activation Buffer
        self.sae_batch_size: int = 2000
        self.device: str = "cuda:0"
        self.seed: int = 42
        self.steps = int(self.num_total_tokens / self.sae_batch_size)

        # Saving trained sae artifacts
        now_string = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.trained_sae_save_dir = "trained_sae_llama_sweep_" + now_string
        self.save_steps = self.relative_log_steps_to_absolute(t.logspace(-3, 0, 7))
        self.backup_steps = None

        # Wandb logging
        self.use_wandb: bool = True
        self.wandb_project_name: str = "llama_sweep_" + now_string
        self.log_steps: int = 100

        # Evaluation
        self.eval_sae_dir = "trained_sae_llama_sweep_2025-09-05--03-31-44"
        self.eval_num_sequences: int = 200
        self.eval_batch_size: int = 10
        self.eval_mean_metric_over_sequence: bool = False

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
        env_config = BaseConfig()
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
        self.wandb_project_name = (f"StandardTrainer-{self.lm_name}-{self.submodule_name}",)

        self.dict_size: int | List[int] = self.activation_dim * 4
        self.lr: float | List[float] = [1e-4]
        self.warmup_steps: int | List[int] = 10
        self.l1_penalty: float | List[float] = [0.015, 0.06]
        self.sparsity_warmup_steps: Optional[int] = 10
        self.seed: int | List[int] = 0


class TopKTrainerConfig(BaseTrainerConfig):
    def __init__(self) -> None:
        super().__init__()

        self.architecture_name: str = "top_k"
        self.trainer: Type[Any] = TopKTrainer
        self.dict_class: Type[Any] = AutoEncoderTopK
        self.wandb_project_name = f"TopKTrainer-{self.lm_name}-{self.submodule_name}"

        self.dict_size: int | List[int] = self.activation_dim * 4
        self.lr: float = [1e-4]
        self.warmup_steps: int | List[int] = 10
        self.k: int | List[int] = [80, 160]
        self.auxk_alpha: float = 1 / 32
        self.threshold_beta: float = 0.999
        self.threshold_start_step: int = 1000  # when to begin tracking the average threshold
        self.seed: int | List[int] = 0


class BatchTopKTrainerConfig(BaseTrainerConfig):
    def __init__(self) -> None:
        super().__init__()

        self.architecture_name: str = "batch_top_k"
        self.trainer: Type[Any] = BatchTopKTrainer
        self.dict_class: Type[Any] = BatchTopKSAE
        self.wandb_project_name = f"BatchTopKTrainer-{self.lm_name}-{self.submodule_name}"

        self.dict_size: int | List[int] = self.activation_dim * 4
        self.lr: float = [1e-4]
        self.warmup_steps: int | List[int] = 10
        self.k: int | List[int] = [80, 160]
        self.auxk_alpha: float = 1 / 32
        self.threshold_beta: float = 0.999
        self.threshold_start_step: int = 1000  # when to begin tracking the average threshold
        self.seed: int | List[int] = 0


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

    TRAINER_CONFIG_LIST = get_trainer_configs([StandardTrainerConfig(), TopKTrainerConfig()])
    print(f"Found {len(TRAINER_CONFIG_LIST)} in total.")
