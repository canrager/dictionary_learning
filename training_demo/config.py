from typing import Optional, Type, Any, List, Dict
import torch as t

from training_demo.config_defaults import get_trainer_configs

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


class LLMConfig:
    def __init__(self) -> None:
        # self.llm_name: str = "EleutherAI/pythia-160m-deduped"
        self.llm_name: str = "google/gemma-2-2b"
        self.llm_batch_size: int = 500
        self.sae_batch_size: int = 100
        self.dtype: t.dtype = t.bfloat16
        self.layer: int = 6
        self.submodule_name: str = f"resid_post_layer_{self.layer}"
        self.submodule_dim: int = 768
        self.device: str = "cuda:0"

        self.normalize_input_acts: bool = True


llm_config = LLMConfig()


class DataConfig:
    def __init__(self) -> None:
        self.dataset_name: str = "HuggingFaceFW/fineweb"
        self.dataset_split: str = "train"
        self.num_total_tokens: int = 50_000_000
        self.num_tokens_per_file: int = 100_000  # Roughly 100 tokens
        self.ctx_len: int = 50
        self.add_special_tokens: bool = False
        self.trained_sae_save_dir = f"trained_sae_splinterp_sweep"
        self.precomputed_act_save_dir = f"precomputed_activations_{self.dataset_name.split("/")[-1]}_{self.num_total_tokens}_tokens"
        self.seed: int = 42
        self.log_steps: int = 100

        # save_steps
        desired_checkpoints = t.logspace(-3, 0, 7).tolist()
        desired_checkpoints = [0.0] + desired_checkpoints[:-1]
        desired_checkpoints.sort()

        self.steps = int(self.num_total_tokens / llm_config.sae_batch_size)
        self.save_steps = [int(self.steps * step) for step in desired_checkpoints]
        self.save_steps.sort()

        self.use_wandb: bool = True
        self.wandb_project_name: str = "splinterp_sae_sweep"

data_config = DataConfig()


class BaseTrainerConfig:
    def __init__(self) -> None:
        self.activation_dim: int = llm_config.submodule_dim
        self.layer: str = llm_config.layer
        self.lm_name: str = llm_config.llm_name
        self.submodule_name: str
        self.device: str = llm_config.device
        self.steps: int = data_config.steps

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert all hyperparameter attributes to a dictionary.
        """
        param_dict = {}

        skipped_arguments = ["architecture", "to_dict", "wandb_project_name"]

        for attr_name in dir(self):
            # Skip private attributes and methods
            if attr_name.startswith("_") or attr_name in skipped_arguments:
                continue
            param_dict[attr_name] = getattr(self, attr_name)

        return param_dict


class StandardTrainerConfig(BaseTrainerConfig):
    def __init__(self) -> None:
        super().__init__()

        self.architecture: str = "standard"
        self.trainer: Type[Any] = StandardTrainerAprilUpdate
        self.dict_class: Type[Any] = AutoEncoder
        self.wandb_project_name = (
            f"StandardTrainer-{llm_config.llm_name}-{llm_config.submodule_name}",
        )

        self.dict_size: int | List[int] = 10_000
        self.lr: float | List[float] = [3e-4]
        self.warmup_steps: int | List[int] = 10
        self.l1_penalty: float | List[float] = [0.015, 0.06]
        self.sparsity_warmup_steps: Optional[int] = 10
        self.seed: int | List[int] = [0]


class TopKTrainerConfig(BaseTrainerConfig):
    def __init__(self) -> None:
        super().__init__()

        self.architecture: str = "top_k"
        self.trainer: Type[Any] = TopKTrainer
        self.dict_class: Type[Any] = AutoEncoderTopK
        self.wandb_project_name = (
            f"TopKTrainer-{llm_config.llm_name}-{llm_config.submodule_name}"
        )

        self.dict_size: int | List[int] = 10_000
        self.lr: float = [3e-4]
        self.warmup_steps: int | List[int] = 10
        self.k: int | List[int] = [80, 160]
        self.auxk_alpha: float = 1 / 32
        self.threshold_beta: float = 0.999
        self.threshold_start_step: int = 1000  # when to begin tracking the average threshold
        self.seed: int | List[int] = [0]


architecture_configs = [StandardTrainerConfig(), TopKTrainerConfig()]


if __name__ == "__main__":
    all_configs = get_trainer_configs(architecture_configs)
    print(f"Found {len(all_configs)} in total.")
    print(all_configs)
