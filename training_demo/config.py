from typing import Optional, Type, Any, List, Dict
import torch as t

from dictionary_learning.config_defaults import get_trainer_configs

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
    llm_name: str = "EleutherAI/pythia-160m-deduped"
    llm_batch_size: int = 64
    context_length: int = 1024
    sae_batch_size: int = 2048
    dtype: t.dtype = t.bfloat16

    submodule_name: str = "resid_layer_XX"


llm_config = LLMConfig()


class EnvConfig:
    dataset_name: str = "monology/pile-uncopyrighted"
    wandb_project_name: str = "splinterp_sae_sweep"


env_config = EnvConfig()


class BaseTrainerConfig:
    input_size: int
    device: str
    layer: str
    lm_name: str
    submodule_name: str
    num_total_tokens: int = 500_000_000

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert all hyperparameter attributes to a dictionary.
        """
        param_dict = {}
        
        for attr_name in dir(self):
            # Skip private attributes and methods
            if attr_name.startswith('_') or attr_name == 'to_dict':
                continue
            param_dict[attr_name] = getattr(self, attr_name)
            
        return param_dict


class StandardTrainerConfig(BaseTrainerConfig):
    architecture: str = "standard"
    trainer: Type[Any] = StandardTrainerAprilUpdate
    dict_class: Type[Any] = AutoEncoder
    wandb_project_name = (
        f"StandardTrainer-{llm_config.llm_name}-{llm_config.submodule_name}",
    )

    dict_size: int | List[int]
    lr: float | List[float] = [3e-4]
    lr_warmup_steps: Optional[int] = 1000
    l1_penalty: float | List[float] = [0.015, 0.06]
    sparsity_warmup_steps: Optional[int] = 5000
    resample_steps: Optional[int] = None
    seeds: int | List[int] = [0]


class TopKTrainerConfig(BaseTrainerConfig):
    architecture: str = "top_k"
    trainer: Type[Any] = TopKTrainer
    dict_class: Type[Any] = AutoEncoderTopK
    wandb_project_name = (
        f"TopKTrainer-{llm_config.llm_name}-{llm_config.submodule_name}"
    )

    dict_size: int | List[int]
    lr: float = [3e-4]
    k: int | List[int] = [80, 160]
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000  # when to begin tracking the average threshold
    seeds: int | List[int] = [0]


architecture_configs = [StandardTrainerConfig(), TopKTrainerConfig()]


if __name__ == "__main__":
    all_configs = get_trainer_configs(architecture_configs)
    print(f"Found {len(all_configs)} in total.")
    print(all_configs)
