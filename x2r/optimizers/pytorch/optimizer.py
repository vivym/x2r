from dataclasses import dataclass
from typing import Tuple

from hydra.core.config_store import ConfigStore

from x2r.configs import OptimizerConfig


@dataclass(kw_only=True)
class AdamConfig(OptimizerConfig):
    _target_: str = "torch.optim.Adam"
    _partial_: bool = True

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.
    amsgrad: bool = False


@dataclass(kw_only=True)
class AdamWConfig(OptimizerConfig):
    _target_: str = "torch.optim.AdamW"
    _partial_: bool = True

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    amsgrad: bool = False


cs = ConfigStore.instance()
cs.store(group="optimizer", name="Adam", node=AdamConfig)
cs.store(group="optimizer", name="AdamW", node=AdamWConfig)
