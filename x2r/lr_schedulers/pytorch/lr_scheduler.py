from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from x2r.configs import LRSchedulerConfig


@dataclass(kw_only=True)
class WarmupLRSchedulerConfig(LRSchedulerConfig):
    _target_: str = "x2r.optimizers.pytorch.lr_scheduler.WarmupScheduler"
    _partial_: bool = True

    lr_scheduler: LRSchedulerConfig
    warmup_start_value: float = 0.
    warmup_end_value: float = 1.
    warmup_duration: int = 0


@dataclass(kw_only=True)
class CosineAnnealingLRConfig(LRSchedulerConfig):
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingLR"
    _partial_: bool = True

    T_max: int
    eta_min: float = 0.
    last_epoch: int = -1
    verbose: bool = False


cs = ConfigStore.instance()
cs.store(group="lr_scheduler", name="WarmupLRScheduler", node=WarmupLRSchedulerConfig)
cs.store(group="lr_scheduler", name="CosineAnnealingLR", node=CosineAnnealingLRConfig)
