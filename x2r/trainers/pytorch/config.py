import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union

import torch.nn as nn
from ray.air.config import (
    DatasetConfig as DatasetConfigBase,
    RunConfig as RunConfigBase,
)
from ray.train.torch.config import TorchConfig as TorchConfigBase
from hydra.core.config_store import ConfigStore

from x2r.configs import TrainerConfig


@dataclass(kw_only=True)
class TorchConfig(TorchConfigBase):
    _target_: str = "ray.train.torch.TorchTrainer"


@dataclass(kw_only=True)
class ScalingConfig:
    _target_: str = "ray.air.config.ScalingConfig"

    trainer_resources: Optional[Dict] = None
    num_workers: Optional[int] = None
    use_gpu: bool = False
    resources_per_worker: Optional[Dict] = None
    placement_strategy: str = "PACK"
    _max_cpu_fraction_per_node: Optional[float] = None


@dataclass(kw_only=True)
class DatasetConfig(DatasetConfigBase):
    _target_: str = "ray.air.config.DatasetConfig"


@dataclass(kw_only=True)
class RunConfig(RunConfigBase):
    _target_: str = "ray.air.config.RunConfig"


@dataclass(kw_only=True)
class CheckpointConfig:
    _target_: str = "ray.air.checkpoint.Checkpoint"

    local_path: Optional[Union[str, os.PathLike]] = None
    uri: Optional[str] = None


class Precision(Enum):
    f32 = "f32"
    f16 = "f16"
    bf16 = "bf16"


class ClipGradType(Enum):
    value = "value"
    norm = "norm"


@dataclass(kw_only=True)
class TrainLoopConfig:
    _target_: str = "x2r.trainers.pytorch.training_loop.default_training_loop_per_worker"
    _partial_: bool = True

    batch_size: Optional[int] = None
    train_batch_size: Optional[int] = None
    val_batch_size: Optional[int] = None
    max_epochs: int = 1000
    max_steps: Optional[int] = None
    val_every_n_epochs: int = 1
    val_every_n_steps: Optional[int] = None
    lr_scheduler_on_step: bool = False
    precision: Precision = Precision.f32
    clip_grad_type: Optional[ClipGradType] = None
    clip_grad_value: Optional[float] = None
    clip_grad_norm_type: float = 2.0
    accumulate_grad_batches: int = 1

    def __post_init__(self):
        if self.batch_size is None and self.train_batch_size is None:
            raise ValueError(
                "Must specify either `batch_size` or `train_batch_size`"
            )

        if self.batch_size is None and self.val_batch_size is None:
            raise ValueError(
                "Must specify either `batch_size` or `val_batch_size`"
            )

        assert self.accumulate_grad_batches > 0, (
            "accumulate_grad_batches must be a positive integer"
        )


@dataclass(kw_only=True)
class TorchTrainerConfig(TrainerConfig):
    _target_: str = "x2r.trainers.pytorch.TorchTrainer"

    train_loop_per_worker: Optional[TrainLoopConfig] = None
    torch_config: Optional[TorchConfig] = None
    scaling_config: Optional[ScalingConfig] = None
    dataset_config: Optional[Dict[str, DatasetConfig]] = None
    run_config: Optional[RunConfig] = None
    resume_from_checkpoint: Optional[CheckpointConfig] = None

    def __post_init__(self):
        if self.train_loop_per_worker is None:
            self.train_loop_per_worker = TrainLoopConfig()


cs = ConfigStore.instance()
cs.store(group="trainer", name="TorchTrainer", node=TorchTrainerConfig)
