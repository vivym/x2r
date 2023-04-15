import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union, List, Tuple

from ray.air.config import FailureConfig
from ray.tune import SyncConfig
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
class DatasetConfig:
    _target_: str = "ray.air.config.DatasetConfig"

    fit: Optional[bool] = None
    split: Optional[bool] = None
    required: Optional[bool] = None
    transform: Optional[bool] = None
    max_object_store_memory_fraction: Optional[float] = None
    global_shuffle: Optional[bool] = None
    randomize_block_order: Optional[bool] = None
    per_epoch_preprocessor: Optional[Dict[str, Any]] = None


@dataclass(kw_only=True)
class CheckpointStrategy:
    _target_: str = "ray.air.config.CheckpointConfig"

    num_to_keep: Optional[int] = None
    checkpoint_score_attribute: Optional[str] = None
    checkpoint_score_order: str = "max"


@dataclass(kw_only=True)
class CLIReporterConfig:
    _target_: str = "ray.tune.CLIReporter"

    metric_columns: Optional[List[str]] = None
    parameter_columns: Optional[List[str]] = None
    total_samples: Optional[int] = None
    max_progress_rows: int = 20
    max_error_rows: int = 20
    max_column_length: int = 20
    max_report_frequency: int = 5
    infer_limit: int = 3
    print_intermediate_tables: Optional[bool] = None
    metric: Optional[str] = None
    mode: Optional[str] = None
    sort_by_metric: bool = False


@dataclass(kw_only=True)
class RunConfig:
    _target_: str = "ray.air.config.RunConfig"

    name: Optional[str] = None
    local_dir: Optional[str] = None
    callbacks: Optional[List[Dict[str, Any]]] = None
    failure_config: Optional[FailureConfig] = None
    sync_config: Optional[SyncConfig] = None
    checkpoint_config: Optional[CheckpointStrategy] = None
    progress_reporter: Optional[CLIReporterConfig] = None
    verbose: int = 3
    log_to_file: Optional[bool] = False


@dataclass(kw_only=True)
class CheckpointConfig:
    _target_: str = "ray.air.checkpoint.Checkpoint"

    local_path: Optional[str] = None
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
    lr_scheduler_on_step: bool = True
    precision: Precision = Precision.f32
    clip_grad_type: Optional[ClipGradType] = None
    clip_grad_value: Optional[float] = None
    clip_grad_norm_type: float = 2.0
    accumulate_grad_batches: int = 1
    checkpoint_every_n_epochs: int = 1
    checkpoint_every_n_steps: Optional[int] = None
    local_shuffle_buffer_size: Optional[int] = None

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
