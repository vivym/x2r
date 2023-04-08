from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, List

from hydra.core.config_store import ConfigStore

from x2r.io import FileSystemConfig


class TaskType(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    TUNE = "tune"
    SERVE = "serve"


@dataclass(kw_only=True)
class PrepareConfig:
    _target_: str

    output_dir: str
    cache_dir: Optional[str] = None
    filesystem: Optional[FileSystemConfig] = None


# TODO:
@dataclass(kw_only=True)
class PreprocessorConfig:
    _target_: str


@dataclass(kw_only=True)
class DatasetConfig:
    _target_: str


@dataclass(kw_only=True)
class DatasetsConfig:
    prepare: Optional[PrepareConfig] = None
    preprocessors: Optional[List[Dict[str, Any]]] = None
    train: Optional[DatasetConfig] = None
    val: Optional[DatasetConfig] = None
    test: Optional[DatasetConfig] = None


@dataclass(kw_only=True)
class ModelConfig:
    _target_: str


@dataclass(kw_only=True)
class OptimizerConfig:
    _target_: str


@dataclass(kw_only=True)
class LRSchedulerConfig:
    _target_: str


@dataclass(kw_only=True)
class TrainerConfig:
    _target_: str


@dataclass(kw_only=True)
class Config:
    task: TaskType = TaskType.TRAIN

    filesystem: Optional[FileSystemConfig] = None

    datasets: DatasetsConfig

    model: ModelConfig

    optimizer: OptimizerConfig

    lr_scheduler: Optional[LRSchedulerConfig] = None

    trainer: TrainerConfig


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
