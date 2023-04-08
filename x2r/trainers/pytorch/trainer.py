from functools import partial
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from ray.air.checkpoint import Checkpoint
from ray.air.config import DatasetConfig, RunConfig, ScalingConfig
from ray.data.preprocessor import Preprocessor
from ray.train.torch import TorchTrainer as TorchTrainerBase
from ray.train.torch.config import TorchConfig
from ray.train.trainer import GenDataset
from omegaconf import OmegaConf


class TorchTrainer(TorchTrainerBase):
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
        lr_scheduler_factory: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler],
        train_loop_per_worker: Union[Callable[[], None], Callable[[Dict], None]],
        *,
        train_loop_config: Optional[Dict] = None,
        torch_config: Optional[TorchConfig] = None,
        scaling_config: Optional[ScalingConfig] = None,
        dataset_config: Optional[Dict[str, DatasetConfig]] = None,
        run_config: Optional[RunConfig] = None,
        datasets: Optional[Dict[str, GenDataset]] = None,
        preprocessor: Optional[Preprocessor] = None,
        resume_from_checkpoint: Optional[Checkpoint] = None,
    ):
        if datasets is not None:
            datasets: Dict[str, GenDataset] = OmegaConf.to_container(datasets)

        super().__init__(
            train_loop_per_worker=lambda: train_loop_per_worker(
                model_factory=model_factory,
                optimizer_factory=optimizer_factory,
                lr_scheduler_factory=lr_scheduler_factory,
            ),
            train_loop_config=train_loop_config,
            torch_config=torch_config,
            scaling_config=scaling_config,
            dataset_config=dataset_config,
            run_config=run_config,
            datasets=datasets,
            preprocessor=preprocessor,
            resume_from_checkpoint=resume_from_checkpoint,
        )
