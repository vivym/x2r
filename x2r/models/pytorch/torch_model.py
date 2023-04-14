from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torchmetrics import Metric as BaseMetric


class Metric(nn.Module):
    metric_fn: BaseMetric
    on_step: bool = True
    on_epoch: bool = True

    def __init__(
        self,
        metric_fn: BaseMetric,
        on_step: bool = True,
        on_epoch: bool = True,
    ) -> None:
        super().__init__()

        self.metric_fn = metric_fn
        self.on_step = on_step
        self.on_epoch = on_epoch

        if self.on_step:
            self._metric_fn_step = self.metric_fn.clone()

        if self.on_epoch:
            self._metric_fn_epoch = self.metric_fn.clone()

    def update(self, *args, **kwargs):
        if self.on_step:
            self._metric_fn_step.update(*args, **kwargs)

        if self.on_epoch:
            self._metric_fn_epoch.update(*args, **kwargs)

    def get_step_metric(self) -> Optional[torch.Tensor]:
        if self.on_step:
            val = self._metric_fn_step.compute()
            self._metric_fn_step.reset()
            return val
        else:
            return None

    def get_epoch_metric(self) -> Optional[torch.Tensor]:
        if self.on_epoch:
            val = self._metric_fn_epoch.compute()
            self._metric_fn_epoch.reset()
            return val


class TorchModel(nn.Module):
    def get_training_metrics(self) -> Optional[Dict[str, Metric]]:
        return None

    def training_step(self, batch: Any):
        raise NotImplementedError("training_step not implemented")

    def get_validation_metrics(self) -> Optional[Dict[str, Metric]]:
        return None

    def validation_step(self, batch: Any):
        raise NotImplementedError("validation_step not implemented")

    def get_testing_metrics(self) -> Optional[Dict[str, Metric]]:
        return None

    def test_step(self, batch: Any):
        raise NotImplementedError("test_step not implemented")
