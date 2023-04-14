from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torchmetrics import Metric as BaseMetric


class Metric(nn.Module):
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

        self._last_step_value = None

    def update(self, *args, **kwargs):
        self._last_step_value = self.metric_fn(*args, **kwargs).item()

    def get_step_metric(self) -> Optional[float]:
        if self.on_step:
            assert self._last_step_value is not None
            return self._last_step_value
        else:
            return None

    def get_epoch_metric(self) -> Optional[float]:
        if self.on_epoch:
            val = self.metric_fn.compute().item()
            self.metric_fn.reset()
            return val
        else:
            return None


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
