from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch.nn as nn


@dataclass(kw_only=True)
class MetricConfig:
    metric_fn: Any
    on_epoch: bool = True
    on_step: bool = True


class TorchModel(nn.Module):
    def get_training_metrics(self) -> Optional[Dict[MetricConfig]]:
        return None

    def training_step(self, batch: Any):
        raise NotImplementedError("training_step not implemented")

    def get_validation_metrics(self) -> Optional[Dict[MetricConfig]]:
        return None

    def validation_step(self, batch: Any):
        raise NotImplementedError("validation_step not implemented")

    def get_testing_metrics(self) -> Optional[Dict[MetricConfig]]:
        return None

    def test_step(self, batch: Any):
        raise NotImplementedError("test_step not implemented")
