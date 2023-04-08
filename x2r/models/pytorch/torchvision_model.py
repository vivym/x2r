from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from torchvision.models import get_model, get_model_weights, resnet50

from x2r.configs import ModelConfig


@dataclass(kw_only=True)
class TorchvisionModelConfig(ModelConfig):
    _target_: str = "x2r.models.pytorch.TorchvisionModel"

    model_name: str
    pretrained: bool = True
    extra_kwargs: Optional[Dict[str, Any]] = None


cs = ConfigStore.instance()
cs.store(group="model", name="TorchvisionModel", node=TorchvisionModelConfig)


class TorchvisionModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        extra_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__()

        if pretrained:
            weights = get_model_weights(model_name).DEFAULT
        else:
            weights = None

        if extra_kwargs is None:
            extra_kwargs = {}

        self.model = get_model(model_name, weights=weights, **extra_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        images, labels = batch["image"], batch["label"]
        images = images.permute(0, 3, 1, 2)

        logits = self(images)

        return F.cross_entropy(logits, labels)

    def validation_step(self, batch, batch_idx: int):
        ...
