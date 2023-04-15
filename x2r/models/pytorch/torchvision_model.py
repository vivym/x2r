from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from torchvision.models import get_model, get_model_weights
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy

from x2r.configs import ModelConfig
from .torch_model import TorchModel, Metric


@dataclass(kw_only=True)
class TorchvisionModelConfig(ModelConfig):
    _target_: str = "x2r.models.pytorch.TorchvisionModel"

    model_name: str
    pretrained: bool = True
    num_classes: int = 1000


cs = ConfigStore.instance()
cs.store(group="model", name="TorchvisionModel", node=TorchvisionModelConfig)


class TorchvisionModel(TorchModel):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        num_classes: int = 1000,
    ):
        super().__init__()

        if pretrained:
            weights = get_model_weights(model_name).DEFAULT
        else:
            weights = None

        self.model = get_model(model_name, weights=weights, num_classes=num_classes)

        self.train_loss_metric = Metric(
            MeanMetric(),
            on_step=True,
            on_epoch=True,
        )
        self.train_acc_metric = Metric(
            MulticlassAccuracy(num_classes=num_classes),
            on_step=True,
            on_epoch=True,
        )
        self.val_loss_metric = Metric(
            MeanMetric(),
            on_step=True,
            on_epoch=True,
        )
        self.val_acc_metric = Metric(
            MulticlassAccuracy(num_classes=num_classes),
            on_step=True,
            on_epoch=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_training_metrics(self) -> Optional[Dict[str, Metric]]:
        return {
            "loss": self.train_loss_metric,
            "acc": self.train_acc_metric,
        }

    def training_step(self, batch):
        images, labels = batch["image"], batch["label"]
        images = images.permute(0, 3, 1, 2)

        logits = self(images)

        loss = F.cross_entropy(logits, labels)

        self.train_loss_metric.update(loss)
        self.train_acc_metric.update(logits, labels)

        return loss

    def get_validation_metrics(self) -> Optional[Dict[str, Metric]]:
        return {
            "loss": self.val_loss_metric,
            "acc": self.val_acc_metric,
        }

    def validation_step(self, batch):
        images, labels = batch["image"], batch["label"]
        images = images.permute(0, 3, 1, 2)

        logits = self(images)

        loss = F.cross_entropy(logits, labels)

        self.val_loss_metric.update(loss)
        self.val_acc_metric.update(logits, labels)
