from typing import Any

import torch.nn as nn


class TorchModel(nn.Module):
    def training_step(self, batch: Any):
        raise NotImplementedError("training_step not implemented")

    def validation_step(self, batch: Any):
        raise NotImplementedError("validation_step not implemented")

    def test_step(self, batch: Any):
        raise NotImplementedError("test_step not implemented")
