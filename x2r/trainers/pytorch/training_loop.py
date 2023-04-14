from dataclasses import dataclass, field, fields
from typing import Any, Callable, Optional, Dict, Iterator

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from ray.air import session, Checkpoint
from ray.data import DatasetIterator
from ray.train.torch import prepare_model, get_device

from x2r.models.pytorch import TorchModel, Metric
from .config import Precision, ClipGradType


class HasNextIterator(Iterator):
    def __init__(self, it):
        self.it = iter(it)
        self._has_next = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._has_next:
            result = self._the_next
        else:
            result = next(self.it)
        self._has_next = None
        return result

    @property
    def has_next(self) -> bool:
        if self._has_next is None:
            try:
                self._the_next = next(self.it)
            except StopIteration:
                self._has_next = False
            else:
                self._has_next = True
        return self._has_next


def default_training_loop_per_worker(
    model_factory: Callable[[], nn.Module],
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    lr_scheduler_factory: Callable[[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LRScheduler]],
    batch_size: Optional[int],
    train_batch_size: Optional[int],
    val_batch_size: Optional[int],
    max_epochs: int,
    max_steps: Optional[int],
    val_every_n_epochs: int,
    val_every_n_steps: Optional[int],
    lr_scheduler_on_step: bool,
    precision: Precision,
    clip_grad_type: Optional[ClipGradType],
    clip_grad_value: Optional[float],
    clip_grad_norm_type: float,
    accumulate_grad_batches: int,
    checkpoint_every_n_epochs: int,
    checkpoint_every_n_steps: Optional[int],
):
    device = get_device()

    model: TorchModel = prepare_model(model_factory())

    optimizer = optimizer_factory(model.parameters())
    lr_scheduler = lr_scheduler_factory(optimizer)

    train_dataset_shard = session.get_dataset_shard("train")
    val_dataset_shard = session.get_dataset_shard("val")
    # TODO: support test dataset (using the best ckpt)

    has_validation_step = hasattr(model, "validation_step")
    assert has_validation_step

    train_batch_size = train_batch_size if train_batch_size is not None else batch_size
    assert train_batch_size is not None
    val_batch_size = val_batch_size if val_batch_size is not None else batch_size
    assert val_batch_size is not None

    loop = TrainingLoop(
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr_scheduler_on_step=lr_scheduler_on_step,
        train_dataset_shard=train_dataset_shard,
        train_batch_size=train_batch_size,
        max_epochs=max_epochs,
        max_steps=max_steps,
        num_steps_per_epoch=None,
        has_validation_step=has_validation_step,
        val_dataset_shard=val_dataset_shard,
        val_batch_size=val_batch_size,
        val_every_n_epochs=val_every_n_epochs,
        val_every_n_steps=val_every_n_steps,
        precision=precision,
        grad_scaler=None,
        clip_grad_type=clip_grad_type,
        clip_grad_value=clip_grad_value,
        clip_grad_norm_type=clip_grad_norm_type,
        accumulate_grad_batches=accumulate_grad_batches,
        checkpoint_every_n_epochs=checkpoint_every_n_epochs,
        checkpoint_every_n_steps=checkpoint_every_n_steps,
        train_metrics=model.get_training_metrics(),
        val_metrics=model.get_validation_metrics(),
        current_epoch=0,
        current_step=0,
    )

    if loop.amp_enabled:
        loop.grad_scaler = GradScaler()

    last_ckpt = session.get_checkpoint()
    if last_ckpt is not None:
        loop.load_state_dict(last_ckpt.to_dict())

    loop.run()


@dataclass
class TrainingLoop:
    device: torch.device

    model: TorchModel

    optimizer: torch.optim.Optimizer

    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    lr_scheduler_on_step: bool

    train_dataset_shard: DatasetIterator
    train_batch_size: int
    max_epochs: int
    max_steps: Optional[int]
    num_steps_per_epoch: Optional[int]

    has_validation_step: bool
    val_dataset_shard: DatasetIterator
    val_batch_size: int
    val_every_n_epochs: int
    val_every_n_steps: Optional[int]

    precision: str
    grad_scaler: Optional[GradScaler]

    clip_grad_type: Optional[str]
    clip_grad_value: Optional[float]
    clip_grad_norm_type: float

    accumulate_grad_batches: int

    checkpoint_every_n_epochs: int
    checkpoint_every_n_steps: Optional[int]

    train_metrics: Optional[Dict[str, Metric]]
    val_metrics: Optional[Dict[str, Metric]]

    current_epoch: int
    current_step: int

    _done: bool = False
    _num_steps_per_epoch: int = 0
    _metrics_to_report: Dict[str, Any] = field(default_factory=dict)
    _state_dict_to_report: Optional[Dict[str, Any]] = None

    @property
    def done(self) -> bool:
        if self._done:
            return self._done

        if self.max_steps is not None and self.current_step >= self.max_steps:
            self._done = True

        if self.max_steps is None and self.current_epoch >= self.max_epochs:
            self._done = True

        return self._done

    @property
    def amp_enabled(self) -> bool:
        return self.precision != Precision.f32

    @property
    def autocast_params(self) -> Dict[str, Any]:
        if self.precision == Precision.f32:
            dtype = None
        elif self.precision == Precision.f16:
            dtype = torch.float16
        elif self.precision == Precision.bf16:
            dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown precision: {self.precision}")

        return {
            "device_type": self.device.type,
            "dtype": dtype,
            "enabled": self.amp_enabled,
        }

    @property
    def should_run_val_on_step(self) -> bool:
        if not self.has_validation_step or self.val_dataset_shard is None:
            return False

        if self.val_every_n_steps is None:
            return False
        else:
            return (self.current_step + 1) % self.val_every_n_steps == 0

    @property
    def should_run_val_on_epoch(self) -> bool:
        if not self.has_validation_step or self.val_dataset_shard is None:
            return False

        if self.val_every_n_steps is not None:
            return False
        else:
            return (self.current_epoch + 1) % self.val_every_n_epochs == 0

    @property
    def should_checkpoint_on_step(self) -> bool:
        if self.checkpoint_every_n_steps is None:
            return False
        else:
            return self.current_step % self.checkpoint_every_n_steps == 0

    @property
    def should_checkpoint_on_epoch(self) -> bool:
        if self.checkpoint_every_n_steps is not None:
            return False
        else:
            return self.current_epoch % self.checkpoint_every_n_epochs == 0

    @property
    def should_checkpoint_on_finish(self) -> bool:
        if self.checkpoint_every_n_steps is not None:
            return self.current_step % self.checkpoint_every_n_steps != 0
        else:
            return self.current_epoch % self.checkpoint_every_n_epochs != 0

    def advance_epoch(self):
        self.current_epoch += 1
        self.num_steps_per_epoch = self._num_steps_per_epoch
        self._num_steps_per_epoch = 0

    def advance_step(self):
        self.current_step += 1
        self._num_steps_per_epoch += 1

    def run(self):
        while not self.done:
            train_dataset_iter = self.train_dataset_shard.iter_torch_batches(
                batch_size=self.train_batch_size, device=self.device
            )
            train_dataset_iter = HasNextIterator(train_dataset_iter)
            self._num_steps_per_epoch = 0

            while train_dataset_iter.has_next and not self.done:
                self.train_one_step(next(train_dataset_iter))

                if self.should_run_val_on_step:
                    self.val_one_epoch()

                self.advance_step()

                if self.should_checkpoint_on_step:
                    self._state_dict_to_report = self.state_dict()

                if train_dataset_iter.has_next and not self.done:
                    self.report()

            if self.lr_scheduler is not None and not self.lr_scheduler_on_step:
                self.lr_scheduler.step()

            if self.should_run_val_on_epoch:
                self.val_one_epoch()

            self._compute_metrics(self.train_metrics, stage="train", on_step=False)

            if not train_dataset_iter.has_next:
                self.advance_epoch()

            if self.should_checkpoint_on_epoch or self.done:
                self._state_dict_to_report = self.state_dict()

            self.report()

    def train_one_step(self, batch):
        model, optimizer, lr_scheduler = self.model, self.optimizer, self.lr_scheduler
        grad_scaler = self.grad_scaler

        # zero the parameter gradients
        optimizer.zero_grad()

        with autocast(**self.autocast_params):
            # forward + backward + optimize
            loss: Optional[torch.Tensor] = model.training_step(batch)

        if loss is not None:
            loss /= self.accumulate_grad_batches

            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (self.current_step + 1) % self.accumulate_grad_batches == 0:
                self._apply_gradient_clippping()

                if grad_scaler is not None:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()

        self._metrics_to_report["epoch"] = self.current_epoch
        self._metrics_to_report["step"] = self.current_step
        self._metrics_to_report["lr"] = self.optimizer.param_groups[0]["lr"]

        if lr_scheduler is not None and self.lr_scheduler_on_step:
            lr_scheduler.step()

        self._compute_metrics(self.train_metrics, stage="train", on_step=True)

    @torch.no_grad()
    def val_one_epoch(self):
        val_dataset_iter = self.val_dataset_shard.iter_torch_batches(
            batch_size=self.val_batch_size, device=self.device
        )

        model = self.model
        model.eval()

        for batch in val_dataset_iter:
            with autocast(**self.autocast_params):
                model.validation_step(batch)

        self._compute_metrics(self.val_metrics, stage="val", on_step=False)

    def report(self):
        metrics = self._metrics_to_report
        state_dict = self._state_dict_to_report
        self._metrics_to_report = {}
        self._state_dict_to_report = None

        if state_dict is not None:
            for key, val in metrics.items():
                if key.startswith("train/") or key.startswith("val/"):
                    state_dict[key] = val
            ckpt = Checkpoint.from_dict(state_dict)
        else:
            ckpt = None

        # TODO: estimated time to finish
        session.report(metrics, checkpoint=ckpt)

    def state_dict(self) -> dict:
        state_dict = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if hasattr(value, "state_dict"):
                state_dict[field.name] = value.state_dict()
            elif field.name in ["current_epoch", "current_step", "num_steps_per_epoch"]:
                state_dict[field.name] = value
        return state_dict

    def load_state_dict(self, state_dict: dict):
        for key, value in state_dict.items():
            if key in ["current_epoch", "current_step", "num_steps_per_epoch"]:
                setattr(self, key, value)
            elif hasattr(self, key):
                getattr(self, key).load_state_dict(value)

    def _compute_metrics(self, metrics: Dict[str, Metric], stage: str, on_step: bool):
        for name, metric in metrics.items():
            if on_step:
                val = metric.get_step_metric()
            else:
                val = metric.get_epoch_metric()

            if val is not None:
                self._metrics_to_report[f"{stage}/{name}_{'step' if on_step else 'epoch'}"] = val

    def _apply_gradient_clippping(self):
        model, optimizer = self.model, self.optimizer
        grad_scaler = self.grad_scaler

        if self.clip_grad_type is not None:
            if grad_scaler is not None:
                # Unscales the gradients of optimizer's assigned params in-place
                grad_scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            if self.clip_grad_type == ClipGradType.norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=self.clip_grad_value,
                    norm_type=self.clip_grad_norm_type,
                )
            elif self.clip_grad_type == ClipGradType.value:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(),
                    clip_value=self.clip_grad_value,
                )
            else:
                raise ValueError(f"Unknown clip_grad_type: {self.clip_grad_type}")
