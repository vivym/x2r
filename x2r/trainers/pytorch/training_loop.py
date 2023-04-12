from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, Dict

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from ray.air import session
from ray.data import DatasetIterator
from ray.train.torch import prepare_model, get_device

from x2r.models.pytorch import TorchModel
from .config import Precision, ClipGradType


@dataclass
class Context:
    device: torch.device

    model: TorchModel

    optimizer: torch.optim.Optimizer

    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
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

    current_epoch: int
    current_step: int

    _done: bool = False
    _num_steps_per_epoch: int = 0

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

    def stop(self):
        self._done = True

    def should_run_val(self, on_epoch: bool) -> bool:
        if not self.has_validation_step or self.val_dataset_shard is None:
            return False

        if on_epoch:
            return self.current_epoch % self.val_every_n_epochs == 0
        else:
            return self.current_step % self.val_every_n_steps == 0

    def advance_epoch(self):
        self.current_epoch += 1
        self.num_steps_per_epoch = self._num_steps_per_epoch
        self._num_steps_per_epoch = 0

    def advance_step(self):
        self.current_step += 1
        self._num_steps_per_epoch += 1


@torch.no_grad()
def val_on_epoch(context: Context):
    val_dataset_iter = context.val_dataset_shard.iter_torch_batches(
        batch_size=context.val_batch_size, device=context.device
    )

    model = context.model
    model.eval()

    for i, batch in enumerate(val_dataset_iter):
        with autocast(**context.autocast_params):
            model.validation_step(batch, i)


def apply_gradient_clippping(context: Context):
    model, optimizer = context.model, context.optimizer
    grad_scaler = context.grad_scaler

    if context.clip_grad_type is not None:
        if grad_scaler is not None:
            # Unscales the gradients of optimizer's assigned params in-place
            grad_scaler.unscale_(optimizer)

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        if context.clip_grad_type == ClipGradType.norm:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=context.clip_grad_value,
                norm_type=context.clip_grad_norm_type,
            )
        elif context.clip_grad_type == ClipGradType.value:
            torch.nn.utils.clip_grad_value_(
                model.parameters(),
                clip_value=context.clip_grad_value,
            )
        else:
            raise ValueError(f"Unknown clip_grad_type: {context.clip_grad_type}")


def train_one_epoch(context: Context):
    train_dataset_iter = context.train_dataset_shard.iter_torch_batches(
        batch_size=context.train_batch_size, device=context.device
    )
    context.num_steps_per_epoch = 0

    model, optimizer, lr_scheduler = context.model, context.optimizer, context.lr_scheduler
    model.train()

    for batch in train_dataset_iter:
        if context.done:
            break

        # zero the parameter gradients
        optimizer.zero_grad()

        with autocast(**context.autocast_params):
            # forward + backward + optimize
            res = model.training_step(batch)

        loss: Optional[torch.Tensor] = None
        metrics: Optional[Dict[str, Any]] = None
        if res is not None:
            if isinstance(res, torch.Tensor):
                loss = res
            elif isinstance(res, tuple) and len(res) == 2:
                if isinstance(res[0], torch.Tensor):
                    loss = res[0]

                if isinstance(res[1], dict):
                    metrics = res[1]

        if loss is not None:
            loss /= context.accumulate_grad_batches

            grad_scaler = context.grad_scaler
            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (context.current_step + 1) % context.accumulate_grad_batches == 0:
                apply_gradient_clippping(context)

                if grad_scaler is not None:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()

        if metrics is not None:
            # TODO: checkpoint
            metrics["epoch"] = context.current_epoch
            metrics["lr"] = optimizer.param_groups[0]["lr"]
            # TODO: estimated time to finish
            session.report(metrics)

        if lr_scheduler is not None and context.lr_scheduler_on_step:
            lr_scheduler.step()

        context.advance_step()

        if context.should_run_val(on_epoch=False):
            val_on_epoch(context)


def default_training_loop_per_worker(
    model_factory: Callable[[], nn.Module],
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    lr_scheduler_factory: Callable[[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]],
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
):
    device = get_device()

    model: TorchModel = prepare_model(model_factory())

    optimizer = optimizer_factory(model.parameters())
    lr_scheduler = lr_scheduler_factory(optimizer)

    train_dataset_shard = session.get_dataset_shard("train")
    val_dataset_shard = session.get_dataset_shard("val")

    has_validation_step = hasattr(model, "validation_step")

    train_batch_size = train_batch_size if train_batch_size is not None else batch_size
    assert train_batch_size is not None
    val_batch_size = val_batch_size if val_batch_size is not None else batch_size
    assert val_batch_size is not None

    context = Context(
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
        current_epoch=0,
        current_step=0,    # TODO: restore from checkpoint
    )

    if context.amp_enabled:
        context.grad_scaler = GradScaler()

    while not context.done:
        train_one_epoch(context)

        if context.lr_scheduler is not None and not context.lr_scheduler_on_step:
            context.lr_scheduler.step()

        context.advance_epoch()

        if context.should_run_val(on_epoch=True):
            val_on_epoch(context)
