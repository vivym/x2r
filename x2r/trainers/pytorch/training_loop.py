from dataclasses import dataclass, fields
from typing import Any, Callable, Optional, Dict, Iterator

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from ray.air import session, Checkpoint
from ray.data import DatasetIterator
from ray.train.torch import prepare_model, get_device

from x2r.models.pytorch import TorchModel, MetricConfig
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


@dataclass
class CheckpointData:
    model: Dict[str, torch.Tensor]
    optimizer: torch.optim.Optimizer
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    grad_scaler: Optional[GradScaler]
    current_epoch: int
    current_step: int


@dataclass
class Context:
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

    train_metrics: Optional[Dict[str, MetricConfig]]
    val_metrics: Optional[Dict[str, MetricConfig]]

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
            if self.val_every_n_steps is not None:
                return False
            else:
                return (self.current_epoch + 1) % self.val_every_n_epochs == 0
        else:
            if self.val_every_n_steps is None:
                return False
            else:
                return (self.current_step + 1) % self.val_every_n_steps == 0

    def should_checkpoint(self, on_epoch: bool = True, on_finish: bool = False) -> bool:
        if on_finish:
            if self.checkpoint_every_n_steps is not None:
                return self.current_step % self.checkpoint_every_n_steps != 0
            else:
                return self.current_epoch % self.checkpoint_every_n_epochs != 0

        if on_epoch:
            if self.checkpoint_every_n_steps is not None:
                return False
            else:
                return (self.current_epoch + 1) % self.checkpoint_every_n_epochs == 0
        else:
            if self.checkpoint_every_n_steps is None:
                return False
            else:
                return (self.current_step + 1) % self.checkpoint_every_n_steps == 0

    def dump_to_checkpoint(self) -> dict:
        ckpt = {}
        for field in fields(self):
            if field.name in ["model", "optimizer", "lr_scheduler", "grad_scaler"]:
                value = getattr(self, field.name)
                ckpt["model"] = None if value is None else value.state_dict()
            elif field.name in ["train_dataset_shard", "val_dataset_shard", "train_metrics", "val_metrics"]:
                # ignore
                continue
            else:
                ckpt[field.name] = getattr(self, field.name)

        return ckpt

    def load_from_checkpoint(self, ckpt: dict):
        for key, value in ckpt.items():
            if key in ["model", "optimizer", "lr_scheduler", "grad_scaler"]:
                if value is None:
                    setattr(self, key, value)
                else:
                    getattr(self, key).load_state_dict(value)
            else:
                # TODO: override config
                setattr(self, key, value)

    def get_metrics(self, train_stage: bool):
        prefix = "train/" if train_stage else "val/"

        metric_configs = self.train_metrics if train_stage else self.val_metrics

        metrics = {}
        for key, config in metric_configs.items():
            metric = config.metric_fn.compute()
            suffix = "_step"
            metrics[prefix + key + suffix] = metric

        return metrics

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
    train_dataset_iter = HasNextIterator(train_dataset_iter)
    context.num_steps_per_epoch = 0

    model, optimizer, lr_scheduler = context.model, context.optimizer, context.lr_scheduler
    model.train()

    while train_dataset_iter.has_next and not context.done:
        batch = next(train_dataset_iter)

        # zero the parameter gradients
        optimizer.zero_grad()

        with autocast(**context.autocast_params):
            # forward + backward + optimize
            loss: Optional[torch.Tensor] = model.training_step(batch)

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

        if lr_scheduler is not None and context.lr_scheduler_on_step:
            lr_scheduler.step()

        if context.should_run_val(on_epoch=False):
            val_on_epoch(context)

        if context.should_checkpoint(on_epoch=False):
            checkpoint = Checkpoint.from_dict(context.dump_to_checkpoint())
        else:
            checkpoint = None

        # metrics["epoch"] = context.current_epoch
        # metrics["step"] = context.current_step
        # metrics["lr"] = optimizer.param_groups[0]["lr"]
        # # TODO: estimated time to finish
        # session.report(metrics, checkpoint=checkpoint)

        context.advance_step()

        # TODO: if no next iter, delay to epoch end
        metrics = context.get_metrics(train_stage=True, on_step=True)
        session.report(metrics, checkpoint=checkpoint)


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
        checkpoint_every_n_epochs=checkpoint_every_n_epochs,
        checkpoint_every_n_steps=checkpoint_every_n_steps,
        train_metrics=model.get_training_metrics(),
        val_metrics=model.get_validation_metrics(),
        current_epoch=0,
        current_step=0,    # TODO: restore from checkpoint
    )

    if context.amp_enabled:
        context.grad_scaler = GradScaler()

    last_ckpt = session.get_checkpoint()
    if last_ckpt is not None:
        context.load_from_checkpoint(last_ckpt.to_dict())

    # TODO: merge epoch and step loop, to avoid duplicated code
    while not context.done:
        train_one_epoch(context)

        if context.lr_scheduler is not None and not context.lr_scheduler_on_step:
            context.lr_scheduler.step()

        if context.should_run_val(on_epoch=True):
            val_on_epoch(context)

        if context.should_checkpoint(on_epoch=True):
            session.report({}, checkpoint=Checkpoint.from_dict(context.dump_to_checkpoint()))

        context.advance_epoch()

    if context.should_checkpoint(on_finish=True):
        session.report({}, checkpoint=Checkpoint.from_dict(context.dump_to_checkpoint()))
