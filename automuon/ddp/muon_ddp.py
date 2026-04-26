"""
automuon/muon_ddp.py

DistributedDataParallel (DDP) wrapper for AutoMuon.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from  automuon.optimizer import AutoMuon


class DDPMuon(AutoMuon):
    """
    AutoMuon with explicit DDP gradient-sync helpers:

      - sync_gradients(): manual all-reduce for non-DDP setups.
      - grad_scale property: the factor applied during sync (1 / world_size).
      - Checks that dist is initialized before step() when sync is active.
    """

    def __init__(
        self,
        module: nn.Module,
        ddp_module: DDP | None = None,
        sync_grads: bool = False,
        **kwargs,
    ) -> None:
        # Remove DDPMuon-specific keys that AutoMuon doesn't understand.
        kwargs.pop("ddp_model", None)
        kwargs.pop("sync_grads", None)

        if ddp_module is not None:
            if not isinstance(ddp_module, DDP):
                raise TypeError(
                    f"ddp_module must be an nn.parallel.DistributedDataParallel "
                    f"instance, got {type(ddp_module).__name__}. "
                    f"Pass the unwrapped model as the first argument."
                )
            if ddp_module.module is not module:
                raise ValueError(
                    "ddp_module.module must be the same object as `module`. "
                    "Pass ddp_model.module as the first arg and ddp_model as ddp_module."
                )

        super().__init__(module, **kwargs)
        self._ddp_module = ddp_module
        self._sync_grads = sync_grads

    # ------------------------------------------------------------------
    # Distributed helpers
    # ------------------------------------------------------------------

    @property
    def world_size(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    @property
    def rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @property
    def grad_scale(self) -> float:
        """The factor applied to gradients during sync (1 / world_size)."""
        return 1.0 / self.world_size

    def sync_gradients(self) -> None:
        """
        Manually all-reduce gradients across ranks and divide by world_size.

        Call this *after* loss.backward() and *before* step() when you are
        NOT using DDP's automatic gradient sync (e.g. custom training loops,
        pipeline parallelism, or gradient accumulation with no_sync()).

        When using standard DDP (nn.parallel.DistributedDataParallel), DDP
        already performs this all-reduce inside backward() — do not call this.
        """
        if not dist.is_available() or not dist.is_initialized():
            return  # single-process, nothing to do

        scale = self.grad_scale
        handles = []

        # Collect all parameters with gradients and issue async all-reduces.
        params_with_grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grads.append(p)
                    handles.append(dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True))

        # Wait for all-reduces and scale.
        for handle, p in zip(handles, params_with_grads):
            handle.wait()
            p.grad.mul_(scale)

    # step() — optionally sync before delegating to AutoMuon

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> float | None:
        if self._sync_grads:
            self.sync_gradients()
        return super().step(closure)

    # Context manager: gradient accumulation with no_sync()

    def no_sync(self):
        """
        Context manager that disables DDP gradient synchronization for
        gradient accumulation. Wraps ddp_module.no_sync() if available.

        Usage:
            for i, batch in enumerate(dataloader):
                ctx = optimizer.no_sync() if i % accum_steps != 0 else nullcontext()
                with ctx:
                    loss = model(batch) / accum_steps
                    loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        """
        if self._ddp_module is not None:
            return self._ddp_module.no_sync()
        # No DDP module; return a no-op context manager.
        from contextlib import nullcontext
        return nullcontext()

    def __repr__(self) -> str:
        base = super().__repr__()
        return (
            f"DDPMuon(\n"
            f"  world_size={self.world_size}, rank={self.rank},\n"
            f"  sync_grads={self._sync_grads},\n"
            f"  {base}\n"
            f")"
        )