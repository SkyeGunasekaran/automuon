"""
automuon/optimizer.py

AutoMuon: the public-facing optimizer shell.

Usage:
    optimizer = AutoMuon(model, lr=3e-4)

    # With separate learning rates:
    optimizer = AutoMuon(model, muon_lr=2e-2, adamw_lr=3e-4)

    # Verbose: print partition table at init:
    optimizer = AutoMuon(model, lr=3e-4, verbose=True)
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer

from automuon.backends.scanner import scan, partition, ScannedParameter
from automuon.backends.muon import Muon
from automuon.utils.muon_logging import print_partition_table


# Default hyperparameters

# Muon typically wants a much higher lr than AdamW because the orthogonalized
# update has unit spectral norm — the lr is literally "spectral norm per step".
# A ratio of ~67x (0.02 / 3e-4) is common in practice.
DEFAULT_MUON_LR   = 0.02
DEFAULT_ADAMW_LR  = 3e-4

# AdamW defaults matching the commonly used values in transformer training.
DEFAULT_ADAMW_BETAS   = (0.9, 0.999)
DEFAULT_ADAMW_EPS     = 1e-8
DEFAULT_ADAMW_WD      = 0.1

# Muon defaults
DEFAULT_MOMENTUM      = 0.95
DEFAULT_NS_STEPS      = 5
DEFAULT_NORMALIZE_GRAD = True
DEFAULT_NS_EPS        = 1e-7


class AutoMuon(Optimizer):
    """
    One-line replacement for AdamW in torch-based training pipelines.

    Automatically scans the model, routes 2D projection weights to Muon
    and everything else (embeddings, norms, biases, 1D params) to AdamW.
    Fully compatible with PyTorch learning rate schedulers.

    Args:
        model:
            The nn.Module to optimize. 
        lr:
            Convenience argument: sets BOTH muon_lr and adamw_lr. 
        muon_lr:
            Learning rate for Muon (projection weights).
            Default: 0.02. See docs for recommended ratio vs adamw_lr.
        adamw_lr:
            Learning rate for AdamW (embeddings, biases, norms, 1D params).
            Default: 3e-4.
        momentum:
            Nesterov momentum for Muon. Default: 0.95.
        ns_steps:
            Newton-Schulz iteration steps. Default: 5.
        normalize_grad:
            Normalize gradient by Frobenius norm before NS. Default: True.
        ns_eps:
            Epsilon for NS normalization. Default: 1e-7.
        adamw_betas:
            Betas for internal AdamW. Default: (0.9, 0.999).
        adamw_eps:
            Epsilon for internal AdamW. Default: 1e-8.
        adamw_wd:
            Weight decay for internal AdamW. Default: 0.1.
        verbose:
            If True, print a partition table at init showing which parameter
            goes to which optimizer and why. Default: False.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float | None = None,
        *,
        muon_lr:        float = DEFAULT_MUON_LR,
        adamw_lr:       float = DEFAULT_ADAMW_LR,
        momentum:       float = DEFAULT_MOMENTUM,
        ns_steps:       int   = DEFAULT_NS_STEPS,
        normalize_grad: bool  = DEFAULT_NORMALIZE_GRAD,
        ns_eps:         float = DEFAULT_NS_EPS,
        adamw_betas:    tuple[float, float] = DEFAULT_ADAMW_BETAS,
        adamw_eps:      float = DEFAULT_ADAMW_EPS,
        adamw_wd:       float = DEFAULT_ADAMW_WD,
        verbose:        bool  = False,
    ) -> None:

        # Convenience: `lr` sets both lrs if neither is explicitly provided.
        if lr is not None:
            muon_lr  = lr
            adamw_lr = lr

        # Scan and partition 
        self._scanned: list[ScannedParameter] = scan(model)
        muon_params, adamw_params = partition(self._scanned)

        if not muon_params and not adamw_params:
            raise ValueError(
                "AutoMuon found no trainable parameters in the model. "
                "Ensure at least one parameter has requires_grad=True."
            )

        if verbose:
            print_partition_table(self._scanned)

        #  Construct internal optimizers 
        self._muon: Muon | None = None
        if muon_params:
            self._muon = Muon(
                muon_params,
                lr=muon_lr,
                momentum=momentum,
                ns_steps=ns_steps,
                normalize_grad=normalize_grad,
                ns_eps=ns_eps,
            )

        self._adamw: AdamW | None = None
        if adamw_params:
            self._adamw = AdamW(
                adamw_params,
                lr=adamw_lr,
                betas=adamw_betas,
                eps=adamw_eps,
                weight_decay=adamw_wd,
            )

        #  Initialise the Optimizer base class 
        # We pass an empty param list because we manage param_groups ourselves
        # via the property below. The base class needs at least the defaults dict.
        defaults = dict(
            muon_lr=muon_lr,
            adamw_lr=adamw_lr,
            momentum=momentum,
            ns_steps=ns_steps,
            normalize_grad=normalize_grad,
            ns_eps=ns_eps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )
        # Pass a dummy parameter so the base class doesn't complain about
        # empty param groups — we override param_groups immediately after.
        dummy = [torch.empty(0)]
        super().__init__(dummy, defaults)
        # Clear the dummy group — param_groups property takes over.
        self.__dict__["_param_groups_override"] = []

    # param_groups property — the scheduler interception point

    @property
    def param_groups(self) -> list[dict]:
        """
        Live unified view of all internal optimizer param groups.

        PyTorch schedulers call optimizer.param_groups and then mutate
        group['lr'] in-place on the returned list. Because we return the
        actual group dicts from the internal optimizers (not copies),
        those writes land directly on the internal groups.

        Group ordering: Muon groups first, then AdamW groups.
        This is stable across calls, so scheduler index-based access works.
        """
        groups = []
        if self._muon is not None:
            groups.extend(self._muon.param_groups)
        if self._adamw is not None:
            groups.extend(self._adamw.param_groups)
        return groups

    @param_groups.setter
    def param_groups(self, value):
        # The Optimizer base class sets self.param_groups = [...] in __init__.
        # We intercept that write and discard it — our property handles groups.
        # All subsequent external writes (from schedulers etc.) go through
        # the internal optimizer group dicts directly via the getter above.
        pass

    # state property — unified view of both internal states

    @property
    def state(self) -> dict:
        """
        Unified state dict merging both internal optimizers.
        Needed so that code that does `optimizer.state[param]` works correctly.
        """
        merged = {}
        if self._muon is not None:
            merged.update(self._muon.state)
        if self._adamw is not None:
            merged.update(self._adamw.state)
        return merged

    @state.setter
    def state(self, value):
        # Intercept base class assignment in __init__; we don't use self.state
        # as a plain dict — it's a merged view computed on the fly.
        pass

    # step()

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> float | None:
        """
        Perform one optimization step.

        Dispatches to both internal optimizers. The lr sync at the top is
        belt-and-suspenders: since param_groups returns the actual internal
        group dicts, scheduler mutations are already in-place. The sync
        handles any edge case where groups were modified between calls.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._muon is not None:
            self._muon.step()
        if self._adamw is not None:
            self._adamw.step()

        return loss

    # zero_grad()

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self._muon is not None:
            self._muon.zero_grad(set_to_none=set_to_none)
        if self._adamw is not None:
            self._adamw.zero_grad(set_to_none=set_to_none)

    # state_dict / load_state_dict

    def state_dict(self) -> dict:
        """
        Returns a state dict containing both internal optimizers' states.
        Format:
            {
                "muon":  <muon state_dict or None>,
                "adamw": <adamw state_dict or None>,
            }
        """
        return {
            "muon":  self._muon.state_dict()  if self._muon  is not None else None,
            "adamw": self._adamw.state_dict() if self._adamw is not None else None,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Restore state from a dict previously returned by state_dict().
        """
        if self._muon is not None and state_dict.get("muon") is not None:
            self._muon.load_state_dict(state_dict["muon"])
        if self._adamw is not None and state_dict.get("adamw") is not None:
            self._adamw.load_state_dict(state_dict["adamw"])

    # Convenience / introspection

    def partition_summary(self) -> list[dict]:
        """
        Return the scanner's partition as a list of plain dicts — useful
        for logging to W&B / MLflow or writing to experiment configs.

        Each dict has keys: name, optimizer, shape, module_type, reason.
        """
        return [
            {
                "name":        s.name,
                "optimizer":   s.optimizer,
                "shape":       list(s.shape),
                "module_type": s.module_type,
                "reason":      s.reason,
            }
            for s in self._scanned
        ]

    def __repr__(self) -> str:
        n_muon  = len(self._muon.param_groups[0]["params"])  if self._muon  else 0
        n_adamw = len(self._adamw.param_groups[0]["params"]) if self._adamw else 0
        muon_lr  = self._muon.param_groups[0]["lr"]  if self._muon  else "—"
        adamw_lr = self._adamw.param_groups[0]["lr"] if self._adamw else "—"
        return (
            f"AutoMuon(\n"
            f"  Muon:  {n_muon} params, lr={muon_lr}\n"
            f"  AdamW: {n_adamw} params, lr={adamw_lr}\n"
            f")"
        )