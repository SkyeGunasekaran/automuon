"""
automuon/muon.py

Update rule (per step):
    buf  = momentum * buf + grad       # momentum accumulation
    g    = grad + momentum * buf       # Nesterov lookahead
    g    = orthogonalize(g, steps)     # NS polar factor
    g   *= max(rows, cols) ** 0.5      # RMS scaling
    param -= lr * g                    # apply update

References:
    https://kellerjordan.github.io/posts/muon/
    https://github.com/KellerJordan/modded-nanogpt
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
import copy 

from automuon.backends.newton_schulz import (
    orthogonalize,
    DEFAULT_NS_STEPS,
)


class Muon(Optimizer):
    """
    Muon optimizer for 2D weight matrices.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = DEFAULT_NS_STEPS,
        normalize_grad: bool = True,
        ns_eps: float = 1e-7,
    ) -> None:
        if lr < 0:
            raise ValueError(f"lr must be non-negative, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            normalize_grad=normalize_grad,
            ns_eps=ns_eps,
        )
        super().__init__(params, defaults)
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load optimizer state without aliasing tensors from the source state_dict.

        This matters when users/tests do:
            opt_b.load_state_dict(opt_a.state_dict())

        without copy.deepcopy(...). If we pass the state_dict through directly,
        the loaded momentum_buffer may share storage with opt_a's buffer.
        """
        return super().load_state_dict(copy.deepcopy(state_dict))

    @staticmethod
    def _to_2d(t: Tensor) -> Tensor:
        """
        Reshape to 2D for orthogonalization.
        Conv weights (C_out, C_in, kH, kW) -> (C_out, C_in*kH*kW).
        """
        if t.ndim == 2:
            return t
        return t.reshape(t.shape[0], -1)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr             = group["lr"]
            momentum       = group["momentum"]
            ns_steps       = group["ns_steps"]
            normalize_grad = group["normalize_grad"]
            ns_eps         = group["ns_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.ndim < 2:
                    raise RuntimeError(
                        f"Muon received a gradient with ndim={grad.ndim} for "
                        f"parameter of shape {tuple(p.shape)}. Muon requires "
                        f"ndim >= 2. Use AdamW for 1D parameters (biases, norms)."
                    )

                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["step"] = 0

                buf = state["momentum_buffer"]
                state["step"] += 1

                # Nesterov momentum
                buf.mul_(momentum).add_(grad)
                nesterov_grad = grad.add(buf, alpha=momentum)

                # Reshape to 2D, orthogonalize, reshape back
                original_shape = nesterov_grad.shape
                g2d = self._to_2d(nesterov_grad)

                g_orth = orthogonalize(
                    g2d,
                    steps=ns_steps,
                    normalize_grad=normalize_grad,
                    eps=ns_eps,
                )

                # RMS scaling: makes effective step size independent of
                # matrix shape, matching the reference implementation.
                # scale = max(rows, cols) ** 0.5
                m, n = g2d.shape
                scale = max(m, n) ** 0.5

                update = g_orth.reshape(original_shape).to(dtype=p.dtype)
                p.add_(update, alpha=-lr * scale)

        return loss
