"""
automuon/backends/newton_schulz.py

Newton-Schulz iteration for approximate matrix orthogonalization.

Key insight from the reference (important for correctness):
    The iteration is always run in "wide" form (cols >= rows).
    For tall matrices (rows > cols) we transpose before iterating
    and transpose back after. This means A = X @ X.T is always the
    smaller (rows x rows) Gram matrix, keeping matmuls cheap.

The quintic polynomial per step:
    A = X @ X.T
    B = b*A + c*(A @ A)
    X = a*X + B @ X

With coefficients (a, b, c) = (3.4445, -4.7750, 2.0315).

The number of steps (default 5) controls accuracy vs cost. 5 steps
is the standard used in all published Muon results. 
"""

from __future__ import annotations

import torch
from torch import Tensor


# Coefficients

# Reference: https://github.com/KellerJordan/modded-nanogpt
# Confirmed in PyTorch's own implementation: pytorch/torch/optim/_muon.py
NS_A: float = 3.4445
NS_B: float = -4.7750
NS_C: float = 2.0315

DEFAULT_NS_STEPS: int = 5


# Core implementation

def orthogonalize(
    G: Tensor,
    steps: int = DEFAULT_NS_STEPS,
    normalize_grad: bool = True,
    eps: float = 1e-7,
) -> Tensor:
    """
    Compute the approximate orthogonal polar factor of G via Newton-Schulz.
    """

    if G.ndim != 2:
        raise ValueError(
            f"orthogonalize expects a 2D tensor, got shape {tuple(G.shape)}. "
            f"Reshape conv/higher-dim weights before calling."
        )
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    # Upcast to float32 for numerical stability; restore dtype at end.
    original_dtype = G.dtype
    X = G.to(dtype=torch.float32)

    # Frobenius normalization ensures spectral norm is approximately <= 1,
    # which is the convergence regime of the NS polynomial.
    if normalize_grad:
        X = X / (X.norm() + eps)

    # Always iterate in wide form (cols >= rows).
    # Transpose tall matrices before iterating, transpose back after.
    # This ensures A = X @ X.T is the smaller (m x m) Gram matrix.
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T

    # Newton-Schulz iterations.
    # Reference loop from modded-nanogpt:
    #   A = X @ X.T
    #   B = b * A + c * A @ A
    #   X = a * X + B @ X
    for _ in range(steps):
        A = X @ X.T
        B = NS_B * A + NS_C * (A @ A)
        X = NS_A * X + B @ X

    if transposed:
        X = X.T

    return X.to(dtype=original_dtype)


# Residual diagnostic, not used during training but useful for testing and debugging.

@torch.no_grad()
def orthogonality_residual(X: Tensor) -> float:
    """
    Measure how far X is from orthogonality.

    Returns ||I - X @ X.T||_F for wide matrices (cols >= rows),
    or ||I - X.T @ X||_F for tall matrices. Returns 0.0 for a
    perfectly semi-orthogonal matrix.
    """
    X_f = X.to(torch.float32)
    if X_f.shape[0] <= X_f.shape[1]:
        # Wide or square: rows should be orthonormal
        gram = X_f @ X_f.T
        eye  = torch.eye(X_f.shape[0], device=X_f.device, dtype=torch.float32)
    else:
        # Tall: cols should be orthonormal
        gram = X_f.T @ X_f
        eye  = torch.eye(X_f.shape[1], device=X_f.device, dtype=torch.float32)
    return (gram - eye).norm(p="fro").item()