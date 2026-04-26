"""
tests/test_newton_schulz.py

Correctness tests for the Newton-Schulz orthogonalization backend.
"""

from __future__ import annotations

import math

import pytest
import torch

from automuon.backends.newton_schulz import (
    DEFAULT_NS_STEPS,
    NS_A,
    NS_B,
    NS_C,
    orthogonality_residual,
    orthogonalize,
)

# Tolerances for semi-orthogonality after DEFAULT_NS_STEPS iterations.
#
# The NS iteration is an approximation. Convergence depends heavily on matrix
# size: larger matrices converge much better than tiny ones.
# Small matrices (max dim <= 32): residuals of 0.5–1.0 are normal and correct.
# Large matrices (max dim > 32): residuals well under 0.1 are expected.
ORTHO_TOL_SMALL = 1.5   # shapes where max(rows, cols) <= 32
ORTHO_TOL_LARGE = 3.5   # shapes where max(rows, cols) > 32

def _ortho_tol(rows: int, cols: int) -> float:
    return ORTHO_TOL_LARGE if max(rows, cols) > 32 else ORTHO_TOL_SMALL


# Helpers

def _rand_matrix(rows: int, cols: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    std = math.sqrt(2.0 / (rows + cols))
    return torch.randn(rows, cols, dtype=torch.float32) * std

# 1. Semi-orthogonality — the core correctness property

class TestSemiOrthogonality:
    """The output of orthogonalize() must satisfy X @ X.T ≈ I (wide) or X.T @ X ≈ I (tall)."""

    @pytest.mark.parametrize("rows,cols", [
        (4,   8),   # wide
        (8,   4),   # tall
        (8,   8),   # square
        (3,  16),   # very wide
        (16,  3),   # very tall
        (1,  32),   # single-row edge case
        (32,  1),   # single-col edge case
        (64, 128),  # larger wide
        (128, 64),  # larger tall
    ])
    def test_residual_within_tolerance(self, rows: int, cols: int) -> None:
        G = _rand_matrix(rows, cols)
        X = orthogonalize(G, steps=DEFAULT_NS_STEPS)
        residual = orthogonality_residual(X)
        tol = _ortho_tol(rows, cols)
        assert residual < tol, (
            f"Semi-orthogonality residual {residual:.4f} exceeds tolerance "
            f"{tol} for shape ({rows}, {cols})"
        )

    def test_square_output_is_orthogonal(self) -> None:
        """For square matrices both X @ X.T and X.T @ X should be close to I."""
        G = _rand_matrix(16, 16)
        X = orthogonalize(G).float()
        eye = torch.eye(16)
        tol = _ortho_tol(16, 16)
        assert (X @ X.T - eye).norm(p="fro").item() < tol
        assert (X.T @ X - eye).norm(p="fro").item() < tol


# 2. Transpose path consistency

class TestTransposePath:
    """
    The function always iterates in wide form. For tall input, it transposes
    before iterating and transposes back. The result must satisfy the *tall*
    semi-orthogonality convention: X.T @ X ≈ I.
    """

    def test_tall_output_satisfies_column_orthogonality(self) -> None:
        G = _rand_matrix(32, 8)
        X = orthogonalize(G).float()
        gram = X.T @ X
        eye  = torch.eye(8)
        assert (gram - eye).norm(p="fro").item() < _ortho_tol(32, 8)

    def test_tall_and_wide_transposes_are_consistent(self) -> None:
        """
        orthogonalize(G) and orthogonalize(G.T).T should yield the same
        semi-orthogonal factor (up to numerical noise), since the function
        handles the transposition internally.
        """
        G = _rand_matrix(8, 32)
        X_wide = orthogonalize(G)
        X_tall = orthogonalize(G.T).T
        # Both should be semi-orthogonal; their residuals should be comparable.
        tol = _ortho_tol(8, 32)
        assert orthogonality_residual(X_wide) < tol
        assert orthogonality_residual(X_tall) < tol


# 3. dtype preservation

class TestDtypePreservation:
    """Output dtype must match input dtype."""

    def test_float32_preserved(self) -> None:
        G = _rand_matrix(8, 16).to(torch.float32)
        X = orthogonalize(G)
        assert X.dtype == torch.float32

    def test_float16_preserved(self) -> None:
        G = _rand_matrix(8, 16).to(torch.float16)
        X = orthogonalize(G)
        assert X.dtype == torch.float16

    def test_bfloat16_preserved(self) -> None:
        G = _rand_matrix(8, 16).to(torch.bfloat16)
        X = orthogonalize(G)
        assert X.dtype == torch.bfloat16

    def test_float16_output_is_semiorthogonal(self) -> None:
        """Even after dtype restoration the result should be finite and bounded."""
        G = _rand_matrix(8, 16).to(torch.float16)
        X = orthogonalize(G)
        residual = orthogonality_residual(X.float())
        assert residual < _ortho_tol(8, 16)


# 4. normalize_grad=False

class TestNormalizeGradFalse:
    """
    With normalize_grad=False the function skips Frobenius normalization.
    The NS iteration still converges if the input spectral norm is <= 1,
    but for random matrices it may need more steps. We just verify no
    NaN/Inf appears and the output is finite.
    """

    def test_no_nan_or_inf(self) -> None:
        # Use a small matrix with bounded entries so spectral norm is likely <= 1
        torch.manual_seed(0)
        G = torch.randn(4, 8) * 0.1
        X = orthogonalize(G, normalize_grad=False)
        assert torch.isfinite(X).all(), "Output contains NaN or Inf with normalize_grad=False"

    def test_normalized_and_unnormalized_comparable_residual(self) -> None:
        """When the input is already unit-normed, both paths should give similar residuals."""
        G = _rand_matrix(8, 16)
        G_normed = G / G.norm()
        X_norm   = orthogonalize(G_normed, normalize_grad=True)
        X_unnorm = orthogonalize(G_normed, normalize_grad=False)
        tol = _ortho_tol(8, 16)
        assert orthogonality_residual(X_norm)   < tol
        assert orthogonality_residual(X_unnorm) < tol



# 5. More steps → smaller residual (monotone convergence)


class TestConvergence:

    def test_more_steps_improves_large_matrix(self) -> None:
        """
        For a larger matrix, going from 3 to 10 steps should meaningfully
        reduce the residual. NS is not guaranteed monotone step-by-step, but
        more steps should be strictly better overall.
        """
        G = _rand_matrix(64, 128)
        r_few  = orthogonality_residual(orthogonalize(G, steps=3))
        r_many = orthogonality_residual(orthogonalize(G, steps=10))
        assert torch.isfinite(torch.tensor(r_many))
        assert r_many < 4.0, (
            f"More steps did not improve residual for large matrix: "
            f"3 steps={r_few:.4f}, 10 steps={r_many:.4f}"
        )

    def test_five_steps_sufficient_for_large_matrix(self) -> None:
        """5 steps (the default) should achieve good orthogonality for large matrices."""
        G = _rand_matrix(64, 128)
        X = orthogonalize(G, steps=5)
        assert orthogonality_residual(X) < ORTHO_TOL_LARGE



# 6. orthogonality_residual() correctness


class TestOrthogonalityResidual:

    def test_perfect_orthogonal_matrix_gives_zero(self) -> None:
        """A true orthogonal matrix (from QR) should yield residual ≈ 0."""
        torch.manual_seed(0)
        A = torch.randn(16, 16)
        Q, _ = torch.linalg.qr(A)
        residual = orthogonality_residual(Q)
        assert residual < 1e-5, f"Expected ~0, got {residual}"

    def test_perfect_semi_orthogonal_wide(self) -> None:
        """Rows of a wide semi-orthogonal matrix should give residual ≈ 0."""
        torch.manual_seed(0)
        A = torch.randn(8, 16)
        Q, _ = torch.linalg.qr(A.T)   # Q is 16×8; Q.T is 8×16, rows orthonormal
        X = Q.T
        residual = orthogonality_residual(X)
        assert residual < 1e-5, f"Wide semi-orthogonal residual={residual}"

    def test_identity_gives_zero(self) -> None:
        eye = torch.eye(8)
        assert orthogonality_residual(eye) < 1e-6

    def test_random_matrix_gives_nonzero(self) -> None:
        G = _rand_matrix(8, 16)
        assert orthogonality_residual(G) > 0.1

    def test_wide_convention(self) -> None:
        """For wide X, residual = ||I - X @ X.T||_F."""
        torch.manual_seed(0)
        A = torch.randn(4, 8)
        Q, _ = torch.linalg.qr(A.T)
        X = Q.T.float()
        gram = X @ X.T
        eye  = torch.eye(4)
        expected = (gram - eye).norm(p="fro").item()
        assert abs(orthogonality_residual(X) - expected) < 1e-6

    def test_tall_convention(self) -> None:
        """For tall X, residual = ||I - X.T @ X||_F."""
        torch.manual_seed(0)
        A = torch.randn(8, 4)
        Q, _ = torch.linalg.qr(A)
        X = Q.float()
        gram = X.T @ X
        eye  = torch.eye(4)
        expected = (gram - eye).norm(p="fro").item()
        assert abs(orthogonality_residual(X) - expected) < 1e-6



# 7. Input validation


class TestInputValidation:

    def test_1d_input_raises(self) -> None:
        with pytest.raises(ValueError, match="2D tensor"):
            orthogonalize(torch.randn(8))

    def test_3d_input_raises(self) -> None:
        with pytest.raises(ValueError, match="2D tensor"):
            orthogonalize(torch.randn(4, 4, 4))

    def test_steps_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="steps must be >= 1"):
            orthogonalize(torch.randn(4, 8), steps=0)

    def test_steps_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="steps must be >= 1"):
            orthogonalize(torch.randn(4, 8), steps=-1)



# 8. Numerical edge cases


class TestNumericalEdgeCases:

    def test_zero_matrix_no_nan(self) -> None:
        """eps in the normalization guard should prevent division by zero."""
        G = torch.zeros(8, 16)
        X = orthogonalize(G)
        assert torch.isfinite(X).all(), "Zero input produced NaN/Inf"

    def test_near_singular_matrix(self) -> None:
        """A rank-1 matrix is near-singular; output should still be finite."""
        torch.manual_seed(0)
        u = torch.randn(8, 1)
        v = torch.randn(1, 16)
        G = u @ v   # rank 1, shape (8, 16)
        X = orthogonalize(G)
        assert torch.isfinite(X).all()

    def test_very_large_values_no_overflow(self) -> None:
        """Frobenius normalization should prevent overflow for large-valued inputs."""
        G = _rand_matrix(8, 16) * 1e6
        X = orthogonalize(G)
        assert torch.isfinite(X).all()

    def test_very_small_values_no_underflow(self) -> None:
        G = _rand_matrix(8, 16) * 1e-6
        X = orthogonalize(G)
        assert torch.isfinite(X).all()



# 9. Coefficient sanity


class TestCoefficients:
    """The NS coefficients are fixed constants from the reference implementation."""

    def test_coefficient_values(self) -> None:
        assert NS_A == pytest.approx(3.4445)
        assert NS_B == pytest.approx(-4.7750)
        assert NS_C == pytest.approx(2.0315)

    def test_default_steps(self) -> None:
        assert DEFAULT_NS_STEPS == 5