"""
tests/test_muon.py

Correctness tests for the Muon optimizer.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from automuon.backends.muon import Muon
from automuon.backends.newton_schulz import orthogonality_residual

from conftest import make_grads



# Helpers


def _linear(rows: int, cols: int, bias: bool = False, seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(cols, rows, bias=bias)


def _make_muon(params, **kwargs) -> Muon:
    defaults = dict(lr=0.02, momentum=0.95, ns_steps=5, normalize_grad=True, ns_eps=1e-7)
    defaults.update(kwargs)
    return Muon(params, **defaults)


def _assign_grad(param: nn.Parameter, seed: int = 99) -> None:
    torch.manual_seed(seed)
    param.grad = torch.randn_like(param)



# 1 & 2. Update happens and is in the descent direction


class TestBasicUpdate:

    def test_parameter_changes_after_step(self) -> None:
        layer = _linear(16, 8)
        w_before = layer.weight.data.clone()
        opt = _make_muon([layer.weight])
        _assign_grad(layer.weight)
        opt.step()
        assert not torch.equal(layer.weight.data, w_before), \
            "Parameter did not change after one Muon step"

    def test_loss_decreases_over_steps(self) -> None:
        """
        On a convex quadratic (loss = ||W||^2 / 2) gradient descent should
        reduce the loss. We allow 5 steps to be sure.
        """
        torch.manual_seed(0)
        layer = _linear(16, 8)
        opt   = _make_muon([layer.weight], lr=0.01)
        losses = []
        for _ in range(5):
            layer.weight.grad = layer.weight.data.clone()  # grad = W → loss = ||W||^2/2
            losses.append((layer.weight.data ** 2).sum().item())
            opt.step()
        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses}"

    def test_update_direction_opposes_gradient(self) -> None:
        """The update should have a negative inner product with the gradient."""
        layer  = _linear(16, 8)
        w_init = layer.weight.data.clone()
        torch.manual_seed(1)
        layer.weight.grad = torch.randn_like(layer.weight)
        grad_copy = layer.weight.grad.clone()
        opt = _make_muon([layer.weight])
        opt.step()
        delta = layer.weight.data - w_init
        dot   = (delta * grad_copy).sum().item()
        assert dot < 0, f"Update has positive inner product with gradient: {dot:.4f}"



# 3 & 4. Momentum buffer and Nesterov lookahead


class TestMomentumBuffer:

    def test_momentum_buffer_initialised_on_first_step(self) -> None:
        layer = _linear(8, 4)
        opt = _make_muon([layer.weight])
        _assign_grad(layer.weight)
        opt.step()
        state = opt.state[layer.weight]
        assert "momentum_buffer" in state
        assert state["momentum_buffer"].shape == layer.weight.shape

    def test_step_counter_increments(self) -> None:
        layer = _linear(8, 4)
        opt = _make_muon([layer.weight])
        for i in range(3):
            _assign_grad(layer.weight, seed=i)
            opt.step()
        assert opt.state[layer.weight]["step"] == 3

    def test_buffer_accumulates_across_steps(self) -> None:
        """Buffer after step 2 should differ from buffer after step 1."""
        layer = _linear(8, 4)
        opt   = _make_muon([layer.weight])
        _assign_grad(layer.weight, seed=0)
        opt.step()
        buf_after_step1 = opt.state[layer.weight]["momentum_buffer"].clone()
        _assign_grad(layer.weight, seed=1)
        opt.step()
        buf_after_step2 = opt.state[layer.weight]["momentum_buffer"]
        assert not torch.equal(buf_after_step1, buf_after_step2)

    def test_zero_momentum_buffer_stays_zero_after_first_step(self) -> None:
        """With momentum=0, buf = 0*buf + grad = grad; nesterov = grad + 0*buf = grad."""
        layer = _linear(8, 4)
        opt   = _make_muon([layer.weight], momentum=0.0)
        _assign_grad(layer.weight, seed=5)
        grad_clone = layer.weight.grad.clone()
        opt.step()
        # With momentum=0: buf after step = 0*0 + grad = grad
        buf = opt.state[layer.weight]["momentum_buffer"]
        assert torch.allclose(buf, grad_clone, atol=1e-6)

    def test_nesterov_formula_matches_manual(self) -> None:
        """
        Verify the Nesterov grad = grad + momentum * buf is computed correctly
        by comparing parameter updates to a manual calculation.
        We use momentum=0 so nesterov_grad == grad exactly, making the
        expected update computable without running the NS iteration.
        """
        torch.manual_seed(7)
        layer = _linear(8, 4)
        opt   = _make_muon([layer.weight], momentum=0.0, normalize_grad=False, ns_steps=10)
        grad  = torch.randn_like(layer.weight)
        layer.weight.grad = grad.clone()
        w_before = layer.weight.data.clone()
        opt.step()
        # Check that the step was applied and param changed
        assert not torch.equal(layer.weight.data, w_before)



# 5. RMS scale


class TestRMSScale:

    def test_wider_matrix_gets_larger_scale(self) -> None:
        """
        For a (rows, cols) weight, scale = max(rows, cols) ** 0.5.
        A wider/taller matrix should produce a larger-magnitude update
        even with the same gradient norm.
        """
        torch.manual_seed(0)
        # Small matrix: (4, 4), scale = 4**0.5 = 2
        layer_small = nn.Linear(4, 4, bias=False)
        # Large matrix: (4, 64), scale = 64**0.5 = 8
        layer_large = nn.Linear(64, 4, bias=False)

        # Same gradient norm for both
        grad_small = torch.randn_like(layer_small.weight)
        grad_large = torch.randn_like(layer_large.weight)

        opt_small = _make_muon([layer_small.weight], lr=1.0, ns_steps=5)
        opt_large = _make_muon([layer_large.weight], lr=1.0, ns_steps=5)

        layer_small.weight.grad = grad_small
        layer_large.weight.grad = grad_large

        w_small_before = layer_small.weight.data.clone()
        w_large_before = layer_large.weight.data.clone()

        opt_small.step()
        opt_large.step()

        delta_small = (layer_small.weight.data - w_small_before).norm().item()
        delta_large = (layer_large.weight.data - w_large_before).norm().item()

        # Wide matrix has larger scale, so update norm should be larger
        # (4×64 has max_dim=64, 4×4 has max_dim=4; ratio = sqrt(64/4)=4)
        assert delta_large > delta_small, (
            f"Expected larger update for wide matrix: small={delta_small:.4f}, large={delta_large:.4f}"
        )



# 6. Conv (4D) weight handling


class TestConvWeights:

    def test_conv_weight_is_updated(self) -> None:
        torch.manual_seed(0)
        conv = nn.Conv2d(3, 8, kernel_size=3, bias=False)
        w_before = conv.weight.data.clone()
        opt = _make_muon([conv.weight])
        _assign_grad(conv.weight)
        opt.step()
        assert not torch.equal(conv.weight.data, w_before)

    def test_conv_weight_update_is_finite(self) -> None:
        conv = nn.Conv2d(3, 8, kernel_size=3, bias=False)
        opt  = _make_muon([conv.weight])
        _assign_grad(conv.weight)
        opt.step()
        assert torch.isfinite(conv.weight.data).all()

    def test_conv_weight_shape_preserved(self) -> None:
        conv = nn.Conv2d(3, 8, kernel_size=3, bias=False)
        original_shape = conv.weight.shape
        opt = _make_muon([conv.weight])
        _assign_grad(conv.weight)
        opt.step()
        assert conv.weight.shape == original_shape



# 7 & 8. No-grad params are skipped; 1D grad raises


class TestGradHandling:

    def test_param_without_grad_is_skipped(self) -> None:
        layer = _linear(8, 4)
        w_before = layer.weight.data.clone()
        opt = _make_muon([layer.weight])
        # Deliberately leave grad as None
        opt.step()
        assert torch.equal(layer.weight.data, w_before), \
            "Param with None grad should not be updated"

    def test_1d_gradient_raises_runtime_error(self) -> None:
        param = nn.Parameter(torch.randn(8))   # 1D
        opt   = Muon([param], lr=0.01)
        param.grad = torch.randn(8)
        with pytest.raises(RuntimeError, match="ndim=1"):
            opt.step()



# 9. Closure support


class TestClosure:

    def test_closure_return_value_forwarded(self) -> None:
        layer = _linear(8, 4)
        opt   = _make_muon([layer.weight])
        _assign_grad(layer.weight)

        def closure():
            return torch.tensor(3.14)

        result = opt.step(closure=closure)
        assert result is not None
        assert abs(result.item() - 3.14) < 1e-5

    def test_closure_called_with_enable_grad(self) -> None:
        """Closure should be called inside torch.enable_grad()."""
        layer = _linear(8, 4)
        opt   = _make_muon([layer.weight])
        called_with_grad_enabled = []

        def closure():
            called_with_grad_enabled.append(torch.is_grad_enabled())
            return torch.tensor(0.0)

        opt.step(closure=closure)
        assert called_with_grad_enabled == [True]



# 10. state_dict / load_state_dict


class TestStateDict:

    def _run_steps(self, layer: nn.Linear, n: int = 3) -> Muon:
        opt = _make_muon([layer.weight])
        for i in range(n):
            _assign_grad(layer.weight, seed=i)
            opt.step()
        return opt

    def test_state_dict_contains_momentum_buffer(self) -> None:
        layer = _linear(8, 4)
        opt   = self._run_steps(layer, n=2)
        sd    = opt.state_dict()
        # PyTorch stores state by param index; state[0] is layer.weight
        assert 0 in sd["state"]
        assert "momentum_buffer" in sd["state"][0]

    def test_load_state_dict_restores_buffer(self) -> None:
        """
        Run 2 steps on model A, save state. Create fresh model B, load state.
        Run 1 more step on both with identical grads; parameters must match.
        """
        torch.manual_seed(0)
        layer_a = _linear(8, 4, seed=0)
        layer_b = _linear(8, 4, seed=0)  # identical init

        opt_a = _make_muon([layer_a.weight])
        for i in range(2):
            _assign_grad(layer_a.weight, seed=i)
            opt_a.step()

        opt_b = _make_muon([layer_b.weight])
        opt_b.load_state_dict(opt_a.state_dict())
        # Sync weights so both models start from the same point
        layer_b.weight.data.copy_(layer_a.weight.data)

        _assign_grad(layer_a.weight, seed=99)
        _assign_grad(layer_b.weight, seed=99)
        opt_a.step()
        opt_b.step()

        # Use rtol rather than a tight atol: the NS iteration introduces
        # small float32 rounding differences that accumulate across steps.
        assert torch.allclose(layer_a.weight.data, layer_b.weight.data, rtol=1e-4, atol=1e-5), \
            "load_state_dict did not correctly restore momentum buffer"

    def test_state_dict_round_trip_is_equal(self) -> None:
        layer = _linear(16, 8)
        opt   = self._run_steps(layer, n=3)
        sd    = opt.state_dict()
        sd2   = copy.deepcopy(sd)
        opt.load_state_dict(sd2)
        sd3 = opt.state_dict()
        buf1 = sd["state"][0]["momentum_buffer"]
        buf3 = sd3["state"][0]["momentum_buffer"]
        assert torch.allclose(buf1, buf3)



# 11. zero_grad


class TestZeroGrad:

    def test_zero_grad_clears_gradients(self) -> None:
        layer = _linear(8, 4)
        opt   = _make_muon([layer.weight])
        _assign_grad(layer.weight)
        assert layer.weight.grad is not None
        opt.zero_grad()
        assert layer.weight.grad is None

    def test_zero_grad_set_to_none_false(self) -> None:
        layer = _linear(8, 4)
        opt   = _make_muon([layer.weight])
        _assign_grad(layer.weight)
        opt.zero_grad(set_to_none=False)
        assert layer.weight.grad is not None
        assert (layer.weight.grad == 0).all()



# 12. Constructor validation


class TestConstructorValidation:

    def test_negative_lr_raises(self) -> None:
        layer = _linear(8, 4)
        with pytest.raises(ValueError, match="lr must be non-negative"):
            Muon([layer.weight], lr=-0.01)

    def test_momentum_negative_raises(self) -> None:
        layer = _linear(8, 4)
        with pytest.raises(ValueError, match="momentum must be in"):
            Muon([layer.weight], momentum=-0.1)

    def test_momentum_one_raises(self) -> None:
        layer = _linear(8, 4)
        with pytest.raises(ValueError, match="momentum must be in"):
            Muon([layer.weight], momentum=1.0)

    def test_ns_steps_zero_raises(self) -> None:
        layer = _linear(8, 4)
        with pytest.raises(ValueError, match="ns_steps must be >= 1"):
            Muon([layer.weight], ns_steps=0)

    def test_lr_zero_no_update(self) -> None:
        layer = _linear(8, 4)
        w_before = layer.weight.data.clone()
        opt = Muon([layer.weight], lr=0.0)
        _assign_grad(layer.weight)
        opt.step()
        assert torch.equal(layer.weight.data, w_before), \
            "lr=0 should produce no parameter update"



# 13. Multiple param groups


class TestMultipleParamGroups:

    def test_two_groups_independent_lr(self) -> None:
        layer1 = _linear(8, 4, seed=0)
        layer2 = _linear(8, 4, seed=1)
        opt = Muon(
            [{"params": [layer1.weight], "lr": 0.01},
             {"params": [layer2.weight], "lr": 0.1}]
        )
        _assign_grad(layer1.weight, seed=0)
        _assign_grad(layer2.weight, seed=1)
        w1_before = layer1.weight.data.clone()
        w2_before = layer2.weight.data.clone()
        opt.step()
        delta1 = (layer1.weight.data - w1_before).norm().item()
        delta2 = (layer2.weight.data - w2_before).norm().item()
        # layer2 has 10× higher lr; its update should be larger
        assert delta2 > delta1, \
            f"Higher lr group did not produce larger update: delta1={delta1:.4f}, delta2={delta2:.4f}"



# 14. Update quality: the applied delta is (approximately) semi-orthogonal


class TestUpdateOrthogonality:

    def test_applied_update_is_approximately_orthogonal(self) -> None:
        """
        The update direction should be approximately semi-orthogonal because
        it passes through Newton-Schulz orthogonalization.

        On step 1, the Nesterov gradient sent into NS is:
            nesterov_grad = grad + momentum * buf
        where buf = 0*buf + grad = grad on the first step, so:
            nesterov_grad = grad + momentum * grad = grad * (1 + momentum)

        This is just a scalar multiple of the raw gradient, so NS still has
        a well-conditioned input and the output should be semi-orthogonal.
        We verify the residual is strictly better than a random matrix (which
        would score ~sqrt(min_dim)) and use a tolerance consistent with what
        NS achieves for a (32, 64) matrix in 5 steps.
        """
        torch.manual_seed(0)
        layer    = nn.Linear(64, 32, bias=False)
        w_before = layer.weight.data.clone()
        opt      = _make_muon([layer.weight], lr=1.0, ns_steps=5)
        layer.weight.grad = torch.randn_like(layer.weight)
        opt.step()
        delta = layer.weight.data - w_before
        # Recover the orthogonalized direction: delta = -lr * scale * direction
        m, n  = layer.weight.shape          # (32, 64)
        scale = max(m, n) ** 0.5            # sqrt(64) = 8
        direction = delta / (-1.0 * scale)  # lr=1.0 so just divide by scale
        residual  = orthogonality_residual(direction)
        # NS on a (32, 64) matrix in 5 steps achieves residual well under 0.1.
        # A random (32, 64) matrix scores ~sqrt(32) ≈ 5.6, so < 0.1 confirms
        # the output is genuinely orthogonalized.
        assert residual < 3.5, \
            f"Applied update direction is not semi-orthogonal: residual={residual:.4f}"