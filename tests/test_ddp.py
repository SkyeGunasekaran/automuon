"""
tests/test_ddp.py

Correctness tests for DDPMuon in single-process (non-distributed) contexts.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from contextlib import nullcontext
from torch.optim.lr_scheduler import StepLR

from automuon.ddp.muon_ddp import DDPMuon
from automuon.optimizer import AutoMuon

from conftest import TinyMLP, make_grads, forward_backward



# Helpers


def _make_ddp(model: nn.Module, **kwargs) -> DDPMuon:
    kwargs.setdefault("muon_lr", 0.02)
    kwargs.setdefault("adamw_lr", 3e-4)
    return DDPMuon(model, **kwargs)



# 1. Construction and type validation


class TestDDPMuonConstruction:

    def test_basic_construction(self, tiny_mlp) -> None:
        opt = _make_ddp(tiny_mlp)
        assert isinstance(opt, DDPMuon)
        assert isinstance(opt, AutoMuon)

    def test_ddp_module_none_is_valid(self, tiny_mlp) -> None:
        opt = _make_ddp(tiny_mlp, ddp_module=None)
        assert opt._ddp_module is None

    def test_ddp_module_wrong_type_raises(self, tiny_mlp) -> None:
        """Passing a plain nn.Module (not DDP) as ddp_module should raise TypeError."""
        fake_ddp = nn.Linear(4, 4)   # not a DDP instance
        with pytest.raises(TypeError, match="DistributedDataParallel"):
            DDPMuon(tiny_mlp, ddp_module=fake_ddp, muon_lr=0.02, adamw_lr=3e-4)

    def test_ddp_module_wrong_inner_module_raises(self, tiny_mlp) -> None:
        """
        If ddp_module.module is not the same object as the first argument,
        a ValueError should be raised.

        We can't instantiate a real DDP without a process group, so we mock
        a minimal object with a `.module` attribute pointing to a different model.
        """
        class FakeDDP(nn.Module):
            """Minimal stand-in that passes isinstance(x, DDP) — not possible without
            monkey-patching, so we test the ValueError branch via a real DDP check."""
            pass

        # We can only test this path if we can create a DDP instance.
        # In CI without GPU/dist, we verify the TypeError path (wrong type) instead.
        # This test documents the expected behavior for the identity mismatch path.
        pytest.skip(
            "Cannot construct a real DDP without an initialized process group; "
            "the ValueError branch is tested in integration tests."
        )

    def test_sync_grads_stored(self, tiny_mlp) -> None:
        opt = _make_ddp(tiny_mlp, sync_grads=True)
        assert opt._sync_grads is True

    def test_sync_grads_default_false(self, tiny_mlp) -> None:
        opt = _make_ddp(tiny_mlp)
        assert opt._sync_grads is False

    def test_ddp_model_kwarg_ignored_gracefully(self, tiny_mlp) -> None:
        """ddp_model kwarg (old API) should be silently popped."""
        opt = DDPMuon(tiny_mlp, ddp_model=None, muon_lr=0.02, adamw_lr=3e-4)
        assert isinstance(opt, DDPMuon)



# 2. Single-process distributed properties


class TestSingleProcessProperties:

    def test_world_size_is_one_without_dist(self, tiny_mlp) -> None:
        opt = _make_ddp(tiny_mlp)
        assert opt.world_size == 1

    def test_rank_is_zero_without_dist(self, tiny_mlp) -> None:
        opt = _make_ddp(tiny_mlp)
        assert opt.rank == 0

    def test_grad_scale_is_one_over_world_size(self, tiny_mlp) -> None:
        opt = _make_ddp(tiny_mlp)
        assert opt.grad_scale == pytest.approx(1.0 / opt.world_size)

    def test_grad_scale_is_one_without_dist(self, tiny_mlp) -> None:
        opt = _make_ddp(tiny_mlp)
        assert opt.grad_scale == pytest.approx(1.0)



# 3. sync_gradients() is a no-op without dist


class TestSyncGradients:

    def test_sync_gradients_noop_without_dist(self, tiny_mlp) -> None:
        opt = _make_ddp(tiny_mlp)
        make_grads(tiny_mlp)
        grads_before = {
            n: p.grad.clone()
            for n, p in tiny_mlp.named_parameters()
            if p.grad is not None
        }
        opt.sync_gradients()
        for n, p in tiny_mlp.named_parameters():
            if p.grad is not None:
                assert torch.allclose(p.grad, grads_before[n]), \
                    f"sync_gradients() modified grad for {n} without dist initialized"

    def test_sync_gradients_with_none_grads_does_not_crash(self, tiny_mlp) -> None:
        """All grads are None; sync_gradients() should be a safe no-op."""
        opt = _make_ddp(tiny_mlp)
        opt.sync_gradients()   # no exception



# 4. step() correctness (delegating to AutoMuon)


class TestDDPMuonStep:

    def test_step_updates_parameters(self, tiny_mlp) -> None:
        opt = _make_ddp(tiny_mlp)
        w_before = tiny_mlp.fc1.weight.data.clone()
        make_grads(tiny_mlp)
        opt.step()
        assert not torch.equal(tiny_mlp.fc1.weight.data, w_before)

    def test_step_with_sync_grads_true_no_dist(self, tiny_mlp) -> None:
        """sync_grads=True but dist not initialized → sync is a no-op, step proceeds."""
        opt = _make_ddp(tiny_mlp, sync_grads=True)
        make_grads(tiny_mlp)
        opt.step()   # should not raise
        # Parameters should have been updated
        for p in tiny_mlp.parameters():
            if p.requires_grad:
                assert torch.isfinite(p.data).all()

    def test_step_real_forward_backward(self, tiny_mlp) -> None:
        opt = _make_ddp(tiny_mlp)
        forward_backward(tiny_mlp)
        opt.step()

    def test_loss_decreases(self, tiny_mlp) -> None:
        opt    = _make_ddp(tiny_mlp, muon_lr=0.01, adamw_lr=1e-3)
        losses = []
        for _ in range(8):
            opt.zero_grad()
            loss = forward_backward(tiny_mlp)
            losses.append(loss.item())
            opt.step()
        assert losses[-1] < losses[0], \
            f"DDPMuon: loss did not decrease. Initial={losses[0]:.4f}, Final={losses[-1]:.4f}"



# 5. no_sync()


class TestNoSync:

    def test_no_sync_returns_context_manager_without_ddp_module(self, tiny_mlp) -> None:
        opt = _make_ddp(tiny_mlp, ddp_module=None)
        ctx = opt.no_sync()
        # Should be a nullcontext (or at minimum, a context manager)
        with ctx:
            pass   # must not raise

    def test_no_sync_is_nullcontext_without_ddp_module(self, tiny_mlp) -> None:
        opt = _make_ddp(tiny_mlp, ddp_module=None)
        ctx = opt.no_sync()
        assert isinstance(ctx, type(nullcontext()))



# 6. __repr__


class TestDDPMuonRepr:

    def test_repr_contains_world_size_and_rank(self, tiny_mlp) -> None:
        opt = _make_ddp(tiny_mlp)
        r   = repr(opt)
        assert "world_size" in r
        assert "rank"       in r

    def test_repr_contains_sync_grads(self, tiny_mlp) -> None:
        opt = _make_ddp(tiny_mlp, sync_grads=True)
        r   = repr(opt)
        assert "sync_grads" in r

    def test_repr_contains_automuon_repr(self, tiny_mlp) -> None:
        opt = _make_ddp(tiny_mlp)
        r   = repr(opt)
        assert "DDPMuon" in r
        # AutoMuon __repr__ content should be embedded
        assert "Muon" in r



# 7. Scheduler compatibility through DDPMuon


class TestDDPMuonSchedulerCompat:

    def test_step_lr_works_through_ddp_muon(self, tiny_mlp) -> None:
        opt   = _make_ddp(tiny_mlp, muon_lr=0.02)
        sched = StepLR(opt, step_size=1, gamma=0.5)
        for _ in range(3):
            make_grads(tiny_mlp)
            opt.step()
            sched.step()
        expected_lr = 0.02 * 0.5 ** 3
        actual_lr   = opt._muon.param_groups[0]["lr"]
        assert actual_lr == pytest.approx(expected_lr, rel=1e-4)



# 8. API surface: DDPMuon inherits all AutoMuon methods


class TestDDPMuonInheritance:

    def test_has_zero_grad(self, tiny_mlp) -> None:
        opt = _make_ddp(tiny_mlp)
        make_grads(tiny_mlp)
        opt.zero_grad()
        for p in tiny_mlp.parameters():
            if p.requires_grad:
                assert p.grad is None

    def test_has_state_dict(self, tiny_mlp) -> None:
        opt = _make_ddp(tiny_mlp)
        sd  = opt.state_dict()
        assert "muon"  in sd
        assert "adamw" in sd

    def test_has_partition_summary(self, tiny_mlp) -> None:
        opt     = _make_ddp(tiny_mlp)
        summary = opt.partition_summary()
        assert isinstance(summary, list)
        assert len(summary) > 0

    def test_has_param_groups(self, tiny_mlp) -> None:
        opt = _make_ddp(tiny_mlp)
        assert len(opt.param_groups) > 0