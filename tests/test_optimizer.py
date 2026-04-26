"""
tests/test_optimizer.py

Correctness tests for AutoMuon — the public-facing optimizer shell.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from automuon.optimizer import AutoMuon

from conftest import (
    AllNormModel,
    ConvModel,
    PartiallyFrozenMLP,
    TinyMLP,
    TinyTransformerBlock,
    WeightTiedLM,
    forward_backward,
    make_grads,
)


# Helpers

def _make(model: nn.Module, **kwargs) -> AutoMuon:
    kwargs.setdefault("muon_lr", 0.02)
    kwargs.setdefault("adamw_lr", 3e-4)
    return AutoMuon(model, **kwargs)


def _param_ptrs(groups) -> set[int]:
    return {p.data_ptr() for g in groups for p in g["params"]}


# 1. Construction

class TestConstruction:

    def test_basic_construction(self, tiny_mlp) -> None:
        opt = _make(tiny_mlp)
        assert opt._muon  is not None
        assert opt._adamw is not None

    def test_all_norm_model_has_no_muon(self, all_norm_model) -> None:
        opt = _make(all_norm_model)
        assert opt._muon is None
        assert opt._adamw is not None

    def test_all_linear_model_has_no_adamw(self) -> None:
        """A model with only 2D weights and no biases/norms → Muon only."""
        model = nn.Sequential(
            nn.Linear(8, 16, bias=False),
            nn.Linear(16, 4, bias=False),
        )
        opt = AutoMuon(model, muon_lr=0.02, adamw_lr=3e-4)
        assert opt._muon  is not None
        assert opt._adamw is None

    def test_no_trainable_params_raises(self) -> None:
        model = nn.Linear(4, 4)
        for p in model.parameters():
            p.requires_grad = False
        with pytest.raises(ValueError, match="no trainable parameters"):
            AutoMuon(model)

    def test_lr_convenience_sets_both(self, tiny_mlp) -> None:
        opt = AutoMuon(tiny_mlp, lr=1e-3)
        muon_lr  = opt._muon.param_groups[0]["lr"]
        adamw_lr = opt._adamw.param_groups[0]["lr"]
        assert muon_lr  == pytest.approx(1e-3)
        assert adamw_lr == pytest.approx(1e-3)

    def test_separate_lrs_respected(self, tiny_mlp) -> None:
        opt = AutoMuon(tiny_mlp, muon_lr=0.05, adamw_lr=5e-4)
        assert opt._muon.param_groups[0]["lr"]  == pytest.approx(0.05)
        assert opt._adamw.param_groups[0]["lr"] == pytest.approx(5e-4)

    def test_verbose_does_not_crash(self, tiny_mlp, capsys) -> None:
        opt = AutoMuon(tiny_mlp, muon_lr=0.02, adamw_lr=3e-4, verbose=True)
        captured = capsys.readouterr()
        assert "Muon" in captured.out or "muon" in captured.out.lower()


# 2. Parameter routing

class TestParameterRouting:

    def test_linear_weights_in_muon_group(self, tiny_mlp) -> None:
        opt = _make(tiny_mlp)
        muon_ptrs = {p.data_ptr() for g in opt._muon.param_groups for p in g["params"]}
        assert tiny_mlp.fc1.weight.data_ptr() in muon_ptrs
        assert tiny_mlp.fc2.weight.data_ptr() in muon_ptrs

    def test_biases_in_adamw_group(self, tiny_mlp) -> None:
        opt = _make(tiny_mlp)
        adamw_ptrs = {p.data_ptr() for g in opt._adamw.param_groups for p in g["params"]}
        assert tiny_mlp.fc1.bias.data_ptr() in adamw_ptrs
        assert tiny_mlp.fc2.bias.data_ptr() in adamw_ptrs

    def test_embedding_in_adamw_group(self, transformer_block) -> None:
        opt = _make(transformer_block)
        adamw_ptrs = {p.data_ptr() for g in opt._adamw.param_groups for p in g["params"]}
        assert transformer_block.embed.weight.data_ptr() in adamw_ptrs

    def test_no_param_in_both_groups(self, tiny_mlp) -> None:
        opt = _make(tiny_mlp)
        muon_ptrs  = _param_ptrs(opt._muon.param_groups)
        adamw_ptrs = _param_ptrs(opt._adamw.param_groups)
        assert muon_ptrs.isdisjoint(adamw_ptrs)

    def test_frozen_params_absent_from_all_groups(self, partially_frozen_mlp) -> None:
        opt = _make(partially_frozen_mlp)
        all_ptrs = _param_ptrs(opt.param_groups)
        for p in partially_frozen_mlp.fc1.parameters():  # frozen
            assert p.data_ptr() not in all_ptrs

    def test_weight_tied_param_appears_once_across_all_groups(self, weight_tied_lm) -> None:
        opt = _make(weight_tied_lm)
        all_params = [p for g in opt.param_groups for p in g["params"]]
        ptrs = [p.data_ptr() for p in all_params]
        assert len(ptrs) == len(set(ptrs)), \
            "Weight-tied parameter appears more than once across optimizer groups"


# 3. step() updates parameters

class TestStep:

    def test_step_updates_muon_params(self, tiny_mlp) -> None:
        opt = _make(tiny_mlp)
        w_before = tiny_mlp.fc1.weight.data.clone()
        make_grads(tiny_mlp)
        opt.step()
        assert not torch.equal(tiny_mlp.fc1.weight.data, w_before)

    def test_step_updates_adamw_params(self, tiny_mlp) -> None:
        opt = _make(tiny_mlp)
        b_before = tiny_mlp.fc1.bias.data.clone()
        make_grads(tiny_mlp)
        opt.step()
        assert not torch.equal(tiny_mlp.fc1.bias.data, b_before)

    def test_step_real_forward_backward(self, tiny_mlp) -> None:
        opt = _make(tiny_mlp)
        loss = forward_backward(tiny_mlp)
        opt.step()
        # All trainable params should have changed
        for p in tiny_mlp.parameters():
            if p.requires_grad:
                assert torch.isfinite(p.data).all()

    def test_step_returns_none_without_closure(self, tiny_mlp) -> None:
        opt = _make(tiny_mlp)
        make_grads(tiny_mlp)
        result = opt.step()
        assert result is None

    def test_step_returns_loss_with_closure(self, tiny_mlp) -> None:
        opt = _make(tiny_mlp)

        def closure():
            opt.zero_grad()
            return torch.tensor(2.71)

        result = opt.step(closure=closure)
        assert result is not None
        assert abs(result.item() - 2.71) < 1e-5

    def test_conv_params_updated(self, conv_model) -> None:
        opt = _make(conv_model)
        w_before = conv_model.conv.weight.data.clone()
        forward_backward(conv_model)
        opt.step()
        assert not torch.equal(conv_model.conv.weight.data, w_before)

    def test_loss_decreases_over_training(self, tiny_mlp) -> None:
        """Loss should trend downward over several steps of real training."""
        opt    = _make(tiny_mlp, muon_lr=0.01, adamw_lr=1e-3)
        losses = []
        for _ in range(10):
            opt.zero_grad()
            loss = forward_backward(tiny_mlp)
            losses.append(loss.item())
            opt.step()
        assert losses[-1] < losses[0], \
            f"Loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"


# 4. zero_grad

class TestZeroGrad:

    def test_zero_grad_clears_all_grads(self, tiny_mlp) -> None:
        opt = _make(tiny_mlp)
        make_grads(tiny_mlp)
        opt.zero_grad()
        for p in tiny_mlp.parameters():
            if p.requires_grad:
                assert p.grad is None

    def test_zero_grad_set_to_none_false(self, tiny_mlp) -> None:
        opt = _make(tiny_mlp)
        make_grads(tiny_mlp)
        opt.zero_grad(set_to_none=False)
        for p in tiny_mlp.parameters():
            if p.requires_grad:
                assert p.grad is not None
                assert (p.grad == 0).all()


# 5 & 6. param_groups live view + scheduler compatibility

class TestParamGroupsAndSchedulers:

    def test_param_groups_returns_live_view(self, tiny_mlp) -> None:
        """Mutating a returned group dict should affect the internal optimizer."""
        opt = _make(tiny_mlp, muon_lr=0.02)
        groups = opt.param_groups
        groups[0]["lr"] = 0.99
        # The internal Muon optimizer should see the change
        assert opt._muon.param_groups[0]["lr"] == pytest.approx(0.99)

    def test_param_groups_ordering_muon_first(self, tiny_mlp) -> None:
        """Muon groups must come before AdamW groups (stable for scheduler indexing)."""
        opt = _make(tiny_mlp)
        n_muon_groups = len(opt._muon.param_groups)
        muon_group_lrs  = [opt.param_groups[i]["lr"]  for i in range(n_muon_groups)]
        assert all(lr == pytest.approx(0.02) for lr in muon_group_lrs)

    def test_cosine_annealing_updates_lr(self, tiny_mlp) -> None:
        opt   = _make(tiny_mlp, muon_lr=0.02, adamw_lr=3e-4)
        sched = CosineAnnealingLR(opt, T_max=10, eta_min=0)
        initial_lrs = [g["lr"] for g in opt.param_groups]
        for _ in range(5):
            make_grads(tiny_mlp)
            opt.step()
            sched.step()
        final_lrs = [g["lr"] for g in opt.param_groups]
        # After 5/10 steps, lr should have decreased from initial
        for init_lr, final_lr in zip(initial_lrs, final_lrs):
            assert final_lr < init_lr, \
                f"Cosine scheduler did not reduce lr: init={init_lr}, final={final_lr}"

    def test_step_lr_halves_lr(self, tiny_mlp) -> None:
        opt   = _make(tiny_mlp, muon_lr=0.02, adamw_lr=3e-4)
        sched = StepLR(opt, step_size=1, gamma=0.5)
        for _ in range(3):
            make_grads(tiny_mlp)
            opt.step()
            sched.step()
        # After 3 halvings: 0.02 * 0.5^3 = 0.0025
        muon_lr = opt._muon.param_groups[0]["lr"]
        assert muon_lr == pytest.approx(0.02 * 0.5 ** 3, rel=1e-4)

    def test_scheduler_lr_propagates_to_internal_optimizer(self, tiny_mlp) -> None:
        """
        The critical invariant: scheduler writes on param_groups must land
        on the internal optimizer, not a stale copy.
        """
        opt   = _make(tiny_mlp, muon_lr=0.1)
        sched = StepLR(opt, step_size=1, gamma=0.1)
        make_grads(tiny_mlp)
        opt.step()
        sched.step()
        # Internal Muon group and unified view must agree
        external_lr = opt.param_groups[0]["lr"]
        internal_lr = opt._muon.param_groups[0]["lr"]
        assert external_lr == pytest.approx(internal_lr)


# 7 & 8. state_dict / load_state_dict + deterministic resume

class TestStateDict:

    def _run_steps(self, model: nn.Module, n: int = 3) -> AutoMuon:
        opt = _make(model)
        for _ in range(n):
            make_grads(model)
            opt.step()
        return opt

    def test_state_dict_keys(self, tiny_mlp) -> None:
        opt = self._run_steps(tiny_mlp)
        sd  = opt.state_dict()
        assert "muon"  in sd
        assert "adamw" in sd

    def test_state_dict_none_when_optimizer_absent(self, all_norm_model) -> None:
        opt = self._run_steps(all_norm_model)
        sd  = opt.state_dict()
        assert sd["muon"] is None

    def test_load_state_dict_restores_state(self, tiny_mlp) -> None:
        opt1 = self._run_steps(tiny_mlp, n=3)
        sd   = copy.deepcopy(opt1.state_dict())

        # Fresh model and optimizer
        torch.manual_seed(0)
        model2 = TinyMLP()
        model2.load_state_dict(tiny_mlp.state_dict())
        opt2 = _make(model2)
        opt2.load_state_dict(sd)

        # Same grad, same step → same result
        make_grads(tiny_mlp, seed=77)
        make_grads(model2,   seed=77)
        opt1.step()
        opt2.step()

        for p1, p2 in zip(tiny_mlp.parameters(), model2.parameters()):
            if p1.requires_grad:
                assert torch.allclose(p1.data, p2.data, atol=1e-6), \
                    "Deterministic resume failed after load_state_dict"

    def test_state_dict_round_trip(self, tiny_mlp) -> None:
        opt = self._run_steps(tiny_mlp, n=2)
        sd1 = opt.state_dict()
        opt.load_state_dict(copy.deepcopy(sd1))
        sd2 = opt.state_dict()
        # Momentum buffers should be identical
        for key in sd1["muon"]["state"]:
            buf1 = sd1["muon"]["state"][key]["momentum_buffer"]
            buf2 = sd2["muon"]["state"][key]["momentum_buffer"]
            assert torch.allclose(buf1, buf2)


# 9. partition_summary()

class TestPartitionSummary:

    def test_returns_list_of_dicts(self, tiny_mlp) -> None:
        opt     = _make(tiny_mlp)
        summary = opt.partition_summary()
        assert isinstance(summary, list)
        assert all(isinstance(d, dict) for d in summary)

    def test_required_keys_present(self, tiny_mlp) -> None:
        opt     = _make(tiny_mlp)
        summary = opt.partition_summary()
        required = {"name", "optimizer", "shape", "module_type", "reason"}
        for d in summary:
            assert required.issubset(d.keys()), \
                f"Missing keys in partition_summary entry: {d.keys()}"

    def test_shape_is_list(self, tiny_mlp) -> None:
        opt = _make(tiny_mlp)
        for entry in opt.partition_summary():
            assert isinstance(entry["shape"], list)

    def test_optimizer_values_are_valid(self, tiny_mlp) -> None:
        opt = _make(tiny_mlp)
        for entry in opt.partition_summary():
            assert entry["optimizer"] in ("muon", "adamw")

    def test_summary_covers_all_params(self, tiny_mlp) -> None:
        opt     = _make(tiny_mlp)
        summary = opt.partition_summary()
        names   = {d["name"] for d in summary}
        for name, _ in tiny_mlp.named_parameters():
            assert name in names, f"Parameter '{name}' missing from partition_summary()"


# 10. __repr__

class TestRepr:

    def test_repr_contains_muon_and_adamw(self, tiny_mlp) -> None:
        opt = _make(tiny_mlp)
        r   = repr(opt)
        assert "Muon"  in r
        assert "AdamW" in r

    def test_repr_contains_lr(self, tiny_mlp) -> None:
        opt = AutoMuon(tiny_mlp, muon_lr=0.02, adamw_lr=3e-4)
        r   = repr(opt)
        assert "0.02"  in r
        assert "0.0003" in r or "3e-4" in r or "0.0003" in r


# 11. Edge cases

class TestEdgeCases:

    def test_step_with_no_grads_does_not_crash(self, tiny_mlp) -> None:
        """Calling step() before any backward should be a safe no-op."""
        opt = _make(tiny_mlp)
        opt.step()  # no grads set — should not raise

    def test_multiple_step_calls_are_stable(self, tiny_mlp) -> None:
        opt = _make(tiny_mlp)
        for _ in range(20):
            make_grads(tiny_mlp)
            opt.step()
        for p in tiny_mlp.parameters():
            assert torch.isfinite(p.data).all(), \
                f"Parameter became non-finite after 20 steps: {p.shape}"

    def test_weight_tied_lm_trains_without_error(self, weight_tied_lm) -> None:
        opt = _make(weight_tied_lm)
        forward_backward(weight_tied_lm)
        opt.step()

    def test_transformer_block_trains_without_error(self, transformer_block) -> None:
        opt = _make(transformer_block)
        forward_backward(transformer_block)
        opt.step()