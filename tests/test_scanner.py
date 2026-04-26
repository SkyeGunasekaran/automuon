"""
tests/test_scanner.py

Correctness tests for the AutoMuon parameter scanner.

The scanner is the most important correctness component: wrong routing
silently degrades training. Every routing rule in _classify() has dedicated
tests here, and the overall scan() / partition() API is tested end-to-end.

Coverage matrix:
  Rule 1 — weight-tied parameters go to adamw (canonical name wins)
  Rule 2 — non-floating-point tensors go to adamw
  Rule 3 — frozen (requires_grad=False) parameters go to adamw
  Rule 4 — module-type exclusions: Embedding, EmbeddingBag, all norm variants
  Rule 5 — name-suffix exclusions: .bias, _bias
  Rule 6 — shape check: ndim < 2 → adamw
  Rule 7 — numel floor: numel < MIN_NUMEL_FOR_MUON → adamw
  Rule 8 — passing all checks → muon

Additional:
  - scan() returns one entry per unique tensor (no duplicates)
  - partition() excludes frozen params from both lists
  - ScannedParameter fields are populated correctly
  - Deep nested modules are traversed correctly
  - Subclasses of excluded module types are caught (isinstance check)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from automuon.backends.scanner import (
    ADAMW_MODULE_TYPES,
    ADAMW_NAME_SUFFIXES,
    MIN_NUMEL_FOR_MUON,
    ScannedParameter,
    _classify,
    partition,
    scan,
)

from conftest import (
    AllNormModel,
    ConvModel,
    PartiallyFrozenMLP,
    TinyMLP,
    TinyTransformerBlock,
    WeightTiedLM,
)


# Helpers

def scanned_by_name(scanned: list[ScannedParameter]) -> dict[str, ScannedParameter]:
    return {s.name: s for s in scanned}


def names_for_tag(scanned: list[ScannedParameter], tag: str) -> set[str]:
    return {s.name for s in scanned if s.optimizer == tag}


# Rule 8 (baseline) — 2D Linear weights → muon

class TestLinearWeightGoesToMuon:

    def test_linear_weight_is_muon(self, tiny_mlp: TinyMLP) -> None:
        result = scanned_by_name(scan(tiny_mlp))
        assert result["fc1.weight"].optimizer == "muon"
        assert result["fc2.weight"].optimizer == "muon"

    def test_linear_weight_reason_contains_2d(self, tiny_mlp: TinyMLP) -> None:
        result = scanned_by_name(scan(tiny_mlp))
        assert "2D+" in result["fc1.weight"].reason

    def test_linear_weight_shape_recorded(self, tiny_mlp: TinyMLP) -> None:
        result = scanned_by_name(scan(tiny_mlp))
        assert result["fc1.weight"].shape == tiny_mlp.fc1.weight.shape

    def test_module_type_recorded_as_linear(self, tiny_mlp: TinyMLP) -> None:
        result = scanned_by_name(scan(tiny_mlp))
        assert result["fc1.weight"].module_type == "Linear"


# Rule 5 — name-suffix exclusions (.bias, _bias)

class TestNameSuffixExclusions:

    def test_dot_bias_goes_to_adamw(self, tiny_mlp: TinyMLP) -> None:
        result = scanned_by_name(scan(tiny_mlp))
        assert result["fc1.bias"].optimizer == "adamw"
        assert result["fc2.bias"].optimizer == "adamw"

    def test_dot_bias_reason_mentions_suffix(self, tiny_mlp: TinyMLP) -> None:
        result = scanned_by_name(scan(tiny_mlp))
        assert ".bias" in result["fc1.bias"].reason

    def test_underscore_bias_suffix_excluded(self) -> None:
        """A 2D parameter whose name ends in '_bias' must go to AdamW."""
        class ModuleWithUnderscoreBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight   = nn.Parameter(torch.randn(8, 8))
                self.my_bias  = nn.Parameter(torch.randn(8, 8))  # doesn't end in _bias
                self.rel_bias = nn.Parameter(torch.randn(4, 4))  # ends in _bias? No, 'rel_bias' ends in 'bias' but suffix is '_bias'

            def forward(self, x): return x

        # Create a module where the parameter name literally ends in _bias
        class StrictUnderscoreBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(8, 8))

            def forward(self, x): return x

        m = StrictUnderscoreBias()
        # Register a parameter with _bias suffix via register_parameter
        m.register_parameter("proj_bias", nn.Parameter(torch.randn(8, 8)))
        result = scanned_by_name(scan(m))
        assert result["proj_bias"].optimizer == "adamw", (
            "2D parameter ending in '_bias' should go to AdamW"
        )
        assert "_bias" in result["proj_bias"].reason

    def test_adamw_name_suffixes_constant(self) -> None:
        """The suffix tuple must contain the documented values."""
        assert ".bias" in ADAMW_NAME_SUFFIXES
        assert "_bias" in ADAMW_NAME_SUFFIXES


# Rule 6 — shape check: ndim < 2 → adamw

class TestShapeCheck:

    def test_1d_param_goes_to_adamw(self) -> None:
        class OneDModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(8))  # 1D, no bias suffix
            def forward(self, x): return x * self.scale

        result = scanned_by_name(scan(OneDModel()))
        assert result["scale"].optimizer == "adamw"
        assert "ndim=1" in result["scale"].reason

    def test_layernorm_weight_and_bias_are_adamw(self, transformer_block) -> None:
        result = scanned_by_name(scan(transformer_block))
        assert result["norm.weight"].optimizer == "adamw"
        assert result["norm.bias"].optimizer   == "adamw"


# Rule 7 — numel floor

class TestNumelFloor:

    def test_1x1_matrix_goes_to_adamw(self) -> None:
        class ScalarGate(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Parameter(torch.ones(1, 1))  # numel == 1 < MIN_NUMEL_FOR_MUON (2)
            def forward(self, x): return x * self.gate

        result = scanned_by_name(scan(ScalarGate()))
        assert result["gate"].optimizer == "adamw"
        assert "numel" in result["gate"].reason

    def test_min_numel_constant(self) -> None:
        assert MIN_NUMEL_FOR_MUON >= 2

    def test_2_element_matrix_is_muon_eligible(self) -> None:
        """A (1, 2) matrix has numel == MIN_NUMEL_FOR_MUON; should be muon."""
        class TwoElementParam(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(1, 2))
            def forward(self, x): return x

        result = scanned_by_name(scan(TwoElementParam()))
        assert result["weight"].optimizer == "muon"


# Rule 4 — module-type exclusions

class TestModuleTypeExclusions:

    def test_embedding_weight_goes_to_adamw(self, transformer_block) -> None:
        result = scanned_by_name(scan(transformer_block))
        assert result["embed.weight"].optimizer == "adamw"

    def test_embedding_reason_mentions_module_type(self, transformer_block) -> None:
        result = scanned_by_name(scan(transformer_block))
        assert "Embedding" in result["embed.weight"].reason

    @pytest.mark.parametrize("norm_cls,in_features", [
        (nn.LayerNorm,    8),
        (nn.GroupNorm,    None),   # special-cased below
        (nn.BatchNorm1d,  8),
        (nn.BatchNorm2d,  8),
        (nn.BatchNorm3d,  8),
        (nn.InstanceNorm1d, 8),
        (nn.InstanceNorm2d, 8),
        (nn.InstanceNorm3d, 8),
    ])
    def test_norm_parameters_go_to_adamw(self, norm_cls, in_features) -> None:
        if norm_cls is nn.GroupNorm:
            norm = norm_cls(num_groups=2, num_channels=8)
        else:
            norm = norm_cls(in_features)
        scanned = scan(norm)
        for s in scanned:
            assert s.optimizer == "adamw", (
                f"{norm_cls.__name__}.{s.name} was routed to {s.optimizer}, expected adamw"
            )

    def test_rmsnorm_goes_to_adamw(self) -> None:
        """nn.RMSNorm was added in PyTorch 2.4; skip if unavailable."""
        if not hasattr(nn, "RMSNorm"):
            pytest.skip("nn.RMSNorm not available in this PyTorch version")
        norm = nn.RMSNorm(8)
        scanned = scan(norm)
        for s in scanned:
            assert s.optimizer == "adamw"

    def test_embedding_bag_goes_to_adamw(self) -> None:
        emb = nn.EmbeddingBag(32, 16)
        scanned = scan(emb)
        for s in scanned:
            assert s.optimizer == "adamw"

    def test_subclass_of_embedding_excluded(self) -> None:
        """isinstance check means subclasses of excluded types are also excluded."""
        class MyEmbedding(nn.Embedding):
            pass

        model = MyEmbedding(32, 16)
        scanned = scan(model)
        for s in scanned:
            assert s.optimizer == "adamw", (
                f"Subclass of nn.Embedding should be excluded, "
                f"but {s.name} went to {s.optimizer}"
            )

    def test_adamw_module_types_constant_contains_key_types(self) -> None:
        assert nn.Embedding    in ADAMW_MODULE_TYPES
        assert nn.EmbeddingBag in ADAMW_MODULE_TYPES
        assert nn.LayerNorm    in ADAMW_MODULE_TYPES
        assert nn.BatchNorm1d  in ADAMW_MODULE_TYPES


# Rule 3 — frozen parameters

class TestFrozenParameters:

    def test_frozen_param_tagged_adamw(self, partially_frozen_mlp) -> None:
        result = scanned_by_name(scan(partially_frozen_mlp))
        # fc1 is frozen; its weight is 2D but should be tagged adamw
        assert result["fc1.weight"].optimizer == "adamw"
        assert "frozen" in result["fc1.weight"].reason

    def test_frozen_param_excluded_from_partition(self, partially_frozen_mlp) -> None:
        scanned = scan(partially_frozen_mlp)
        muon_params, adamw_params = partition(scanned)
        all_ptrs = {p.data_ptr() for p in muon_params + adamw_params}
        for p in partially_frozen_mlp.fc1.parameters():
            assert p.data_ptr() not in all_ptrs, "Frozen param must not appear in any optimizer group"

    def test_trainable_params_still_partitioned(self, partially_frozen_mlp) -> None:
        scanned = scan(partially_frozen_mlp)
        muon_params, adamw_params = partition(scanned)
        muon_ptrs  = {p.data_ptr() for p in muon_params}
        adamw_ptrs = {p.data_ptr() for p in adamw_params}
        # fc2.weight is 2D and trainable → muon; fc2.bias → adamw
        assert partially_frozen_mlp.fc2.weight.data_ptr() in muon_ptrs
        assert partially_frozen_mlp.fc2.bias.data_ptr()   in adamw_ptrs


# Rule 2 — non-floating-point tensors

class TestNonFloatingPoint:

    def test_integer_param_goes_to_adamw(self) -> None:
        class IntParam(nn.Module):
            def __init__(self):
                super().__init__()
                # integer parameter (e.g. quantization step)
                self.scale = nn.Parameter(
                    torch.zeros(4, 4, dtype=torch.int32), requires_grad=False
                )
                self.weight = nn.Parameter(torch.randn(4, 4))
            def forward(self, x): return x

        result = scanned_by_name(scan(IntParam()))
        assert result["scale"].optimizer == "adamw"
        assert "non-floating-point" in result["scale"].reason
        # float weight is still eligible
        assert result["weight"].optimizer == "muon"


# Rule 1 — weight tying

class TestWeightTying:

    def test_tied_param_appears_exactly_once(self, weight_tied_lm) -> None:
        """scan() deduplicates by data_ptr; tied param must appear once."""
        scanned = scan(weight_tied_lm)
        names = [s.name for s in scanned]
        # embed.weight and lm_head.weight share a tensor; only one should appear
        both_present = "embed.weight" in names and "lm_head.weight" in names
        assert not both_present, (
            "Weight-tied tensor appeared twice in scan() output"
        )

    def test_tied_param_counted_once_in_partition(self, weight_tied_lm) -> None:
        scanned = scan(weight_tied_lm)
        muon_p, adamw_p = partition(scanned)
        # Embedding weight is adamw; it must appear at most once
        embed_weight = weight_tied_lm.embed.weight
        muon_count  = sum(1 for p in muon_p  if p is embed_weight)
        adamw_count = sum(1 for p in adamw_p if p is embed_weight)
        assert muon_count + adamw_count <= 1

    def test_canonical_entry_gets_correct_tag(self, weight_tied_lm) -> None:
        """The first-seen name owns the tensor; it's embed.weight → adamw."""
        scanned = scan(weight_tied_lm)
        result = scanned_by_name(scanned)
        # embed.weight is first in named_parameters(); should be the canonical name
        assert result["embed.weight"].optimizer == "adamw"


# scan() API correctness

class TestScanAPI:

    def test_returns_list_of_scanned_parameters(self, tiny_mlp) -> None:
        scanned = scan(tiny_mlp)
        assert isinstance(scanned, list)
        assert all(isinstance(s, ScannedParameter) for s in scanned)

    def test_no_duplicate_tensors(self, tiny_mlp) -> None:
        scanned = scan(tiny_mlp)
        ptrs = [s.param.data_ptr() for s in scanned]
        assert len(ptrs) == len(set(ptrs)), "scan() returned duplicate parameter tensors"

    def test_all_trainable_params_covered(self, tiny_mlp) -> None:
        """Every parameter returned by model.parameters() must appear in scan()."""
        scanned = scan(tiny_mlp)
        scanned_ptrs = {s.param.data_ptr() for s in scanned}
        for p in tiny_mlp.parameters():
            assert p.data_ptr() in scanned_ptrs, (
                f"Parameter with shape {p.shape} missing from scan() output"
            )

    def test_scanned_parameter_fields_populated(self, tiny_mlp) -> None:
        scanned = scan(tiny_mlp)
        for s in scanned:
            assert isinstance(s.name,       str)
            assert isinstance(s.optimizer,  str)
            assert s.optimizer in ("muon", "adamw")
            assert isinstance(s.reason,     str)
            assert len(s.reason) > 0
            assert isinstance(s.module_type, str)
            assert isinstance(s.shape,       torch.Size)
            assert s.shape == s.param.shape

    def test_deeply_nested_model(self) -> None:
        """Scanner must recurse through arbitrarily nested nn.Sequential."""
        model = nn.Sequential(
            nn.Sequential(
                nn.Linear(8, 16),
                nn.LayerNorm(16),
            ),
            nn.Linear(16, 4),
        )
        scanned = scan(model)
        result  = scanned_by_name(scanned)
        # Inner linear weight → muon
        assert result["0.0.weight"].optimizer == "muon"
        # LayerNorm → adamw
        assert result["0.1.weight"].optimizer == "adamw"
        # Outer linear → muon
        assert result["1.weight"].optimizer   == "muon"

    def test_conv_weights_tagged_muon(self, conv_model) -> None:
        result = scanned_by_name(scan(conv_model))
        # Conv2d weight is 4D but Muon flattens it; scanner marks it muon
        assert result["conv.weight"].optimizer == "muon"
        assert result["conv.bias"].optimizer   == "adamw"

    def test_all_norm_model_has_no_muon_params(self, all_norm_model) -> None:
        scanned = scan(all_norm_model)
        muon_tagged = [s for s in scanned if s.optimizer == "muon"]
        assert len(muon_tagged) == 0


# partition() API correctness

class TestPartitionAPI:

    def test_returns_two_lists(self, tiny_mlp) -> None:
        scanned = scan(tiny_mlp)
        result  = partition(scanned)
        assert isinstance(result, tuple)
        assert len(result) == 2
        muon_p, adamw_p = result
        assert isinstance(muon_p, list)
        assert isinstance(adamw_p, list)

    def test_no_param_in_both_groups(self, tiny_mlp) -> None:
        scanned = scan(tiny_mlp)
        muon_p, adamw_p = partition(scanned)
        muon_ptrs  = {p.data_ptr() for p in muon_p}
        adamw_ptrs = {p.data_ptr() for p in adamw_p}
        assert muon_ptrs.isdisjoint(adamw_ptrs), (
            "A parameter appeared in both Muon and AdamW groups"
        )

    def test_frozen_excluded_from_both(self, partially_frozen_mlp) -> None:
        scanned = scan(partially_frozen_mlp)
        muon_p, adamw_p = partition(scanned)
        muon_ptrs  = {p.data_ptr() for p in muon_p}
        adamw_ptrs = {p.data_ptr() for p in adamw_p}
        for p in partially_frozen_mlp.parameters():
            if not p.requires_grad:
                assert p.data_ptr() not in muon_ptrs
                assert p.data_ptr() not in adamw_ptrs

    def test_linear_weights_in_muon_group(self, tiny_mlp) -> None:
        scanned = scan(tiny_mlp)
        muon_p, _ = partition(scanned)
        muon_ptrs = {p.data_ptr() for p in muon_p}
        assert tiny_mlp.fc1.weight.data_ptr() in muon_ptrs
        assert tiny_mlp.fc2.weight.data_ptr() in muon_ptrs

    def test_biases_in_adamw_group(self, tiny_mlp) -> None:
        _, adamw_p = partition(scan(tiny_mlp))
        adamw_ptrs = {p.data_ptr() for p in adamw_p}
        assert tiny_mlp.fc1.bias.data_ptr() in adamw_ptrs
        assert tiny_mlp.fc2.bias.data_ptr() in adamw_ptrs

    def test_transformer_block_partition(self, transformer_block) -> None:
        scanned = scan(transformer_block)
        muon_p, adamw_p = partition(scanned)
        muon_ptrs = {p.data_ptr() for p in muon_p}
        # proj.weight is the only Muon candidate
        assert transformer_block.proj.weight.data_ptr()    in muon_ptrs
        assert transformer_block.embed.weight.data_ptr() not in muon_ptrs
        assert transformer_block.norm.weight.data_ptr()  not in muon_ptrs


# _classify() unit tests (the inner routing function)

class TestClassifyDirect:
    """
    Exercise _classify() directly with synthetic parameters to test
    each branch in isolation, independent of module context.
    """

    def _param(self, shape, dtype=torch.float32, requires_grad=True):
        return nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=requires_grad)

    def test_2d_float_goes_to_muon(self) -> None:
        p = self._param((8, 8))
        tag, reason = _classify("weight", p, None, "weight")
        assert tag == "muon"

    def test_1d_float_goes_to_adamw(self) -> None:
        p = self._param((8,))
        tag, reason = _classify("scale", p, None, "scale")
        assert tag == "adamw"
        assert "ndim" in reason

    def test_frozen_goes_to_adamw(self) -> None:
        p = self._param((8, 8), requires_grad=False)
        tag, reason = _classify("weight", p, None, "weight")
        assert tag == "adamw"
        assert "frozen" in reason

    def test_weight_tied_goes_to_adamw(self) -> None:
        p = self._param((8, 8))
        tag, reason = _classify("lm_head.weight", p, None, "embed.weight")
        assert tag == "adamw"
        assert "weight-tied" in reason

    def test_dot_bias_suffix(self) -> None:
        p = self._param((8, 8))
        tag, reason = _classify("layer.bias", p, None, "layer.bias")
        assert tag == "adamw"

    def test_underscore_bias_suffix(self) -> None:
        p = self._param((8, 8))
        tag, reason = _classify("attn_bias", p, None, "attn_bias")
        assert tag == "adamw"

    def test_embedding_module_excluded(self) -> None:
        p = self._param((32, 16))
        emb = nn.Embedding(32, 16)
        tag, reason = _classify("weight", p, emb, "weight")
        assert tag == "adamw"
        assert "Embedding" in reason

    def test_numel_below_floor(self) -> None:
        p = self._param((1, 1))  # numel == 1
        tag, reason = _classify("gate", p, None, "gate")
        assert tag == "adamw"
        assert "numel" in reason

    def test_non_float_excluded(self) -> None:
        p = nn.Parameter(torch.zeros(4, 4, dtype=torch.int32), requires_grad=False)
        tag, reason = _classify("quant_scale", p, None, "quant_scale")
        assert tag == "adamw"
        assert "non-floating-point" in reason