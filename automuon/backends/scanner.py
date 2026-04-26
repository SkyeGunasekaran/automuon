"""
automuon/scanner.py

Parameter scanner: walks an nn.Module tree, resolves each parameter's
owning module type, applies eligibility rules, and returns a typed
partition of the model's parameters into Muon vs AdamW groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

OptimizerTag = Literal["muon", "adamw"]


@dataclass
class ScannedParameter:
    """
    The scanner's output unit. One entry per unique parameter tensor.

    'reason' is a short human-readable string that powers verbose logging
    and is useful for unit testing scanner decisions independently of the
    optimizer construction.
    """
    name: str
    param: nn.Parameter
    optimizer: OptimizerTag
    reason: str
    module_type: str    # e.g. "Linear", "Embedding", "LayerNorm"
    shape: torch.Size


# Module-type exclusion rules

# Parameters owned by any of these module types are always sent to AdamW,
# regardless of tensor shape. The reasoning for each:

#   Embedding       - 2D but Muon on token/positional embeddings is harmful;
#                     the rows are looked up sparsely, not multiplied densely.
#   *Norm layers    - scale/bias are 1D but we make this explicit rather than
#                     relying on the shape check, so custom norm layers that
#                     happen to have 2D parameters don't slip through.
#   *Bias params    - caught by name suffix check below, but belt-and-suspenders.

# This is intentionally conservative. Better to send an ambiguous parameter
# to AdamW (safe) than to Muon (potentially wrong).

# Please feel free to submit PRs to expand this list if you encounter more cases in the wild!
ADAMW_MODULE_TYPES: tuple[type[nn.Module], ...] = (
    nn.Embedding,
    nn.EmbeddingBag,
    nn.LayerNorm,
    nn.RMSNorm,         
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)

# Parameter name suffixes that are always AdamW regardless of shape.
# Catches biases on custom modules that aren't in ADAMW_MODULE_TYPES.
ADAMW_NAME_SUFFIXES: tuple[str, ...] = (
    ".bias",
    "_bias",
)

# Minimum number of elements for Muon to be meaningful. 
# Tiny 2D tensors (e.g. (1,1) scalar gates) are not good targets for orthogonalization.
MIN_NUMEL_FOR_MUON: int = 2

# Weight-sharing resolution

def _build_ptr_to_name(model: nn.Module) -> dict[int, str]:
    """
    Map each unique data_ptr -> the *first* parameter name that owns it.
    Used to detect weight-tied parameters (e.g. input embedding == LM head).
    """
    ptr_to_name: dict[int, str] = {}
    for name, param in model.named_parameters():
        ptr = param.data_ptr()
        if ptr not in ptr_to_name:
            ptr_to_name[ptr] = name
    return ptr_to_name


# Per-parameter eligibility logic

def _classify(
    name: str,
    param: nn.Parameter,
    owning_module: nn.Module | None,
    canonical_name: str,
) -> tuple[OptimizerTag, str]:
    """
    Returns (optimizer_tag, reason_string) for a single parameter.

    Checks are ordered from most specific to most general so the reason
    string is always maximally informative — the first matching rule wins.
    """

    # 1. Weight-sharing / duplicate tensor.
    #    If canonical_name != name, another parameter already owns this
    #    tensor. Route to AdamW; the canonical entry handles Muon eligibility.
    if canonical_name != name:
        return "adamw", f"weight-tied to '{canonical_name}'"

    # 2. Non-floating-point parameters (e.g. integer quantization scales).
    #    Newton-Schulz requires floating-point arithmetic.
    if not param.is_floating_point():
        return "adamw", "non-floating-point tensor"

    # 3. Frozen parameters.
    #    Tag adamw as a safe fallback; the optimizer shell will exclude them
    #    from all param groups via the requires_grad filter in partition().
    if not param.requires_grad:
        return "adamw", "frozen (requires_grad=False)"

    # 4. Module-type exclusions.
    #    Checked against the owning module's class, not the param name,
    #    so exotic subclasses of nn.Embedding etc. are caught correctly.
    if owning_module is not None:
        if isinstance(owning_module, ADAMW_MODULE_TYPES):
            module_cls = type(owning_module).__name__
            return "adamw", f"module type '{module_cls}' excluded"

    # 5. Name-suffix exclusions.
    #    Belt-and-suspenders for biases on custom modules not in
    #    ADAMW_MODULE_TYPES. Also catches positional bias tables, etc.
    for suffix in ADAMW_NAME_SUFFIXES:
        if name.endswith(suffix):
            return "adamw", f"name matches suffix '{suffix}'"

    # 6. Shape check — Muon orthogonalization requires ndim >= 2.
    if param.ndim < 2:
        return "adamw", f"ndim={param.ndim} (requires >= 2)"

    # 7. Numel floor — tiny 2D tensors (e.g. (1,1) scalar gates) are
    #    not meaningful targets for orthogonalization.
    if param.numel() < MIN_NUMEL_FOR_MUON:
        return "adamw", f"numel={param.numel()} below threshold ({MIN_NUMEL_FOR_MUON})"

    # 8. Passed all checks — Muon eligible.
    return "muon", f"2D+ projection, shape={tuple(param.shape)}"

# Public scanner API

def scan(model: nn.Module) -> list[ScannedParameter]:
    """
    Walk the model and classify every parameter as 'muon' or 'adamw'.
    """

    # Step 1: Build param_name -> owning module map.
    
    # named_parameters(recurse=False) on each module yields only the params
    # *directly* owned by that module, not its children. This is exactly
    # what we want — it tells us e.g. that 'transformer.wte.weight' is owned
    # by an nn.Embedding, not by the root transformer module.
    
    param_to_module: dict[str, nn.Module] = {}
    for module_name, module in model.named_modules():
        for param_name, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            param_to_module[full_name] = module

    # Step 2: Build data_ptr -> canonical name map for weight-tie detection.
    ptr_to_canonical: dict[int, str] = _build_ptr_to_name(model)

    # Step 3: Classify each unique parameter.
    results: list[ScannedParameter] = []
    seen_ptrs: set[int] = set()

    for name, param in model.named_parameters():
        ptr = param.data_ptr()

        # Skip if we've already emitted an entry for this tensor.
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)

        canonical_name = ptr_to_canonical[ptr]
        owning_module = param_to_module.get(name)
        module_type = type(owning_module).__name__ if owning_module else "unknown"

        tag, reason = _classify(name, param, owning_module, canonical_name)

        results.append(ScannedParameter(
            name=name,
            param=param,
            optimizer=tag,
            reason=reason,
            module_type=module_type,
            shape=param.shape,
        ))

    return results


def partition(
    scanned: list[ScannedParameter],
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """
    Convenience split of scan() output into (muon_params, adamw_params).
    Frozen parameters (requires_grad=False) are excluded from both lists.
    """
    muon_params = [
        s.param for s in scanned
        if s.optimizer == "muon" and s.param.requires_grad
    ]
    adamw_params = [
        s.param for s in scanned
        if s.optimizer == "adamw" and s.param.requires_grad
    ]
    return muon_params, adamw_params