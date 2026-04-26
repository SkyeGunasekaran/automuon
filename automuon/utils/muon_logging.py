"""
automuon/utils/logging.py

Verbose partition table renderer for AutoMuon.
"""

from __future__ import annotations
from  automuon.backends.scanner import ScannedParameter


def print_partition_table(scanned: list[ScannedParameter]) -> None:
    """
    Print a formatted partition table to stdout.
    """
    if not scanned:
        print("AutoMuon: no parameters found.")
        return

    # Column widths — derived from content so the table is always readable.
    w_name   = max(len(s.name)              for s in scanned)
    w_name   = max(w_name, len("Parameter"))
    w_opt    = max(len(s.optimizer.upper()) for s in scanned)
    w_opt    = max(w_opt, len("Optimizer"))
    w_shape  = max(len(str(tuple(s.shape))) for s in scanned)
    w_shape  = max(w_shape, len("Shape"))
    w_reason = max(len(s.reason)            for s in scanned)
    w_reason = max(w_reason, len("Reason"))

    sep = "─" * (w_name + w_opt + w_shape + w_reason + 13)

    print()
    print("AutoMuon parameter partition")
    print(sep)
    print(
        f" {'Parameter':<{w_name}}  "
        f"{'Optimizer':<{w_opt}}  "
        f"{'Shape':<{w_shape}}  "
        f"{'Reason'}"
    )
    print(sep)

    for s in scanned:
        tag = s.optimizer.upper()
        frozen_marker = " [frozen]" if not s.param.requires_grad else ""
        print(
            f" {s.name:<{w_name}}  "
            f"{tag:<{w_opt}}  "
            f"{str(tuple(s.shape)):<{w_shape}}  "
            f"{s.reason}{frozen_marker}"
        )

    print(sep)

    # Summary stats
    trainable = [s for s in scanned if s.param.requires_grad]
    muon_p    = [s for s in trainable if s.optimizer == "muon"]
    adamw_p   = [s for s in trainable if s.optimizer == "adamw"]
    frozen_p  = [s for s in scanned  if not s.param.requires_grad]

    muon_numel  = sum(s.param.numel() for s in muon_p)
    adamw_numel = sum(s.param.numel() for s in adamw_p)
    total_numel = muon_numel + adamw_numel

    muon_pct = f"  ({muon_numel / total_numel * 100:.1f}% of trainable elements)" \
               if total_numel > 0 else ""

    print(f" Muon:   {len(muon_p)} params{muon_pct}")
    print(f" AdamW:  {len(adamw_p)} params")
    if frozen_p:
        print(f" Frozen: {len(frozen_p)} params (excluded from both groups)")
    print(sep)
    print()