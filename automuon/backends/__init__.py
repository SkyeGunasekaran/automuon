"""
automuon/backends/__init__.py
"""

from automuon.backends.muon import Muon
from automuon.backends.scanner import scan, partition, ScannedParameter
from automuon.backends.newton_schulz import (
    orthogonalize,
    orthogonality_residual,
    DEFAULT_NS_STEPS,
)

__all__ = [
    "Muon",
    "scan",
    "partition",
    "ScannedParameter",
    "orthogonalize",
    "orthogonality_residual",
    "DEFAULT_NS_STEPS",
]