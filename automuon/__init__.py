"""
automuon/__init__.py
"""

from automuon.optimizer import AutoMuon
from automuon.backends.muon import Muon
from automuon.backends.scanner import scan, partition, ScannedParameter
from automuon.backends.newton_schulz import (
    orthogonalize,
    orthogonality_residual,
    DEFAULT_NS_STEPS,
)
from automuon.utils.muon_logging import print_partition_table
from automuon.ddp.muon_ddp import DDPMuon

__all__ = [
    "AutoMuon",
    "Muon",
    "DDPMuon",
    "scan",
    "partition",
    "ScannedParameter",
    "orthogonalize",
    "orthogonality_residual",
    "DEFAULT_NS_STEPS",
    "print_partition_table",
]

__version__ = "0.1.0"