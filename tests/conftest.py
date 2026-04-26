"""
tests/conftest.py

Shared fixtures and model factories used across the AutoMuon test suite.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest


# Minimal model factories
class TinyMLP(nn.Module):
    """
    Two linear layers + bias, no norms or embeddings.
    Expected partition: both weight matrices → Muon, both biases → AdamW.
    """
    def __init__(self, in_dim: int = 8, hidden: int = 16, out_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class TinyTransformerBlock(nn.Module):
    """
    Single transformer-style block: embedding + linear projection + layernorm.
    Expected partition:
      - embed.weight       → AdamW  (nn.Embedding)
      - proj.weight        → Muon   (nn.Linear 2D)
      - proj.bias          → AdamW  (bias suffix)
      - norm.weight/bias   → AdamW  (nn.LayerNorm)
    """
    def __init__(self, vocab: int = 32, dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.proj  = nn.Linear(dim, dim)
        self.norm  = nn.LayerNorm(dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        return self.norm(self.proj(x))


class WeightTiedLM(nn.Module):
    """
    Minimal language model with weight tying: embed.weight == lm_head.weight.
    The tied tensor should appear exactly once in the partition (as adamw,
    because it is owned by nn.Embedding).
    """
    def __init__(self, vocab: int = 32, dim: int = 16):
        super().__init__()
        self.embed   = nn.Embedding(vocab, dim)
        self.proj    = nn.Linear(dim, dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        # Tie weights
        self.lm_head.weight = self.embed.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        x = self.proj(x)
        return self.lm_head(x)


class ConvModel(nn.Module):
    """
    Tiny conv net. Conv weights are 4D; Muon should flatten them to 2D.
    Expected partition:
      - conv.weight (4D) → Muon
      - conv.bias        → AdamW
      - fc.weight        → Muon
      - fc.bias          → AdamW
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.fc   = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        return self.fc(x.mean(dim=(-2, -1)))


class AllNormModel(nn.Module):
    """
    A model whose only trainable parameters are norm scale/bias.
    All should go to AdamW. AutoMuon should construct with no Muon optimizer.
    """
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(8)
        self.norm2 = nn.BatchNorm1d(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm2(self.norm1(x))


class PartiallyFrozenMLP(nn.Module):
    """
    fc1 is frozen; fc2 is trainable.
    Frozen params must be excluded from both optimizer groups.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)
        for p in self.fc1.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class CustomBiasModule(nn.Module):
    """
    Module with a 2D parameter that has a name ending in '_bias'.
    Should be routed to AdamW by the name-suffix rule even though it is 2D.
    """
    def __init__(self):
        super().__init__()
        self.weight  = nn.Parameter(torch.randn(8, 8))
        self.my_bias = nn.Parameter(torch.randn(8, 8))  # name ends with '_bias'? No.

    def forward(self, x):
        return x @ self.weight


class DotBiasModule(nn.Module):
    """
    Registers a parameter whose full dotted name ends with '.bias'.
    Simulates a custom module with a 2D bias-like tensor.
    """
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8))

    def forward(self, x):
        return x @ self.weight


# Pytest fixtures

@pytest.fixture
def tiny_mlp() -> TinyMLP:
    torch.manual_seed(0)
    return TinyMLP()


@pytest.fixture
def transformer_block() -> TinyTransformerBlock:
    torch.manual_seed(0)
    return TinyTransformerBlock()


@pytest.fixture
def weight_tied_lm() -> WeightTiedLM:
    torch.manual_seed(0)
    return WeightTiedLM()


@pytest.fixture
def conv_model() -> ConvModel:
    torch.manual_seed(0)
    return ConvModel()


@pytest.fixture
def all_norm_model() -> AllNormModel:
    torch.manual_seed(0)
    return AllNormModel()


@pytest.fixture
def partially_frozen_mlp() -> PartiallyFrozenMLP:
    torch.manual_seed(0)
    return PartiallyFrozenMLP()


# Gradient helpers

def make_grads(model: nn.Module, seed: int = 42) -> None:
    """
    Assign synthetic gradients to all trainable parameters.
    Gradients match parameter shape exactly. Used to drive optimizer steps
    without a forward pass or loss function.
    """
    torch.manual_seed(seed)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)


def forward_backward(model: nn.Module, loss_fn=None) -> torch.Tensor:
    """
    Run a real forward + backward pass on a tiny batch.
    Returns the scalar loss. Works for TinyMLP and TinyTransformerBlock.
    """
    torch.manual_seed(1)
    if isinstance(model, (TinyMLP,)):
        x    = torch.randn(4, 8)
        out  = model(x)
        loss = out.pow(2).mean()
    elif isinstance(model, TinyTransformerBlock):
        idx  = torch.randint(0, 32, (4, 6))
        out  = model(idx)
        loss = out.pow(2).mean()
    elif isinstance(model, WeightTiedLM):
        idx  = torch.randint(0, 32, (4, 6))
        out  = model(idx)
        loss = out.pow(2).mean()
    elif isinstance(model, ConvModel):
        x    = torch.randn(4, 3, 8, 8)
        out  = model(x)
        loss = out.pow(2).mean()
    elif isinstance(model, AllNormModel):
        x    = torch.randn(4, 8)
        out  = model(x)
        loss = out.pow(2).mean()
    elif isinstance(model, PartiallyFrozenMLP):
        x    = torch.randn(4, 8)
        out  = model(x)
        loss = out.pow(2).mean()
    else:
        raise ValueError(f"No forward_backward recipe for {type(model).__name__}")
    loss.backward()
    return loss