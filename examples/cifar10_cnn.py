"""
examples/cifar10_cnn.py

CIFAR-10 image classification with AutoMuon vs AdamW — head-to-head comparison.

This example demonstrates that AutoMuon works beyond language modelling: it is
a drop-in replacement for AdamW in a standard CNN training pipeline on a
vision task. Both optimizers are trained from the same random initialisation
so the comparison is fair.

Usage:
    # Quick smoke test (CPU, no GPU required)
    python examples/cifar10_cnn.py --epochs 3 --batch-size 128 --num-workers 0

    # Full run (recommended: CUDA GPU)
    python examples/cifar10_cnn.py --epochs 30 --batch-size 512

    # Force fp16 instead of bfloat16 (older GPUs)
    python examples/cifar10_cnn.py --epochs 30 --fp16

    # Reproducible run (slower — disables cudnn.benchmark)
    python examples/cifar10_cnn.py --epochs 30 --deterministic
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from automuon import AutoMuon


# Reproducibility

def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True lets cuDNN pick the fastest kernel per input shape.
    # Disable only when byte-exact reproducibility is required.
    torch.backends.cudnn.benchmark     = not deterministic
    torch.backends.cudnn.deterministic = deterministic


# Model

class SmallCIFARCNN(nn.Module):
    """
    VGG-style CNN for CIFAR-10.

    Five conv blocks (64 → 64 → 128 → 128 → 256 channels) with BatchNorm
    and ReLU, followed by a global average pool and a linear classifier.
    No biases on conv layers (BatchNorm makes them redundant).

    AutoMuon partition:
        Muon  — all conv weights (4D, reshaped to 2D by the scanner)
        AdamW — BatchNorm scale/bias, classifier weight + bias
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,   64,  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,  64,  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64,  128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

        # Normalisation constants registered as buffers so they move to the
        # target device with the model and the operation fuses into the first
        # conv layer in the torch.compile() graph.
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        self.register_buffer("norm_mean", mean)
        self.register_buffer("norm_std",  std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.norm_mean) / self.norm_std
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# Data

def make_loaders(
    batch_size:     int,
    num_workers:    int,
    prefetch_factor: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Build CIFAR-10 train and test DataLoaders.

    Uses torchvision.transforms.v2 when available (faster CPU augmentation).
    Normalisation is intentionally omitted here — it is handled on-GPU inside
    SmallCIFARCNN.forward() so the CPU->GPU transfer is a compact float32 tensor.
    """
    try:
        from torchvision.transforms import v2
        train_tfms = v2.Compose([
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        test_tfms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
    except ImportError:
        train_tfms = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        test_tfms = T.ToTensor()

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=train_tfms)
    test_set  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_tfms)

    loader_kwargs: dict = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
    )

    return (
        DataLoader(train_set, shuffle=True,  **loader_kwargs),
        DataLoader(test_set,  shuffle=False, **loader_kwargs),
    )


# Training helpers

@dataclass
class EpochStats:
    epoch:      int
    train_loss: float
    train_acc:  float
    test_loss:  float
    test_acc:   float
    seconds:    float


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=1) == y).sum()


def _make_scaler(amp_dtype: torch.dtype, device: torch.device):
    """
    Return a GradScaler for fp16, or a no-op scaler for bf16/CPU.

    bfloat16 has a wider dynamic range than float16 and does not require
    loss scaling. On CPU, GradScaler is not supported at all.
    """
    use_scaler = (amp_dtype == torch.float16) and device.type == "cuda"
    if use_scaler:
        # Use the non-deprecated API when available (PyTorch >= 2.3).
        if hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler("cuda")
        return torch.cuda.amp.GradScaler()
    # Disabled scaler: scale/unscale/update are all no-ops.
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=False)
    return torch.cuda.amp.GradScaler(enabled=False)


def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    scaler,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    model.train()

    # Accumulate losses as tensors to avoid a GPU->CPU sync (.item()) per step.
    total_loss    = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0,   device=device)
    total_seen    = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            logits = model(x)
            loss   = F.cross_entropy(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs             = y.size(0)
        total_loss    += loss.detach() * bs
        total_correct += _accuracy(logits.detach(), y)
        total_seen    += bs

    # Single GPU->CPU sync per epoch.
    return (
        (total_loss / total_seen).item(),
        (total_correct / total_seen).item(),
    )


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    device:    torch.device,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    model.eval()

    total_loss    = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0,   device=device)
    total_seen    = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            logits = model(x)
            loss   = F.cross_entropy(logits, y)

        bs             = y.size(0)
        total_loss    += loss * bs
        total_correct += _accuracy(logits, y)
        total_seen    += bs

    return (
        (total_loss / total_seen).item(),
        (total_correct / total_seen).item(),
    )


def run_experiment(
    name:         str,
    model:        nn.Module,
    optimizer:    torch.optim.Optimizer,
    scheduler:    torch.optim.lr_scheduler.LRScheduler,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    device:       torch.device,
    epochs:       int,
    amp_dtype:    torch.dtype,
) -> list[EpochStats]:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(optimizer)

    scaler  = _make_scaler(amp_dtype, device)
    history: list[EpochStats] = []

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            device=device, scaler=scaler, amp_dtype=amp_dtype,
        )
        scheduler.step()

        test_loss, test_acc = evaluate(
            model=model, loader=test_loader, device=device, amp_dtype=amp_dtype,
        )

        seconds = time.perf_counter() - t0
        stats   = EpochStats(epoch, train_loss, train_acc, test_loss, test_acc, seconds)
        history.append(stats)

        print(
            f"  epoch {epoch:02d}/{epochs} | "
            f"train loss {train_loss:.4f} | train acc {train_acc*100:5.2f}% | "
            f"test loss {test_loss:.4f} | test acc {test_acc*100:5.2f}% | "
            f"{seconds:.1f}s"
        )

    return history


def summarize(histories: dict[str, list[EpochStats]]) -> None:
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for name, hist in histories.items():
        best       = max(hist, key=lambda s: s.test_acc)
        final      = hist[-1]
        total_time = sum(s.seconds for s in hist)
        print(
            f"  {name:<10} | "
            f"best {best.test_acc*100:5.2f}% @ epoch {best.epoch:02d} | "
            f"final {final.test_acc*100:5.2f}% | "
            f"total {total_time:.1f}s"
        )


# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CIFAR-10 CNN: AutoMuon vs AdamW head-to-head comparison."
    )
    parser.add_argument("--epochs",          type=int,   default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size",      type=int,   default=512,
                        help="Batch size (default: 512)")
    parser.add_argument("--num-workers",     type=int,   default=8,
                        help="DataLoader worker processes (default: 8; use 0 for CPU)")
    parser.add_argument("--prefetch-factor", type=int,   default=4,
                        help="DataLoader prefetch factor (default: 4)")
    parser.add_argument("--seed",            type=int,   default=1337,
                        help="Random seed (default: 1337)")
    # AdamW baseline hyperparameters
    parser.add_argument("--adamw-lr",        type=float, default=8e-4,
                        help="AdamW baseline learning rate (default: 8e-4)")
    parser.add_argument("--adamw-wd",        type=float, default=0.05,
                        help="AdamW weight decay (default: 0.05)")
    # AutoMuon hyperparameters
    parser.add_argument("--muon-lr",         type=float, default=2e-3,
                        help="AutoMuon Muon learning rate for conv weights (default: 2e-3)")
    parser.add_argument("--automuon-adamw-lr", type=float, default=8e-4,
                        help="AutoMuon AdamW learning rate for norms/biases (default: 8e-4)")
    parser.add_argument("--automuon-wd",     type=float, default=0.05,
                        help="AutoMuon AdamW weight decay (default: 0.05)")
    # Hardware / precision
    parser.add_argument("--fp16",            action="store_true",
                        help="Use float16 autocast instead of bfloat16")
    parser.add_argument("--no-compile",      action="store_true",
                        help="Disable torch.compile() (useful for debugging)")
    parser.add_argument("--deterministic",   action="store_true",
                        help="Disable cudnn.benchmark for exact reproducibility (slower)")
    args = parser.parse_args()

    seed_everything(args.seed, deterministic=args.deterministic)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if args.fp16 else torch.bfloat16

    print(f"device:     {device}")
    print(f"amp dtype:  {amp_dtype}")
    print(f"batch size: {args.batch_size}")
    print(f"epochs:     {args.epochs}")

    train_loader, test_loader = make_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    # Both experiments start from the same random initialisation.
    base_model    = SmallCIFARCNN()
    initial_state = copy.deepcopy(base_model.state_dict())

    def _build_model() -> nn.Module:
        m = SmallCIFARCNN().to(device)
        m.load_state_dict(initial_state)
        if not args.no_compile and torch.cuda.is_available():
            # reduce-overhead is best for fixed input shapes (CIFAR-10 is always 32×32).
            m = torch.compile(m, mode="reduce-overhead")
        return m

    histories: dict[str, list[EpochStats]] = {}

    
    # AdamW baseline
    
    adamw_model     = _build_model()
    adamw_optimizer = torch.optim.AdamW(
        adamw_model.parameters(),
        lr=args.adamw_lr,
        weight_decay=args.adamw_wd,
    )
    adamw_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        adamw_optimizer, T_max=args.epochs,
    )

    histories["AdamW"] = run_experiment(
        name="AdamW",
        model=adamw_model, optimizer=adamw_optimizer, scheduler=adamw_scheduler,
        train_loader=train_loader, test_loader=test_loader,
        device=device, epochs=args.epochs, amp_dtype=amp_dtype,
    )

    
    # AutoMuon
    
    # Important: AutoMuon must receive the *unwrapped* nn.Module so its
    # parameter scanner can resolve module types (e.g. detect nn.Conv2d vs
    # nn.BatchNorm2d). torch.compile() is applied afterwards, and the
    # compiled model is what gets passed to run_experiment.
    
    automuon_raw   = SmallCIFARCNN().to(device)
    automuon_raw.load_state_dict(initial_state)

    automuon_optimizer = AutoMuon(
        automuon_raw,                         # <-- unwrapped model for scanning
        muon_lr=args.muon_lr,
        adamw_lr=args.automuon_adamw_lr,
        adamw_wd=args.automuon_wd,
        verbose=True,                         # prints the partition table
    )

    # Compile *after* building the optimizer.
    automuon_model = automuon_raw
    if not args.no_compile and torch.cuda.is_available():
        automuon_model = torch.compile(automuon_raw, mode="reduce-overhead")

    automuon_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        automuon_optimizer, T_max=args.epochs,
    )

    histories["AutoMuon"] = run_experiment(
        name="AutoMuon",
        model=automuon_model, optimizer=automuon_optimizer,
        scheduler=automuon_scheduler,
        train_loader=train_loader, test_loader=test_loader,
        device=device, epochs=args.epochs, amp_dtype=amp_dtype,
    )

    summarize(histories)


if __name__ == "__main__":
    main()