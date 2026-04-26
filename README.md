# AutoMuon
 
A one-line drop-in replacement for AdamW.
 
AutoMuon automatically routes each parameter in your model to the right optimizer: 2D projection weights go to [Muon](https://kellerjordan.github.io/posts/muon/), everything else (embeddings, norms, biases) goes to AdamW. You pass the model once and never think about it again (ideally)!
 
```python
# Before
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
 
# After
optimizer = AutoMuon(model)
```

---
 
## Installation
 
```bash
pip install https://github.com/SkyeGunasekaran/automuon.git
```
 
Requires Python ≥ 3.10 and PyTorch ≥ 2.0.
 
---
 
## Quickstart
 
```python
from automuon import AutoMuon
 
optimizer = AutoMuon(model)
 
# Training loop is unchanged
for x, y in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```
 
**Separate learning rates** (recommended for best performance):
 
```python
optimizer = AutoMuon(
    model,
    muon_lr=0.02,    # for 2D projection weights
    adamw_lr=3e-4,   # for everything else
)
```
 
The default `muon_lr` is `0.02` and `adamw_lr` is `3e-4` - a ~67× (haha) ratio that reflects the fact that Muon's orthogonalized update has unit spectral norm, so its learning rate is literally "spectral norm per step."
 
**Schedulers work without any changes:**
 
```python
optimizer = AutoMuon(model, muon_lr=0.02, adamw_lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
 
# In your loop:
optimizer.step()
scheduler.step()
```
 
---
 
## What goes where
 
The scanner walks your model and classifies every parameter:
 
| Goes to **Muon** | Goes to **AdamW** |
|---|---|
| `nn.Linear` weights | `nn.Embedding`, `nn.EmbeddingBag` |
| `nn.Conv*` weights (any shape) | All norm layers (LayerNorm, BatchNorm, RMSNorm, GroupNorm, …) |
| Any other 2D+ float parameter | Any parameter whose name ends in `.bias` or `_bias` |
| | Frozen parameters (excluded from both) |
| | Weight-tied parameters (deduplicated, canonical entry wins) |
 
The routing is conservative by design; when in doubt, a parameter goes to AdamW. Conv weights are reshaped to 2D before orthogonalization and reshaped back, so the shape is transparent.
 
**See the routing for your model:**
 
```python
optimizer = AutoMuon(model, verbose=True)
```
 
```
AutoMuon parameter partition
────────────────────────────────────────────────────────────────
 Parameter              Optimizer  Shape          Reason
────────────────────────────────────────────────────────────────
 features.0.weight      MUON       (64, 3, 3, 3)  2D+ projection, shape=(64, 3, 3, 3)
 features.1.weight      ADAMW      (64,)           module type 'BatchNorm2d' excluded
 features.1.bias        ADAMW      (64,)           module type 'BatchNorm2d' excluded
 ...
 classifier.weight      ADAMW      (10, 256)       2D+ projection, shape=(10, 256)
 classifier.bias        ADAMW      (10,)           name matches suffix '.bias'
────────────────────────────────────────────────────────────────
 Muon:   5 params  (97.1% of trainable elements)
 AdamW:  12 params
────────────────────────────────────────────────────────────────
```
 
---
 
## Multi-GPU (DDP)
 
Use `DDPMuon` instead of `AutoMuon`. The API is identical, just pass the unwrapped model.
 
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from automuon.ddp.muon_ddp import DDPMuon
 
# Standard DDP setup
dist.init_process_group("nccl")
model     = MyModel().to(device)
ddp_model = DDP(model, device_ids=[local_rank])
 
# Pass the unwrapped model; DDP handles gradient sync automatically
optimizer = DDPMuon(
    model,                  # ddp_model.module - unwrapped
    ddp_module=ddp_model,   # for no_sync() support
    muon_lr=0.02,
    adamw_lr=3e-4,
)
 
# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = ddp_model(batch).mean()
    loss.backward()         # DDP all-reduces gradients here
    optimizer.step()
```
 
**Gradient accumulation with DDP:**
 
```python
ACCUM_STEPS = 4
 
for i, batch in enumerate(dataloader):
    # Suppress DDP sync on all but the last accumulation step
    ctx = optimizer.no_sync() if (i + 1) % ACCUM_STEPS != 0 else contextlib.nullcontext()
    with ctx:
        loss = ddp_model(batch).mean() / ACCUM_STEPS
        loss.backward()
 
    if (i + 1) % ACCUM_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
```
 
---
 
## Checkpointing
 
```python
# Save
torch.save({
    "model":     model.state_dict(),
    "optimizer": optimizer.state_dict(),
}, "checkpoint.pt")
 
# Load
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
```
 
---
 
## Beyond language modelling
 
Muon was originally developed for transformer training, but the orthogonalized update is beneficial for any model with dense 2D weight multiplications. The `examples/cifar10_cnn.py` script trains a small CNN on CIFAR-10 and compares AutoMuon against AdamW from the same random seed:
 
```bash
python examples/cifar10_cnn.py --epochs 30 --batch-size 512
```
 
My result (RTX 5090 w/ 30 epochs, batch size 512, bfloat16):
 
```
AdamW    | best 86.65% @ epoch 29 | total 52.5s
AutoMuon | best 90.39% @ epoch 27 | total 40.9s
```
 
AutoMuon reaches AdamW's final accuracy by epoch 8 and finishes training faster in wall time!
 
---
 
## Hyperparameter reference
 
| Argument | Default | Notes |
|---|---|---|
| `muon_lr` | `0.02` | LR for Muon (projection weights). Try `0.01`–`0.05`. |
| `adamw_lr` | `3e-4` | LR for AdamW (embeddings, norms, biases). |
| `momentum` | `0.95` | Nesterov momentum for Muon. |
| `ns_steps` | `5` | Newton-Schulz iteration steps. More = more orthogonal, more compute. |
| `adamw_betas` | `(0.9, 0.999)` | AdamW betas. |
| `adamw_wd` | `0.1` | AdamW weight decay. |
| `verbose` | `False` | Print partition table at init. |
 
---
 
## References
 
- Keller Jordan - [Muon: An optimizer for hidden layers](https://kellerjordan.github.io/posts/muon/)
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) - reference implementation
- PyTorch built-in: [`torch.optim.Muon`](https://pytorch.org/docs/stable/generated/torch.optim.Muon.html)
---
 
## License
 
MIT. See [LICENSE](LICENSE).