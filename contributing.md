# Contributing to AutoMuon
 
Thanks for your interest in contributing! Contributions are most welcome in the areas described below.
 
---
 
## Table of Contents
 
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [What to Contribute](#what-to-contribute)
- [Adding a New Module-Type Exclusion](#adding-a-new-module-type-exclusion)
- [Code Style](#code-style)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Bugs](#reporting-bugs)
---
 
## Getting Started
 
1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/skyegunasekaran/automuon.git
   cd automuon
   ```
 
2. Create a branch for your change:
   ```bash
   git checkout -b my-feature
   ```
 
---
 
## Development Setup
 
AutoMuon's only runtime dependency is PyTorch (≥ 2.0). For development, install
the package in editable mode with the `dev` extras:
 
```bash
pip install -e ".[dev]"
```
 
This installs `pytest` and `pytest-cov` alongside the package itself.
 
---
 
## Running Tests
 
```bash
# Run the full test suite
pytest
 
# With coverage report
pytest --cov=automuon --cov-report=term-missing
 
# Run a specific file
pytest tests/test_scanner.py -v
```
 
All tests must pass before a PR will be merged. CI runs the suite on Python
3.10, 3.11, and 3.12 against PyTorch 2.0 and the latest stable release.
 
---
 
## What to Contribute
 
### High-value areas
 
- **New module-type exclusions** - if you find a module type whose 2D weights
  should go to AdamW rather than Muon (e.g. a sparse attention variant, a
  custom embedding layer), please open a PR. See the section below.
- **Bug fixes** - especially in the scanner's weight-tie detection, the
  Newton-Schulz backend, or the DDP sync helpers.
- **Tests** - more coverage of edge cases (frozen params, weight-tied models,
  empty param groups, single-param models, etc.) is always welcome.
- **Documentation** - clarifications, additional examples, and docstring
  improvements are all appreciated.

### Out of scope (for now)
 
- FSDP support - this is planned but not yet ready.
- Optimizers other than Muon and AdamW - AutoMuon is intentionally minimal.

---
 
## Adding a New Module-Type Exclusion
 
The canonical list lives in `automuon/backends/scanner.py`:
 
```python
ADAMW_MODULE_TYPES: tuple[type[nn.Module], ...] = (
    nn.Embedding,
    nn.EmbeddingBag,
    nn.LayerNorm,
    ...
)
```
 
To add an exclusion:
 
1. Add the class to `ADAMW_MODULE_TYPES` with a comment explaining *why*
   Muon is wrong for this module type. The reasoning matters - please don't
   just add it without explanation.
2. Add a test in `tests/test_scanner.py` that constructs a minimal model
   containing the module and asserts the parameter is routed to `"adamw"`.
3. Update the "What goes to AdamW and why" section of `README.md`.

---
 
## Code Style
 
- **Type hints** on all public functions and class methods.
- **Docstrings** on all public symbols. Format: plain prose, no reStructuredText
  or NumPy-style sections - keep it readable in source.
- **No external dependencies** beyond PyTorch. Do not add `numpy`, `einops`,
  or similar as runtime dependencies.
- Line length: 100 characters. Enforced loosely - clarity beats strict limits.
- Prefer explicit over implicit. The scanner's `_classify()` function is a good
  example of the style: ordered checks, each with a reason string, early returns.
---
 
## Submitting a Pull Request
 
1. Make sure `pytest` passes with no failures.
2. If your change affects public API or behavior, update `CHANGELOG.md` under
   an `[Unreleased]` section at the top.
3. Keep PRs focused - one logical change per PR makes review much easier.
4. Write a clear PR description explaining *what* changed and *why*.
---
 
## Reporting Bugs
 
Open a GitHub issue with:
 
- A minimal reproducible example (model definition + optimizer construction + the
  failing call).
- The full traceback.
- Your PyTorch version (`torch.__version__`) and Python version.
- Whether you're running single-GPU, DDP, or another setup.
---
 
Thanks again, AutoMuon is better with your help!