# determl -- Deterministic ML Inference Library

**Detect, prevent, and ENFORCE determinism in ML inference.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## The Problem

Machine learning models can give **different outputs every time you run them** -- even with the exact same input. This happens because of:

- **Random seeds** -- PyTorch, NumPy, and Python each have their own random number generators
- **GPU non-determinism** -- CUDA operations like `scatter_add` and Flash Attention use non-deterministic algorithms by default for speed
- **cuDNN auto-tuning** -- picks different algorithms on different runs
- **Floating point arithmetic** -- the order of addition can change results across hardware

This is a problem when you need to **prove** that a model was executed correctly -- for example, in decentralized AI networks where multiple nodes must agree on the output.

## What Makes determl Different

Most "determinism" solutions just set `torch.manual_seed()` and call it a day. **determl goes further:**

| Feature | Other tools | determl v2 |
|---------|------------|------------|
| Seed locking | Sets seeds | Sets seeds |
| Op detection | Not available | Scans model for non-deterministic ops |
| Op replacement | Not available | **Auto-replaces** Flash Attention with math backend, Dropout with identity |
| Cross-hardware | Not addressed | **Canonicalizes** outputs so hashes match across A100/V100/RTX |
| Environment enforcement | Not addressed | **Refuses** to compare results from incompatible hardware |
| Verification | Not available | Runs N times, compares SHA-256 hashes |

---

## Quick Start

### Installation

```bash
# Basic (torch + numpy only)
pip install -e .

# With HuggingFace wrapper support
pip install -e ".[transformers]"

# With dev tools (pytest)
pip install -e ".[dev]"
```

### One-Line Setup (like rl-swarm)

```bash
./run_determl.sh
# Creates venv, installs everything, prompts for model name, launches
```

---

### The DeterministicEngine (v2 API)

```python
from determl import DeterministicEngine

# Load model + auto-fix all non-deterministic ops
engine = DeterministicEngine(seed=42, precision="high")
engine.load("Qwen/Qwen2.5-Coder-0.5B-Instruct")

# Run deterministic inference
result = engine.run("Write hello world in Python")
print(result.text)            # Same every time
print(result.canonical_hash)  # Same hash across different GPUs

# Export proof of execution
proof = result.to_proof()
# Send proof to another node for verification
```

### Scan + Auto-Fix

```python
from determl import DeterministicEngine

engine = DeterministicEngine(seed=42)
report = engine.load("your-model")
print(report)
# Enforcement Report for 'your-model' (142 modules scanned)
# ==========================================================
#   [FIXED] 'attention' (MultiheadAttention): Wrapped with deterministic SDPA
#   [FIXED] 'dropout' (Dropout): Replaced with identity
#   [SKIP] 'pool' (FractionalMaxPool2d): Uses random samples by design
#
# Summary: 2 fixed, 1 skipped
```

### Cross-Hardware Verification

```python
from determl import OutputCanonicalizer

canon = OutputCanonicalizer(precision="high")

# Node A (A100 GPU)
result_a = canon.canonicalize(model_output_a)

# Node B (V100 GPU)
result_b = canon.canonicalize(model_output_b)

# Canonical hashes match even though raw floats differ!
assert result_a.canonical_hash == result_b.canonical_hash
```

### Environment Enforcement

```python
from determl import EnvironmentGuardian

guardian = EnvironmentGuardian()

# Capture this machine's fingerprint
local = guardian.create_fingerprint()
print(local)
# Environment [a1b2c3d4]
#   PyTorch:  2.2.0
#   CUDA:     12.1
#   GPU:      NVIDIA A100 (Ampere)
#   Deterministic: True

# Compare with remote node
result = guardian.compare(local, remote_fingerprint)
# STRICT / COMPATIBLE / INCOMPATIBLE

# Enforce -- raises if incompatible
guardian.enforce(remote_fingerprint)
```

---

## CLI

```bash
# Interactive deterministic inference
determl run Qwen/Qwen2.5-Coder-0.5B-Instruct

# Scan model for non-deterministic ops
determl scan Qwen/Qwen2.5-Coder-0.5B-Instruct

# Verify determinism (auto-prompt, 5 runs)
determl verify Qwen/Qwen2.5-Coder-0.5B-Instruct

# Show environment info
determl info
```

---

## API Reference

### `DeterministicEngine` (v2 -- recommended)

| Method | Description |
|--------|-------------|
| `DeterministicEngine(seed=42, precision="high")` | Create engine |
| `.load(model_name)` | Load HuggingFace model, auto-fix non-det ops |
| `.load_model(model, tokenizer)` | Use pre-loaded model |
| `.run(prompt)` | Deterministic text generation |
| `.run_tensor(tensor)` | Deterministic tensor inference |
| `.verify(prompt, num_runs=5)` | Built-in verification |
| `.scan()` | Show enforcement report |
| `.get_info()` | Engine + environment info |

### `DeterministicEnforcer`

| Method | Description |
|--------|-------------|
| `.enforce(model)` | Patch model in-place, return `EnforcementReport` |
| `.deterministic_context()` | Context manager for deterministic execution |

### `OutputCanonicalizer`

| Method | Description |
|--------|-------------|
| `.canonicalize(tensor)` | Round + hash for cross-hardware consistency |
| `.canonicalize_logits(logits, top_k=10)` | Token-level comparison for LLMs |
| `.canonical_hash(tensor)` | Shorthand for just the hash |
| `.compare(a, b, tolerance)` | Compare two tensors with tolerance |

### `EnvironmentGuardian`

| Method | Description |
|--------|-------------|
| `.create_fingerprint()` | Capture this machine's environment |
| `.compare(local, remote)` | Compare two fingerprints |
| `.enforce(required)` | Raise if environments are incompatible |

### v1 API (still available)

| Class | Description |
|-------|-------------|
| `DeterministicConfig` | Seed locking + flag setting |
| `NonDeterminismDetector` | Static model scanning |
| `InferenceVerifier` | Run-N-times verification |
| `DeterministicLLM` | Simple HuggingFace wrapper |

### Utilities

| Function | Description |
|----------|-------------|
| `hash_tensor(tensor)` | SHA-256 of tensor bytes |
| `hash_string(text)` | SHA-256 of UTF-8 string |
| `get_environment_snapshot()` | Full compute environment dict |

---

## Architecture

```
determl/
  config.py          # Seed locking (v1)
  detector.py        # Static op scanning (v1)
  verifier.py        # Hash verification (v1)
  wrapper.py         # Simple LLM wrapper (v1)
  enforcer.py        # Runtime op interception + auto-fix (v2)
  canonicalizer.py   # Cross-hardware output normalization (v2)
  guardian.py        # Environment enforcement (v2)
  engine.py          # High-level DeterministicEngine (v2)
  cli.py             # CLI entry point (v2)
  utils.py           # Hashing + env snapshots
```

---

## Deep Dive: Sources of Non-Determinism

### 1. Random Seeds
Multiple RNG systems (Python `random`, NumPy, PyTorch CPU, PyTorch CUDA) must ALL be seeded. Missing even one breaks determinism.

### 2. CUDA Operations
Some CUDA kernels use `atomicAdd` which is inherently non-deterministic. `torch.use_deterministic_algorithms(True)` forces deterministic alternatives (slower but reproducible).

### 3. Flash Attention
`scaled_dot_product_attention` uses Flash Attention or memory-efficient kernels by default. determl forces the math backend via `sdpa_kernel(SDPBackend.MATH)`.

### 4. cuDNN Auto-Tuning
`torch.backends.cudnn.benchmark = True` (default) lets cuDNN pick the fastest algorithm, which may vary between runs. We disable this.

### 5. cuBLAS Workspace
Matrix multiplications on GPU can produce different results depending on workspace size. Setting `CUBLAS_WORKSPACE_CONFIG=:4096:8` fixes this.

### 6. Floating-Point Ordering
Different GPUs (A100 vs V100) use different FMA units and reduction orders, producing slightly different floating-point results. The `OutputCanonicalizer` addresses this by rounding outputs to a configurable precision before hashing.

### 7. Sampling in LLMs
`do_sample=True` (temperature, top-k, top-p) introduces randomness by design. Greedy decoding (`do_sample=False`) eliminates this entirely.

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All tests use tiny randomly-initialized models -- no large downloads, CPU only.

---

## License

MIT -- see [LICENSE](LICENSE).
