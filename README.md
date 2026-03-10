# detinfer — Deterministic Inference & Replay Toolkit

**Deterministic runtime controls, session tracing, and replay verification for supported LLM workflows.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-88%20passed-brightgreen.svg)]()
[![PyPI](https://img.shields.io/badge/pypi-v0.2.2-blue.svg)](https://pypi.org/project/detinfer/)

---

## Why?

Most reproducibility guides stop at setting RNG seeds. In practice, LLM runs can still drift because of decoding settings, backend behavior, prompt rendering, tokenizer differences, and attention/runtime choices.

detinfer focuses on supported workflows where deterministic settings, token tracing, replay, and diffing can be used together to verify and debug LLM outputs.

```bash
pip install detinfer
detinfer agent gpt2 --prompt "What is 2+2?" --export run.json
detinfer replay run.json
detinfer diff run_a.json run_b.json
```

---

## What detinfer is

- A **deterministic runtime and verification toolkit** for supported LLM inference paths
- A **replay/debugging tool** for agent sessions and token-level generation traces
- A **reproducibility aid** for benchmarking, CI, and regression testing

## What detinfer is not

- A guarantee of universal determinism for all PyTorch workloads
- A guarantee of bitwise-identical outputs across all hardware and library combinations
- A replacement for careful control of tokenizer, prompt formatting, backend, and environment

---

## Installation

```bash
pip install detinfer
```

With HuggingFace model support (recommended):

```bash
pip install "detinfer[transformers]"
```

With INT8 quantization (experimental):

```bash
pip install "detinfer[quantized]"
```

Update to latest version:

```bash
pip install --upgrade detinfer
```

### Setting up a virtual environment

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

pip install "detinfer[transformers]"
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- Any NVIDIA GPU (recommended) or CPU

### Quick Reference — All CLI Commands

```bash
# Replace <model> with any HuggingFace model, e.g. gpt2, Qwen/Qwen2.5-0.5B-Instruct

# ── Agent ──
detinfer agent <model>                               # Multi-turn deterministic agent
detinfer agent <model> --prompt "What is 2+2?"       # Non-interactive (single question)
detinfer agent <model> --stream                      # Stream tokens in real-time
detinfer agent <model> --system "You are a tutor"    # Set system prompt
detinfer agent <model> --export session.json         # Export session trace
detinfer agent <model> --quantize int8               # Experimental INT8 mode
detinfer agent <model> --verbose-trace               # Record top-k tokens per step

# ── Inference ──
detinfer run <model>                                # Interactive deterministic inference
detinfer run <model> --seed 42 --max-tokens 512     # Custom seed and token limit

# ── Verify & Replay ──
detinfer verify <model>                             # Run 5 times, compare hashes
detinfer replay session.json                        # Replay a saved session
detinfer replay session.json --strict               # Step-by-step verification
detinfer verify-session session.json                # Verify session as execution proof
detinfer verify-session session.json --strict       # Strict proof verification
detinfer diff run_a.json run_b.json                 # Token-level comparison of two runs

# ── Analysis ──
detinfer scan <model>                               # Scan for non-deterministic ops
detinfer compare <model>                            # Before vs after detinfer comparison
detinfer benchmark <model>                          # Full benchmark (auto-scales)

# ── Cross-GPU Proofs ──
detinfer export <model> -o proof.json               # Export proof for cross-GPU verification
detinfer cross-verify proof.json                    # Verify proof from another machine

# ── Info ──
detinfer info                                       # Show GPU and environment details
```

---

## Getting Started

```python
import detinfer

# Apply deterministic runtime settings
detinfer.enforce(seed=42)

# Now supported PyTorch inference paths use deterministic settings:
output = model(input)
```

---

## Usage

### 1. Runtime Enforcement

```python
import detinfer

# Apply deterministic runtime settings for supported workflows
detinfer.enforce(seed=42)
```

This locks RNG seeds, disables cuDNN benchmarking, enables `torch.use_deterministic_algorithms`, and sets cuBLAS workspace config.

### 2. DeterministicEngine (for LLM inference)

```python
from detinfer import DeterministicEngine

# Load any HuggingFace model + apply deterministic runtime settings
engine = DeterministicEngine(seed=42)
engine.load("<model>")  # e.g., "Qwen/Qwen2.5-0.5B-Instruct", "gpt2"

# Run inference under deterministic settings
result = engine.run("Write hello world in Python")
print(result.text)            # The generated text
print(result.canonical_hash)  # SHA-256 hash for verification
```

### 3. Deterministic Agent

Deterministic agent sessions with replayable execution traces.

`detinfer agent` runs supported generation workflows under deterministic settings, records token-level traces, and exports replayable session files for verification and debugging.

Both `chat()` and `chat_stream()` use manual token-by-token generation with deterministic argmax (smallest-token-ID tie-breaking).

```python
from detinfer import DeterministicAgent

# Multi-turn deterministic agent
agent = DeterministicAgent("gpt2", seed=42)
response = agent.chat("What is 2+2?")
print(response)

# With system prompt
agent = DeterministicAgent(
    "Qwen/Qwen2.5-0.5B-Instruct",
    seed=42,
    system_prompt="You are a math tutor"
)
response = agent.chat("What is calculus?")

# Streaming output (tokens appear one by one)
for chunk in agent.chat_stream("Explain gravity"):
    print(chunk, end="", flush=True)

# Export full session trace
agent.export_session("session.json")
```

### 4. Session Export & Trace

Exported sessions contain:

```json
{
  "schema_version": "1",
  "model": "gpt2",
  "seed": 42,
  "messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
  ],
  "generations": [
    {
      "input_tokens": [464, 318],
      "output_tokens": [17, 10],
      "steps": [
        {"step": 0, "chosen_token": 17},
        {"step": 1, "chosen_token": 10}
      ]
    }
  ]
}
```

With `--verbose-trace`, each step also includes `top_tokens` and `top_scores` (top-10 candidates).

### 5. Replay & Verification

`detinfer replay` re-runs a saved session and verifies prompt hashes, input tokens, output tokens, and stop conditions under the current runtime.

```bash
detinfer replay run.json
```

If a divergence occurs, the first mismatch is reported:

```
Replay verification: FAILED
Mismatch at turn 2 step 18
Expected token: 287
Observed token: 318
```

### 6. Session Diffing

`detinfer diff` compares two saved sessions and reports the first divergence point at the token/step level.

```bash
detinfer diff run_a.json run_b.json
```

### 7. Session Proof Verification

```bash
detinfer verify-session session.json
```

```
  ══════════════════════════════════════════════════════════════
    DETERMINISTIC EXECUTION PROOF VERIFICATION
  ══════════════════════════════════════════════════════════════

    Model:        gpt2
    Seed:         42
    Turns:        3

    ✓ VERIFIED — All 3 turns match exactly
    ✓ This session is a valid deterministic execution proof.
  ══════════════════════════════════════════════════════════════
```

### 8. Cross-GPU Verification

```bash
# Machine A
detinfer export <model> -o proof.json

# Transfer proof.json to Machine B

# Machine B
detinfer cross-verify proof.json
```

### 9. Verify Determinism

```python
# Run 5 times, compare all hashes
result = engine.verify(num_runs=5)
print(result)
# DETERMINISTIC: All 5 runs produced identical output
```

---

## CLI Reference

### `detinfer agent`

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | — | Non-interactive mode (single question) |
| `--stream` | off | Stream tokens as they are generated |
| `--system` | — | System prompt (e.g., "You are a math tutor") |
| `--seed` | 42 | Random seed |
| `--max-tokens` | 256 | Max tokens per turn |
| `--device` | auto | Device (cpu, cuda, auto) |
| `--export` | — | Export session trace to JSON |
| `--quantize` | — | Quantization mode (`int8`, experimental) |
| `--verbose-trace` | off | Record top-k tokens and scores per step |

### `detinfer verify-session`

| Flag | Default | Description |
|------|---------|-------------|
| `--strict` | off | Verify every generation step |
| `--model` | — | Override model |

### `detinfer replay`

| Flag | Default | Description |
|------|---------|-------------|
| `--strict` | off | Step-by-step verification |
| `--model` | — | Override model |

### `detinfer verify`

| Flag | Default | Description |
|------|---------|-------------|
| `--runs` | 5 | Number of runs to compare |
| `--seed` | 42 | Random seed |

---

## How It Works

detinfer applies deterministic runtime settings for 7 sources of non-determinism:

| Source | Problem | detinfer Setting |
|--------|---------|-----------------|
| Random seeds | Separate RNGs in Python, NumPy, PyTorch, CUDA | Locks all seeds in one call |
| CUDA atomics | `scatter_add`, `index_add` use non-deterministic `atomicAdd` | Enables `torch.use_deterministic_algorithms(True)` |
| Flash Attention | `scaled_dot_product_attention` is non-deterministic | Replaces with deterministic math backend |
| cuDNN tuning | Auto-selects different algorithms per run | Disables benchmark mode |
| cuBLAS workspace | Matrix multiplications vary with workspace config | Sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Float ordering | Different GPUs may produce different float results | Canonicalizes outputs before hashing |
| LLM decoding | `temperature`, `top_k`, `top_p` add randomness | Uses deterministic argmax with stable tie-breaking |

---

## Architecture

```
detinfer/
  __init__.py       # Top-level API: enforce(), status(), checkpoint_hash()
  cli.py            # CLI entry point (14 commands)

  inference/        # Deterministic inference library
    config.py       # Seed locking + deterministic flags
    enforcer.py     # Runtime op patching (Dropout, Flash Attention)
    canonicalizer.py # Cross-hardware output normalization
    guardian.py     # Environment fingerprinting + compatibility
    engine.py       # High-level DeterministicEngine for LLMs
    benchmark.py    # Auto-scaling benchmark suite (8 tiers, 36 prompts)
    proof.py        # Cross-GPU proof export/import/verify
    detector.py     # Static model scanning
    verifier.py     # Hash-based verification
    wrapper.py      # Simple HuggingFace wrapper
    utils.py        # Hashing + env snapshots

  agent/            # Deterministic agent system
    runtime.py      # DeterministicAgent — multi-turn agent with deterministic argmax
    trace.py        # Token-level trace recording + session schema
    replay.py       # Session replay verification + diff
```

---

## API Reference

### Top-Level API

```python
import detinfer

detinfer.enforce(seed=42)              # Apply deterministic runtime settings
detinfer.status()                       # Check enforcement state
detinfer.checkpoint_hash(model)         # Hash model weights (for training)
```

### DeterministicEngine

| Method | Description |
|--------|-------------|
| `DeterministicEngine(seed, precision, device)` | Create engine |
| `.load(model_name)` | Load HuggingFace model, apply deterministic settings |
| `.load(model_name, quantize="int8")` | Load with INT8 quantization (experimental) |
| `.run(prompt, max_new_tokens)` | Run inference under deterministic settings |
| `.verify(prompt, num_runs)` | Run N times, compare hashes |
| `.scan()` | Show enforcement report |

### DeterministicAgent

| Method | Description |
|--------|-------------|
| `DeterministicAgent(model, seed, system_prompt)` | Create agent |
| `.chat(message)` | Send message, get response (deterministic argmax) |
| `.chat_stream(message)` | Stream tokens as generated |
| `.export_session(path)` | Export full token trace to JSON |
| `.get_session_hash()` | Get canonical session hash |

### Replay & Diff

| Function | Description |
|----------|-------------|
| `replay_session(trace_path)` | Re-run session, verify token-by-token |
| `diff_sessions(path_a, path_b)` | Compare two traces, find first mismatch |

---

## GitHub Action

Add determinism verification to your CI pipeline:

```yaml
# .github/workflows/determinism.yml
name: Verify Determinism
on: [push]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: xailong-6969/detinfer@v2-enforcement
        with:
          command: verify-session
          session-file: baseline.json
          strict: true
```

---

## Support Status

### Supported

- Single-process HuggingFace causal LM inference
- Deterministic agent session export and replay
- Token-level diffing for saved sessions
- Greedy decoding paths under controlled runtime settings

### Experimental

- INT8 quantized mode (may improve consistency, not guaranteed bitwise identical)
- Cross-device consistency checks
- Streaming/verbose trace paths

### Not Guaranteed

- Universal determinism for arbitrary PyTorch code
- Bitwise-identical results across all GPU architectures
- Distributed training or asynchronous multi-node systems
- External API or tool call determinism

### Detailed Compatibility

| Feature | Status | Notes |
|---|---|---|
| PyTorch eager mode | **Supported** | Default execution mode |
| Greedy decoding | **Supported** | Enforced via deterministic argmax |
| fp32 / fp16 inference | **Supported** | Deterministic on supported backends |
| CPU inference | **Supported** | Fully deterministic |
| NVIDIA GPU (single) | **Supported** | T4, V100, A100, RTX 3070/4090, etc. |
| HuggingFace CausalLM | **Supported** | GPT-2, Qwen, TinyLlama, LLaMA, etc. |
| bf16 inference | Partial | Hardware-dependent rounding may differ |
| Multi-GPU (`device_map="auto"`) | Partial | May have split-order edge cases |
| Flash Attention | Partial | Auto-replaced with MATH backend |
| INT8 (bitsandbytes) | **Experimental** | May improve consistency |
| GPTQ/AWQ quantization | **Not supported** | Kernel-specific rounding |
| `torch.compile` | **Not supported** | Graph autotuning |
| Beam search | **Not supported** | Tie-breaking is implementation-specific |
| vLLM / paged attention | **Not supported** | KV cache paging |
| AMD GPUs (ROCm) | Untested | |
| Apple Silicon (MPS) | Untested | |

---

## Running Tests

```bash
pip install "detinfer[dev]"
pytest tests/ -v
```

---

## License

MIT — see [LICENSE](LICENSE).
