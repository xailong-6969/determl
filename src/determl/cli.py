"""
determl.cli -- Command-Line Interface

Provides the `determl` command with subcommands:
  determl run <model>              -- Interactive deterministic inference
  determl scan <model>             -- Scan for non-deterministic ops
  determl verify <model>           -- Verify determinism (auto-prompt)
  determl info                     -- Show environment info
"""

from __future__ import annotations

import argparse
import sys

import torch


def cmd_info(args: argparse.Namespace) -> None:
    """Show environment information."""
    from determl.guardian import EnvironmentGuardian

    guardian = EnvironmentGuardian()
    fingerprint = guardian.create_fingerprint()
    print(fingerprint)


def cmd_scan(args: argparse.Namespace) -> None:
    """Scan a model for non-deterministic ops."""
    from determl.engine import DeterministicEngine

    print(f"Loading model: {args.model}...")
    engine = DeterministicEngine(
        seed=args.seed,
        device=args.device,
    )
    report = engine.load(args.model)
    print(f"\n{report}")


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify a model produces deterministic output."""
    from determl.engine import DeterministicEngine

    print(f"Loading model: {args.model}...")
    engine = DeterministicEngine(
        seed=args.seed,
        precision=args.precision,
        device=args.device,
    )
    engine.load(args.model)

    prompt = args.prompt or "What is 2 + 2? Answer with just the number."
    print(f"Verifying with prompt: {prompt!r}")
    print(f"Running {args.runs} times...\n")

    result = engine.verify(prompt=prompt, num_runs=args.runs)
    print(result)


def cmd_run(args: argparse.Namespace) -> None:
    """Interactive deterministic inference."""
    from determl.engine import DeterministicEngine

    print(f"Loading model: {args.model}...")
    engine = DeterministicEngine(
        seed=args.seed,
        precision=args.precision,
        device=args.device,
    )
    report = engine.load(args.model)

    print(f"\n{report}")
    print(f"\nEngine: {engine}")
    print("\nReady! Type your prompt (Ctrl+C to exit):\n")

    try:
        while True:
            prompt = input("> ").strip()
            if not prompt:
                continue

            result = engine.run(prompt, max_new_tokens=args.max_tokens)
            print(f"\n{result}\n")

    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="determl",
        description="Deterministic ML Inference Tool",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- determl info --
    subparsers.add_parser("info", help="Show environment information")

    # -- determl scan <model> --
    scan_parser = subparsers.add_parser("scan", help="Scan model for non-deterministic ops")
    scan_parser.add_argument("model", help="HuggingFace model name (e.g., Qwen/Qwen2.5-Coder-0.5B-Instruct)")
    scan_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    scan_parser.add_argument("--device", default=None, help="Device (cpu/cuda, default: auto)")

    # -- determl verify <model> --
    verify_parser = subparsers.add_parser("verify", help="Verify model determinism")
    verify_parser.add_argument("model", help="HuggingFace model name")
    verify_parser.add_argument("--prompt", default=None, help="Test prompt (default: auto)")
    verify_parser.add_argument("--runs", type=int, default=5, help="Number of runs (default: 5)")
    verify_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    verify_parser.add_argument("--precision", default="high", help="Canonical precision (default: high)")
    verify_parser.add_argument("--device", default=None, help="Device (default: auto)")

    # -- determl run <model> --
    run_parser = subparsers.add_parser("run", help="Interactive deterministic inference")
    run_parser.add_argument("model", help="HuggingFace model name")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    run_parser.add_argument("--precision", default="high", help="Canonical precision (default: high)")
    run_parser.add_argument("--device", default=None, help="Device (default: auto)")
    run_parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens (default: 256)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "info": cmd_info,
        "scan": cmd_scan,
        "verify": cmd_verify,
        "run": cmd_run,
    }

    handlers[args.command](args)


if __name__ == "__main__":
    main()
