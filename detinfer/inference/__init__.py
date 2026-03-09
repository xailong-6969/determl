"""
detinfer.inference -- Deterministic Inference Library

Core modules for enforcing, detecting, and verifying
determinism in ML inference and training.
"""

from detinfer.inference.config import DeterministicConfig
from detinfer.inference.detector import NonDeterminismDetector
from detinfer.inference.enforcer import DeterministicEnforcer
from detinfer.inference.canonicalizer import OutputCanonicalizer
from detinfer.inference.guardian import EnvironmentGuardian
from detinfer.inference.engine import DeterministicEngine
from detinfer.inference.verifier import InferenceVerifier
from detinfer.inference.utils import hash_tensor, hash_string, get_environment_snapshot

try:
    from detinfer.inference.wrapper import DeterministicLLM
except ImportError:
    DeterministicLLM = None

__all__ = [
    "DeterministicConfig",
    "NonDeterminismDetector",
    "DeterministicEnforcer",
    "OutputCanonicalizer",
    "EnvironmentGuardian",
    "DeterministicEngine",
    "InferenceVerifier",
    "DeterministicLLM",
    "hash_tensor",
    "hash_string",
    "get_environment_snapshot",
]
