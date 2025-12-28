"""Inference engine version detection."""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EngineInfo:
    """Inference engine information."""

    name: str
    version: str | None = None


def detect_vllm_version() -> EngineInfo | None:
    """
    Detect vLLM version.

    Tries Python import first, falls back to CLI.
    """
    # Try Python import
    try:
        import vllm  # type: ignore[import-not-found]

        version = getattr(vllm, "__version__", None)
        return EngineInfo(name="vllm", version=version)
    except ImportError:
        pass

    # Try CLI
    try:
        result = subprocess.run(
            ["vllm", "version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse version from output
            match = re.search(r"(\d+\.\d+\.?\d*)", result.stdout)
            if match:
                return EngineInfo(name="vllm", version=match.group(1))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception as e:
        logger.debug(f"Failed to detect vLLM version via CLI: {e}")

    return None


def detect_tgi_version() -> EngineInfo | None:
    """
    Detect TGI (Text Generation Inference) version.

    Uses CLI since TGI is a Rust binary.
    """
    try:
        result = subprocess.run(
            ["text-generation-launcher", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Output format: "text-generation-launcher X.Y.Z"
            output = result.stdout.strip() or result.stderr.strip()
            match = re.search(r"(\d+\.\d+\.?\d*)", output)
            if match:
                return EngineInfo(name="tgi", version=match.group(1))
            # If no version found but command succeeded, engine exists
            return EngineInfo(name="tgi", version=None)
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        logger.debug("TGI version detection timed out")
    except Exception as e:
        logger.debug(f"Failed to detect TGI version: {e}")

    return None


def detect_engine(backend_type: str) -> EngineInfo | None:
    """
    Detect engine version based on backend type.

    Args:
        backend_type: Backend type string (e.g., "vllm", "tgi")

    Returns:
        EngineInfo if detected, None otherwise
    """
    detectors = {
        "vllm": detect_vllm_version,
        "tgi": detect_tgi_version,
    }

    detector = detectors.get(backend_type)
    if detector:
        return detector()

    # Unknown backend type - return basic info
    return EngineInfo(name=backend_type, version=None)
