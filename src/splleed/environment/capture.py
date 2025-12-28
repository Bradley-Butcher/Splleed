"""Environment capture functionality."""

from __future__ import annotations

from .engine import detect_engine
from .gpu import detect_gpus
from .system import detect_system
from .types import EnvironmentInfo, ModelInfo


def _get_splleed_version() -> str | None:
    """Get splleed version."""
    try:
        from splleed import __version__

        return __version__
    except (ImportError, AttributeError):
        return None


def capture_environment(
    backend_type: str,
    model_name: str | None = None,
    model_revision: str | None = None,
    quantization: str | None = None,
    dtype: str | None = None,
) -> EnvironmentInfo:
    """
    Capture complete environment fingerprint.

    Args:
        backend_type: Backend type string (e.g., "vllm", "tgi")
        model_name: Model name if known
        model_revision: Model revision/commit if known
        quantization: Quantization method if applicable
        dtype: Data type if known

    Returns:
        Complete EnvironmentInfo with all detected information
    """
    model_info = None
    if model_name:
        model_info = ModelInfo(
            name=model_name,
            revision=model_revision,
            quantization=quantization,
            dtype=dtype,
        )

    return EnvironmentInfo(
        gpus=detect_gpus(),
        engine=detect_engine(backend_type),
        model=model_info,
        system=detect_system(),
        splleed_version=_get_splleed_version(),
    )
