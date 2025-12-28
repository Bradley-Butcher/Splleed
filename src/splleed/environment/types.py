"""Environment data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .engine import EngineInfo
from .gpu import GPUInfo, format_gpu_info
from .system import SystemInfo


@dataclass
class ModelInfo:
    """Model information."""

    name: str
    revision: str | None = None
    quantization: str | None = None
    dtype: str | None = None


@dataclass
class EnvironmentInfo:
    """Complete environment fingerprint for benchmark reproducibility."""

    gpus: list[GPUInfo] = field(default_factory=list)
    engine: EngineInfo | None = None
    model: ModelInfo | None = None
    system: SystemInfo | None = None
    splleed_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {}

        if self.gpus:
            result["gpus"] = [
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "vram_mb": gpu.vram_mb,
                    "driver_version": gpu.driver_version,
                    "cuda_version": gpu.cuda_version,
                }
                for gpu in self.gpus
            ]

        if self.engine:
            result["engine"] = {
                "name": self.engine.name,
                "version": self.engine.version,
            }

        if self.model:
            result["model"] = {
                "name": self.model.name,
                "revision": self.model.revision,
                "quantization": self.model.quantization,
                "dtype": self.model.dtype,
            }

        if self.system:
            result["system"] = {
                "python_version": self.system.python_version,
                "platform": self.system.platform,
                "hostname": self.system.hostname,
                "cpu_count": self.system.cpu_count,
                "memory_gb": self.system.memory_gb,
            }

        if self.splleed_version:
            result["splleed_version"] = self.splleed_version

        return result

    def format_summary(self) -> str:
        """Format as a brief human-readable summary."""
        parts = []

        if self.gpus:
            parts.append(f"GPU: {format_gpu_info(self.gpus)}")

        if self.engine:
            version_str = f" {self.engine.version}" if self.engine.version else ""
            parts.append(f"Engine: {self.engine.name}{version_str}")

        if self.model:
            quant_str = f" ({self.model.quantization})" if self.model.quantization else ""
            parts.append(f"Model: {self.model.name}{quant_str}")

        if self.system:
            parts.append(f"System: Python {self.system.python_version}")

        return ", ".join(parts) if parts else "Unknown environment"
