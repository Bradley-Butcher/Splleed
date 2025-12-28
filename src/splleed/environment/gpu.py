"""GPU information detection via nvidia-smi."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Structured GPU information."""

    index: int
    name: str
    vram_mb: int
    driver_version: str
    cuda_version: str | None = None


def _parse_cuda_version(nvidia_smi_output: str) -> str | None:
    """Extract CUDA version from nvidia-smi header output."""
    for line in nvidia_smi_output.split("\n"):
        if "CUDA Version:" in line:
            # Format: "| NVIDIA-SMI 550.54    Driver Version: 550.54    CUDA Version: 12.4  |"
            parts = line.split("CUDA Version:")
            if len(parts) > 1:
                version = parts[1].strip().split()[0].rstrip("|").strip()
                return version
    return None


def detect_gpus() -> list[GPUInfo]:
    """
    Detect available GPUs using nvidia-smi.

    Returns:
        List of GPUInfo for each detected GPU, empty list if detection fails
    """
    gpus: list[GPUInfo] = []

    try:
        # Get CUDA version from header
        header_result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        cuda_version = None
        if header_result.returncode == 0:
            cuda_version = _parse_cuda_version(header_result.stdout)

        # Query GPU properties
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            logger.debug(f"nvidia-smi query failed: {result.stderr}")
            return []

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                idx_str, name, vram_str, driver = parts[:4]

                try:
                    gpus.append(
                        GPUInfo(
                            index=int(idx_str),
                            name=name,
                            vram_mb=int(float(vram_str)),
                            driver_version=driver,
                            cuda_version=cuda_version,
                        )
                    )
                except ValueError as e:
                    logger.debug(f"Failed to parse GPU info line '{line}': {e}")

    except FileNotFoundError:
        logger.debug("nvidia-smi not found - no NVIDIA GPUs available")
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")
    except Exception as e:
        logger.warning(f"Failed to detect GPUs: {e}")

    return gpus


def format_gpu_info(gpus: list[GPUInfo]) -> str:
    """Format GPU info as a human-readable string."""
    if not gpus:
        return "No GPUs detected"

    parts = []
    for gpu in gpus:
        vram_gb = gpu.vram_mb / 1024
        cuda_str = f", CUDA {gpu.cuda_version}" if gpu.cuda_version else ""
        parts.append(f"{gpu.name} ({vram_gb:.0f}GB{cuda_str})")

    return ", ".join(parts)
