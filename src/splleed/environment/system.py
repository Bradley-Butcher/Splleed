"""System information detection."""

from __future__ import annotations

import os
import platform
import socket
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SystemInfo:
    """System environment information."""

    python_version: str
    platform: str
    hostname: str
    cpu_count: int
    memory_gb: float


def _get_memory_gb() -> float:
    """Get total system memory in GB."""
    # Try psutil if available
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

    # Try /proc/meminfo on Linux
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        try:
            with meminfo_path.open() as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # Format: "MemTotal:       16384000 kB"
                        parts = line.split()
                        if len(parts) >= 2:
                            kb = int(parts[1])
                            return kb / (1024**2)
        except (OSError, ValueError):
            pass

    # Fallback: return 0
    return 0.0


def detect_system() -> SystemInfo:
    """Detect system information using stdlib."""
    return SystemInfo(
        python_version=platform.python_version(),
        platform=platform.platform(),
        hostname=socket.gethostname(),
        cpu_count=os.cpu_count() or 0,
        memory_gb=round(_get_memory_gb(), 1),
    )
