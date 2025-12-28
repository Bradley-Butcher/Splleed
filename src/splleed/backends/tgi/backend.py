"""TGI backend implementation."""

from __future__ import annotations

import logging

from splleed.backends.base import OpenAIServerBackend

from .config import TGIConfig

logger = logging.getLogger(__name__)


class TGIBackend(OpenAIServerBackend[TGIConfig]):
    """
    Text Generation Inference (TGI) backend.

    Supports both:
    - Connect mode: Connect to an existing TGI server
    - Managed mode: Start and manage a TGI server

    TGI exposes an OpenAI-compatible API, so we inherit from OpenAIServerBackend
    which handles SSE parsing and common HTTP logic.
    """

    def _get_port(self) -> int:
        """Get port from config."""
        return self.config.port

    def health_endpoint(self) -> str:
        """TGI health check endpoint."""
        return "/health"

    def default_executable(self) -> str:
        """Default executable for TGI."""
        return "text-generation-launcher"

    def quiet_env_vars(self) -> dict[str, str]:
        """Suppress TGI server logs."""
        return {"RUST_LOG": "warn"}

    def adjust_payload(self, payload: dict) -> dict:
        """Adjust payload for TGI quirks."""
        # TGI requires 0 < top_p < 1 (exclusive), but OpenAI allows 1.0
        if payload.get("top_p", 1.0) >= 1.0:
            payload["top_p"] = 0.9999
        if payload.get("top_p", 0.0) <= 0.0:
            payload["top_p"] = 0.0001
        return payload

    def build_launch_command(self) -> list[str]:
        """Build the TGI server launch command."""
        if not self.config.model:
            raise ValueError("Cannot start server without 'model' specified in config")

        executable = self.get_executable()
        self.check_executable(executable)

        cmd = [
            executable,
            "--model-id",
            self.config.model,
            "--hostname",
            self.config.host,
            "--port",
            str(self.config.port),
            "--max-concurrent-requests",
            str(self.config.max_concurrent_requests),
            "--cuda-memory-fraction",
            str(self.config.cuda_memory_fraction),
        ]

        # Sharding
        if self.config.num_shard is not None:
            cmd.extend(["--num-shard", str(self.config.num_shard)])

        if self.config.sharded is not None:
            cmd.extend(["--sharded", str(self.config.sharded).lower()])

        # Token limits
        if self.config.max_total_tokens is not None:
            cmd.extend(["--max-total-tokens", str(self.config.max_total_tokens)])

        if self.config.max_input_tokens is not None:
            cmd.extend(["--max-input-tokens", str(self.config.max_input_tokens)])

        if self.config.max_batch_size is not None:
            cmd.extend(["--max-batch-size", str(self.config.max_batch_size)])

        # Quantization and dtype
        if self.config.quantize is not None:
            cmd.extend(["--quantize", self.config.quantize])

        if self.config.dtype is not None:
            cmd.extend(["--dtype", self.config.dtype])

        # Boolean flags
        if self.config.trust_remote_code:
            cmd.append("--trust-remote-code")

        if self.config.disable_custom_kernels:
            cmd.append("--disable-custom-kernels")

        if self.config.json_output:
            cmd.append("--json-output")

        # Revision
        if self.config.revision is not None:
            cmd.extend(["--revision", self.config.revision])

        # Extra args (pass-through)
        for key, value in self.config.extra_args.items():
            arg_key = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(arg_key)
            else:
                cmd.extend([arg_key, str(value)])

        return cmd

    async def initialize(self) -> None:
        """
        Initialize the backend based on config.

        Automatically connects or starts server based on whether
        endpoint or model is configured.
        """
        if self.config.endpoint:
            await self.connect(self.config.endpoint)
        elif self.config.model:
            await self.start()
        else:
            raise ValueError("Either 'endpoint' or 'model' must be configured")
