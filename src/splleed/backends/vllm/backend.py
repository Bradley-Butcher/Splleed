"""vLLM backend implementation."""

from __future__ import annotations

import logging

from splleed.backends.base import OpenAIServerBackend

from .config import VLLMConfig

logger = logging.getLogger(__name__)


class VLLMBackend(OpenAIServerBackend[VLLMConfig]):
    """
    vLLM inference backend.

    Supports both:
    - Connect mode: Connect to an existing vLLM server
    - Managed mode: Start and manage a vLLM server

    vLLM exposes an OpenAI-compatible API, so we inherit from OpenAIServerBackend
    which handles SSE parsing and common HTTP logic.
    """

    def _get_port(self) -> int:
        """Get port from config."""
        return self.config.port

    def health_endpoint(self) -> str:
        """vLLM health check endpoint."""
        return "/health"

    def default_executable(self) -> str:
        """Default executable for vLLM."""
        return "vllm"

    def quiet_env_vars(self) -> dict[str, str]:
        """Suppress vLLM server logs."""
        return {"VLLM_LOGGING_LEVEL": "WARNING"}

    def build_launch_command(self) -> list[str]:
        """Build the vLLM server launch command."""
        if not self.config.model:
            raise ValueError("Cannot start server without 'model' specified in config")

        executable = self.get_executable()
        self.check_executable(executable)

        cmd = [
            executable,
            "serve",
            self.config.model,
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--tensor-parallel-size",
            str(self.config.tensor_parallel),
            "--pipeline-parallel-size",
            str(self.config.pipeline_parallel),
            "--gpu-memory-utilization",
            str(self.config.gpu_memory_utilization),
            "--max-num-seqs",
            str(self.config.max_num_seqs),
            "--dtype",
            self.config.dtype,
            "--kv-cache-dtype",
            self.config.kv_cache_dtype,
        ]

        # Boolean flags
        if self.config.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")

        if self.config.chunked_prefill:
            cmd.append("--enable-chunked-prefill")

        if self.config.disable_log_requests:
            cmd.append("--disable-log-requests")

        # Optional parameters
        if self.config.max_num_batched_tokens is not None:
            cmd.extend(["--max-num-batched-tokens", str(self.config.max_num_batched_tokens)])

        if self.config.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.config.max_model_len)])

        if self.config.quantization is not None:
            cmd.extend(["--quantization", self.config.quantization])

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
