"""vLLM backend configuration."""

from typing import Literal, Self

from pydantic import Field, model_validator

from splleed.backends.base import BackendConfigBase


class VLLMConfig(BackendConfigBase):
    """
    Configuration for vLLM backend.

    Either 'endpoint' (connect to existing server) or 'model' (start new server)
    must be provided.
    """

    type: Literal["vllm"] = "vllm"

    # Connection mode: connect to existing server
    endpoint: str | None = Field(
        default=None,
        description="Endpoint of existing vLLM server (e.g., http://localhost:8000)",
    )

    # Managed mode: start our own server
    model: str | None = Field(
        default=None,
        description="HuggingFace model name or path (e.g., meta-llama/Llama-3.1-8B-Instruct)",
    )

    # Server configuration (only used in managed mode)
    host: str = Field(default="127.0.0.1", description="Host to bind server to")
    port: int = Field(default=8000, description="Port to bind server to")
    tensor_parallel: int = Field(default=1, description="Number of GPUs for tensor parallelism")
    pipeline_parallel: int = Field(default=1, description="Number of GPUs for pipeline parallelism")

    # vLLM-specific options
    enable_prefix_caching: bool = Field(
        default=True,
        description="Enable prefix caching (improves throughput for similar prompts)",
    )
    gpu_memory_utilization: float = Field(
        default=0.9,
        description="Fraction of GPU memory to use for KV cache (0.0-1.0)",
    )
    max_num_batched_tokens: int | None = Field(
        default=None,
        description="Maximum number of tokens to batch together",
    )
    max_num_seqs: int = Field(
        default=256,
        description="Maximum number of sequences to process in parallel",
    )
    max_model_len: int | None = Field(
        default=None,
        description="Maximum context length (defaults to model's max)",
    )
    chunked_prefill: bool = Field(
        default=True,
        description="Enable chunked prefill for better latency on long prompts",
    )
    disable_log_requests: bool = Field(
        default=True,
        description="Disable request logging for cleaner output",
    )

    # Quantization
    quantization: Literal["awq", "gptq", "squeezellm", "fp8", "bitsandbytes"] | None = Field(
        default=None,
        description="Quantization method",
    )
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = Field(
        default="auto",
        description="Model data type",
    )
    kv_cache_dtype: Literal["fp8", "auto"] = Field(
        default="auto",
        description="KV cache data type",
    )

    # Advanced options (pass-through)
    extra_args: dict = Field(
        default_factory=dict,
        description="Additional arguments passed to vLLM server",
    )

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        """Ensure either endpoint or model is provided."""
        if not self.endpoint and not self.model:
            raise ValueError("Either 'endpoint' or 'model' must be provided")
        return self
