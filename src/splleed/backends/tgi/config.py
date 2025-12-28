"""TGI backend configuration."""

from typing import Literal, Self

from pydantic import Field, model_validator

from splleed.backends.base import BackendConfigBase


class TGIConfig(BackendConfigBase):
    """
    Configuration for Text Generation Inference (TGI) backend.

    Either 'endpoint' (connect to existing server) or 'model' (start new server)
    must be provided.
    """

    type: Literal["tgi"] = "tgi"

    # Connection mode: connect to existing server
    endpoint: str | None = Field(
        default=None,
        description="Endpoint of existing TGI server (e.g., http://localhost:3000)",
    )

    # Managed mode: start our own server
    model: str | None = Field(
        default=None,
        description="HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B-Instruct)",
    )

    # Server configuration (only used in managed mode)
    host: str = Field(default="0.0.0.0", description="Host to bind server to")
    port: int = Field(default=3000, description="Port to bind server to")
    num_shard: int | None = Field(
        default=None,
        description="Number of GPU shards for tensor parallelism",
    )
    sharded: bool | None = Field(
        default=None,
        description="Whether to shard model across GPUs (auto-detected if num_shard set)",
    )

    # TGI-specific options
    max_concurrent_requests: int = Field(
        default=128,
        description="Maximum concurrent requests",
    )
    max_total_tokens: int | None = Field(
        default=None,
        description="Maximum total tokens (input + output) for memory budget",
    )
    max_input_tokens: int | None = Field(
        default=None,
        description="Maximum input prompt length",
    )
    max_batch_size: int | None = Field(
        default=None,
        description="Maximum requests per batch",
    )

    # Quantization
    quantize: (
        Literal["awq", "gptq", "eetq", "exl2", "marlin", "bitsandbytes", "bitsandbytes-nf4", "fp8"]
        | None
    ) = Field(
        default=None,
        description="Quantization method",
    )
    dtype: Literal["float16", "bfloat16"] | None = Field(
        default=None,
        description="Model data type (auto-detected if not set)",
    )

    # Memory
    cuda_memory_fraction: float = Field(
        default=1.0,
        description="Fraction of CUDA memory to use (0.0-1.0)",
    )

    # Advanced options
    trust_remote_code: bool = Field(
        default=False,
        description="Trust and execute remote code from HuggingFace Hub",
    )
    revision: str | None = Field(
        default=None,
        description="Model revision (commit ID or branch)",
    )
    disable_custom_kernels: bool = Field(
        default=False,
        description="Disable custom CUDA kernels (for non-A100 hardware)",
    )
    json_output: bool = Field(
        default=False,
        description="Output logs in JSON format",
    )

    # Extra args (pass-through)
    extra_args: dict = Field(
        default_factory=dict,
        description="Additional arguments passed to TGI launcher",
    )

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        """Ensure either endpoint or model is provided."""
        if not self.endpoint and not self.model:
            raise ValueError("Either 'endpoint' or 'model' must be provided")
        return self
