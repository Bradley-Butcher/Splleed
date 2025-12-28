"""
Template backend configuration.

TODO: Rename this class and customize for your backend.

Your config class should include:
- Connection settings (endpoint for existing servers)
- Launch settings (model, resources for managed mode)
- Engine-specific options (quantization, caching, etc.)
"""

from typing import Literal, Self

from pydantic import Field, model_validator

from splleed.backends.base import BackendConfigBase


class TemplateConfig(BackendConfigBase):
    """
    Configuration for template backend.

    TODO: Rename to YourEngineConfig and customize.
    """

    # TODO: Change "template" to your engine name (e.g., "tgi", "ollama")
    type: Literal["template"] = "template"

    # Connection mode: connect to existing server
    endpoint: str | None = Field(
        default=None,
        description="Endpoint of existing server (e.g., http://localhost:8000)",
    )

    # Managed mode: start our own server
    model: str | None = Field(
        default=None,
        description="Model name or path",
    )

    # Server configuration (only used in managed mode)
    host: str = Field(default="127.0.0.1", description="Host to bind server to")
    port: int = Field(default=8000, description="Port to bind server to")

    # TODO: Add engine-specific options below
    # example_option: bool = Field(default=True, description="An example option")
    # quantization: str | None = Field(default=None, description="Quantization method")

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        """Ensure either endpoint or model is provided."""
        if not self.endpoint and not self.model:
            raise ValueError(
                "Either 'endpoint' (connect mode) or 'model' (managed mode) is required"
            )
        return self
