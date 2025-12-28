"""Backend registry and factory."""

from typing import Annotated

from pydantic import Field

from .base import (
    Backend,
    BackendConfigBase,
    ConnectableBackend,
    GenerateRequest,
    ManagedBackend,
    OpenAIServerBackend,
)
from .tgi import TGIBackend, TGIConfig
from .vllm import VLLMBackend, VLLMConfig

# Discriminated union of all backend configs
BackendConfig = Annotated[
    VLLMConfig | TGIConfig,
    Field(discriminator="type"),
]

# Registry mapping type string to (config_class, backend_class)
BACKENDS: dict[str, tuple[type[BackendConfigBase], type[Backend]]] = {
    "vllm": (VLLMConfig, VLLMBackend),
    "tgi": (TGIConfig, TGIBackend),
}


def get_backend(config: BackendConfig) -> Backend:
    """Instantiate the appropriate backend from config."""
    _, backend_cls = BACKENDS[config.type]
    return backend_cls(config)


__all__ = [
    "Backend",
    "BackendConfig",
    "BackendConfigBase",
    "BACKENDS",
    "ConnectableBackend",
    "GenerateRequest",
    "get_backend",
    "ManagedBackend",
    "OpenAIServerBackend",
    "TGIBackend",
    "TGIConfig",
    "VLLMBackend",
    "VLLMConfig",
]
