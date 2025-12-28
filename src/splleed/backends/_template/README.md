# Adding a New Backend

This directory is a template for adding new inference backends to splleed.

## Quick Start

1. **Copy this directory:**
   ```bash
   cp -r src/splleed/backends/_template src/splleed/backends/my_engine
   ```

2. **Edit `config.py`:**
   - Rename `TemplateConfig` to `MyEngineConfig`
   - Change `type: Literal["template"]` to `type: Literal["my_engine"]`
   - Add your engine-specific configuration options

3. **Edit `backend.py`:**
   - Rename `TemplateBackend` to `MyEngineBackend`
   - Implement the required methods (see below)

4. **Register in `backends/__init__.py`:**
   ```python
   from .my_engine import MyEngineBackend, MyEngineConfig

   # Add to BACKENDS dict:
   BACKENDS = {
       ...
       "my_engine": (MyEngineConfig, MyEngineBackend),
   }

   # Add to BackendConfig union:
   BackendConfig = Annotated[
       VLLMConfig | TGIConfig | MyEngineConfig,
       Field(discriminator="type"),
   ]
   ```

5. **Run tests:**
   ```bash
   pytest tests/backends/test_my_engine.py
   ```

## Backend Types

Choose the appropriate base class for your backend:

### `Backend` (simplest)
For backends that just generate tokens. Implement:
- `generate_stream(request) -> AsyncIterator[str]`
- `health() -> bool`

### `ConnectableBackend`
For backends that connect to external servers. Adds:
- `connect(endpoint: str) -> None`

### `ManagedBackend`
For backends that can start/stop their own server. Adds:
- `start() -> None`
- `shutdown() -> None`

### `OpenAIServerBackend`
For OpenAI-compatible API servers (vLLM, TGI, etc.). Just implement:
- `build_launch_command() -> list[str]`
- `health_endpoint() -> str`

SSE parsing and HTTP logic is handled automatically.

## Example: Adding TGI Backend

```python
# config.py
class TGIConfig(BackendConfigBase):
    type: Literal["tgi"] = "tgi"
    endpoint: str | None = None
    model: str | None = None
    max_batch_prefill_tokens: int | None = None
    max_total_tokens: int | None = None

# backend.py
class TGIBackend(OpenAIServerBackend[TGIConfig]):
    def health_endpoint(self) -> str:
        return "/health"

    def build_launch_command(self) -> list[str]:
        return [
            "text-generation-launcher",
            "--model-id", self.config.model,
            "--port", str(self.config.port),
        ]
```

## Testing Your Backend

Create a test file at `tests/backends/test_my_engine.py`:

```python
import pytest
from splleed.backends.my_engine import MyEngineBackend, MyEngineConfig

@pytest.fixture
def config():
    return MyEngineConfig(endpoint="http://localhost:8000")

@pytest.mark.asyncio
async def test_health(config):
    backend = MyEngineBackend(config)
    await backend.connect(config.endpoint)
    assert await backend.health()
```
