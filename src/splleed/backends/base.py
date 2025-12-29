"""Base classes for inference backends."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Generic, TypeVar

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    """Request for text generation."""

    prompt: str
    max_tokens: int = 128
    min_tokens: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stream: bool = True


class BackendConfigBase(BaseModel):
    """Base class for all backend configurations."""

    # Note: 'type' field is defined in each subclass with Literal type
    # for Pydantic discriminated unions to work correctly

    # For managed mode: how to find the engine executable
    executable: str | None = None

    # Logging control for managed servers
    quiet: bool = Field(
        default=True,
        description="Suppress server logs during benchmarking (downloads still shown)",
    )


ConfigT = TypeVar("ConfigT", bound=BackendConfigBase)


class Backend(ABC, Generic[ConfigT]):
    """
    Abstract base class for inference backends.

    Backends are responsible for:
    - Connecting to or starting an inference server
    - Sending generation requests and yielding tokens
    - Health checking

    Timing is NOT the backend's responsibility - that's handled by RequestExecutor.
    """

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    @abstractmethod
    def generate_stream(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
        """
        Generate tokens from a prompt.

        Yields tokens as they are received. Timing is handled by the caller.

        Args:
            request: Generation request parameters

        Yields:
            Token strings as they are generated
        """
        ...

    @abstractmethod
    async def health(self) -> bool:
        """
        Check if the backend is ready to accept requests.

        Returns:
            True if healthy, False otherwise
        """
        ...

    async def initialize(self) -> None:
        """
        Initialize the backend (optional custom initialization).

        Override this for backends that need custom init beyond connect/start.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support initialize()")

    async def connect(self, endpoint: str) -> None:
        """
        Connect to an existing inference server.

        Args:
            endpoint: Server URL to connect to
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support connect()")

    async def start(self) -> None:
        """Start the inference server (managed mode)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support start()")

    async def shutdown(self) -> None:
        """Shutdown the backend and cleanup resources."""
        pass  # Default: no-op for backends that don't need cleanup


class ConnectableBackend(Backend[ConfigT]):
    """Backend that can connect to an existing server."""

    @abstractmethod
    async def connect(self, endpoint: str) -> None:
        """
        Connect to an existing server.

        Args:
            endpoint: Server endpoint URL (e.g., "http://localhost:8000")
        """
        ...


class ManagedBackend(ConnectableBackend[ConfigT]):
    """Backend that can start and stop its own server."""

    @abstractmethod
    async def start(self) -> None:
        """
        Start the inference server.

        Uses configuration from self.config.
        Returns when the server is ready to accept requests.
        """
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shutdown the server."""
        ...


class OpenAIServerBackend(ManagedBackend[ConfigT]):
    """
    Base class for OpenAI-compatible API servers (vLLM, TGI, etc.).

    Handles common logic:
    - HTTP client management
    - Server process lifecycle
    - SSE (Server-Sent Events) parsing for streaming
    - Health checking

    Subclasses must implement:
    - build_launch_command(): Return the command to start the server
    - health_endpoint(): Return the health check URL path
    """

    def __init__(self, config: ConfigT) -> None:
        super().__init__(config)
        self.endpoint: str | None = None
        self.client: httpx.AsyncClient | None = None
        self.process: asyncio.subprocess.Process | None = None
        self._managed = False

    @abstractmethod
    def build_launch_command(self) -> list[str]:
        """
        Build the command to launch the server.

        Returns:
            Command as list of strings (e.g., ["vllm", "serve", "model", ...])
        """
        ...

    @abstractmethod
    def health_endpoint(self) -> str:
        """
        Return the health check endpoint path.

        Returns:
            URL path (e.g., "/health" or "/v1/models")
        """
        ...

    @abstractmethod
    def default_executable(self) -> str:
        """
        Return the default executable name for this backend.

        Returns:
            Executable name (e.g., "vllm", "text-generation-launcher")
        """
        ...

    def quiet_env_vars(self) -> dict[str, str]:
        """
        Return environment variables to suppress server logs.

        Override in subclasses to set backend-specific log levels.
        Called when config.quiet is True.

        Returns:
            Dict of env var name -> value
        """
        return {}

    def _get_port(self) -> int:
        """Get the port to use. Override if needed."""
        return 8000

    def _executable_env_var(self) -> str:
        """Get the environment variable name for this backend's executable."""
        # Extract backend type from config (e.g., "vllm" -> "SPLLEED_VLLM_PATH")
        backend_type = getattr(self.config, "type", "unknown")
        return f"SPLLEED_{backend_type.upper()}_PATH"

    def get_executable(self) -> str:
        """
        Resolve the executable path for this backend.

        Resolution order:
        1. Config 'executable' field
        2. Environment variable (SPLLEED_<TYPE>_PATH)
        3. Default executable name (from PATH)
        """
        # 1. Explicit config
        if self.config.executable:
            return self.config.executable

        # 2. Environment variable
        env_var = self._executable_env_var()
        env_path = os.environ.get(env_var)
        if env_path:
            return env_path

        # 3. Default
        return self.default_executable()

    def check_executable(self, executable: str) -> None:
        """
        Check if executable exists and provide helpful error if not.

        Args:
            executable: The executable path to check

        Raises:
            FileNotFoundError: With helpful message if not found
        """
        backend_type = getattr(self.config, "type", "unknown")
        env_var = self._executable_env_var()
        default_exe = self.default_executable()

        # If it's an absolute/relative path, check file exists
        if os.path.sep in executable or executable.startswith("."):
            if not os.path.isfile(executable):
                raise FileNotFoundError(f"{backend_type} executable not found at: {executable}")
            return

        # Otherwise check PATH
        if shutil.which(executable) is None:
            raise FileNotFoundError(
                f"Could not find '{executable}' command.\n\n"
                f"For managed mode, {backend_type} must be accessible. Options:\n"
                f"  1. Install {backend_type}: pip install {backend_type}\n"
                f"  2. Activate your {backend_type} environment before running splleed\n"
                f"  3. Set path in config:\n"
                f"       executable: /path/to/{default_exe}\n"
                f"  4. Set via environment:\n"
                f"       export {env_var}=/path/to/{default_exe}\n"
                f"  5. Use connect mode instead (recommended for production):\n"
                f"       backend:\n"
                f"         endpoint: http://localhost:8000"
            )

    async def connect(self, endpoint: str) -> None:
        """Connect to an existing server."""
        self.endpoint = endpoint.rstrip("/")
        self.client = httpx.AsyncClient(base_url=self.endpoint, timeout=600.0)
        self._managed = False

        # Verify connection
        if not await self.health():
            raise ConnectionError(f"Cannot connect to server at {endpoint}")

        logger.info(f"Connected to server at {endpoint}")

    async def start(self) -> None:
        """Start the server process and wait for it to be ready."""
        cmd = self.build_launch_command()
        port = self._get_port()
        self.endpoint = f"http://localhost:{port}"

        logger.info(f"Starting server with command: {' '.join(cmd)}")

        # Set up environment with quiet mode if configured
        env = None
        if self.config.quiet:
            quiet_vars = self.quiet_env_vars()
            if quiet_vars:
                env = os.environ.copy()
                env.update(quiet_vars)

        # Don't capture stderr - let server output go directly to terminal
        # so user can see download progress, model loading, etc.
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=None,  # Inherit stderr - shows in terminal
            env=env,
        )

        self.client = httpx.AsyncClient(base_url=self.endpoint, timeout=600.0)
        self._managed = True

        # Wait for server to be healthy
        await self._wait_for_healthy(timeout=300.0)

        logger.info(f"Server started and healthy at {self.endpoint}")

    async def _wait_for_healthy(self, timeout: float = 300.0) -> None:
        """Poll until server is healthy or timeout."""
        start = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start < timeout:
            if await self.health():
                return

            # Check if process died
            if self.process and self.process.returncode is not None:
                raise RuntimeError(f"Server process died with code {self.process.returncode}")

            await asyncio.sleep(1.0)

        raise TimeoutError(f"Server did not become healthy within {timeout}s")

    async def health(self) -> bool:
        """Check server health."""
        if not self.client:
            return False

        try:
            response = await self.client.get(self.health_endpoint())
            return response.status_code == 200
        except httpx.RequestError:
            return False

    async def shutdown(self) -> None:
        """Shutdown the server if we started it."""
        if self.client:
            await self.client.aclose()
            self.client = None

        if self._managed and self.process:
            logger.info("Shutting down server...")
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=10.0)
            except TimeoutError:
                logger.warning("Server did not terminate gracefully, killing...")
                self.process.kill()
                await self.process.wait()

            self.process = None

    def adjust_payload(self, payload: dict) -> dict:
        """
        Adjust the request payload for backend-specific requirements.

        Override in subclasses to handle backend quirks (e.g., TGI's top_p range).
        """
        return payload

    async def generate_stream(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
        """
        Send generation request and stream tokens via SSE.

        This implements the OpenAI-compatible streaming API.
        """
        if not self.client:
            raise RuntimeError("Not connected to server")

        payload = {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "stream": True,
        }

        if request.min_tokens is not None:
            payload["min_tokens"] = request.min_tokens

        if request.top_k is not None:
            payload["top_k"] = request.top_k

        payload = self.adjust_payload(payload)

        async with self.client.stream(
            "POST",
            "/v1/completions",
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                # SSE format: "data: {json}" or "data: [DONE]"
                if not line or not line.startswith("data: "):
                    continue

                data = line[6:]  # Remove "data: " prefix

                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    # OpenAI format: choices[0].text or choices[0].delta.content
                    choices = chunk.get("choices", [])
                    if choices:
                        text = choices[0].get("text", "")
                        if text:
                            yield text
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse SSE chunk: {data}")
                    continue
