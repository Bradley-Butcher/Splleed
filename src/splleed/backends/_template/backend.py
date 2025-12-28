"""
Template backend implementation.

TODO: Rename this class and implement the required methods.

Choose the appropriate base class:
- Backend: Simplest, just generate tokens
- ConnectableBackend: Can connect to existing servers
- ManagedBackend: Can start/stop servers
- OpenAIServerBackend: For OpenAI-compatible APIs (handles SSE parsing)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from splleed.backends.base import GenerateRequest, ManagedBackend

from .config import TemplateConfig


class TemplateBackend(ManagedBackend[TemplateConfig]):
    """
    Template backend implementation.

    TODO: Rename to YourEngineBackend and implement methods.

    If your engine uses an OpenAI-compatible API, inherit from
    OpenAIServerBackend instead - it handles SSE parsing for you.
    """

    async def connect(self, endpoint: str) -> None:
        """
        Connect to an existing server.

        TODO: Implement connection logic.
        - Store the endpoint
        - Create HTTP client
        - Verify the server is reachable
        """
        raise NotImplementedError("TODO: Implement connect()")

    async def start(self) -> None:
        """
        Start the inference server.

        TODO: Implement server startup.
        - Build the launch command from self.config
        - Start the subprocess
        - Wait for health check to pass
        """
        raise NotImplementedError("TODO: Implement start()")

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the server.

        TODO: Implement shutdown logic.
        - Terminate the subprocess
        - Clean up resources
        """
        raise NotImplementedError("TODO: Implement shutdown()")

    async def health(self) -> bool:
        """
        Check if the backend is ready.

        TODO: Implement health check.
        - Make request to health endpoint
        - Return True if healthy, False otherwise
        """
        raise NotImplementedError("TODO: Implement health()")

    def default_executable(self) -> str:
        """
        Return the default executable name.

        TODO: Change to your engine's CLI command.
        Examples: "vllm", "text-generation-launcher", "ollama"
        """
        return "template"

    async def generate_stream(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
        """
        Generate tokens from a prompt.

        TODO: Implement token generation.
        - Send request to server
        - Parse streaming response
        - Yield tokens as they arrive

        Note: Timing is handled by RequestExecutor, not here.
        Just yield the tokens.
        """
        # Type hint requires this to be a generator
        if False:
            yield ""
        raise NotImplementedError("TODO: Implement generate_stream()")
