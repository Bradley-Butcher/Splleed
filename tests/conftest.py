"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_prompts() -> list[str]:
    """Sample prompts for testing."""
    return [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about the ocean.",
    ]
