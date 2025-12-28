"""Tests for backend implementations."""

import pytest

from splleed.backends import BACKENDS, TGIBackend, TGIConfig, VLLMBackend, VLLMConfig, get_backend


class TestBackendRegistry:
    """Tests for backend registry."""

    def test_vllm_registered(self):
        """Test vLLM is registered."""
        assert "vllm" in BACKENDS
        config_cls, backend_cls = BACKENDS["vllm"]
        assert config_cls is VLLMConfig
        assert backend_cls is VLLMBackend

    def test_tgi_registered(self):
        """Test TGI is registered."""
        assert "tgi" in BACKENDS
        config_cls, backend_cls = BACKENDS["tgi"]
        assert config_cls is TGIConfig
        assert backend_cls is TGIBackend

    def test_get_backend_vllm(self):
        """Test getting vLLM backend."""
        config = VLLMConfig(model="test-model")
        backend = get_backend(config)
        assert isinstance(backend, VLLMBackend)

    def test_get_backend_tgi(self):
        """Test getting TGI backend."""
        config = TGIConfig(model="test-model")
        backend = get_backend(config)
        assert isinstance(backend, TGIBackend)


class TestVLLMBackend:
    """Tests for vLLM backend."""

    def test_default_executable(self):
        """Test default executable name."""
        config = VLLMConfig(model="test-model")
        backend = VLLMBackend(config)
        assert backend.default_executable() == "vllm"

    def test_health_endpoint(self):
        """Test health endpoint."""
        config = VLLMConfig(model="test-model")
        backend = VLLMBackend(config)
        assert backend.health_endpoint() == "/health"

    def test_get_port(self):
        """Test port retrieval."""
        config = VLLMConfig(model="test-model", port=9000)
        backend = VLLMBackend(config)
        assert backend._get_port() == 9000

    def test_build_launch_command_basic(self):
        """Test basic launch command."""
        config = VLLMConfig(model="meta-llama/Llama-3.1-8B")
        backend = VLLMBackend(config)

        # Mock check_executable to not fail
        backend.check_executable = lambda executable: None

        cmd = backend.build_launch_command()

        assert cmd[0] == "vllm"
        assert "serve" in cmd
        assert "meta-llama/Llama-3.1-8B" in cmd
        assert "--host" in cmd
        assert "--port" in cmd

    def test_build_launch_command_with_options(self):
        """Test launch command with options."""
        config = VLLMConfig(
            model="test-model",
            port=9000,
            tensor_parallel=2,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.85,
            quantization="awq",
        )
        backend = VLLMBackend(config)
        backend.check_executable = lambda executable: None

        cmd = backend.build_launch_command()

        assert "--port" in cmd
        assert "9000" in cmd
        assert "--tensor-parallel-size" in cmd
        assert "2" in cmd
        assert "--enable-prefix-caching" in cmd
        assert "--gpu-memory-utilization" in cmd
        assert "0.85" in cmd
        assert "--quantization" in cmd
        assert "awq" in cmd

    def test_build_launch_command_extra_args(self):
        """Test launch command with extra args."""
        config = VLLMConfig(
            model="test-model",
            extra_args={"enforce_eager": True, "seed": 42},
        )
        backend = VLLMBackend(config)
        backend.check_executable = lambda executable: None

        cmd = backend.build_launch_command()

        assert "--enforce-eager" in cmd
        assert "--seed" in cmd
        assert "42" in cmd

    def test_build_launch_command_requires_model(self):
        """Test launch command requires model."""
        config = VLLMConfig(endpoint="http://localhost:8000")
        backend = VLLMBackend(config)

        with pytest.raises(ValueError, match="Cannot start server without 'model'"):
            backend.build_launch_command()


class TestTGIBackend:
    """Tests for TGI backend."""

    def test_default_executable(self):
        """Test default executable name."""
        config = TGIConfig(model="test-model")
        backend = TGIBackend(config)
        assert backend.default_executable() == "text-generation-launcher"

    def test_health_endpoint(self):
        """Test health endpoint."""
        config = TGIConfig(model="test-model")
        backend = TGIBackend(config)
        assert backend.health_endpoint() == "/health"

    def test_get_port(self):
        """Test port retrieval."""
        config = TGIConfig(model="test-model", port=8080)
        backend = TGIBackend(config)
        assert backend._get_port() == 8080

    def test_default_port(self):
        """Test default port is 3000."""
        config = TGIConfig(model="test-model")
        backend = TGIBackend(config)
        assert backend._get_port() == 3000

    def test_build_launch_command_basic(self):
        """Test basic launch command."""
        config = TGIConfig(model="meta-llama/Llama-3.1-8B")
        backend = TGIBackend(config)
        backend.check_executable = lambda executable: None

        cmd = backend.build_launch_command()

        assert cmd[0] == "text-generation-launcher"
        assert "--model-id" in cmd
        assert "meta-llama/Llama-3.1-8B" in cmd
        assert "--hostname" in cmd
        assert "--port" in cmd

    def test_build_launch_command_with_options(self):
        """Test launch command with options."""
        config = TGIConfig(
            model="test-model",
            port=8080,
            num_shard=2,
            quantize="gptq",
            dtype="bfloat16",
            max_concurrent_requests=64,
            trust_remote_code=True,
        )
        backend = TGIBackend(config)
        backend.check_executable = lambda executable: None

        cmd = backend.build_launch_command()

        assert "--port" in cmd
        assert "8080" in cmd
        assert "--num-shard" in cmd
        assert "2" in cmd
        assert "--quantize" in cmd
        assert "gptq" in cmd
        assert "--dtype" in cmd
        assert "bfloat16" in cmd
        assert "--max-concurrent-requests" in cmd
        assert "64" in cmd
        assert "--trust-remote-code" in cmd

    def test_build_launch_command_extra_args(self):
        """Test launch command with extra args."""
        config = TGIConfig(
            model="test-model",
            extra_args={"max_waiting_tokens": 30, "speculate": 2},
        )
        backend = TGIBackend(config)
        backend.check_executable = lambda executable: None

        cmd = backend.build_launch_command()

        assert "--max-waiting-tokens" in cmd
        assert "30" in cmd
        assert "--speculate" in cmd
        assert "2" in cmd

    def test_build_launch_command_requires_model(self):
        """Test launch command requires model."""
        config = TGIConfig(endpoint="http://localhost:3000")
        backend = TGIBackend(config)

        with pytest.raises(ValueError, match="Cannot start server without 'model'"):
            backend.build_launch_command()


class TestExecutableResolution:
    """Tests for executable resolution logic."""

    def test_vllm_env_var_name(self):
        """Test vLLM environment variable name."""
        config = VLLMConfig(model="test-model")
        backend = VLLMBackend(config)
        assert backend._executable_env_var() == "SPLLEED_VLLM_PATH"

    def test_tgi_env_var_name(self):
        """Test TGI environment variable name."""
        config = TGIConfig(model="test-model")
        backend = TGIBackend(config)
        assert backend._executable_env_var() == "SPLLEED_TGI_PATH"

    def test_config_executable_takes_priority(self, monkeypatch):
        """Test config executable takes priority over env var."""
        monkeypatch.setenv("SPLLEED_VLLM_PATH", "/env/vllm")

        config = VLLMConfig(model="test-model", executable="/config/vllm")
        backend = VLLMBackend(config)

        assert backend.get_executable() == "/config/vllm"

    def test_env_var_takes_priority_over_default(self, monkeypatch):
        """Test env var takes priority over default."""
        monkeypatch.setenv("SPLLEED_TGI_PATH", "/custom/text-generation-launcher")

        config = TGIConfig(model="test-model")
        backend = TGIBackend(config)

        assert backend.get_executable() == "/custom/text-generation-launcher"

    def test_default_when_no_override(self):
        """Test default executable when no override."""
        config = VLLMConfig(model="test-model")
        backend = VLLMBackend(config)

        assert backend.get_executable() == "vllm"

    def test_check_executable_not_found(self):
        """Test check_executable raises helpful error."""
        config = VLLMConfig(model="test-model")
        backend = VLLMBackend(config)

        with pytest.raises(FileNotFoundError) as exc_info:
            backend.check_executable("nonexistent-command-12345")

        error_msg = str(exc_info.value)
        assert "Could not find 'nonexistent-command-12345'" in error_msg
        assert "SPLLEED_VLLM_PATH" in error_msg
        assert "pip install vllm" in error_msg
