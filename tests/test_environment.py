"""Tests for environment fingerprinting."""

from splleed.environment import (
    EngineInfo,
    EnvironmentInfo,
    GPUInfo,
    ModelInfo,
    SystemInfo,
    capture_environment,
    detect_system,
)


class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_create_basic(self):
        """Test creating basic GPU info."""
        gpu = GPUInfo(
            index=0,
            name="NVIDIA A100",
            vram_mb=81920,
            driver_version="550.54",
            cuda_version="12.4",
        )
        assert gpu.name == "NVIDIA A100"
        assert gpu.vram_mb == 81920


class TestEngineInfo:
    """Tests for EngineInfo dataclass."""

    def test_create_with_version(self):
        """Test creating engine info with version."""
        engine = EngineInfo(name="vllm", version="0.6.3")
        assert engine.name == "vllm"
        assert engine.version == "0.6.3"

    def test_create_without_version(self):
        """Test creating engine info without version."""
        engine = EngineInfo(name="tgi")
        assert engine.name == "tgi"
        assert engine.version is None


class TestSystemInfo:
    """Tests for SystemInfo dataclass."""

    def test_create(self):
        """Test creating system info."""
        system = SystemInfo(
            python_version="3.12.0",
            platform="Linux-6.6.0",
            hostname="benchmark-server",
            cpu_count=32,
            memory_gb=128.0,
        )
        assert system.python_version == "3.12.0"
        assert system.cpu_count == 32


class TestDetectSystem:
    """Tests for detect_system function."""

    def test_returns_system_info(self):
        """Test that detect_system returns valid SystemInfo."""
        system = detect_system()
        assert isinstance(system, SystemInfo)
        assert system.python_version  # Should have some version
        assert system.platform  # Should have some platform
        assert system.cpu_count >= 0


class TestEnvironmentInfo:
    """Tests for EnvironmentInfo dataclass."""

    def test_to_dict_empty(self):
        """Test to_dict with empty environment."""
        env = EnvironmentInfo()
        data = env.to_dict()
        assert isinstance(data, dict)
        # Empty environment should have empty dict
        assert "gpus" not in data or not data["gpus"]

    def test_to_dict_with_gpu(self):
        """Test to_dict with GPU info."""
        env = EnvironmentInfo(
            gpus=[
                GPUInfo(
                    index=0,
                    name="NVIDIA A100",
                    vram_mb=81920,
                    driver_version="550.54",
                    cuda_version="12.4",
                )
            ]
        )
        data = env.to_dict()
        assert "gpus" in data
        assert len(data["gpus"]) == 1
        assert data["gpus"][0]["name"] == "NVIDIA A100"

    def test_to_dict_with_engine(self):
        """Test to_dict with engine info."""
        env = EnvironmentInfo(engine=EngineInfo(name="vllm", version="0.6.3"))
        data = env.to_dict()
        assert data["engine"]["name"] == "vllm"
        assert data["engine"]["version"] == "0.6.3"

    def test_to_dict_with_model(self):
        """Test to_dict with model info."""
        env = EnvironmentInfo(
            model=ModelInfo(
                name="meta-llama/Llama-3.1-8B",
                quantization="awq",
            )
        )
        data = env.to_dict()
        assert data["model"]["name"] == "meta-llama/Llama-3.1-8B"
        assert data["model"]["quantization"] == "awq"

    def test_format_summary_empty(self):
        """Test format_summary with empty environment."""
        env = EnvironmentInfo()
        summary = env.format_summary()
        assert summary == "Unknown environment"

    def test_format_summary_with_data(self):
        """Test format_summary with data."""
        env = EnvironmentInfo(
            gpus=[
                GPUInfo(
                    index=0,
                    name="NVIDIA A100",
                    vram_mb=81920,
                    driver_version="550.54",
                    cuda_version="12.4",
                )
            ],
            engine=EngineInfo(name="vllm", version="0.6.3"),
        )
        summary = env.format_summary()
        assert "NVIDIA A100" in summary
        assert "vllm" in summary


class TestCaptureEnvironment:
    """Tests for capture_environment function."""

    def test_returns_environment_info(self):
        """Test that capture_environment returns EnvironmentInfo."""
        env = capture_environment(backend_type="vllm")
        assert isinstance(env, EnvironmentInfo)

    def test_captures_system_info(self):
        """Test that system info is captured."""
        env = capture_environment(backend_type="vllm")
        assert env.system is not None
        assert env.system.python_version

    def test_with_model_info(self):
        """Test capturing with model info."""
        env = capture_environment(
            backend_type="vllm",
            model_name="test-model",
            quantization="awq",
        )
        assert env.model is not None
        assert env.model.name == "test-model"
        assert env.model.quantization == "awq"

    def test_without_model_info(self):
        """Test capturing without model info."""
        env = capture_environment(backend_type="vllm")
        assert env.model is None
