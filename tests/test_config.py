"""Tests for configuration loading."""

import pytest
import yaml
from pydantic import ValidationError

from splleed.config.loader import (
    generate_example_config,
    load_config,
    load_yaml,
    merge_cli_overrides,
    validate_config,
)


class TestLoadYaml:
    """Tests for YAML loading."""

    def test_load_valid_yaml(self, tmp_path):
        """Test loading valid YAML."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
backend:
  type: vllm
  model: test-model
"""
        )

        data = load_yaml(config_path)
        assert data["backend"]["type"] == "vllm"
        assert data["backend"]["model"] == "test-model"

    def test_load_empty_yaml(self, tmp_path):
        """Test loading empty YAML."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        data = load_yaml(config_path)
        assert data == {}

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_yaml(tmp_path / "nonexistent.yaml")

    def test_load_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("invalid: yaml: content:")

        with pytest.raises(yaml.YAMLError):
            load_yaml(config_path)


class TestMergeCliOverrides:
    """Tests for CLI override merging."""

    def test_simple_override(self):
        """Test simple key override."""
        config = {"key": "value"}
        overrides = {"key": "new_value"}

        result = merge_cli_overrides(config, overrides)
        assert result["key"] == "new_value"

    def test_nested_override(self):
        """Test nested key override using dots."""
        config = {"backend": {"model": "old-model"}}
        overrides = {"backend.model": "new-model"}

        result = merge_cli_overrides(config, overrides)
        assert result["backend"]["model"] == "new-model"

    def test_create_nested_path(self):
        """Test creating nested path that doesn't exist."""
        config = {}
        overrides = {"backend.model": "test-model"}

        result = merge_cli_overrides(config, overrides)
        assert result["backend"]["model"] == "test-model"

    def test_list_from_comma_string(self):
        """Test converting comma-separated string to list."""
        config = {"concurrency": [1]}
        overrides = {"concurrency": "1,2,4,8"}

        result = merge_cli_overrides(config, overrides)
        assert result["concurrency"] == [1, 2, 4, 8]

    def test_skip_none_values(self):
        """Test that None values are skipped."""
        config = {"key": "value"}
        overrides = {"key": None}

        result = merge_cli_overrides(config, overrides)
        assert result["key"] == "value"

    def test_deep_override(self):
        """Test deeply nested override."""
        config = {"a": {"b": {"c": "old"}}}
        overrides = {"a.b.c": "new"}

        result = merge_cli_overrides(config, overrides)
        assert result["a"]["b"]["c"] == "new"


class TestLoadConfig:
    """Tests for full config loading."""

    def test_load_minimal_config(self, tmp_path):
        """Test loading minimal valid config."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
backend:
  type: vllm
  model: test-model
dataset:
  type: inline
  prompts:
    - "Hello world"
"""
        )

        config = load_config(config_path)
        assert config.backend.type == "vllm"
        assert config.backend.model == "test-model"

    def test_load_with_overrides(self, tmp_path):
        """Test loading with CLI overrides."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
backend:
  type: vllm
  model: original-model
dataset:
  type: inline
  prompts:
    - "Hello"
"""
        )

        config = load_config(
            config_path,
            overrides={"backend.model": "overridden-model"},
        )
        assert config.backend.model == "overridden-model"

    def test_load_full_config(self, tmp_path):
        """Test loading full configuration."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
backend:
  type: vllm
  model: test-model
  tensor_parallel: 2
  enable_prefix_caching: false

dataset:
  type: inline
  prompts:
    - "Test prompt"
  num_samples: 50

benchmark:
  mode: serve
  concurrency: [1, 2, 4]
  warmup: 3
  runs: 10

sampling:
  max_tokens: 100
  temperature: 0.5

output:
  path: results/test.json
  format: json
"""
        )

        config = load_config(config_path)

        assert config.backend.type == "vllm"
        assert config.backend.tensor_parallel == 2
        assert config.backend.enable_prefix_caching is False

        assert config.dataset.type == "inline"
        assert config.dataset.num_samples == 50

        assert config.benchmark.mode == "serve"
        assert config.benchmark.concurrency == [1, 2, 4]

        assert config.sampling.max_tokens == 100
        assert config.sampling.temperature == 0.5

    def test_load_invalid_config(self, tmp_path):
        """Test loading invalid config raises error."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
backend:
  type: invalid_backend_type
"""
        )

        with pytest.raises(ValidationError):
            load_config(config_path)


class TestValidateConfig:
    """Tests for configuration validation."""

    def test_valid_config(self):
        """Test validating valid config."""
        data = {
            "backend": {"type": "vllm", "model": "test"},
            "dataset": {"type": "inline", "prompts": ["hello"]},
        }

        errors = validate_config(data)
        assert errors == []

    def test_missing_required_field(self):
        """Test validation with missing required field."""
        data = {
            "backend": {"type": "vllm"},  # Missing model
        }

        errors = validate_config(data)
        assert len(errors) > 0

    def test_invalid_type(self):
        """Test validation with invalid type."""
        data = {
            "backend": {"type": "invalid", "model": "test"},
            "dataset": {"type": "inline", "prompts": ["hello"]},
        }

        errors = validate_config(data)
        assert len(errors) > 0


class TestGenerateExampleConfig:
    """Tests for example config generation."""

    def test_generate_valid_yaml(self):
        """Test that generated example is valid YAML."""
        example = generate_example_config()
        data = yaml.safe_load(example)

        assert data["backend"]["type"] == "vllm"
        assert "model" in data["backend"]

    def test_generate_loadable_config(self, tmp_path):
        """Test that generated example can be loaded."""
        example = generate_example_config()

        config_path = tmp_path / "config.yaml"
        config_path.write_text(example)

        # Should not raise
        config = load_config(config_path)
        assert config.backend.type == "vllm"


class TestTGIConfig:
    """Tests for TGI backend configuration."""

    def test_load_tgi_config(self, tmp_path):
        """Test loading TGI config."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
backend:
  type: tgi
  model: meta-llama/Llama-3.1-8B-Instruct
dataset:
  type: inline
  prompts:
    - "Hello world"
"""
        )

        config = load_config(config_path)
        assert config.backend.type == "tgi"
        assert config.backend.model == "meta-llama/Llama-3.1-8B-Instruct"

    def test_tgi_with_endpoint(self, tmp_path):
        """Test TGI config with endpoint (connect mode)."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
backend:
  type: tgi
  endpoint: http://localhost:3000
dataset:
  type: inline
  prompts:
    - "Hello"
"""
        )

        config = load_config(config_path)
        assert config.backend.type == "tgi"
        assert config.backend.endpoint == "http://localhost:3000"

    def test_tgi_full_config(self, tmp_path):
        """Test TGI config with all options."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
backend:
  type: tgi
  model: test-model
  port: 8080
  num_shard: 2
  quantize: awq
  dtype: bfloat16
  max_concurrent_requests: 64
  trust_remote_code: true
dataset:
  type: inline
  prompts:
    - "Test"
"""
        )

        config = load_config(config_path)
        assert config.backend.type == "tgi"
        assert config.backend.port == 8080
        assert config.backend.num_shard == 2
        assert config.backend.quantize == "awq"
        assert config.backend.dtype == "bfloat16"
        assert config.backend.max_concurrent_requests == 64
        assert config.backend.trust_remote_code is True

    def test_tgi_requires_model_or_endpoint(self, tmp_path):
        """Test TGI config requires either model or endpoint."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
backend:
  type: tgi
dataset:
  type: inline
  prompts:
    - "Hello"
"""
        )

        with pytest.raises(ValidationError):
            load_config(config_path)
