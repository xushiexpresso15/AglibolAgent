"""Configuration manager for Aglibol Agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from aglibol.core.types import HardwareTier


class HardwareConfig(BaseModel):
    auto_detect: bool = True
    vram_gb: float | None = None
    ram_gb: float | None = None
    tier: HardwareTier | None = None


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
    timeout: int = 180
    max_retries: int = 3
    retry_delay: float = 2.0


class ModelsConfig(BaseModel):
    chat: str = "qwen2.5:7b"
    planner: str = "qwen2.5:7b"
    researcher: str = "qwen2.5:7b"
    coder: str = "qwen2.5-coder:7b"
    reviewer: str = "qwen2.5:7b"
    writer: str = "qwen2.5:7b"
    tester: str = "qwen2.5-coder:7b"
    doc_writer: str = "phi3:mini"
    vision: str = "moondream2"
    embedder: str = "nomic-embed-text"


class OptimizationConfig(BaseModel):
    num_ctx: Any = "auto"
    keep_alive: str = "auto"
    flash_attention: bool = True
    num_gpu: int = -1
    max_output_tokens: int = 4096


class WorkflowConfig(BaseModel):
    name: str = "default"
    max_self_correction: int = 3
    enable_human_approval: bool = False
    checkpoint_every_step: bool = True


class StorageConfig(BaseModel):
    brain_dir: str = "~/.aglibol"
    max_checkpoints: int = 50
    persist_messages: bool = True

    @property
    def resolved_brain_dir(self) -> Path:
        return Path(os.path.expanduser(self.brain_dir)).resolve()


class AppConfig(BaseModel):
    """Top-level Aglibol Agent configuration."""

    version: str = "0.1.0"
    agent_mode: str = "auto"
    safety_mode: str = "balanced"
    default_workspace: str | None = None
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)


class ConfigManager:
    """Loads, merges, and validates configuration from YAML files and environment variables."""

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> AppConfig:
        """Load configuration hierarchy: default.yaml -> ~/.aglibol/config.yaml -> custom config_path."""
        base_dict: dict[str, Any] = {}

        # 1. Project default.yaml
        default_file = Path(__file__).parent.parent.parent.parent / "config" / "default.yaml"
        if default_file.exists():
            try:
                with open(default_file, encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                    if isinstance(content, dict) and "aglibol" in content:
                        base_dict = content["aglibol"]
            except Exception as e:
                import logging

                logging.getLogger("aglibol.config").warning("Config error: %s", e)
                pass

        # 2. User home config (~/.aglibol/config.yaml)
        user_file = Path(os.path.expanduser("~/.aglibol/config.yaml"))
        if user_file.exists():
            try:
                with open(user_file, encoding="utf-8") as f:
                    user_dict = yaml.safe_load(f)
                    if isinstance(user_dict, dict):
                        cls._deep_merge(base_dict, user_dict.get("aglibol", user_dict))
            except Exception as e:
                import logging

                logging.getLogger("aglibol.config").warning("Config error: %s", e)
                pass

        # 3. Explicit config path override
        if config_path:
            p = Path(config_path)
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    custom_dict = yaml.safe_load(f)
                    if isinstance(custom_dict, dict):
                        cls._deep_merge(base_dict, custom_dict.get("aglibol", custom_dict))

        # 4. Environment variable overrides (e.g., OLLAMA_HOST)
        if "OLLAMA_HOST" in os.environ:
            base_dict.setdefault("ollama", {})["host"] = os.environ["OLLAMA_HOST"]

        return AppConfig(**base_dict)

    @classmethod
    def load_tier_profile(cls, tier: HardwareTier) -> dict[str, Any]:
        """Load profile configuration YAML for a given tier if available."""
        profile_file = (
            Path(__file__).parent.parent.parent.parent
            / "config"
            / "profiles"
            / f"{tier.value}.yaml"
        )
        if profile_file.exists():
            try:
                with open(profile_file, encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                import logging

                logging.getLogger("aglibol.config").warning("Config error: %s", e)
                pass
        return {}

    @staticmethod
    def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                ConfigManager._deep_merge(target[key], value)
            else:
                target[key] = value

    @classmethod
    def set_user_config_value(cls, key_path: str, value: Any) -> Path:
        """Set a configuration value using dot notation (e.g. 'models.coder') and persist to ~/.aglibol/config.yaml."""
        user_file = Path(os.path.expanduser("~/.aglibol/config.yaml"))
        user_file.parent.mkdir(parents=True, exist_ok=True)

        existing_data: dict[str, Any] = {}
        if user_file.exists():
            try:
                with open(user_file, encoding="utf-8") as f:
                    existing_data = yaml.safe_load(f) or {}
            except Exception as e:
                import logging

                logging.getLogger("aglibol.config").warning("Config error: %s", e)
                existing_data = {}

        root = existing_data.setdefault("aglibol", {})

        parts = key_path.split(".")
        current = root
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        val_to_save = (
            value.value
            if hasattr(value, "value")
            else (str(value) if isinstance(value, Path) else value)
        )
        current[parts[-1]] = val_to_save

        temp_file = user_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                yaml.dump(existing_data, f, default_flow_style=False)
            os.replace(temp_file, user_file)
        except Exception as e:
            import logging

            logging.getLogger("aglibol.config").warning("Config error: %s", e)
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)

        return user_file
