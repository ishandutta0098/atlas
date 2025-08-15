"""Utility functions for the Atlas project.

This module contains utility functions for configuration loading, file operations,
and common functionality used across the project.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Global config cache
_config: Optional[Dict[str, Any]] = None
_config_path: Optional[str] = None


def load_config(
    config_path: Optional[str] = None, force_reload: bool = False
) -> Dict[str, Any]:
    """Load project configuration from YAML file.

    Args:
        config_path (Optional[str]): Path to configuration file.
            If None, uses default config.yaml in src/configs/
        force_reload (bool): Force reload even if config is already loaded.

    Returns:
        Dict[str, Any]: Loaded configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    global _config, _config_path

    # Resolve config path
    if config_path is None:
        current_dir = Path(__file__).parent
        config_path = str((current_dir / "configs" / "config.yaml").absolute())
    else:
        config_path = os.path.abspath(config_path)

    # Return cached config if available and not forcing reload
    if _config is not None and _config_path == config_path and not force_reload:
        return _config

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            _config = yaml.safe_load(file)
            _config_path = config_path
            return _config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing config file {config_path}: {e}")


def get_config(key_path: str, default: Any = None) -> Any:
    """Get a configuration value using dot notation.

    Args:
        key_path (str): Dot-separated path to the config value (e.g., 'api.openai.model').
        default (Any): Default value if key is not found.

    Returns:
        Any: Configuration value or default.

    Examples:
        >>> get_config('api.openai.model')
        'gpt-5'
        >>> get_config('api.openai.timeout', 60)
        120
    """
    if _config is None:
        load_config()

    keys = key_path.split(".")
    value = _config

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def get_worker_count(num_workers: Optional[int] = None) -> int:
    """Determine the number of workers based on config and parameters.

    Args:
        num_workers (Optional[int]): User-specified number of workers.

    Returns:
        int: Number of workers to use (0 means sequential processing).
    """
    if num_workers is not None:
        return max(0, num_workers)

    workers_config = get_config("processing.workers", {})

    if not workers_config.get("auto_detect", True):
        return workers_config.get("min_workers", 2)

    # Auto-detect based on CPU count
    cpu_count = os.cpu_count() or 1
    cpu_ratio = workers_config.get("cpu_ratio", 0.5)
    min_workers = workers_config.get("min_workers", 2)
    max_workers = workers_config.get("max_workers", 16)

    auto_workers = max(min_workers, min(max_workers, int(cpu_count * cpu_ratio)))
    return auto_workers


def get_prompt_path(prompt_name: Optional[str] = None) -> str:
    """Get the full path to a prompt template file.

    Args:
        prompt_name (Optional[str]): Name of the prompt file.
            If None, uses default summarizer from config.

    Returns:
        str: Absolute path to the prompt file.
    """
    prompts_config = get_config("paths.prompts", {})
    base_dir = prompts_config.get("base_dir", "src/prompts")

    if prompt_name is None:
        prompt_name = prompts_config.get(
            "default_summarizer", "summarizer_youtube_v2.yaml"
        )

    # Resolve relative to project root
    current_dir = Path(__file__).parent.parent  # Go up from src/ to project root
    prompt_path = current_dir / base_dir / prompt_name
    return str(prompt_path.absolute())


def setup_logging() -> None:
    """Setup logging based on configuration."""
    log_config = get_config("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper())
    format_str = log_config.get("format", "[%(levelname)s] %(message)s")

    # Configure basic logging
    logging.basicConfig(level=level, format=format_str, force=True)

    # Setup file logging if enabled
    if log_config.get("file_logging", False):
        log_file = log_config.get("log_file", "logs/atlas.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_str))

        # Add to root logger
        logging.getLogger().addHandler(file_handler)


def ensure_output_folder(folder_path: str) -> str:
    """Ensure output folder exists and return absolute path.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        str: Absolute path to the folder.
    """
    abs_path = os.path.abspath(folder_path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path
