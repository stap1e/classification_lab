"""Load ExperimentConfig from YAML profiles and environment variables."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from config.defaults import ExperimentConfig

_ENV_PATTERN = re.compile(
    r"\$\{([^}:]+)(?::-([^}]*))?\}"
)


def _expand_env_string(value: str) -> str:
    """Replace ``${VAR}`` and ``${VAR:-default}`` in a string."""

    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        return ""

    return _ENV_PATTERN.sub(repl, value)


def _resolve_string(value: str, context: dict[str, str]) -> str:
    expanded = _expand_env_string(value)
    return expanded.format(**context)


def _resolve_value(value: Any, context: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _resolve_string(value, context)
    if isinstance(value, dict):
        return {k: _resolve_value(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(v, context) for v in value]
    return value


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "experiments.yaml"


def list_profiles(config_path: Path | None = None) -> list[str]:
    path = config_path or _default_config_path()
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    experiments = data.get("experiments", {})
    return sorted(experiments.keys())


def load_experiment_config(
    profile: str,
    config_path: Path | str | None = None,
) -> ExperimentConfig:
    """
    Load one experiment profile from YAML.

    Raises:
        FileNotFoundError: config file missing
        KeyError: unknown profile name
    """
    path = Path(config_path) if config_path else _default_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    experiments = data.get("experiments") or {}
    if profile not in experiments:
        available = ", ".join(sorted(experiments.keys())) or "(none)"
        raise KeyError(f"Unknown profile {profile!r}. Available: {available}")

    raw_paths = data.get("paths") or {}
    context: dict[str, str] = {}
    for key, raw in raw_paths.items():
        context[key] = _resolve_string(str(raw), context)

    resolved = _resolve_value(experiments[profile], context)
    return ExperimentConfig.from_mapping(resolved)
