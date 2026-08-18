"""Runtime environment loading without requiring an extra production package."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv as _load_dotenv
except ModuleNotFoundError:
    _load_dotenv = None


def _parse_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_simple_dotenv(path: Path) -> bool:
    if not path.is_file():
        return False

    loaded = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        os.environ[key] = _parse_value(value)
        loaded = True

    return loaded


def load_environment(dotenv_path: str | Path | None = None) -> bool:
    """Load a local environment file when available.

    Compose injects environment variables directly, so production does not
    need a dotenv dependency. Local development keeps compatible loading for
    the simple KEY=value format used by .env.example.
    """

    path = Path(dotenv_path or ".env")
    if _load_dotenv is not None:
        return bool(_load_dotenv(path))
    return _load_simple_dotenv(path)
