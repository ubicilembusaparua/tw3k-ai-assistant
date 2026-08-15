from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_approved_rag_dependencies_are_locked() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = {
        dependency.split(">=", maxsplit=1)[0]
        for dependency in project["project"]["dependencies"]
    }

    assert {"openai", "qdrant-client", "sentence-transformers"} <= dependencies
    assert (ROOT / "uv.lock").is_file()


def test_environment_example_lists_configuration_without_a_secret() -> None:
    variables = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", maxsplit=1)
            variables[key] = value

    assert "OPENAI_API_KEY" in variables
    assert variables["OPENAI_API_KEY"] == ""
    assert variables["QDRANT_URL"] == "http://localhost:6333"
