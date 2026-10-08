from __future__ import annotations

import tomllib
from pathlib import Path

import finetune

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def project() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]


def test_the_dev_tools_are_pinned_exactly():
    """A floor on a linter is a floor on the output of CI. ruff adds rules in
    minor releases, so a range means a green branch can go red with no commit
    behind it."""
    dev = project()["optional-dependencies"]["dev"]
    assert sorted(dev) == ["pytest==9.1.1", "ruff==0.16.10"]
    assert all("==" in requirement for requirement in dev)


def test_the_version_is_declared_once_and_agrees():
    assert project()["version"] == finetune.__version__


def test_the_runtime_dependencies_are_not_optional():
    """Nothing here is an extra. A repository whose subject is fine-tuning
    cannot make torch opt in, and the offline guarantee is bought with lazy
    imports and HF_HUB_OFFLINE rather than by leaving the library out."""
    declared = {
        requirement.split(">")[0].split("=")[0].split("<")[0].strip()
        for requirement in project()["dependencies"]
    }
    assert {"torch", "transformers", "peft", "scikit-learn", "mlflow"} <= declared
    assert "optional-dependencies" in project()
    assert set(project()["optional-dependencies"]) == {"dev"}


def test_the_capped_dependencies_say_why_in_a_comment():
    # Both caps are behaviour changes that were hit, not guesses, and the
    # reason sits next to the pin so it can be re-checked when it is lifted.
    text = PYPROJECT.read_text()
    assert "transformers>=5.0.0,<6.0.0" in text
    assert "mlflow>=3.0.0" in text
    assert '"maintenance mode"' in text
    assert "vocab_file" in text


def test_the_console_script_points_at_the_cli():
    assert project()["scripts"] == {"ft": "finetune.cli:main"}
