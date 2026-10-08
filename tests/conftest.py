"""Offline by construction, and the env vars are set before any import.

``HF_HUB_OFFLINE`` and ``TRANSFORMERS_OFFLINE`` are read by huggingface_hub at
import time, so they have to be set before anything imports transformers. That
works here because every torch and transformers import in ``finetune`` sits
inside the function that needs it, not at module scope. If one of them were
moved to the top of a module, the package would be imported during test
collection and these would arrive too late, which is the sort of regression
that only shows up as a test suite that quietly starts downloading a model.

With them set, any call that would reach the Hub raises instead.
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# mlflow prints a hint to stderr on import. Nothing in the suite imports it,
# and this keeps the output clean if something ever does.
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A scratch copy of the real corpus and configs.

    The real corpus, not a toy one, because the counts the README quotes come
    out of it and a test on different data cannot check them.
    """
    (tmp_path / "data").mkdir()
    shutil.copy(REPO / "data" / "intents.jsonl", tmp_path / "data" / "intents.jsonl")
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    return tmp_path


@pytest.fixture
def smoke_config(project: Path):
    from finetune.config import load_config

    return load_config(project / "configs" / "smoke.toml")


@pytest.fixture
def corpus(smoke_config, project: Path):
    from finetune.splitting import load_corpus

    return load_corpus(smoke_config, project)


@pytest.fixture
def recorded_split(smoke_config, corpus, project: Path):
    from finetune.splitting import make_split, write_split

    split = make_split(corpus, smoke_config)
    write_split(split, smoke_config, project)
    return split
