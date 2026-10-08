"""Tracking behind one small interface, with a null implementation.

The null tracker is not a no-op: it keeps everything it was handed, so a test
can assert what a run would have logged without a tracking server, a database
or an import of mlflow. That matters more than it sounds, because importing
mlflow costs about a second and pulls in sqlalchemy and a web stack, and a run
with tracking switched off should pay none of it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from .config import Config


class Tracker(Protocol):
    @contextmanager
    def run(self, name: str) -> Iterator[None]: ...

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...

    def log_artifact(self, path: Path) -> None: ...


class NullTracker:
    """Records calls in memory. Used by every test and by any run with
    ``tracking.enabled = false``, which is the default."""

    def __init__(self) -> None:
        self.runs: list[str] = []
        self.params: dict[str, Any] = {}
        self.metrics: list[tuple[int, dict[str, float]]] = []
        self.artifacts: list[str] = []

    @contextmanager
    def run(self, name: str) -> Iterator[None]:
        self.runs.append(name)
        yield

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.metrics.append((step, dict(metrics)))

    def log_artifact(self, path: Path) -> None:
        self.artifacts.append(str(path))

    def metric_series(self, key: str) -> list[float]:
        return [values[key] for _, values in self.metrics if key in values]


class MlflowTracker:
    """MLflow, imported on construction rather than at module import."""

    def __init__(self, uri: str, experiment: str) -> None:
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)

    @contextmanager
    def run(self, name: str) -> Iterator[None]:
        with self._mlflow.start_run(run_name=name):
            yield

    def log_params(self, params: dict[str, Any]) -> None:
        # MLflow params are immutable once set and stringified on the way in,
        # so every value is flattened here rather than at each call site.
        self._mlflow.log_params({key: str(value) for key, value in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        if path.is_dir():
            self._mlflow.log_artifacts(str(path), artifact_path=path.name)
        else:
            self._mlflow.log_artifact(str(path))


def make_tracker(cfg: Config) -> Tracker:
    if not cfg.tracking.enabled:
        return NullTracker()
    return MlflowTracker(cfg.tracking.uri, cfg.tracking.experiment)
