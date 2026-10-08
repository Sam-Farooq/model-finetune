from __future__ import annotations

import inspect
import sys
from dataclasses import replace
from pathlib import Path

from finetune.tracking import MlflowTracker, NullTracker, make_tracker


def test_a_disabled_config_gets_the_null_tracker(smoke_config):
    assert isinstance(make_tracker(smoke_config), NullTracker)


def test_choosing_the_null_tracker_does_not_import_mlflow(smoke_config):
    """The reason mlflow is imported inside MlflowTracker's constructor.

    Importing mlflow costs about a second and pulls in sqlalchemy and a web
    stack. A run with tracking off should pay none of it, and the only way to
    keep that true is to assert it.
    """
    make_tracker(smoke_config)
    assert "mlflow" not in sys.modules


def test_the_two_trackers_offer_the_same_surface(smoke_config):
    """Checked by inspection rather than by construction, because building the
    mlflow one opens a store. If the interfaces drift, a run with tracking on
    fails at the first log call and not before."""
    expected = {"run", "log_params", "log_metrics", "log_artifact"}
    assert expected <= set(dir(NullTracker))
    assert expected <= set(dir(MlflowTracker))
    for name in sorted(expected):
        null = inspect.signature(getattr(NullTracker, name))
        live = inspect.signature(getattr(MlflowTracker, name))
        assert list(null.parameters) == list(live.parameters), name

    enabled = replace(smoke_config, tracking=replace(smoke_config.tracking, enabled=True))
    assert enabled.tracking.enabled is True


def test_the_null_tracker_keeps_everything_it_was_handed():
    tracker = NullTracker()
    with tracker.run("a-run"):
        tracker.log_params({"lora.r": 4})
        tracker.log_metrics({"val_macro_f1": 0.1, "train_loss": 1.3}, step=1)
        tracker.log_metrics({"val_macro_f1": 0.4, "train_loss": 1.1}, step=2)
        tracker.log_artifact(Path("runs/x"))

    assert tracker.runs == ["a-run"]
    assert tracker.params == {"lora.r": 4}
    assert [step for step, _ in tracker.metrics] == [1, 2]
    assert tracker.metric_series("val_macro_f1") == [0.1, 0.4]
    assert tracker.artifacts == ["runs/x"]


def test_metric_series_skips_a_step_that_did_not_log_the_key():
    tracker = NullTracker()
    tracker.log_metrics({"a": 1.0}, step=1)
    tracker.log_metrics({"b": 2.0}, step=2)
    assert tracker.metric_series("a") == [1.0]
