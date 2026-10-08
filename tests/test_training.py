from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from finetune.errors import SplitError
from finetune.splitting import load_corpus
from finetune.tracking import NullTracker
from finetune.training import (
    evaluate_adapter,
    label_set,
    resolve_device,
    run_dir_for,
    seed_everything,
    train,
)

pytestmark = pytest.mark.torch


def test_label_set_is_sorted_and_deduplicated(corpus):
    assert label_set(corpus) == ["billing", "cancel", "shipping", "technical"]


def test_resolve_device_honours_cpu(smoke_config):
    assert resolve_device(smoke_config) == "cpu"


def test_seeding_twice_gives_the_same_draw():
    import torch

    seed_everything(7)
    first = torch.rand(4)
    seed_everything(7)
    assert torch.equal(first, torch.rand(4))


@pytest.fixture
def run(smoke_config, recorded_split, project: Path):
    tracker = NullTracker()
    result = train(smoke_config, root=project, tracker=tracker)
    return result, tracker


def test_the_rule_stops_the_run_before_the_epoch_budget(run):
    result, _ = run
    # smoke.toml asks for 4 epochs with patience 2, and the majority class
    # predictor never improves, so patience wins on epoch 3.
    assert result.epochs_ran == 3
    assert len(result.history) == 3
    assert result.stopped_early is True
    assert "has not improved" in result.stop_reason
    assert result.best_epoch == 1


def test_the_history_has_one_row_per_epoch_with_the_expected_keys(run):
    result, _ = run
    for index, row in enumerate(result.history, start=1):
        assert row["epoch"] == index
        assert {"train_loss", "val_loss", "val_macro_f1", "val_accuracy"} <= set(row)


def test_the_training_loss_falls(run):
    result, _ = run
    losses = [row["train_loss"] for row in result.history]
    assert losses == sorted(losses, reverse=True)


def test_a_randomly_initialised_model_learns_the_majority_class(run):
    """Not a result. A statement of what 45 examples and 1,156 trainable
    parameters on a random base actually produce, so that nobody reads the
    smoke run as a score."""
    result, _ = run
    assert result.test["accuracy"] == pytest.approx(1 / 3, abs=1e-6)
    assert result.test["macro_f1"] == 0.125
    assert result.test["n"] == 9
    # Every row predicted as the same class: one column of the confusion
    # matrix holds everything.
    columns_used = [
        index for index in range(4) if any(row[index] for row in result.test["confusion"])
    ]
    assert len(columns_used) == 1


def test_the_parameter_report_matches_the_readme(run):
    result, _ = run
    assert result.parameters == {
        "trainable_parameters": 1156,
        "total_parameters": 45320,
        "trainable_percent": pytest.approx(2.5508, abs=0.0001),
        "adapter_parameters": 1024,
        "head_parameters": 132,
    }


def test_two_runs_of_the_same_config_agree(smoke_config, recorded_split, project: Path):
    first = train(smoke_config, root=project, tracker=NullTracker())
    second = train(smoke_config, root=project, tracker=NullTracker())
    assert [row["train_loss"] for row in first.history] == [
        row["train_loss"] for row in second.history
    ]
    assert first.test["macro_f1"] == second.test["macro_f1"]


def test_the_tracker_gets_the_config_the_parameter_counts_and_the_split(run):
    _, tracker = run
    assert tracker.runs and tracker.runs[0].startswith("intents-v1-")
    assert tracker.params["lora.r"] == 4
    assert tracker.params["train.lr"] == 0.005
    assert tracker.params["trainable_parameters"] == 1156
    assert tracker.params["labels"] == "billing,cancel,shipping,technical"
    assert json.loads(tracker.params["split_counts"])["train"]["billing"] == 12
    # The base has no name on any hub, so it is identified by its weights.
    assert tracker.params["base_checksum"].startswith("sha256:")


def test_tracking_excludes_itself_from_the_logged_params(run):
    _, tracker = run
    assert not [key for key in tracker.params if key.startswith("tracking.")]


def test_one_metric_point_per_epoch_plus_the_test_row(run):
    result, tracker = run
    assert tracker.metric_series("val_macro_f1") == [
        row["val_macro_f1"] for row in result.history
    ]
    assert tracker.metric_series("test_macro_f1") == [result.test["macro_f1"]]


def test_the_test_split_is_scored_once(run):
    _, tracker = run
    assert len(tracker.metric_series("test_macro_f1")) == 1


def test_the_artefacts_land_in_a_directory_keyed_on_the_config(
    run, smoke_config, project: Path
):
    result, tracker = run
    run_dir = run_dir_for(smoke_config, project)
    assert Path(result.run_dir) == run_dir
    assert smoke_config.fingerprint() in run_dir.name
    assert (run_dir / "adapter" / "adapter_config.json").exists()
    assert (run_dir / "adapter" / "adapter_model.safetensors").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "confusion.txt").exists()
    assert tracker.artifacts == [str(run_dir)]


def test_the_saved_adapter_is_small(run, smoke_config, project: Path):
    run_dir = run_dir_for(smoke_config, project)
    weights = run_dir / "adapter" / "adapter_model.safetensors"
    # 1,156 trainable parameters at four bytes is under 5 KB of tensor data.
    # The point of LoRA is that this is what a checkpoint costs.
    assert weights.stat().st_size < 20_000


def test_labels_json_records_what_the_adapter_was_trained_against(
    run, smoke_config, project: Path
):
    meta = json.loads((run_dir_for(smoke_config, project) / "labels.json").read_text())
    assert meta["labels"] == ["billing", "cancel", "shipping", "technical"]
    assert meta["config_fingerprint"] == smoke_config.fingerprint()
    assert meta["corpus_fingerprint"].startswith("sha256:")
    assert meta["base_source"] == "random"


def test_metrics_json_holds_the_whole_run(run, smoke_config, project: Path):
    saved = json.loads((run_dir_for(smoke_config, project) / "metrics.json").read_text())
    assert saved["best_epoch"] == 1
    assert len(saved["history"]) == 3
    assert saved["test"]["macro_f1"] == 0.125
    assert saved["parameters"]["adapter_parameters"] == 1024


def test_confusion_txt_covers_validation_and_test(run, smoke_config, project: Path):
    text = (run_dir_for(smoke_config, project) / "confusion.txt").read_text()
    assert "validation" in text
    assert "test" in text
    assert "technical" in text


def test_evaluating_the_saved_adapter_reproduces_the_run(run, smoke_config, project: Path):
    """The round trip that matters: a seeded random base rebuilt from the
    config, plus the saved adapter, scores exactly what training reported.

    Worth knowing why it holds. For source = "random" the base weights come
    from the seed, so rebuilding is reproducible. A real checkpoint is
    identified by its name instead, and this property is a feature of the
    smoke config rather than of the design.
    """
    result, _ = run
    adapter = run_dir_for(smoke_config, project) / "adapter"
    metrics = evaluate_adapter(smoke_config, adapter, root=project)
    assert metrics["macro_f1"] == result.test["macro_f1"]
    assert metrics["accuracy"] == result.test["accuracy"]
    assert metrics["confusion"] == result.test["confusion"]


def test_evaluating_against_a_changed_label_set_is_refused(run, smoke_config, project: Path):
    adapter = run_dir_for(smoke_config, project) / "adapter"
    meta_path = adapter.parent / "labels.json"
    meta = json.loads(meta_path.read_text())
    meta["labels"] = ["cancel", "billing", "shipping", "technical"]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(SplitError, match="permute every prediction"):
        evaluate_adapter(smoke_config, adapter, root=project)


def test_evaluating_without_the_label_record_is_refused(run, smoke_config, project: Path):
    adapter = run_dir_for(smoke_config, project) / "adapter"
    (adapter.parent / "labels.json").unlink()
    with pytest.raises(SplitError, match="cannot map ids back to labels"):
        evaluate_adapter(smoke_config, adapter, root=project)


def test_training_without_a_recorded_split_is_refused(smoke_config, project: Path):
    with pytest.raises(SplitError, match="ft split"):
        train(smoke_config, root=project, tracker=NullTracker())


def test_training_against_a_stale_split_is_refused(
    smoke_config, recorded_split, project: Path
):
    corpus = project / "data" / "intents.jsonl"
    with corpus.open("a") as handle:
        handle.write('{"id": "x-01", "text": "a brand new message", "label": "billing"}\n')
    with pytest.raises(SplitError, match="does not match the corpus"):
        train(smoke_config, root=project, tracker=NullTracker())


def test_an_empty_split_is_refused(smoke_config, recorded_split, project: Path):
    """A ratio set that starves one split is caught at training time with the
    name of the split, not as a division by zero inside the loop."""
    examples = load_corpus(smoke_config, project)
    record = json.loads((project / "splits" / "intents-v1.json").read_text())
    record["assignments"] = {example.id: "train" for example in examples}
    record["corpus_fingerprint"] = recorded_split.corpus_fingerprint
    (project / "splits" / "intents-v1.json").write_text(json.dumps(record))
    with pytest.raises(SplitError, match="the validation split is empty"):
        train(smoke_config, root=project, tracker=NullTracker())


def test_a_longer_budget_runs_more_epochs(smoke_config, recorded_split, project: Path):
    patient = replace(
        smoke_config,
        train=replace(smoke_config.train, epochs=6),
        early_stopping=replace(smoke_config.early_stopping, patience=10),
    )
    result = train(patient, root=project, tracker=NullTracker())
    assert result.epochs_ran == 6
    assert result.stopped_early is False
    assert result.stop_reason == "ran out of epochs"
    assert set(result.test["per_label"]) == {"billing", "cancel", "shipping", "technical"}
