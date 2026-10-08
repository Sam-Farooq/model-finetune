from __future__ import annotations

import pytest

from finetune.metrics import classification_metrics, confusion_table, scalars

LABELS = ["billing", "cancel", "shipping", "technical"]


def test_macro_f1_on_a_hand_worked_example():
    # Two classes, four rows. Class 0: tp 1, fp 1, fn 0 -> p 0.5, r 1.0,
    # f1 0.667. Class 1: tp 1, fp 0, fn 1 -> p 1.0, r 0.5, f1 0.667.
    metrics = classification_metrics([0, 0, 1, 1], [0, 0, 0, 1], ["a", "b"])
    assert metrics["per_label"]["a"]["precision"] == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["per_label"]["a"]["recall"] == 1.0
    assert metrics["per_label"]["b"]["recall"] == 0.5
    assert metrics["macro_f1"] == pytest.approx(0.733333, abs=1e-5)
    assert metrics["accuracy"] == 0.75
    assert metrics["n"] == 4


def test_an_absent_class_still_counts_against_the_macro_average():
    """The reason the label set is an argument.

    Three rows, all of class 0, predicted correctly. Over the four declared
    labels that is a macro F1 of 1.0/4 = 0.25. If the labels were inferred from
    the data it would be 1.0, and nothing in the output would say that the
    denominator had changed.
    """
    with pytest.warns(UserWarning, match="A single label was found"):
        metrics = classification_metrics([0, 0, 0], [0, 0, 0], LABELS)
    assert metrics["macro_f1"] == 0.25
    assert metrics["accuracy"] == 1.0
    assert metrics["per_label"]["cancel"]["support"] == 0
    assert metrics["per_label"]["cancel"]["f1"] == 0.0


def test_balanced_accuracy_and_mcc_do_not_take_the_declared_label_set():
    """An asymmetry worth knowing about rather than hiding.

    ``precision_recall_fscore_support`` and ``confusion_matrix`` take a
    ``labels`` argument and are given one, so macro_f1 and the per-class table
    are computed over all four declared classes. ``balanced_accuracy_score``
    and ``matthews_corrcoef`` have no such argument: they average over the
    classes present in the data, and sklearn emits a warning saying so. One
    row of one class therefore reports macro_f1 0.25 and balanced_accuracy
    1.0, and both numbers are correct for what they measure.
    """
    with pytest.warns(UserWarning):
        metrics = classification_metrics([0, 0], [0, 0], LABELS)
    assert metrics["macro_f1"] == 0.25
    assert metrics["balanced_accuracy"] == 1.0


def test_a_class_never_predicted_scores_zero_rather_than_nan():
    metrics = classification_metrics([0, 1], [0, 0], ["a", "b"])
    assert metrics["per_label"]["b"]["precision"] == 0.0
    assert metrics["macro_f1"] == metrics["macro_f1"]  # not NaN


def test_the_majority_class_predictor_on_the_real_test_shape():
    """Four classes with supports 3, 1, 2, 3 and everything predicted as the
    last one. This is the smoke run's actual behaviour and its actual score."""
    y_true = [0, 0, 0, 1, 2, 2, 3, 3, 3]
    y_pred = [3] * 9
    metrics = classification_metrics(y_true, y_pred, LABELS)
    assert metrics["accuracy"] == pytest.approx(1 / 3, abs=1e-6)
    assert metrics["macro_f1"] == 0.125
    assert metrics["mcc"] == 0.0
    assert metrics["per_label"]["technical"]["recall"] == 1.0
    assert metrics["per_label"]["technical"]["precision"] == pytest.approx(1 / 3, abs=1e-6)


def test_mcc_goes_negative_when_the_model_is_worse_than_chance():
    # The metric that makes an early stopping floor of 0.0 a bug.
    metrics = classification_metrics([0, 0, 1, 1], [1, 1, 0, 0], ["a", "b"])
    assert metrics["mcc"] == -1.0


def test_balanced_accuracy_differs_from_accuracy_under_imbalance():
    y_true = [0] * 9 + [1]
    y_pred = [0] * 10
    metrics = classification_metrics(y_true, y_pred, ["a", "b"])
    assert metrics["accuracy"] == 0.9
    assert metrics["balanced_accuracy"] == 0.5


def test_confusion_matrix_is_true_by_predicted():
    metrics = classification_metrics([0, 1], [1, 1], ["a", "b"])
    assert metrics["confusion"] == [[0, 1], [0, 1]]


def test_an_empty_label_set_is_refused():
    with pytest.raises(ValueError, match="needs its own denominator"):
        classification_metrics([0], [0], [])


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="y_true has 2 rows, y_pred has 1"):
        classification_metrics([0, 1], [0], ["a", "b"])


def test_no_rows_is_refused():
    with pytest.raises(ValueError, match="no rows to score"):
        classification_metrics([], [], ["a"])


def test_a_label_id_outside_the_declared_set_is_refused():
    with pytest.raises(ValueError, match=r"label ids \[4\] are outside"):
        classification_metrics([0, 4], [0, 0], LABELS)


def test_scalars_prefixes_and_drops_the_nested_tables():
    metrics = classification_metrics([0, 1], [0, 1], ["a", "b"])
    flat = scalars(metrics, "val")
    assert flat["val_macro_f1"] == 1.0
    assert flat["val_n"] == 2.0
    assert "val_per_label" not in flat
    assert "val_confusion" not in flat
    assert all(isinstance(value, float) for value in flat.values())


def test_confusion_table_has_one_row_per_label():
    metrics = classification_metrics([0, 1, 2, 3], [0, 1, 2, 3], LABELS)
    lines = confusion_table(metrics, LABELS).splitlines()
    assert len(lines) == 5
    assert lines[1].startswith("billing")
    assert lines[4].startswith("technical")
