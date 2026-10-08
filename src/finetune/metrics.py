"""Metrics over integer label ids, with the label set passed in explicitly.

The explicit label set is the point of this module. ``sklearn`` infers the
labels from the data when the ``labels`` argument is left out, so a validation
fold that happens to contain no examples of one class gets a macro average over
the three classes present, while the fold before it averaged four. Both are
reported as ``macro_f1`` and neither is comparable to the other. Early stopping
on a metric whose denominator moves between epochs is not early stopping.
"""

from __future__ import annotations

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

Number = float


def classification_metrics(
    y_true: list[int], y_pred: list[int], labels: list[str]
) -> dict[str, object]:
    """Macro F1 first, with the per-class table that explains it.

    Macro and not accuracy, because the corpus this was built for runs about
    three to one between its commonest and rarest class. Accuracy on that is
    mostly a report on the majority class, and the rare class is the one the
    model is being asked to learn.

    ``zero_division=0`` is deliberate. A class the model never predicts has
    undefined precision, and the alternatives are a NaN that poisons the macro
    average or a warning nobody reads.

    One asymmetry to know about. ``precision_recall_fscore_support`` and
    ``confusion_matrix`` accept the label set and are given it, so ``macro_f1``
    and the per-class table always span all declared classes.
    ``balanced_accuracy_score`` and ``matthews_corrcoef`` have no such
    argument and average over the classes present in the data instead, and
    sklearn warns when that set is smaller. So a fold holding one class reports
    ``macro_f1`` 0.25 and ``balanced_accuracy`` 1.0 at the same time. Both are
    right about different denominators, and the one to early-stop on is
    ``macro_f1``, because it is the one whose denominator cannot move between
    epochs.
    """
    if not labels:
        raise ValueError("labels must be non-empty: the metric needs its own denominator")
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} rows, y_pred has {len(y_pred)}")
    if not y_true:
        raise ValueError("no rows to score")

    valid = range(len(labels))
    out_of_range = sorted({v for v in [*y_true, *y_pred] if v not in valid})
    if out_of_range:
        raise ValueError(
            f"label ids {out_of_range} are outside the declared label set of "
            f"{len(labels)}: {labels}"
        )

    ids = list(valid)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=ids, zero_division=0
    )
    macro_f1 = float(f1.mean())

    per_label = {
        labels[i]: {
            "precision": round(float(precision[i]), 6),
            "recall": round(float(recall[i]), 6),
            "f1": round(float(f1[i]), 6),
            "support": int(support[i]),
        }
        for i in ids
    }

    correct = sum(1 for a, b in zip(y_true, y_pred, strict=True) if a == b)
    return {
        "macro_f1": round(macro_f1, 6),
        "accuracy": round(correct / len(y_true), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        # Runs -1 to 1, so it is the metric that catches an early stopping rule
        # that assumed its monitor could not be negative.
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 6),
        "n": len(y_true),
        "per_label": per_label,
        "confusion": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
    }


def scalars(metrics: dict[str, object], prefix: str) -> dict[str, Number]:
    """The scalar metrics only, prefixed, for a tracking backend.

    ``per_label`` and ``confusion`` are dropped rather than flattened. MLflow
    takes one float per key, and a per-class table flattened into forty keys
    makes the run comparison view unreadable for no gain: the table belongs in
    an artefact, which is where the run writes it.
    """
    return {
        f"{prefix}_{key}": float(value)
        for key, value in metrics.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def confusion_table(metrics: dict[str, object], labels: list[str]) -> str:
    """The confusion matrix as text, rows true and columns predicted."""
    matrix = metrics["confusion"]
    width = max([len(name) for name in labels] + [5])
    header = " " * (width + 2) + "  ".join(f"{name[:5]:>5}" for name in labels)
    lines = [header]
    for index, name in enumerate(labels):
        cells = "  ".join(f"{value:>5}" for value in matrix[index])
        lines.append(f"{name:<{width}}  {cells}")
    return "\n".join(lines)
