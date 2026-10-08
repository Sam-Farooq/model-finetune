"""Stop when the validation metric has not moved for ``patience`` epochs."""

from __future__ import annotations

from dataclasses import dataclass

from .config import EarlyStoppingConfig


@dataclass(frozen=True)
class Verdict:
    epoch: int
    value: float
    improved: bool
    should_stop: bool
    best_epoch: int
    best_value: float
    epochs_without_improvement: int
    reason: str


class EarlyStopper:
    """A rule, not a callback. It holds no model and returns no side effects.

    The caller decides what to do with ``should_stop``, which is what makes the
    rule testable without a training loop: feed it a list of numbers and assert
    which epoch it stops on.
    """

    def __init__(self, cfg: EarlyStoppingConfig) -> None:
        self.monitor = cfg.monitor
        self.mode = cfg.mode
        self.patience = cfg.patience
        self.min_delta = cfg.min_delta
        # Not 0.0. A monitor in [-1, 1] such as val_mcc starts below zero on a
        # model that has learned nothing, and a floor of 0.0 means the first
        # real epoch never counts as an improvement, so best_epoch stays unset
        # and the run stops at patience with no best checkpoint to restore.
        self.best_value = float("-inf") if cfg.mode == "max" else float("inf")
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        self.stopped = False

    def _is_better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta

    def update(self, epoch: int, metrics: dict[str, float]) -> Verdict:
        if self.monitor not in metrics:
            # A typo in the monitor key would otherwise read as "never
            # improved" and stop every run at patience, which looks like a
            # training problem and is a config problem.
            raise KeyError(
                f"early stopping monitor {self.monitor!r} is not in this epoch's "
                f"metrics. available: {sorted(metrics)}"
            )
        value = float(metrics[self.monitor])
        improved = self._is_better(value)

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        should_stop = self.epochs_without_improvement >= self.patience
        self.stopped = self.stopped or should_stop

        if improved:
            reason = f"{self.monitor} improved to {value:.4f}"
        elif should_stop:
            reason = (
                f"{self.monitor} has not improved by more than {self.min_delta} for "
                f"{self.epochs_without_improvement} epoch(s); best {self.best_value:.4f} "
                f"at epoch {self.best_epoch}"
            )
        else:
            reason = (
                f"{self.monitor} at {value:.4f}, no improvement on "
                f"{self.best_value:.4f} ({self.epochs_without_improvement} of "
                f"{self.patience})"
            )

        return Verdict(
            epoch=epoch,
            value=value,
            improved=improved,
            should_stop=should_stop,
            best_epoch=self.best_epoch,
            best_value=self.best_value,
            epochs_without_improvement=self.epochs_without_improvement,
            reason=reason,
        )
