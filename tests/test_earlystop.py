from __future__ import annotations

import pytest

from finetune.config import EarlyStoppingConfig
from finetune.earlystop import EarlyStopper


def run(values: list[float], **kwargs) -> list:
    cfg = EarlyStoppingConfig(**{"monitor": "val_macro_f1", **kwargs})
    stopper = EarlyStopper(cfg)
    verdicts = []
    for epoch, value in enumerate(values, start=1):
        verdict = stopper.update(epoch, {"val_macro_f1": value})
        verdicts.append(verdict)
        if verdict.should_stop:
            break
    return verdicts


def test_a_rising_metric_never_stops():
    verdicts = run([0.1, 0.3, 0.5, 0.7], patience=2, min_delta=0.0)
    assert len(verdicts) == 4
    assert not any(v.should_stop for v in verdicts)
    assert verdicts[-1].best_epoch == 4


def test_it_stops_patience_epochs_after_the_best():
    verdicts = run([0.1, 0.8, 0.7, 0.7, 0.7], patience=3, min_delta=0.0)
    assert len(verdicts) == 5
    assert verdicts[-1].should_stop
    assert verdicts[-1].best_epoch == 2
    assert verdicts[-1].best_value == 0.8


def test_patience_of_one_stops_on_the_first_flat_epoch():
    verdicts = run([0.5, 0.5], patience=1, min_delta=0.0)
    assert len(verdicts) == 2
    assert verdicts[0].improved
    assert verdicts[1].should_stop


def test_the_counter_resets_on_an_improvement():
    verdicts = run([0.1, 0.1, 0.2, 0.2, 0.2], patience=3, min_delta=0.0)
    assert [v.epochs_without_improvement for v in verdicts] == [0, 1, 0, 1, 2]
    assert not verdicts[-1].should_stop


def test_min_delta_stops_a_run_that_is_still_rising():
    """The documented cost of min_delta.

    A metric climbing by 0.0005 an epoch never clears a min_delta of 0.002, so
    patience runs out and the run is stopped on epoch 4 while the metric is
    still going up. It had gained 0.0015 over three epochs by then. That is the
    trade min_delta is: it buys immunity to noise and it pays for it by
    stopping a slow but real improvement.
    """
    verdicts = run([0.5000, 0.5005, 0.5010, 0.5015], patience=3, min_delta=0.002)
    assert verdicts[-1].should_stop
    assert verdicts[-1].epoch == 4
    assert verdicts[-1].best_epoch == 1
    assert verdicts[-1].best_value == 0.5
    assert verdicts[-1].value > verdicts[0].value


def test_a_crawl_that_eventually_clears_min_delta_does_count():
    # min_delta is measured against the best value, not the previous epoch, so
    # small gains accumulate rather than being discarded one at a time.
    verdicts = run([0.500, 0.501, 0.502, 0.503], patience=3, min_delta=0.002)
    assert len(verdicts) == 4
    assert verdicts[-1].improved
    assert verdicts[-1].best_epoch == 4


def test_min_delta_accepts_a_jump():
    verdicts = run([0.50, 0.55], patience=1, min_delta=0.002)
    assert verdicts[-1].improved
    assert not verdicts[-1].should_stop


def test_mode_min_treats_a_fall_as_an_improvement():
    cfg = EarlyStoppingConfig(monitor="val_loss", mode="min", patience=2, min_delta=0.0)
    stopper = EarlyStopper(cfg)
    verdicts = [
        stopper.update(epoch, {"val_loss": value})
        for epoch, value in enumerate([1.0, 0.8, 0.9, 0.95], start=1)
    ]
    assert [v.improved for v in verdicts] == [True, True, False, False]
    assert verdicts[-1].should_stop
    assert verdicts[-1].best_value == 0.8
    assert verdicts[-1].best_epoch == 2


def test_a_monitor_that_starts_negative_still_improves():
    """The regression test for the bug this rule shipped with.

    ``best_value`` was initialised to 0.0 for mode=max, which is fine for
    macro F1 and wrong for anything that can be negative. With val_mcc at
    -0.4 then -0.1, neither epoch beat 0.0, so nothing was ever recorded as
    an improvement: best_epoch stayed -1, the run stopped at patience, and
    there was no best state to restore. The floor is -inf now.
    """
    cfg = EarlyStoppingConfig(monitor="val_mcc", mode="max", patience=2, min_delta=0.0)
    stopper = EarlyStopper(cfg)
    first = stopper.update(1, {"val_mcc": -0.4})
    assert first.improved
    assert first.best_epoch == 1
    second = stopper.update(2, {"val_mcc": -0.1})
    assert second.improved
    assert second.best_value == -0.1
    assert not second.should_stop


def test_mode_min_starts_from_positive_infinity():
    cfg = EarlyStoppingConfig(monitor="val_loss", mode="min", patience=2)
    stopper = EarlyStopper(cfg)
    assert stopper.update(1, {"val_loss": 12345.0}).improved


def test_a_misspelled_monitor_raises_rather_than_never_improving():
    cfg = EarlyStoppingConfig(monitor="val_macro_f", patience=2)
    stopper = EarlyStopper(cfg)
    with pytest.raises(KeyError) as exc:
        stopper.update(1, {"val_macro_f1": 0.5, "val_loss": 1.0})
    message = str(exc.value)
    assert "val_macro_f" in message
    assert "val_loss" in message


def test_the_reason_names_the_best_epoch_and_the_threshold():
    verdicts = run([0.9, 0.1, 0.1, 0.1], patience=3, min_delta=0.002)
    reason = verdicts[-1].reason
    assert "has not improved by more than 0.002" in reason
    assert "best 0.9000 at epoch 1" in reason


def test_a_held_verdict_reports_progress_toward_patience():
    verdicts = run([0.9, 0.5], patience=3, min_delta=0.0)
    assert "1 of 3" in verdicts[-1].reason
