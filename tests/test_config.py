from __future__ import annotations

from pathlib import Path

import pytest

from finetune.config import load_config
from finetune.errors import ConfigError


def write(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def test_extends_merges_child_over_parent(project: Path):
    cfg = load_config(project / "configs" / "smoke.toml")
    # base sets epochs nowhere and smoke sets 4; base sets seed and smoke does not.
    assert cfg.train.epochs == 4
    assert cfg.train.seed == 1337
    assert cfg.model.source == "random"


def test_origin_names_the_file_that_set_each_key(project: Path):
    cfg = load_config(project / "configs" / "smoke.toml")
    assert cfg.origin["train.epochs"] == "smoke.toml"
    assert cfg.origin["train.seed"] == "base.toml"
    assert cfg.origin["early_stopping.patience"] == "smoke.toml"
    assert cfg.origin["early_stopping.monitor"] == "base.toml"


def test_a_key_left_at_its_default_has_no_origin(project: Path):
    cfg = load_config(project / "configs" / "smoke.toml")
    assert "model.tokenizer" not in cfg.origin


def test_base_config_alone_is_not_runnable(project: Path):
    with pytest.raises(ConfigError) as exc:
        load_config(project / "configs" / "base.toml")
    assert "model.name is required" in str(exc.value)


def test_unknown_key_is_refused_and_names_the_file(project: Path):
    bad = write(
        project / "configs" / "typo.toml",
        'extends = "base.toml"\n[model]\nname = "x"\n[early_stopping]\nmin_dleta = 0.1\n',
    )
    with pytest.raises(ConfigError) as exc:
        load_config(bad)
    message = str(exc.value)
    assert "typo.toml" in message
    assert "early_stopping.min_dleta" in message
    assert "min_delta" in message


def test_unknown_section_is_refused(project: Path):
    bad = write(
        project / "configs" / "section.toml",
        'extends = "base.toml"\n[model]\nname = "x"\n[optimiser]\nkind = "adamw"\n',
    )
    with pytest.raises(ConfigError, match=r"unknown section \[optimiser\]"):
        load_config(bad)


def test_extends_cycle_is_reported_not_recursed(project: Path):
    write(project / "configs" / "a.toml", 'extends = "b.toml"\n')
    write(project / "configs" / "b.toml", 'extends = "a.toml"\n')
    with pytest.raises(ConfigError, match="extends forms a cycle"):
        load_config(project / "configs" / "a.toml")


def test_missing_file_is_a_config_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="no such config file"):
        load_config(tmp_path / "nope.toml")


def test_malformed_toml_names_the_file(project: Path):
    bad = write(project / "configs" / "broken.toml", "[model\nname = 1\n")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(bad)


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        (
            "[split]\nratios = { train = 0.7, validation = 0.2, test = 0.2 }",
            "must sum to 1.0",
        ),
        ("[split]\nratios = { train = 0.8, validation = 0.2 }", "exactly the keys"),
        ("[split]\nratios = { train = 1.2, validation = -0.1, test = -0.1 }", "positive"),
        ('[split]\nmethod = "kfold"', "quantile or independent"),
        ("[lora]\nr = 0", "lora.r must be >= 1"),
        ("[lora]\ndropout = 1.0", "must be in [0.0, 1.0)"),
        ("[lora]\ntarget_modules = []", "at least one module"),
        ("[lora]\nmodules_to_save = []", "classification head"),
        ("[train]\nepochs = 0", "epochs must be >= 1"),
        ("[train]\nlr = 0.0", "train.lr must be > 0"),
        ('[train]\ndevice = "tpu"', "must be cpu or auto"),
        ("[data]\nmax_length = 4", "max_length must be >= 8"),
        ("[early_stopping]\npatience = 0", "patience must be >= 1"),
        ('[early_stopping]\nmode = "highest"', "must be max or min"),
        ("[early_stopping]\nmin_delta = -0.01", "min_delta must be >= 0"),
    ],
)
def test_out_of_range_values_are_refused(project: Path, override: str, expected: str):
    bad = write(
        project / "configs" / "range.toml",
        f'extends = "smoke.toml"\n{override}\n',
    )
    with pytest.raises(ConfigError) as exc:
        load_config(bad)
    assert expected in str(exc.value)


def test_early_stopping_on_a_training_metric_is_refused(project: Path):
    bad = write(
        project / "configs" / "trainmon.toml",
        'extends = "smoke.toml"\n[early_stopping]\nmonitor = "train_loss"\n',
    )
    with pytest.raises(ConfigError) as exc:
        load_config(bad)
    assert "starting with val_" in str(exc.value)
    assert "trainmon.toml" in str(exc.value)


def test_fingerprint_is_stable_and_ignores_tracking(project: Path):
    first = load_config(project / "configs" / "smoke.toml")
    moved = write(
        project / "configs" / "moved.toml",
        'extends = "smoke.toml"\n[tracking]\nenabled = true\n'
        'uri = "sqlite:///somewhere/else.db"\nexperiment = "other"\n',
    )
    assert load_config(moved).fingerprint() == first.fingerprint()


def test_fingerprint_moves_when_the_experiment_moves(project: Path):
    first = load_config(project / "configs" / "smoke.toml")
    changed = write(
        project / "configs" / "r16.toml", 'extends = "smoke.toml"\n[lora]\nr = 16\n'
    )
    assert load_config(changed).fingerprint() != first.fingerprint()


def test_the_two_shipped_configs_have_different_fingerprints(project: Path):
    smoke = load_config(project / "configs" / "smoke.toml")
    real = load_config(project / "configs" / "distilbert-lora.toml")
    assert smoke.fingerprint() != real.fingerprint()
    # The split is shared on purpose, so the two runs are comparable.
    assert smoke.split.name == real.split.name
    assert smoke.split.salt == real.split.salt
    assert smoke.split.ratios == real.split.ratios
