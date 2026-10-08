from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune.cli import main


def ft(*args: str, root: Path) -> int:
    return main(["--root", str(root), *args])


def test_config_resolves_and_shows_where_each_key_came_from(project: Path, capsys):
    assert ft("config", str(project / "configs" / "smoke.toml"), root=project) == 0
    out = capsys.readouterr().out
    assert "fingerprint" in out
    assert "train.epochs" in out
    assert "smoke.toml" in out
    assert "base.toml" in out


def test_config_json_is_machine_readable(project: Path, capsys):
    assert ft("config", str(project / "configs" / "smoke.toml"), "--json", root=project) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["lora"]["r"] == 4
    assert len(payload["fingerprint"]) == 16


def test_a_bad_config_exits_two_and_says_why(project: Path, capsys):
    assert ft("config", str(project / "configs" / "base.toml"), root=project) == 2
    assert "model.name is required" in capsys.readouterr().err


def test_a_missing_config_exits_two(project: Path, capsys):
    assert ft("config", str(project / "configs" / "nope.toml"), root=project) == 2
    assert "no such config file" in capsys.readouterr().err


def test_split_creates_the_record_and_prints_the_counts(project: Path, capsys):
    assert ft("split", str(project / "configs" / "smoke.toml"), root=project) == 0
    out = capsys.readouterr().out
    assert (project / "splits" / "intents-v1.json").exists()
    assert "method=quantile" in out
    assert "billing" in out
    # The warning that decides how much a validation score is worth.
    assert "thin: validation/cancel has 1" in out


def test_split_is_idempotent_and_says_so(project: Path, capsys):
    ft("split", str(project / "configs" / "smoke.toml"), root=project)
    before = (project / "splits" / "intents-v1.json").read_text()
    capsys.readouterr()

    assert ft("split", str(project / "configs" / "smoke.toml"), root=project) == 0
    assert "unchanged" in capsys.readouterr().out
    assert (project / "splits" / "intents-v1.json").read_text() == before


def test_both_configs_produce_the_same_split_record(project: Path, capsys):
    ft("split", str(project / "configs" / "smoke.toml"), root=project)
    first = (project / "splits" / "intents-v1.json").read_text()
    assert ft("split", str(project / "configs" / "distilbert-lora.toml"), root=project) == 0
    assert (project / "splits" / "intents-v1.json").read_text() == first
    assert "unchanged" in capsys.readouterr().out


def test_split_exits_one_when_the_corpus_has_moved(project: Path, capsys):
    ft("split", str(project / "configs" / "smoke.toml"), root=project)
    capsys.readouterr()
    with (project / "data" / "intents.jsonl").open("a") as handle:
        handle.write('{"id": "x-01", "text": "something new", "label": "cancel"}\n')

    assert ft("split", str(project / "configs" / "smoke.toml"), root=project) == 1
    err = capsys.readouterr().err
    assert "does not match the corpus" in err
    assert "--force" in err


def test_force_rebuilds_and_reports_how_many_ids_moved(project: Path, capsys):
    ft("split", str(project / "configs" / "smoke.toml"), root=project)
    capsys.readouterr()
    with (project / "data" / "intents.jsonl").open("a") as handle:
        for index in range(8):
            handle.write(
                json.dumps(
                    {
                        "id": f"n-{index:02d}",
                        "text": "a new invoice question",
                        "label": "billing",
                    }
                )
                + "\n"
            )

    assert ft("split", str(project / "configs" / "smoke.toml"), "--force", root=project) == 0
    out = capsys.readouterr().out
    assert "3 of 64 ids that were in the old record changed split" in out
    assert "measuring a different test set" in out


@pytest.mark.torch
def test_train_then_eval_through_the_cli(project: Path, capsys):
    config = str(project / "configs" / "smoke.toml")
    assert ft("split", config, root=project) == 0
    capsys.readouterr()

    assert ft("train", config, root=project) == 0
    out = capsys.readouterr().out
    assert "1156 trainable of 45320" in out
    assert "1024 adapter, 132 head" in out
    assert "best epoch   1" in out
    assert "macro_f1 0.1250" in out

    assert ft("eval", config, root=project) == 0
    out = capsys.readouterr().out
    assert "macro_f1 0.1250" in out
    assert "accuracy 0.3333" in out
    assert "technical  precision 0.3333  recall 1.0000" in out


def test_eval_without_an_adapter_exits_one(project: Path, capsys):
    config = str(project / "configs" / "smoke.toml")
    ft("split", config, root=project)
    capsys.readouterr()
    assert ft("eval", config, root=project) == 1
    assert "ft train" in capsys.readouterr().err


def test_train_without_a_split_exits_one(project: Path, capsys):
    assert ft("train", str(project / "configs" / "smoke.toml"), root=project) == 1
    assert "ft split" in capsys.readouterr().err


def test_no_subcommand_is_a_usage_error(project: Path):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
