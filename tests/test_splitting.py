from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from finetune.config import SPLIT_NAMES, load_config
from finetune.errors import SplitError
from finetune.splitting import (
    Example,
    allocate,
    bucket,
    corpus_fingerprint,
    load_corpus,
    make_split,
    read_split,
    split_path,
    thin_cells,
    verify,
    write_split,
)


def with_method(cfg, method: str):
    return replace(cfg, split=replace(cfg.split, method=method))


def grow(examples: list[Example], n: int = 8) -> list[Example]:
    return [
        *examples,
        *[
            Example(id=f"n-{i:02d}", text="a new message about an invoice", label="billing")
            for i in range(n)
        ],
    ]


def test_bucket_is_stable_across_processes():
    # A literal, because the whole point is that this number does not change.
    # hash() would differ per process; sha256 does not.
    assert bucket("intents-v1", "b-01") == pytest.approx(0.49199967685991536, abs=1e-15)
    assert 0.0 <= bucket("s", "x") < 1.0


def test_bucket_depends_on_the_salt():
    assert bucket("one", "b-01") != bucket("two", "b-01")


@pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 64, 65, 999])
def test_allocate_always_sums_to_n(n: int):
    ratios = {"train": 0.7, "validation": 0.15, "test": 0.15}
    counts = allocate(n, ratios)
    assert sum(counts.values()) == n
    assert set(counts) == set(SPLIT_NAMES)


def test_allocate_gives_a_tiny_class_to_train_first():
    # Two examples of a label cannot fill three splits. Train wins on the
    # largest remainder, which is why validation and test lose the class and
    # why thin_cells exists.
    assert allocate(2, {"train": 0.7, "validation": 0.15, "test": 0.15}) == {
        "train": 2,
        "validation": 0,
        "test": 0,
    }


def test_quantile_split_has_the_counts_the_readme_quotes(smoke_config, corpus):
    split = make_split(corpus, smoke_config)
    totals = {name: len(split.ids(name)) for name in SPLIT_NAMES}
    assert totals == {"train": 45, "validation": 10, "test": 9}
    assert sum(totals.values()) == len(corpus) == 64
    assert split.counts(corpus)["validation"] == {
        "billing": 3,
        "cancel": 1,
        "shipping": 2,
        "technical": 4,
    }


def test_every_example_lands_in_exactly_one_split(smoke_config, corpus):
    split = make_split(corpus, smoke_config)
    assert set(split.assignments) == {example.id for example in corpus}
    pooled = [example_id for name in SPLIT_NAMES for example_id in split.ids(name)]
    assert len(pooled) == len(set(pooled)) == 64


def test_the_same_inputs_produce_a_byte_identical_record(smoke_config, corpus):
    first = make_split(corpus, smoke_config)
    shuffled = list(reversed(corpus))
    second = make_split(shuffled, smoke_config)
    assert json.dumps(first.to_record()) == json.dumps(second.to_record())


def test_quantile_is_exact_on_strata_and_moves_a_few_ids_when_the_corpus_grows(
    smoke_config, corpus
):
    cfg = with_method(smoke_config, "quantile")
    before = make_split(corpus, cfg)
    after = make_split(grow(corpus), cfg)
    moved = [e.id for e in corpus if before.assignments[e.id] != after.assignments[e.id]]
    # Measured, and the cost of exact strata: the per-label cut moves when the
    # label grows, so a few ids either side of a boundary change split.
    assert len(moved) == 3


def test_independent_never_moves_an_id_but_misses_a_class(smoke_config, corpus):
    cfg = with_method(smoke_config, "independent")
    before = make_split(corpus, cfg)
    after = make_split(grow(corpus), cfg)
    assert all(before.assignments[e.id] == after.assignments[e.id] for e in corpus)

    totals = {name: len(before.ids(name)) for name in SPLIT_NAMES}
    # 15 where the ratio asks for 9.6. At 64 examples the law of large numbers
    # is not available.
    assert totals == {"train": 38, "validation": 11, "test": 15}
    # And cancel is absent from validation entirely, which is what makes an
    # inferred label set dangerous: macro F1 would average three classes here
    # and four on the test split.
    assert "cancel" not in before.counts(corpus)["validation"]


def test_thin_cells_finds_the_classes_too_small_to_validate_on(smoke_config, corpus):
    split = make_split(corpus, smoke_config)
    thin = thin_cells(split, corpus, minimum=3)
    assert ("validation", "cancel", 1) in thin
    assert ("test", "cancel", 1) in thin
    assert ("validation", "shipping", 2) in thin
    assert not [cell for cell in thin if cell[1] == "technical"]
    assert thin_cells(split, corpus, minimum=0) == []


def test_fingerprint_ignores_text_and_catches_a_relabel(corpus):
    original = corpus_fingerprint(corpus)
    retyped = [replace(corpus[0], text="corrected typo"), *corpus[1:]]
    assert corpus_fingerprint(retyped) == original

    relabelled = [replace(corpus[0], label="cancel"), *corpus[1:]]
    assert corpus_fingerprint(relabelled) != original


def test_fingerprint_is_order_independent(corpus):
    assert corpus_fingerprint(corpus) == corpus_fingerprint(list(reversed(corpus)))


def test_verify_passes_on_the_corpus_it_was_built_from(smoke_config, corpus):
    verify(make_split(corpus, smoke_config), corpus, smoke_config)


def test_verify_reports_added_ids(smoke_config, corpus):
    split = make_split(corpus, smoke_config)
    with pytest.raises(SplitError) as exc:
        verify(split, grow(corpus, 3), smoke_config)
    message = str(exc.value)
    assert "3 id(s) not in the split" in message
    assert "n-00" in message
    assert "--force" in message


def test_verify_reports_removed_ids(smoke_config, corpus):
    split = make_split(corpus, smoke_config)
    with pytest.raises(SplitError, match="2 recorded id"):
        verify(split, corpus[:-2], smoke_config)


def test_verify_spots_a_relabel_when_the_ids_all_still_match(smoke_config, corpus):
    split = make_split(corpus, smoke_config)
    relabelled = [replace(corpus[0], label="cancel"), *corpus[1:]]
    with pytest.raises(SplitError) as exc:
        verify(split, relabelled, smoke_config)
    assert "a label changed" in str(exc.value)


def test_write_then_read_round_trips(smoke_config, corpus, project: Path):
    split = make_split(corpus, smoke_config)
    path = write_split(split, smoke_config, project)
    assert path == split_path(smoke_config, project)
    assert read_split(path).to_record() == split.to_record()


def test_reading_a_missing_record_says_how_to_make_one(smoke_config, project: Path):
    with pytest.raises(SplitError, match="ft split"):
        read_split(split_path(smoke_config, project))


def test_reading_a_truncated_record_names_the_missing_fields(project: Path):
    path = project / "half.json"
    path.write_text(json.dumps({"name": "x", "salt": "y"}))
    with pytest.raises(SplitError) as exc:
        read_split(path)
    assert "missing" in str(exc.value)
    assert "assignments" in str(exc.value)


def test_a_record_naming_an_unknown_split_is_refused(project: Path):
    path = project / "weird.json"
    path.write_text(
        json.dumps(
            {
                "name": "x",
                "salt": "y",
                "method": "quantile",
                "ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
                "corpus_fingerprint": "sha256:abc",
                "assignments": {"a": "train", "b": "holdout"},
            }
        )
    )
    with pytest.raises(SplitError, match=r"unknown splits \['holdout'\]"):
        read_split(path)


def test_ids_rejects_a_split_name_that_does_not_exist(smoke_config, corpus):
    with pytest.raises(SplitError, match="unknown split 'holdout'"):
        make_split(corpus, smoke_config).ids("holdout")


def test_duplicate_id_is_refused_with_both_line_numbers(project: Path):
    corpus = project / "data" / "dupe.jsonl"
    corpus.write_text(
        '{"id": "a", "text": "one", "label": "billing"}\n'
        '{"id": "b", "text": "two", "label": "cancel"}\n'
        '{"id": "a", "text": "three", "label": "billing"}\n'
    )
    cfg = load_config(project / "configs" / "smoke.toml")
    cfg = replace(cfg, data=replace(cfg.data, corpus="data/dupe.jsonl"))
    with pytest.raises(SplitError) as exc:
        load_corpus(cfg, project)
    message = str(exc.value)
    assert "duplicate id 'a'" in message
    assert ":3:" in message
    assert "line 1" in message


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('{"id": "a", "text": "x"}', "missing field(s) ['label']"),
        ('{"id": "a", "label": "billing"}', "missing field(s) ['text']"),
        ('{"id": "a", "text": "   ", "label": "billing"}', "empty text"),
        ("{not json", "not valid JSON"),
    ],
)
def test_malformed_corpus_lines_name_the_line(project: Path, line: str, expected: str):
    corpus = project / "data" / "bad.jsonl"
    corpus.write_text(line + "\n")
    cfg = load_config(project / "configs" / "smoke.toml")
    cfg = replace(cfg, data=replace(cfg.data, corpus="data/bad.jsonl"))
    with pytest.raises(SplitError) as exc:
        load_corpus(cfg, project)
    assert expected in str(exc.value)


def test_blank_lines_and_comments_are_skipped(project: Path):
    corpus = project / "data" / "sparse.jsonl"
    corpus.write_text(
        '// a note\n\n{"id": "a", "text": "one", "label": "billing"}\n\n'
        '{"id": "b", "text": "two", "label": "cancel"}\n'
    )
    cfg = load_config(project / "configs" / "smoke.toml")
    cfg = replace(cfg, data=replace(cfg.data, corpus="data/sparse.jsonl"))
    assert [e.id for e in load_corpus(cfg, project)] == ["a", "b"]


def test_an_empty_corpus_is_refused(project: Path):
    corpus = project / "data" / "empty.jsonl"
    corpus.write_text("\n\n")
    cfg = load_config(project / "configs" / "smoke.toml")
    cfg = replace(cfg, data=replace(cfg.data, corpus="data/empty.jsonl"))
    with pytest.raises(SplitError, match="no examples"):
        load_corpus(cfg, project)


def test_a_missing_corpus_names_the_path(smoke_config, project: Path):
    elsewhere = project / "elsewhere"
    elsewhere.mkdir()
    with pytest.raises(SplitError, match="corpus not found"):
        load_corpus(smoke_config, elsewhere)
