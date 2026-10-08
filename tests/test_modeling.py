from __future__ import annotations

from dataclasses import replace

import pytest

from finetune.errors import ConfigError
from finetune.modeling import (
    SPECIAL_TOKENS,
    adapter_parameter_estimate,
    build_model,
    build_tokenizer,
    linear_module_names,
    parameter_report,
    resolve_targets,
    vocab_from_texts,
    wrap_with_lora,
)

LABELS = ["billing", "cancel", "shipping", "technical"]


# vocab_from_texts is pure python and needs nothing installed.
def test_vocab_starts_with_the_special_tokens_in_order():
    vocab = vocab_from_texts(["refund my card"])
    assert list(vocab)[: len(SPECIAL_TOKENS)] == list(SPECIAL_TOKENS)
    assert vocab["[PAD]"] == 0
    assert vocab["[UNK]"] == 1


def test_vocab_is_lowercased_deduplicated_and_sorted():
    vocab = vocab_from_texts(["Card card CARD", "refund!! a card?"])
    words = [token for token in vocab if token not in SPECIAL_TOKENS]
    assert words == ["a", "card", "refund"]


def test_vocab_ids_are_contiguous_from_zero():
    vocab = vocab_from_texts(["one two three", "four"])
    assert sorted(vocab.values()) == list(range(len(vocab)))


def test_vocab_does_not_depend_on_the_order_of_the_texts():
    assert vocab_from_texts(["b a", "c"]) == vocab_from_texts(["c", "a b"])


def test_vocab_keeps_digits_and_drops_punctuation():
    vocab = vocab_from_texts(["charged twice on the 14th!"])
    assert "14th" in vocab
    assert "twice" in vocab
    assert "twice!" not in vocab


@pytest.mark.torch
def test_the_tiny_model_builds_with_nothing_downloaded(smoke_config, corpus):
    texts = [example.text for example in corpus]
    tokenizer = build_tokenizer(smoke_config, texts)
    assert tokenizer.vocab_size == len(vocab_from_texts(texts))

    model = build_model(smoke_config, num_labels=4, vocab_size=tokenizer.vocab_size)
    assert model.config.num_hidden_layers == 2
    assert model.config.hidden_size == 32
    assert model.config.num_labels == 4


@pytest.mark.torch
def test_the_built_tokenizer_round_trips_real_words_rather_than_unknowns(
    smoke_config, corpus
):
    """The failure that the vocab_file approach produced silently.

    If the vocabulary does not load, every word encodes to [UNK] and training
    still runs. So this asserts the opposite: a sentence from the corpus
    survives the round trip.
    """
    texts = [example.text for example in corpus]
    tokenizer = build_tokenizer(smoke_config, texts)
    encoded = tokenizer("refund the annual charge")
    unknown = tokenizer.convert_tokens_to_ids("[UNK]")
    assert unknown not in encoded["input_ids"]
    assert len(encoded["input_ids"]) == 6  # [CLS] + four words + [SEP]


@pytest.mark.torch
def test_a_word_not_in_the_training_texts_becomes_unknown(smoke_config):
    tokenizer = build_tokenizer(smoke_config, ["refund the charge"])
    unknown = tokenizer.convert_tokens_to_ids("[UNK]")
    assert unknown in tokenizer("quantum chromodynamics")["input_ids"]


@pytest.mark.torch
def test_padding_and_truncation_respect_max_length(smoke_config, corpus):
    tokenizer = build_tokenizer(smoke_config, [e.text for e in corpus])
    batch = tokenizer(
        [corpus[0].text, "short"],
        padding="max_length",
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )
    assert batch["input_ids"].shape == (2, 16)
    assert batch["attention_mask"][1].sum().item() < 16


@pytest.mark.torch
def test_linear_module_names_finds_the_attention_projections(smoke_config, corpus):
    tokenizer = build_tokenizer(smoke_config, [e.text for e in corpus])
    model = build_model(smoke_config, 4, tokenizer.vocab_size)
    names = linear_module_names(model)
    assert {"query", "key", "value", "classifier"} <= set(names)


@pytest.mark.torch
def test_a_target_module_name_from_the_wrong_architecture_is_refused(smoke_config, corpus):
    """DistilBERT names and a BERT model. PEFT's own behaviour here has
    changed across versions, and a version that builds a model with no adapter
    in it hands you a tuning problem that is really a typo."""
    tokenizer = build_tokenizer(smoke_config, [e.text for e in corpus])
    model = build_model(smoke_config, 4, tokenizer.vocab_size)
    with pytest.raises(ConfigError) as exc:
        resolve_targets(model, ["q_lin", "v_lin"])
    message = str(exc.value)
    assert "q_lin" in message
    assert "available:" in message
    assert "query" in message


@pytest.mark.torch
def test_resolve_targets_passes_the_names_through_when_they_all_match(smoke_config, corpus):
    tokenizer = build_tokenizer(smoke_config, [e.text for e in corpus])
    model = build_model(smoke_config, 4, tokenizer.vocab_size)
    assert resolve_targets(model, ["query", "value"]) == ["query", "value"]


@pytest.mark.torch
def test_the_adapter_weighs_what_r_and_the_shapes_say_it_should(
    smoke_config, corpus, recorded_split
):
    # Built from the training texts only, which is what a run does, so these
    # are the same numbers `ft train configs/smoke.toml` prints.
    train_ids = set(recorded_split.ids("train"))
    train_texts = [e.text for e in corpus if e.id in train_ids]
    tokenizer = build_tokenizer(smoke_config, train_texts)
    model = wrap_with_lora(build_model(smoke_config, 4, tokenizer.vocab_size), smoke_config)
    report = parameter_report(model)

    # Two layers, two adapted projections each, r 4, width 32.
    expected = adapter_parameter_estimate(
        smoke_config,
        hidden_size=smoke_config.model.hidden_size,
        targets_per_layer=len(smoke_config.lora.target_modules),
        layers=smoke_config.model.num_hidden_layers,
    )
    assert expected == 1024
    assert report["adapter_parameters"] == expected
    # Linear(32 -> 4) is 128 weights and 4 biases.
    assert report["head_parameters"] == 132
    assert report["trainable_parameters"] == 1156
    assert report["total_parameters"] == 45320
    assert report["trainable_percent"] == pytest.approx(2.5508, abs=0.0001)


@pytest.mark.torch
def test_fitting_the_vocabulary_on_the_whole_corpus_makes_a_bigger_model(
    smoke_config, corpus, recorded_split
):
    """A test for the leak the vocabulary avoids.

    Fitting on every text gives the embedding table rows for words that only
    appear in validation and test. The model grows, and training would be
    updating embeddings for words it is not supposed to have seen.
    """
    train_ids = set(recorded_split.ids("train"))
    on_train = build_tokenizer(smoke_config, [e.text for e in corpus if e.id in train_ids])
    on_everything = build_tokenizer(smoke_config, [e.text for e in corpus])
    assert on_everything.vocab_size > on_train.vocab_size

    leaked = parameter_report(
        wrap_with_lora(build_model(smoke_config, 4, on_everything.vocab_size), smoke_config)
    )
    honest = parameter_report(
        wrap_with_lora(build_model(smoke_config, 4, on_train.vocab_size), smoke_config)
    )
    assert leaked["total_parameters"] > honest["total_parameters"]
    assert leaked["trainable_parameters"] == honest["trainable_parameters"]


@pytest.mark.torch
def test_the_frozen_base_is_the_bulk_of_the_model(smoke_config, corpus):
    tokenizer = build_tokenizer(smoke_config, [e.text for e in corpus])
    model = wrap_with_lora(build_model(smoke_config, 4, tokenizer.vocab_size), smoke_config)
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    assert len(frozen) > 20
    assert not any("lora_" in name for name in frozen)


@pytest.mark.torch
def test_the_classification_head_is_trainable_after_wrapping(smoke_config, corpus):
    tokenizer = build_tokenizer(smoke_config, [e.text for e in corpus])
    model = wrap_with_lora(build_model(smoke_config, 4, tokenizer.vocab_size), smoke_config)
    head = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and "classifier" in name
    ]
    assert head, "the randomly initialised head has to be trainable"


@pytest.mark.torch
def test_a_modules_to_save_that_matches_nothing_is_refused(smoke_config, corpus):
    """Catches the quiet version: an adapter on a frozen random head."""
    cfg = replace(smoke_config, lora=replace(smoke_config.lora, modules_to_save=["score"]))
    tokenizer = build_tokenizer(cfg, [e.text for e in corpus])
    model = build_model(cfg, 4, tokenizer.vocab_size)
    with pytest.raises(Exception) as exc:  # noqa: B017 - peft raises its own type
        wrap_with_lora(model, cfg)
    assert "score" in str(exc.value)


@pytest.mark.torch
def test_raising_r_raises_the_adapter_count_in_proportion(smoke_config, corpus):
    tokenizer = build_tokenizer(smoke_config, [e.text for e in corpus])
    small = parameter_report(
        wrap_with_lora(build_model(smoke_config, 4, tokenizer.vocab_size), smoke_config)
    )
    doubled = replace(smoke_config, lora=replace(smoke_config.lora, r=8))
    large = parameter_report(
        wrap_with_lora(build_model(doubled, 4, tokenizer.vocab_size), doubled)
    )
    assert large["adapter_parameters"] == 2 * small["adapter_parameters"]
    assert large["head_parameters"] == small["head_parameters"]
