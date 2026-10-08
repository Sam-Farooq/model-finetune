"""The training and evaluation loop.

Shape of a run: resolve the config, load the corpus, verify the recorded split
against it, build the model, attach the adapter, train with early stopping on
validation, restore the best adapter state, score the test split once, write
the artefacts, log everything to the tracker.

The test split is read exactly once per run, at the end. Scoring it per epoch
and picking the best epoch by it makes it a second validation set, and the
number stops meaning what the word test implies.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import SPLIT_NAMES, Config
from .earlystop import EarlyStopper
from .errors import SplitError
from .metrics import classification_metrics, confusion_table, scalars
from .modeling import Bundle, build_model, build_tokenizer, parameter_report, wrap_with_lora
from .splitting import Example, Split, load_corpus, read_split, split_path, verify
from .tracking import NullTracker, Tracker


@dataclass
class RunResult:
    config_fingerprint: str
    run_dir: str
    labels: list[str]
    epochs_ran: int
    best_epoch: int
    best_value: float
    stopped_early: bool
    stop_reason: str
    parameters: dict[str, float | int] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    test: dict[str, Any] = field(default_factory=dict)


def seed_everything(seed: int) -> None:
    """Seed Python, numpy and torch.

    What this does not buy: identical results on a different torch version, a
    different device, or a different batch size, since the order of the float
    additions changes. It fixes the initialisation, the shuffle order and the
    dropout masks, which is enough for two runs of the same config on the same
    machine to agree, and that is the claim being made.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def label_set(examples: list[Example]) -> list[str]:
    """Sorted, so the label to id mapping is a function of the corpus.

    Insertion order would make the mapping depend on which line of the JSONL
    came first, and an adapter trained under one order and evaluated under
    another permutes every prediction without raising anything.
    """
    return sorted({example.label for example in examples})


def resolve_device(cfg: Config) -> str:
    if cfg.train.device == "cpu":
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _subset(examples: list[Example], split: Split, name: str) -> list[Example]:
    wanted = set(split.ids(name))
    rows = [example for example in examples if example.id in wanted]
    if not rows:
        raise SplitError(f"the {name} split is empty")
    return sorted(rows, key=lambda example: example.id)


def _encode(tokenizer: Any, texts: list[str], max_length: int) -> dict[str, Any]:
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def _batches(count: int, batch_size: int, generator: Any | None = None) -> list[list[int]]:
    import torch

    order = (
        torch.randperm(count, generator=generator).tolist()
        if generator is not None
        else list(range(count))
    )
    return [order[start : start + batch_size] for start in range(0, count, batch_size)]


def _train_one_epoch(
    bundle: Bundle,
    rows: list[Example],
    label_to_id: dict[str, int],
    cfg: Config,
    optimizer: Any,
    device: str,
    generator: Any,
) -> float:
    import torch

    bundle.model.train()
    trainable = [p for p in bundle.model.parameters() if p.requires_grad]
    running = 0.0
    for indices in _batches(len(rows), cfg.train.batch_size, generator):
        batch = [rows[index] for index in indices]
        encoded = _encode(bundle.tokenizer, [row.text for row in batch], cfg.data.max_length)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        targets = torch.tensor([label_to_id[row.label] for row in batch], device=device)

        optimizer.zero_grad(set_to_none=True)
        output = bundle.model(**encoded, labels=targets)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.train.max_grad_norm)
        optimizer.step()
        running += float(output.loss.item()) * len(batch)
    return running / len(rows)


def evaluate_rows(
    bundle: Bundle,
    rows: list[Example],
    labels: list[str],
    cfg: Config,
    device: str,
) -> dict[str, Any]:
    import torch

    label_to_id = {label: index for index, label in enumerate(labels)}
    bundle.model.eval()
    predicted: list[int] = []
    actual: list[int] = []
    running = 0.0
    with torch.no_grad():
        for indices in _batches(len(rows), cfg.train.batch_size):
            batch = [rows[index] for index in indices]
            encoded = _encode(
                bundle.tokenizer, [row.text for row in batch], cfg.data.max_length
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            targets = torch.tensor([label_to_id[row.label] for row in batch], device=device)
            output = bundle.model(**encoded, labels=targets)
            running += float(output.loss.item()) * len(batch)
            predicted.extend(output.logits.argmax(dim=-1).tolist())
            actual.extend(targets.tolist())

    metrics = classification_metrics(actual, predicted, labels)
    metrics["loss"] = round(running / len(rows), 6)
    return metrics


def build_bundle(cfg: Config, train_texts: list[str], labels: list[str]) -> Bundle:
    tokenizer = build_tokenizer(cfg, train_texts)
    vocab_size = len(tokenizer.get_vocab())
    base = build_model(cfg, num_labels=len(labels), vocab_size=vocab_size)
    model = wrap_with_lora(base, cfg)
    return Bundle(
        model=model, tokenizer=tokenizer, kind=cfg.model.source, vocab_size=vocab_size
    )


def base_checksum(model: Any) -> str:
    """Identify an unnamed base model by its weights.

    Only meaningful for ``source = "random"``. A pretrained base is identified
    by its name on the Hub; a randomly initialised one is identified by nothing
    at all, and an adapter is only valid against the base it was trained on.
    """
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if "lora_" in name:
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return f"sha256:{digest.hexdigest()[:24]}"


def run_dir_for(cfg: Config, root: Path) -> Path:
    """Keyed on the config fingerprint, so the same experiment reuses its own
    directory and two different configs cannot overwrite each other."""
    return root / cfg.train.out_dir / f"{cfg.split.name}-{cfg.fingerprint()}"


def train(cfg: Config, root: Path | None = None, tracker: Tracker | None = None) -> RunResult:
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    root = root or Path()
    tracker = tracker or NullTracker()

    examples = load_corpus(cfg, root)
    split = read_split(split_path(cfg, root))
    verify(split, examples, cfg)

    labels = label_set(examples)
    label_to_id = {label: index for index, label in enumerate(labels)}
    rows = {name: _subset(examples, split, name) for name in SPLIT_NAMES}

    seed_everything(cfg.train.seed)
    bundle = build_bundle(cfg, [row.text for row in rows["train"]], labels)
    device = resolve_device(cfg)
    bundle.model.to(device)
    report = parameter_report(bundle.model)

    adapter_params = [
        parameter
        for name, parameter in bundle.model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    head_params = [
        parameter
        for name, parameter in bundle.model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    optimizer = torch.optim.AdamW(
        [
            # No weight decay on the adapter. B starts at zero, which makes the
            # adapter the identity at step 0, so decay toward zero is a prior
            # that the adapter should do nothing. The head gets the decay,
            # because that is where overfitting 44 training rows shows up.
            {"params": adapter_params, "lr": cfg.train.lr, "weight_decay": 0.0},
            {
                "params": head_params,
                "lr": cfg.train.head_lr,
                "weight_decay": cfg.train.weight_decay,
            },
        ]
    )

    stopper = EarlyStopper(cfg.early_stopping)
    generator = torch.Generator().manual_seed(cfg.train.seed)
    history: list[dict[str, float]] = []
    best_state: dict[str, Any] | None = None
    stop_reason = "ran out of epochs"
    epochs_ran = 0

    run_name = f"{cfg.split.name}-{cfg.fingerprint()}"
    with tracker.run(run_name):
        params: dict[str, Any] = {"config_fingerprint": cfg.fingerprint()}
        for section, values in cfg.resolved().items():
            if section == "tracking":
                continue
            for key, value in values.items():
                params[f"{section}.{key}"] = value
        params.update(report)
        params["labels"] = ",".join(labels)
        params["corpus_fingerprint"] = split.corpus_fingerprint
        params["split_counts"] = json.dumps(split.counts(examples), sort_keys=True)
        if cfg.model.source == "random":
            params["base_checksum"] = base_checksum(bundle.model)
        tracker.log_params(params)

        for epoch in range(1, cfg.train.epochs + 1):
            epochs_ran = epoch
            train_loss = _train_one_epoch(
                bundle, rows["train"], label_to_id, cfg, optimizer, device, generator
            )
            validation = evaluate_rows(bundle, rows["validation"], labels, cfg, device)
            epoch_metrics = {"train_loss": round(train_loss, 6), **scalars(validation, "val")}
            history.append({"epoch": epoch, **epoch_metrics})
            tracker.log_metrics(epoch_metrics, step=epoch)

            verdict = stopper.update(epoch, epoch_metrics)
            if verdict.improved:
                best_state = copy.deepcopy(
                    {
                        key: value.detach().cpu().clone()
                        for key, value in get_peft_model_state_dict(bundle.model).items()
                    }
                )
            if verdict.should_stop:
                stop_reason = verdict.reason
                break

        if best_state is not None:
            # Without this the saved adapter is whatever the last epoch left
            # behind, which is by definition the state early stopping decided
            # against.
            set_peft_model_state_dict(bundle.model, best_state)

        validation = evaluate_rows(bundle, rows["validation"], labels, cfg, device)
        test = evaluate_rows(bundle, rows["test"], labels, cfg, device)
        tracker.log_metrics({**scalars(test, "test")}, step=epochs_ran)

        run_dir = run_dir_for(cfg, root)
        run_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = run_dir / "adapter"
        bundle.model.save_pretrained(str(adapter_dir))
        (run_dir / "labels.json").write_text(
            json.dumps(
                {
                    "labels": labels,
                    "split": cfg.split.name,
                    "corpus_fingerprint": split.corpus_fingerprint,
                    "config_fingerprint": cfg.fingerprint(),
                    "base": cfg.model.name,
                    "base_source": cfg.model.source,
                },
                indent=2,
            )
            + "\n"
        )
        result = RunResult(
            config_fingerprint=cfg.fingerprint(),
            run_dir=str(run_dir),
            labels=labels,
            epochs_ran=epochs_ran,
            best_epoch=stopper.best_epoch,
            best_value=stopper.best_value,
            stopped_early=stopper.stopped,
            stop_reason=stop_reason,
            parameters=report,
            history=history,
            validation=validation,
            test=test,
        )
        (run_dir / "metrics.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
        (run_dir / "confusion.txt").write_text(
            "validation\n"
            + confusion_table(validation, labels)
            + "\n\ntest\n"
            + confusion_table(test, labels)
            + "\n"
        )
        tracker.log_artifact(run_dir)

    return result


def evaluate_adapter(
    cfg: Config, adapter_dir: Path, root: Path | None = None
) -> dict[str, Any]:
    """Score the test split with an adapter already on disk.

    The label order is read from the run directory and checked against the
    corpus. A permuted label list is the failure that produces plausible
    looking metrics from a model that is answering a different question.
    """
    from peft import PeftModel

    root = root or Path()
    examples = load_corpus(cfg, root)
    split = read_split(split_path(cfg, root))
    verify(split, examples, cfg)

    meta_path = adapter_dir.parent / "labels.json"
    if not meta_path.exists():
        raise SplitError(
            f"no labels.json beside {adapter_dir}; cannot map ids back to labels"
        )
    meta = json.loads(meta_path.read_text())
    labels = meta["labels"]
    if labels != label_set(examples):
        raise SplitError(
            f"the adapter was trained on labels {labels} and the corpus now has "
            f"{label_set(examples)}. Scoring would permute every prediction"
        )

    rows = _subset(examples, split, "test")
    seed_everything(cfg.train.seed)
    tokenizer = build_tokenizer(cfg, [row.text for row in _subset(examples, split, "train")])
    base = build_model(cfg, num_labels=len(labels), vocab_size=len(tokenizer.get_vocab()))
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    device = resolve_device(cfg)
    model.to(device)
    bundle = Bundle(
        model=model,
        tokenizer=tokenizer,
        kind=cfg.model.source,
        vocab_size=len(tokenizer.get_vocab()),
    )
    return evaluate_rows(bundle, rows, labels, cfg, device)
