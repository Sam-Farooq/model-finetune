"""ft: resolve a config, record a split, train, evaluate.

Exit codes are the contract:

* 0, it worked
* 1, a check failed. The recorded split no longer matches the corpus, or a
  label set moved under a saved adapter
* 2, the config or the command line is wrong

Anything that needs torch is imported inside the subcommand that needs it, so
``ft config`` and ``ft split`` run in well under a second.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import SPLIT_NAMES, Config, load_config
from .errors import ConfigError, SplitError
from .splitting import (
    load_corpus,
    make_split,
    read_split,
    split_path,
    thin_cells,
    verify,
    write_split,
)


def _print_counts(split, examples, minimum: int) -> None:
    counts = split.counts(examples)
    labels = sorted({example.label for example in examples})
    width = max(len(label) for label in labels)
    header = f"{'label':<{width}}  " + "  ".join(f"{name[:10]:>10}" for name in SPLIT_NAMES)
    print(header)
    for label in labels:
        row = "  ".join(f"{counts[name].get(label, 0):>10}" for name in SPLIT_NAMES)
        print(f"{label:<{width}}  {row}")
    totals = "  ".join(f"{sum(counts[name].values()):>10}" for name in SPLIT_NAMES)
    print(f"{'total':<{width}}  {totals}")

    thin = thin_cells(split, examples, minimum)
    if thin:
        print()
        for name, label, count in thin:
            print(
                f"thin: {name}/{label} has {count}, below min_per_label_per_split "
                f"of {minimum}"
            )
        print(
            "A per-class score computed on a handful of rows moves in steps of "
            "1/n. Read the macro average on this split knowing that."
        )


def cmd_config(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.json:
        print(json.dumps({"fingerprint": cfg.fingerprint(), **cfg.resolved()}, indent=2))
        return 0
    print(f"config      {cfg.path}")
    print(f"fingerprint {cfg.fingerprint()}   (tracking excluded)")
    print()
    rows = [
        (f"{section}.{key}", value, cfg.origin.get(f"{section}.{key}", "default"))
        for section, values in cfg.resolved().items()
        for key, value in values.items()
    ]
    width = max(len(name) for name, _, _ in rows)
    for name, value, origin in rows:
        print(f"  {name:<{width}}  {str(value):<34}  {origin}")
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    cfg: Config = load_config(args.config)
    root = Path(args.root)
    examples = load_corpus(cfg, root)
    path = split_path(cfg, root)

    if path.exists() and not args.force:
        existing = read_split(path)
        verify(existing, examples, cfg)
        print(f"{path} matches the corpus ({existing.corpus_fingerprint}, unchanged)")
        _print_counts(existing, examples, cfg.split.min_per_label_per_split)
        return 0

    previous = read_split(path) if path.exists() else None
    split = make_split(examples, cfg)
    written = write_split(split, cfg, root)
    print(f"wrote {written}  method={split.method}  {split.corpus_fingerprint}")

    if previous is not None:
        shared = set(previous.assignments) & set(split.assignments)
        moved = sum(
            1 for key in shared if previous.assignments[key] != split.assignments[key]
        )
        print(
            f"{moved} of {len(shared)} ids that were in the old record changed split. "
            f"Any metric from a run against the old record is now measuring a "
            f"different test set"
        )
    _print_counts(split, examples, cfg.split.min_per_label_per_split)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .tracking import make_tracker
    from .training import train

    cfg = load_config(args.config)
    root = Path(args.root)
    result = train(cfg, root=root, tracker=make_tracker(cfg, root))

    print()
    print(
        f"{'epoch':>5}  {'train_loss':>10}  {'val_loss':>9}  {'val_macro_f1':>12}  "
        f"{'val_acc':>8}"
    )
    for row in result.history:
        print(
            f"{int(row['epoch']):>5}  {row['train_loss']:>10.4f}  "
            f"{row['val_loss']:>9.4f}  {row['val_macro_f1']:>12.4f}  "
            f"{row['val_accuracy']:>8.4f}"
        )
    print()
    print(
        f"parameters   {result.parameters['trainable_parameters']} trainable of "
        f"{result.parameters['total_parameters']} "
        f"({result.parameters['trainable_percent']}%)"
    )
    print(
        f"             {result.parameters['adapter_parameters']} adapter, "
        f"{result.parameters['head_parameters']} head"
    )
    print(f"best epoch   {result.best_epoch} at val_macro_f1 {result.best_value:.4f}")
    print(f"stopped      {result.stop_reason}")
    print(
        f"test         macro_f1 {result.test['macro_f1']:.4f}  "
        f"accuracy {result.test['accuracy']:.4f}  n {result.test['n']}"
    )
    print(f"artefacts    {result.run_dir}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .metrics import confusion_table
    from .training import evaluate_adapter, run_dir_for

    cfg = load_config(args.config)
    root = Path(args.root)
    adapter = Path(args.adapter) if args.adapter else run_dir_for(cfg, root) / "adapter"
    if not adapter.exists():
        raise SplitError(f"no adapter at {adapter}. Train one with: ft train {args.config}")

    metrics = evaluate_adapter(cfg, adapter, root=root)
    labels = json.loads((adapter.parent / "labels.json").read_text())["labels"]
    print(f"adapter   {adapter}")
    print(
        f"test      macro_f1 {metrics['macro_f1']:.4f}  accuracy "
        f"{metrics['accuracy']:.4f}  mcc {metrics['mcc']:.4f}  n {metrics['n']}"
    )
    print()
    for label, row in metrics["per_label"].items():
        print(
            f"  {label:<10} precision {row['precision']:.4f}  recall "
            f"{row['recall']:.4f}  f1 {row['f1']:.4f}  support {row['support']}"
        )
    print()
    print(confusion_table(metrics, labels))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ft", description="Config driven LoRA fine-tuning of a sequence classifier"
    )
    parser.add_argument(
        "--root",
        default=".",
        help="directory that paths in the config are relative to (default: .)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config", help="resolve and validate a config")
    config_parser.add_argument("config")
    config_parser.add_argument("--json", action="store_true")
    config_parser.set_defaults(func=cmd_config)

    split_parser = subparsers.add_parser(
        "split", help="create the recorded split, or check it against the corpus"
    )
    split_parser.add_argument("config")
    split_parser.add_argument(
        "--force", action="store_true", help="rebuild and report how many ids moved"
    )
    split_parser.set_defaults(func=cmd_split)

    train_parser = subparsers.add_parser("train", help="train the adapter")
    train_parser.add_argument("config")
    train_parser.set_defaults(func=cmd_train)

    eval_parser = subparsers.add_parser(
        "eval", help="score the test split with a saved adapter"
    )
    eval_parser.add_argument("config")
    eval_parser.add_argument("--adapter", default=None)
    eval_parser.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except SplitError as exc:
        print(f"split error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
