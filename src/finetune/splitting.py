"""Build a train/validation/test split once, write it down, and check it later.

The split is a pure function of the corpus ids, their labels, a salt and the
ratios. Run it twice on two machines and the JSON is byte identical, which is
what makes the recorded file worth committing.

Why not ``random.shuffle`` with a seed. A seeded shuffle is reproducible only
against the exact list it was given. Add one example to the corpus and every
assignment downstream of it moves, so the validation set from last week shares
some unknown fraction of its rows with this week's training set. Assigning from
a hash of the id has no such coupling: an id's bucket depends on the id and the
salt, and on nothing else in the file.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import SPLIT_NAMES, Config
from .errors import SplitError

_RECORD_FIELDS = ("name", "salt", "method", "ratios", "corpus_fingerprint", "assignments")


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: str


def load_corpus(cfg: Config, root: Path | None = None) -> list[Example]:
    """Read the JSONL corpus named by the config.

    Order is preserved for reading but never relied on: every downstream step
    sorts by id, because a corpus that gets appended to should not change any
    existing assignment.
    """
    path = (root or Path()) / cfg.data.corpus
    if not path.exists():
        raise SplitError(f"corpus not found: {path}")

    examples: list[Example] = []
    seen: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SplitError(f"{path}:{lineno}: not valid JSON: {exc.msg}") from None
            missing = [
                field
                for field in (cfg.data.id_field, cfg.data.text_field, cfg.data.label_field)
                if field not in row
            ]
            if missing:
                raise SplitError(f"{path}:{lineno}: missing field(s) {missing}")
            example = Example(
                id=str(row[cfg.data.id_field]),
                text=str(row[cfg.data.text_field]),
                label=str(row[cfg.data.label_field]),
            )
            if not example.text.strip():
                raise SplitError(f"{path}:{lineno}: empty text for id {example.id!r}")
            if example.id in seen:
                raise SplitError(
                    f"{path}:{lineno}: duplicate id {example.id!r}, first seen on line "
                    f"{seen[example.id]}. A duplicated id lands in one split and is scored "
                    f"in another"
                )
            seen[example.id] = lineno
            examples.append(example)

    if not examples:
        raise SplitError(f"{path}: no examples")
    return examples


def corpus_fingerprint(examples: list[Example]) -> str:
    """Identity of the (id, label) pairs, insensitive to order and to text.

    Text is excluded so that fixing a typo in an example does not invalidate a
    recorded split. Labels are included because relabelling changes what the
    stratification was balancing.
    """
    lines = sorted(f"{e.id}\t{e.label}" for e in examples)
    digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    return f"sha256:{digest[:24]}"


def bucket(salt: str, example_id: str) -> float:
    """Map an id into [0, 1). Stable across machines and Python versions.

    ``hash()`` is salted per process since 3.3, so it cannot be used here at
    all: a split built in one process would not reproduce in the next.
    """
    digest = hashlib.sha256(f"{salt}\x00{example_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def allocate(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """Split ``n`` items by ratio, with the counts summing to exactly ``n``.

    Largest remainder. Rounding each ratio independently loses or gains rows,
    and a split that quietly drops two examples is the kind of thing that is
    only noticed when two reports disagree by a rounding error.
    """
    exact = {name: n * ratios[name] for name in SPLIT_NAMES}
    counts = {name: int(math.floor(value)) for name, value in exact.items()}
    remaining = n - sum(counts.values())
    order = sorted(
        SPLIT_NAMES,
        key=lambda name: (-(exact[name] - counts[name]), SPLIT_NAMES.index(name)),
    )
    for name in order[:remaining]:
        counts[name] += 1
    return counts


@dataclass(frozen=True)
class Split:
    name: str
    salt: str
    method: str
    ratios: dict[str, float]
    corpus_fingerprint: str
    assignments: dict[str, str]

    def ids(self, split: str) -> list[str]:
        if split not in SPLIT_NAMES:
            raise SplitError(f"unknown split {split!r}, expected one of {list(SPLIT_NAMES)}")
        return sorted(key for key, value in self.assignments.items() if value == split)

    def counts(self, examples: list[Example]) -> dict[str, dict[str, int]]:
        label_of = {e.id: e.label for e in examples}
        table: dict[str, Counter[str]] = {name: Counter() for name in SPLIT_NAMES}
        for example_id, split in self.assignments.items():
            if example_id in label_of:
                table[split][label_of[example_id]] += 1
        return {name: dict(sorted(counter.items())) for name, counter in table.items()}

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "salt": self.salt,
            "method": self.method,
            "ratios": {name: self.ratios[name] for name in SPLIT_NAMES},
            "corpus_fingerprint": self.corpus_fingerprint,
            "assignments": dict(sorted(self.assignments.items())),
        }


def make_split(examples: list[Example], cfg: Config) -> Split:
    """Assign every example to exactly one split.

    Two methods, and the choice is a real one:

    ``quantile`` sorts each label's ids by bucket and cuts at the ratio
    boundaries. The per-label counts come out exact, so a four-class corpus
    gets four stratified splits. The cost is that the boundary moves when the
    corpus grows, so a handful of ids near a cut can change split.

    ``independent`` sends each id to whichever ratio interval its bucket falls
    in, with no reference to any other id. Nothing ever moves when the corpus
    grows, and the counts are only approximate: at 64 examples a 15 percent
    validation share lands anywhere from about 8 to 12, and a rare label can
    miss a split entirely.
    """
    if cfg.split.method == "quantile":
        assignments = _quantile(examples, cfg.split.ratios, cfg.split.salt)
    elif cfg.split.method == "independent":
        assignments = _independent(examples, cfg.split.ratios, cfg.split.salt)
    else:  # pragma: no cover - config validation rejects anything else
        raise SplitError(f"unknown split method {cfg.split.method!r}")

    return Split(
        name=cfg.split.name,
        salt=cfg.split.salt,
        method=cfg.split.method,
        ratios=dict(cfg.split.ratios),
        corpus_fingerprint=corpus_fingerprint(examples),
        assignments=assignments,
    )


def _quantile(examples: list[Example], ratios: dict[str, float], salt: str) -> dict[str, str]:
    by_label: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        by_label[example.label].append(example)

    assignments: dict[str, str] = {}
    for label in sorted(by_label):
        # Sorting on (bucket, id) keeps the order total: two ids with the same
        # bucket would otherwise be ordered by corpus position.
        ordered = sorted(by_label[label], key=lambda e: (bucket(salt, e.id), e.id))
        counts = allocate(len(ordered), ratios)
        cursor = 0
        for name in SPLIT_NAMES:
            for example in ordered[cursor : cursor + counts[name]]:
                assignments[example.id] = name
            cursor += counts[name]
    return assignments


def _independent(
    examples: list[Example], ratios: dict[str, float], salt: str
) -> dict[str, str]:
    edges: list[tuple[float, str]] = []
    cumulative = 0.0
    for name in SPLIT_NAMES:
        cumulative += ratios[name]
        edges.append((cumulative, name))

    assignments: dict[str, str] = {}
    for example in examples:
        value = bucket(salt, example.id)
        # The last edge is 1.0 up to float error, so the final split absorbs
        # anything that lands past the computed boundary.
        assignments[example.id] = next(
            (name for edge, name in edges if value < edge), SPLIT_NAMES[-1]
        )
    return assignments


def write_split(split: Split, cfg: Config, root: Path | None = None) -> Path:
    directory = (root or Path()) / cfg.split.dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{split.name}.json"
    path.write_text(json.dumps(split.to_record(), indent=2, sort_keys=False) + "\n")
    return path


def split_path(cfg: Config, root: Path | None = None) -> Path:
    return (root or Path()) / cfg.split.dir / f"{cfg.split.name}.json"


def read_split(path: Path) -> Split:
    if not path.exists():
        raise SplitError(f"no recorded split at {path}. Create one with: ft split <config>")
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SplitError(f"{path}: not valid JSON: {exc.msg}") from None
    missing = [field for field in _RECORD_FIELDS if field not in record]
    if missing:
        raise SplitError(f"{path}: recorded split is missing {missing}")
    bad = {value for value in record["assignments"].values() if value not in SPLIT_NAMES}
    if bad:
        raise SplitError(f"{path}: assignments name unknown splits {sorted(bad)}")
    return Split(
        name=record["name"],
        salt=record["salt"],
        method=record["method"],
        ratios=record["ratios"],
        corpus_fingerprint=record["corpus_fingerprint"],
        assignments=record["assignments"],
    )


def verify(split: Split, examples: list[Example], cfg: Config) -> None:
    """Refuse a recorded split that no longer describes this corpus.

    Training against a stale split is the quiet version of a leak: the ids that
    were added since are missing from every split and are never scored, and an
    id whose label changed is being stratified on a label it no longer has.
    """
    actual = corpus_fingerprint(examples)
    if actual == split.corpus_fingerprint:
        return

    recorded_ids = set(split.assignments)
    corpus_ids = {e.id for e in examples}
    added = sorted(corpus_ids - recorded_ids)
    removed = sorted(recorded_ids - corpus_ids)
    relabelled = len(corpus_ids & recorded_ids) if not added and not removed else 0

    detail = [f"recorded {split.corpus_fingerprint}, corpus {actual}"]
    if added:
        detail.append(f"{len(added)} id(s) not in the split, first: {added[:3]}")
    if removed:
        detail.append(
            f"{len(removed)} recorded id(s) not in the corpus, first: {removed[:3]}"
        )
    if relabelled:
        detail.append(f"the ids match, so a label changed on some of {relabelled}")
    raise SplitError(
        f"recorded split {split.name!r} does not match the corpus: "
        + "; ".join(detail)
        + f". Rebuild it with: ft split {cfg.path} --force"
    )


def thin_cells(
    split: Split, examples: list[Example], minimum: int
) -> list[tuple[str, str, int]]:
    """(split, label, count) for every cell below ``minimum``.

    This is the number that decides whether a validation score means anything.
    One example of a class in the validation fold makes that class's recall
    either 0.0 or 1.0, and with four classes a macro average built on it moves
    0.25 on a single flip. That is noise, not a signal to early-stop on.
    """
    cells = split.counts(examples)
    labels = sorted({e.label for e in examples})
    thin = []
    for name in SPLIT_NAMES:
        for label in labels:
            count = cells[name].get(label, 0)
            if count < minimum:
                thin.append((name, label, count))
    return thin
