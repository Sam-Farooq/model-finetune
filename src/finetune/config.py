"""Resolve a TOML experiment config, with inheritance and a named origin.

Three things this does that a plain ``tomllib.load`` does not:

* follows ``extends`` so a config can state only what it changes;
* records which file set every leaf, so an error message can name that file
  rather than the file that was asked for;
* refuses unknown keys. A typo in ``min_delta`` is otherwise a silent default,
  and a training run that ignored half its config is worse than one that
  refused to start.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigError

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class DataConfig:
    corpus: str
    text_field: str = "text"
    label_field: str = "label"
    id_field: str = "id"
    max_length: int = 64


@dataclass(frozen=True)
class SplitConfig:
    name: str
    salt: str
    dir: str = "splits"
    method: str = "quantile"
    ratios: dict[str, float] = field(
        default_factory=lambda: {"train": 0.7, "validation": 0.15, "test": 0.15}
    )
    min_per_label_per_split: int = 3


@dataclass(frozen=True)
class ModelConfig:
    name: str
    source: str = "pretrained"
    # Only read when source == "random": a tiny architecture built from scratch
    # so the loop can be exercised with nothing downloaded.
    hidden_size: int = 32
    num_hidden_layers: int = 2
    num_attention_heads: int = 2
    intermediate_size: int = 64
    tokenizer: str = ""


@dataclass(frozen=True)
class LoraConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_lin", "v_lin"])
    modules_to_save: list[str] = field(default_factory=lambda: ["classifier"])


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 1337
    epochs: int = 12
    batch_size: int = 16
    lr: float = 3e-4
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    device: str = "cpu"
    out_dir: str = "runs"


@dataclass(frozen=True)
class EarlyStoppingConfig:
    monitor: str = "val_macro_f1"
    mode: str = "max"
    patience: int = 3
    min_delta: float = 0.002


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = False
    experiment: str = "finetune"
    uri: str = "runs/mlruns"


@dataclass(frozen=True)
class Config:
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    lora: LoraConfig
    train: TrainConfig
    early_stopping: EarlyStoppingConfig
    tracking: TrackingConfig
    origin: dict[str, str]
    path: str

    def fingerprint(self) -> str:
        """A stable id for the experiment this config describes.

        ``tracking`` is excluded on purpose. Moving the MLflow store or naming
        the experiment differently changes nothing about what gets trained, and
        a fingerprint that moves when it does cannot be used to tell two runs
        apart.
        """
        payload = {
            "data": asdict(self.data),
            "split": asdict(self.split),
            "model": asdict(self.model),
            "lora": asdict(self.lora),
            "train": asdict(self.train),
            "early_stopping": asdict(self.early_stopping),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def resolved(self) -> dict[str, Any]:
        return {
            "data": asdict(self.data),
            "split": asdict(self.split),
            "model": asdict(self.model),
            "lora": asdict(self.lora),
            "train": asdict(self.train),
            "early_stopping": asdict(self.early_stopping),
            "tracking": asdict(self.tracking),
        }


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "split": SplitConfig,
    "model": ModelConfig,
    "lora": LoraConfig,
    "train": TrainConfig,
    "early_stopping": EarlyStoppingConfig,
    "tracking": TrackingConfig,
}
_REQUIRED = {"data": ("corpus",), "split": ("name", "salt"), "model": ("name",)}


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        raise ConfigError(f"no such config file: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: not valid TOML: {exc}") from None


def _chain(path: Path, seen: list[Path] | None = None) -> list[tuple[Path, dict[str, Any]]]:
    """Base-first list of (file, body), following ``extends``."""
    seen = seen or []
    resolved = path.resolve()
    if resolved in [p.resolve() for p in seen]:
        trail = " -> ".join(p.name for p in [*seen, resolved])
        raise ConfigError(f"extends forms a cycle: {trail}")
    body = _read_toml(resolved)
    parent = body.pop("extends", None)
    if parent is None:
        return [(resolved, body)]
    if not isinstance(parent, str):
        raise ConfigError(f"{resolved}: extends must be a path, got {type(parent).__name__}")
    base = (resolved.parent / parent).resolve()
    return [*_chain(base, [*seen, resolved]), (resolved, body)]


def load_config(path: str | Path) -> Config:
    """Read a config, apply its ``extends`` chain, validate, and freeze it."""
    merged: dict[str, dict[str, Any]] = {}
    origin: dict[str, str] = {}
    files = _chain(Path(path))

    for file, body in files:
        label = file.name
        for section, values in body.items():
            if section not in _SECTIONS:
                raise ConfigError(
                    f"{label}: unknown section [{section}]. "
                    f"known sections: {', '.join(sorted(_SECTIONS))}"
                )
            if not isinstance(values, dict):
                raise ConfigError(f"{label}: [{section}] must be a table")
            allowed = {f.name for f in _SECTIONS[section].__dataclass_fields__.values()}
            for key, value in values.items():
                if key not in allowed:
                    raise ConfigError(
                        f"{label}: unknown key {section}.{key}. "
                        f"known keys in [{section}]: {', '.join(sorted(allowed))}"
                    )
                merged.setdefault(section, {})[key] = value
                origin[f"{section}.{key}"] = label

    for section, keys in _REQUIRED.items():
        for key in keys:
            if key not in merged.get(section, {}):
                raise ConfigError(f"{Path(path).name}: {section}.{key} is required and unset")

    sections: dict[str, Any] = {}
    for section, cls in _SECTIONS.items():
        try:
            sections[section] = cls(**merged.get(section, {}))
        except TypeError as exc:
            raise ConfigError(f"{Path(path).name}: [{section}]: {exc}") from None

    cfg = Config(
        **sections,
        origin=origin,
        path=str(Path(path)),
    )
    _validate(cfg)
    return cfg


def _where(cfg: Config, key: str) -> str:
    return cfg.origin.get(key, Path(cfg.path).name)


def _validate(cfg: Config) -> None:
    def bad(key: str, message: str) -> ConfigError:
        return ConfigError(f"{_where(cfg, key)}: {key} {message}")

    if set(cfg.split.ratios) != set(SPLIT_NAMES):
        raise bad("split.ratios", f"must have exactly the keys {list(SPLIT_NAMES)}")
    for name, value in cfg.split.ratios.items():
        if not isinstance(value, int | float) or value <= 0:
            raise bad("split.ratios", f"[{name}] must be a positive number, got {value!r}")
    total = sum(cfg.split.ratios.values())
    if abs(total - 1.0) > 1e-9:
        raise bad("split.ratios", f"must sum to 1.0, got {total:.6f}")
    if cfg.split.method not in ("quantile", "independent"):
        raise bad(
            "split.method", f"must be quantile or independent, got {cfg.split.method!r}"
        )
    if cfg.split.min_per_label_per_split < 0:
        raise bad("split.min_per_label_per_split", "must be >= 0")

    if cfg.model.source not in ("pretrained", "random"):
        raise bad("model.source", f"must be pretrained or random, got {cfg.model.source!r}")

    if cfg.lora.r < 1:
        raise bad("lora.r", "must be >= 1")
    if cfg.lora.alpha <= 0:
        raise bad("lora.alpha", "must be > 0")
    if not 0.0 <= cfg.lora.dropout < 1.0:
        raise bad("lora.dropout", "must be in [0.0, 1.0)")
    if not cfg.lora.target_modules:
        raise bad("lora.target_modules", "must name at least one module")
    if not cfg.lora.modules_to_save:
        # A sequence classification head starts randomly initialised. Freeze it
        # and the adapter is asked to make a frozen random projection separable,
        # which trains to nothing and looks like a bad learning rate.
        raise bad("lora.modules_to_save", "must include the classification head")

    if cfg.train.epochs < 1:
        raise bad("train.epochs", "must be >= 1")
    if cfg.train.batch_size < 1:
        raise bad("train.batch_size", "must be >= 1")
    for key, value in (("train.lr", cfg.train.lr), ("train.head_lr", cfg.train.head_lr)):
        if value <= 0:
            raise bad(key, f"must be > 0, got {value!r}")
    if cfg.train.weight_decay < 0:
        raise bad("train.weight_decay", "must be >= 0")
    if cfg.train.max_grad_norm <= 0:
        raise bad("train.max_grad_norm", "must be > 0")
    if cfg.train.device not in ("cpu", "auto"):
        raise bad("train.device", f"must be cpu or auto, got {cfg.train.device!r}")
    if cfg.data.max_length < 8:
        raise bad("data.max_length", "must be >= 8")

    if cfg.early_stopping.mode not in ("max", "min"):
        raise bad(
            "early_stopping.mode", f"must be max or min, got {cfg.early_stopping.mode!r}"
        )
    if cfg.early_stopping.patience < 1:
        raise bad("early_stopping.patience", "must be >= 1")
    if cfg.early_stopping.min_delta < 0:
        raise bad("early_stopping.min_delta", "must be >= 0")
    if cfg.tracking.enabled and "://" not in cfg.tracking.uri:
        # MLflow 3 refuses a bare filesystem tracking store: the FileStore is
        # in maintenance mode and its constructor raises unless
        # MLFLOW_ALLOW_FILE_STORE is set. Caught here because the alternative
        # is a twenty line traceback out of mlflow's store registry, thrown
        # after the model has been built.
        raise bad(
            "tracking.uri",
            f"needs a scheme, got {cfg.tracking.uri!r}. Use "
            f"sqlite:///runs/mlflow.db for a local file, or http://host:5000 "
            f"for a server",
        )

    if not cfg.early_stopping.monitor.startswith("val_"):
        # Stopping on a training metric is not early stopping. Training loss
        # falls monotonically for a while whatever the model is doing to the
        # held-out set, so the rule would never fire until it was too late.
        raise bad(
            "early_stopping.monitor",
            f"must be a validation metric starting with val_, got "
            f"{cfg.early_stopping.monitor!r}",
        )
