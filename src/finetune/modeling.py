"""Build the classifier, pick what LoRA adapts, and count what is trainable.

Everything in here that needs torch is behind a function call, so importing the
module costs nothing. ``vocab_from_texts`` is pure and is tested in the fast
suite; the rest is tested against a randomly initialised two layer model built
in process, which downloads nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .config import Config
from .errors import ConfigError

SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
_WORD = re.compile(r"[a-z0-9]+")


def vocab_from_texts(texts: list[str]) -> dict[str, int]:
    """A word level vocabulary, lowercased, in a deterministic order.

    Built from the training texts only. Fitting a vocabulary on the whole
    corpus leaks: a word that appears once, in a test example, gets an id and
    an embedding row that training could update, and the test score then
    reflects a model that had seen the test vocabulary. It is a small leak at
    this size and it is the same mistake as fitting a scaler on everything.
    """
    words = sorted({word for text in texts for word in _WORD.findall(text.lower())})
    return {token: index for index, token in enumerate([*SPECIAL_TOKENS, *words])}


@dataclass
class Bundle:
    model: Any
    tokenizer: Any
    kind: str
    vocab_size: int


def build_tokenizer(cfg: Config, train_texts: list[str]) -> Any:
    """A real tokenizer for a real checkpoint, a built one for the tiny model.

    The tiny path constructs a ``tokenizers`` WordLevel model in memory rather
    than writing a ``vocab.txt`` and handing it to ``BertTokenizerFast``. That
    was the first attempt and it does not work on transformers 5: the
    ``vocab_file`` argument is accepted and the file is not read, so the
    tokenizer comes back holding the five special tokens and every word in the
    corpus encodes to ``[UNK]``. Nothing raises. The loop trains, the loss
    falls a little, and every input is the same sequence of unknown tokens.
    """
    if cfg.model.source == "pretrained":
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(cfg.model.tokenizer or cfg.model.name)

    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = vocab_from_texts(train_texts)
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def build_model(cfg: Config, num_labels: int, vocab_size: int) -> Any:
    """The base classifier, before any adapter is attached."""
    if cfg.model.source == "pretrained":
        from transformers import AutoModelForSequenceClassification

        return AutoModelForSequenceClassification.from_pretrained(
            cfg.model.name, num_labels=num_labels
        )

    from transformers import AutoModelForSequenceClassification, BertConfig

    architecture = BertConfig(
        vocab_size=vocab_size,
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        num_labels=num_labels,
        pad_token_id=0,
    )
    return AutoModelForSequenceClassification.from_config(architecture)


def linear_module_names(model: Any) -> list[str]:
    """Every leaf name that belongs to an ``nn.Linear``, deduplicated."""
    from torch import nn

    return sorted(
        {
            name.rsplit(".", 1)[-1]
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and name
        }
    )


def resolve_targets(model: Any, requested: list[str]) -> list[str]:
    """Check the requested module names exist on this architecture.

    This is the error worth catching early, because the failure it prevents is
    silent. ``target_modules`` is architecture specific: BERT names its
    attention projections ``query``, ``key``, ``value``, DistilBERT names them
    ``q_lin``, ``k_lin``, ``v_lin``. Point a DistilBERT config at ``q_proj``
    and, depending on the PEFT version, you get either an exception or a model
    with no adapter in it at all. The second case trains the head alone, scores
    a little worse, and reads as a tuning problem.
    """
    available = linear_module_names(model)
    missing = [name for name in requested if name not in available]
    if missing:
        raise ConfigError(
            f"lora.target_modules names {missing}, which no nn.Linear in "
            f"{model.config.model_type} is called. available: {available}"
        )
    return list(requested)


def build_lora_config(cfg: Config, targets: list[str]) -> Any:
    """Translate the config section into a PEFT LoraConfig.

    ``alpha`` over ``r`` is the scale the adapter's output is multiplied by, so
    the pair matters more than either number. The defaults here are r 8 and
    alpha 16, a scale of 2. Raising r alone makes the adapter both larger and
    louder, which is two changes reported as one.
    """
    from peft import LoraConfig as PeftLoraConfig
    from peft import TaskType

    return PeftLoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=targets,
        modules_to_save=list(cfg.lora.modules_to_save),
        bias="none",
    )


def wrap_with_lora(model: Any, cfg: Config) -> Any:
    """Attach the adapter and assert the classification head can still learn.

    A sequence classification head on a fresh checkpoint is randomly
    initialised. If it is frozen along with the rest of the base model, the
    adapter is being asked to make a fixed random projection of the features
    separable, which trains to roughly chance and looks like a learning rate
    problem for as long as you are willing to believe it is one.
    """
    from peft import get_peft_model

    targets = resolve_targets(model, cfg.lora.target_modules)
    wrapped = get_peft_model(model, build_lora_config(cfg, targets))

    head_trainable = [
        name
        for name, parameter in wrapped.named_parameters()
        if parameter.requires_grad
        and any(module in name for module in cfg.lora.modules_to_save)
    ]
    if not head_trainable:
        raise ConfigError(
            f"lora.modules_to_save {cfg.lora.modules_to_save} left no trainable "
            f"parameter in the classification head. The head is randomly "
            f"initialised and has to be trained"
        )
    return wrapped


def parameter_report(model: Any) -> dict[str, float | int]:
    """Trainable against total, which is the number LoRA is chosen for."""
    trainable = 0
    total = 0
    adapter = 0
    head = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            if "lora_" in name:
                adapter += count
            else:
                head += count
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": round(100.0 * trainable / total, 4) if total else 0.0,
        "adapter_parameters": adapter,
        "head_parameters": head,
    }


def adapter_parameter_estimate(
    cfg: Config, hidden_size: int, targets_per_layer: int, layers: int
) -> int:
    """What the adapter should weigh, from r and the shapes, with no model.

    Each adapted linear of shape (d, d) gains an A of (r, d) and a B of (d, r),
    so 2*r*d per module, and every layer is adapted. Checked against the real
    count in the tests, because a formula that disagrees with the model means
    some of the target names matched nothing.

    The layer count is in here because the first version of this function left
    it out and returned 512 for a model whose adapter weighs 1024. Two layers
    of two adapted projections is four modules, not two.
    """
    return 2 * cfg.lora.r * hidden_size * targets_per_layer * layers
