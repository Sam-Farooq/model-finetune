# model-finetune

A LoRA fine-tune of a four class intent classifier, driven by a TOML config,
with the train/validation/test split built once and committed as a file.
PyTorch, Hugging Face Transformers and PEFT do the training, scikit-learn
computes the metrics, MLflow records the runs.

There is no server here, no endpoint and no registry promotion step. A run
reads a config, writes an adapter and a metrics file into `runs/`, and logs the
same numbers to MLflow. What it works hardest at is making two runs of the same
config agree.

**No real checkpoint has been fine-tuned in this repository.**
[configs/distilbert-lora.toml](configs/distilbert-lora.toml) describes a run
against `distilbert-base-uncased` and has never been executed, because that
needs a download. Every number below is either produced by the offline run
described under [The smoke run, exactly](#the-smoke-run-exactly), computed from
files that are in the repository now, or labelled as what a formula predicts.
There is no benchmark in here and no measured accuracy for any model worth
serving.

## The split is a file in git

```
$ ft split configs/smoke.toml
splits/intents-v1.json matches the corpus (sha256:0c5de968b07bfc2796a7a1e1, unchanged)
label           train  validation        test
billing            12           3           3
cancel              6           1           1
shipping           10           2           2
technical          17           4           3
total              45          10           9

thin: validation/cancel has 1, below min_per_label_per_split of 3
thin: validation/shipping has 2, below min_per_label_per_split of 3
thin: test/cancel has 1, below min_per_label_per_split of 3
thin: test/shipping has 2, below min_per_label_per_split of 3
```

[splits/intents-v1.json](splits/intents-v1.json) is 77 lines and holds one
assignment per example id, plus the salt, the ratios and a fingerprint of the
corpus it was built from. It is committed. The adapters and metrics in `runs/`
are not: those are tracked by their MLflow run, and a checkpoint in git history
is a checkpoint whose provenance nobody can reconstruct.

Three decisions sit in that file.

**Assignment comes from a hash of the id, not from a seeded shuffle.** A seeded
shuffle is reproducible only against the exact list it was handed. Append one
example and everything downstream of it moves, so last week's validation set
shares an unknown fraction of its rows with this week's training set and
nothing says so. `sha256(salt + id)` depends on the id and the salt and on
nothing else in the file. Python's own `hash()` cannot be used at all: it has
been salted per process since 3.3, so a split built in one process would not
reproduce in the next.

**The record carries no timestamp.** It is a pure function of the corpus ids,
their labels, the salt, the ratios and the method, so rebuilding it on another
machine produces the same bytes. CI rebuilds it with `--force` on Linux and
runs `git diff --exit-code` against the copy committed from an arm64 Mac. A
creation date would be useful and would also make that check impossible.

**The fingerprint covers ids and labels, not text.** Fixing a typo in an
example leaves the split valid. Changing an example's label invalidates it,
because the label is what the stratification was balancing. Train against a
stale record and the ids added since are in no split and are never scored,
which is the quiet version of a leak; `ft train` refuses rather than warns, and
says how many ids moved.

### What stratifying cost

Two methods are implemented and the choice is real. Both numbers below come
from [tests/test_splitting.py](tests/test_splitting.py), on the 64 example
corpus, measured by adding 8 new `billing` examples.

| | `quantile` (default) | `independent` |
|---|---|---|
| Totals at n=64 | 45 / 10 / 9 | 38 / 11 / 15 |
| Per-label counts | exact | whatever the hash gives |
| `cancel` in validation | 1 | **absent** |
| Ids that change split when the corpus grows by 8 | **3** | **0** |

`quantile` sorts each label's ids by hash and cuts at the ratio boundaries, so
the strata are exact and the boundary moves when a label grows. `independent`
sends each id to whichever ratio interval its hash falls in, so nothing ever
moves and at 64 examples the test split came out at 15 rows where the ratio
asks for 9.6, with `cancel` missing from validation entirely.

That last cell is why [metrics.py](src/finetune/metrics.py) takes the label set
as an argument rather than inferring it. sklearn infers labels from the data
when `labels=` is omitted, so a validation fold holding three classes reports a
macro F1 averaged over three while the test fold averages four. Both print as
`macro_f1` and neither is comparable to the other. Early stopping on a metric
whose denominator moves between epochs is not early stopping.

Stratification won because exact strata are worth more here than stability: the
corpus is 64 rows and is not growing. On a corpus that gets appended to weekly,
`independent` is the right default and the ratio drift is the price.

## What LoRA is pointed at, and why

```toml
target_modules   = ["q_lin", "v_lin"]
modules_to_save  = ["pre_classifier", "classifier"]
```

**The query and value projections, not the key, and not the feed forward.**
Adapting Q and V is the configuration the LoRA paper reports as the best return
per parameter, and the intuition holds up: V changes what gets mixed into a
position and Q changes what it attends to, while K mostly duplicates the degree
of freedom Q already has. Adding the feed forward layers would roughly triple
the adapter for a classification head that reads one pooled vector.

**Both head layers are saved, not adapted.** This is the trap in the whole
exercise. A sequence classification head on a fresh checkpoint is randomly
initialised. Leave it frozen with the rest of the base and the adapter is being
asked to make a fixed random projection of the features separable, which trains
to roughly chance and reads as a learning rate problem for as long as you are
willing to believe it is one. `wrap_with_lora` raises if no trainable parameter
survives in the head.

**The module names are checked against the model.** `target_modules` is
architecture specific. BERT calls its projections `query`, `key`, `value`;
DistilBERT calls them `q_lin`, `k_lin`, `v_lin`. Point a DistilBERT config at
`q_proj` and, depending on the PEFT version, you get an exception or a model
with no adapter in it at all, which trains the head alone and scores a little
worse. `resolve_targets` raises and lists what the architecture does call:

```
lora.target_modules names ['q_lin', 'v_lin'], which no nn.Linear in bert is
called. available: ['classifier', 'dense', 'key', 'query', 'value']
```

## Rank, scale, and the thing that is actually the constraint

`r` and `alpha` are one decision, not two. The adapter's output is scaled by
`alpha / r`, so raising `r` on its own makes the adapter both larger and
quieter. The DistilBERT config is r 8, alpha 16, a scale of 2.

The parameter count follows from the shapes and nothing else. Each adapted
linear of width d gains an A of (r, d) and a B of (d, r), so 2rd per module,
across every layer:

| | r | width | adapted per layer | layers | adapter parameters |
|---|---|---|---|---|---|
| smoke, measured | 4 | 32 | 2 | 2 | **1,024** |
| DistilBERT, predicted | 8 | 768 | 2 | 6 | **147,456** |

The first row is observed and asserted in
[tests/test_modeling.py](tests/test_modeling.py). The second is what
`adapter_parameter_estimate` returns for DistilBERT's shapes and has never been
built, so no measured fraction of that base is quoted here. The formula is
checked against a real model in the tests precisely so that a disagreement
means some target name matched nothing.

Rank is not the binding constraint on either config. On the smoke model r 4
against a width of 32 is already a quarter of the matrix, and the 44,164
parameter random base is what limits it. On DistilBERT with 45 training
examples the constraint is the corpus: 147k free parameters fitting 45 rows,
and the reason that is not hopeless is that none of the base underneath them is
moving. Raising r would not buy capacity that 45 examples can use.

## Why the adapter's learning rate is not a fine-tuning learning rate

`lr = 3e-4` for the adapter against the 2e-5 that a full fine-tune of a BERT
family model usually wants. Two orders of magnitude, for three reasons.

1. **There are no pretrained weights at risk.** 2e-5 is small because a full
   fine-tune is editing representations that took a lot of compute to learn,
   and a large step destroys them. The base is frozen here. The worst a large
   step can do is make a bad adapter.
2. **B starts at zero, so the adapter starts as the identity.** The run begins
   at exactly the frozen model's behaviour and the first gradients have to move
   B away from zero through a rank-r bottleneck. A 2e-5 step on a matrix
   initialised to zero is a very slow start.
3. **1,024 parameters are a much noisier gradient estimate than 66 million.**
   Fewer parameters absorbing the same batch means a larger step is tolerable
   and a smaller one is slow.

The head gets its own group at `head_lr = 1e-3`, higher again, because it is
randomly initialised and has further to travel than an adapter that starts
correct.

**The adapter group has no weight decay.** Decay pulls B toward zero, and zero
is the identity, so decay on the adapter is a prior that the adapter should do
nothing. The head keeps the decay, because the head fitting 45 rows is where
overfitting shows up.

Everything in that paragraph is reasoning about the optimiser, not a measured
result. None of these rates has been compared against another on a real
checkpoint.

## Early stopping, and the bug it shipped with

`monitor = "val_macro_f1"`, `mode = "max"`, `patience = 3`, `min_delta = 0.002`.
The rule is a plain object that takes a number and returns a verdict, so the
tests feed it lists of floats and assert which epoch it stops on; it holds no
model.

Two things in it are worth reading.

**A monitor must start with `val_`, and the config refuses anything else.**
Stopping on training loss is not early stopping. Training loss falls
monotonically for a long time whatever is happening to held-out data, so the
rule would never fire until it no longer mattered.

**`best_value` starts at negative infinity, and the first version started at
0.0.** For macro F1 in [0, 1] that looks harmless. `val_mcc` runs from -1 to 1,
and this module reports it. With MCC at -0.4 then -0.1, neither epoch beat 0.0,
so nothing was ever recorded as an improvement: `best_epoch` stayed at -1, the
run stopped as soon as patience ran out, and there was no best state to
restore, meaning the saved adapter was whichever one the rule had just decided
against. The regression test is
`test_a_monitor_that_starts_negative_still_improves`.

`min_delta` has a cost that is easy to miss. It is measured against the best
value rather than the previous epoch, so small gains accumulate; but a metric
climbing by 0.0005 an epoch never clears 0.002, and the run is stopped on epoch
4 having gained 0.0015 and still rising. Both halves of that are pinned in
[tests/test_earlystop.py](tests/test_earlystop.py).

## Telling overfitting from a bad split

With validation at 10 rows, `cancel` has exactly 1 of them. That one row's
recall is 0.0 or 1.0 and nothing in between.

Here is what that is worth, computed on the test split's shape with the four
declared labels. A model predicting `technical` for all 9 rows scores a macro
F1 of 0.1250. Change one prediction, the single `cancel` row, from `technical`
to `cancel`, and the macro F1 is 0.3864. **One row out of nine moves the
headline metric by 0.26.**

So `min_delta = 0.002` is guarding against float noise while the sampling noise
is two orders of magnitude larger. The rule is correct and on this split it is
reading mostly chance.

That is what `ft split`'s thin-cell warning is for. It prints the size of the
smallest unit of change before anything is trained.

Given that, three shapes are distinguishable and one is not:

| Train loss | Validation | Test at the end | Reading |
|---|---|---|---|
| falling | falling | close to validation | learning |
| falling | rising | close to validation | overfitting, and the head is where to look |
| falling | moving | far from validation | the split, not the model |
| falling | tracks train from epoch 1 | close to validation | suspect a leak |

Row three is the one people misdiagnose. If validation and test disagree by
more than a couple of rows' worth, the two folds are not samples of the same
thing, and raising dropout or lowering the learning rate is chasing a
difference that is not in the model. Row four is why the vocabulary is fitted
on the training texts alone: fitting it on the whole corpus gives the embedding
table rows for words that only occur in validation and test, and training would
update them. It is a small leak at this size and it is the same mistake as
fitting a scaler on everything.

The mechanical version of a leak, an id in two splits, cannot happen: a
duplicate id is refused when the corpus loads, naming both line numbers.

## The smoke run, exactly

```
$ ft train configs/smoke.toml

epoch  train_loss   val_loss  val_macro_f1   val_acc
    1      1.3777     1.3578        0.1429    0.4000
    2      1.3615     1.3392        0.1429    0.4000
    3      1.3486     1.3254        0.1429    0.4000

parameters   1156 trainable of 45320 (2.5508%)
             1024 adapter, 132 head
best epoch   1 at val_macro_f1 0.1429
stopped      val_macro_f1 has not improved by more than 0.002 for 2 epoch(s); best 0.1429 at epoch 1
test         macro_f1 0.1250  accuracy 0.3333  n 9
artefacts    runs/intents-v1-b957fc53a14d0a73
```

What that was, in full:

* **Model**: a BERT architecture built from a config with **random weights**,
  2 layers, hidden size 32, 2 attention heads, 45,320 parameters. Nothing was
  downloaded. It is not a pretrained checkpoint and it has never seen text.
* **Tokenizer**: a word level vocabulary built in memory from the 45
  training texts, 293 words plus 5 special tokens. Fitting it on all 64
  examples would give 369.
* **Data**: 45 training examples, 10 validation, 9 test, from the 64 line
  hand written corpus in [data/](data/README.md).
* **Epochs**: 4 asked for, 3 run. Patience of 2 ended it.
* **Time**: about 4 seconds on a CPU.

**It learned to predict the majority class and nothing else.** Every one of the
9 test rows was labelled `technical`, which is 3 of 9 correct, which is the
0.3333 accuracy above, and a macro F1 of 0.125 because one class has F1 0.5 and
three have 0.0. MCC is exactly 0.0. That is the correct outcome for 1,156
trainable parameters on a random base with 45 examples, and it is what this
config is for: it proves the loop, the seeding, the early stopping, the saving
and the reloading all work. It is not a result and the repository does not have
one.

The adapter it saved is 5,912 bytes. `ft eval` rebuilds the random base from
the seed, loads that file, and reports `macro_f1 0.1250 accuracy 0.3333`, the
same numbers training printed. That round trip holds for this config because
the base weights come from a seed; a real checkpoint is identified by its name
instead, which is why `labels.json` records both the base and the label order.

Running the same config twice gives identical losses to every decimal place,
asserted in `test_two_runs_of_the_same_config_agree`. What seeding does not buy
is reproducibility across a different torch version, device or batch size,
because the order of the float additions changes.

## Running it

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

ft config configs/smoke.toml          # resolve, validate, show where each key came from
ft split configs/smoke.toml           # check the committed split against the corpus
ft train configs/smoke.toml           # about 4 seconds, downloads nothing
ft eval  configs/smoke.toml           # score the test split with the saved adapter

pytest -q
ruff check . && ruff format --check .
```

`ft config` prints the resolved config with the file that set each key, which
is the output worth having when configs inherit:

```
  train.seed                     1337                                base.toml
  train.epochs                   4                                   smoke.toml
  early_stopping.monitor         val_macro_f1                        base.toml
  early_stopping.patience        2                                   smoke.toml
```

An unknown key is refused rather than defaulted, and the message names the file
it came from, not the file you asked for:

```
config error: typo.toml: unknown key early_stopping.min_dleta. known keys in
[early_stopping]: min_delta, mode, monitor, patience
```

[configs/base.toml](configs/base.toml) names no model, so `ft config
configs/base.toml` exits 2. A base config that happens to run is the one people
end up running.

Exit codes: 0 worked, 1 a check failed, 2 the config or command line is wrong.

### Tracking

`tracking.enabled` is `false` by default and the run uses an in-memory tracker
that keeps everything it was handed, which is what the tests assert against.
Switch it on and the same calls go to MLflow:

```toml
[tracking]
enabled = true
uri = "sqlite:///runs/mlflow.db"
experiment = "intent-lora"
```

One run logs 46 parameters and 13 metrics, with `val_macro_f1` and
`train_loss` as per-epoch series and the test metrics at the final step. The
run directory goes up as an artifact. `ft train` on the smoke config with
tracking on produced exactly that, and the parameters include
`split_counts`, the label order, the trainable parameter counts, and a
`base_checksum` over the base weights, because a randomly initialised base has
no name on any hub and an adapter is only valid against the base it was trained
on.

Two things about that URI, both found by running it.

The scheme is not optional. MLflow 3 put the filesystem tracking store in
maintenance mode and its constructor raises on a bare path like `runs/mlruns`
unless `MLFLOW_ALLOW_FILE_STORE` is set. The config validator checks for a
scheme so that arrives as one line before the model is built, rather than as a
traceback out of mlflow's store registry after it.

And a sqlite tracking store does not move the artifacts. MLflow still writes
them to `./mlruns` in the working directory, so a project root collects a
second output tree next to the one it configured. `MlflowTracker` creates the
experiment with an artifact location under `runs/mlartifacts` instead, which
has to happen at creation because an experiment's artifact location cannot be
changed afterwards. `mlruns/` is in `.gitignore` anyway, because an experiment
created any other way will put it back.

## Offline by construction

158 tests, 119 of which touch no model at all. No network, no key, no
downloaded weights, no container.

```
pytest -q                 # 158 passed in about 7 seconds
pytest -q -m "not torch"  # 119 passed in about 4 seconds
```

Four things make that true.

1. **[tests/conftest.py](tests/conftest.py) sets `HF_HUB_OFFLINE` and
   `TRANSFORMERS_OFFLINE` before anything imports transformers.** Any call that
   would reach the Hub raises instead.
2. **Every torch and transformers import sits inside the function that needs
   it.** That is what makes point 1 work: huggingface_hub reads those variables
   at import time, so a module-scope import would have the package loaded
   during test collection and the variables arriving too late. CI runs the pure
   subset separately to keep that honest, and asserts the Hub cache is empty
   afterwards.
3. **The tests that do build a model build a randomly initialised one**, two
   layers wide 32, in process, in under a second. They are marked `torch` and
   the marker is what the pure subset deselects.
4. **No test trains anything that could be mistaken for a result.** The
   training tests assert the mechanism: that the loss falls, that the rule
   stops on the epoch it should, that the saved adapter reloads to the same
   score, that a stale split is refused.

The corpus is 64 messages written by hand for this repository.
[data/README.md](data/README.md) says so beside the file. None of it was
collected from a product, exported from a help desk, or generated by a model,
and no real person, order or invoice appears in it.

## What was given up

| Decision | Cost |
|---|---|
| Stratified `quantile` split | 3 ids of 64 change split when the corpus grows by 8. `independent` moves none and misses a class instead |
| Fingerprint ignores example text | A rewritten example keeps its split, and a corpus whose text changed completely still verifies |
| No timestamp in the split record | No record of when it was built. Byte-reproducibility was worth more |
| Vocabulary fitted on training texts only | More unknown tokens at evaluation time than fitting on everything would give |
| No class weighting in the loss | Imbalance is handled once, in the metric. Weighting the loss as well means two corrections pulling the same way and no way to attribute a change to either |
| Test split scored once per run | No test curve to look at. A test score consulted every epoch is a second validation set |
| No learning rate schedule or warmup | One fewer thing to tune, and one fewer thing to credit. AdamW with two flat groups |
| `runs/` is not committed | No adapter in git history. The MLflow run is the record |

## Not done

* **No run against a pretrained checkpoint.** This is the gap that matters.
  Everything about the loop is exercised, and nothing about whether these
  hyperparameters are any good is known, because the only model trained here
  was random. The next thing is `ft train configs/distilbert-lora.toml` and an
  honest table from it.
* **The validation set is too small to early-stop on.** 10 rows, one of them
  the only `cancel` example. The rule is correct and the signal it reads is
  mostly noise. Cross validation would fix it for a corpus this size and would
  mean five runs per experiment and no single recorded split, which is the
  thing this repository is built around.
* **`r` and `alpha` have never been varied.** They are one decision and the
  repository only tests that the parameter count moves with `r` as the formula
  says. A sweep needs a real base model first.
* **No merge step.** PEFT can fold an adapter back into the base weights for
  inference. Nothing here does, so serving this would mean shipping the base
  and the adapter and loading both.
* **`independent` is implemented and not used by either shipped config.** It is
  the right default for a corpus that grows, and the evidence for that claim is
  one measurement on a 64 row file.
