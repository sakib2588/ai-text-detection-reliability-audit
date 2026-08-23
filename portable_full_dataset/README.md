# NLP Final Term Project -- Full-Dataset Pipeline

Group 02, Section B. This runs the whole study on the **complete** DAIGT V2 and HC3
corpora, not the 6,000-row midterm-matched sample used in the coursework draft.

## What this does

1. Loads and balances both datasets to their full available size:
   - DAIGT V2: 17,497 documents per class = 34,994 total
   - HC3: 26,903 documents per class = 53,806 total
2. Splits each 80/10/10 (train/validation/test) with a **duplicate-aware split** --
   see "Why the duplicate-aware split" below.
3. Fine-tunes BERT and DeBERTa across all 8 hyperparameter configurations from the
   assignment sheet, on both datasets (32 runs).
4. Repeats the winning configuration for each model/dataset at 2 more random seeds
   (123, 456), for 8 more runs, to check how much the numbers move with training noise.
5. Refits Naive Bayes, Logistic Regression, and SVM on the identical split, at the
   same full scale, so the comparison table is not mixing two different sample sizes.
6. Builds the ensemble (validation-weighted soft vote of the best BERT and best
   DeBERTa) and writes out `table1_experiments_full.csv` and `table2_combined_full.csv`.

## Requirements

- **Python 3.12** (tested on 3.12.3). Check with `python --version` or `python3 --version`
  before starting -- this matters because the pinned `torch` and `transformers` versions
  in `requirements.txt` were verified against 3.12 specifically. A different Python 3.12.x
  patch version is fine; a different major version (3.10, 3.11, 3.13) is more likely to
  hit a dependency resolution problem and is not what this was tested against.
- An NVIDIA GPU with CUDA support, for the transformer sweep (see "Disk and memory" below
  for what happens if VRAM is limited -- it degrades gracefully rather than crashing).
- pip (comes with Python).

## How to run it

```bash
# 0. confirm you're on Python 3.12
python --version

# 1. put daigt.csv and hc3.jsonl next to this script
#    (or set NLP_DATA_DIR to point at the folder that has them)

# 2. install dependencies
pip install -r requirements.txt

# 3. run everything
python run_full_pipeline.py
```

If your system's default `python`/`pip` point at a different version, use a Python 3.12
virtual environment instead:

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_full_pipeline.py
```

That is the whole procedure. No Jupyter required -- this is a plain script so you can
leave it running overnight in a terminal (or under `nohup`/`screen`/`tmux`) without
needing to babysit a notebook kernel.

## Expected time

**Roughly 14-15 hours on a single consumer GPU** (measured on an RTX 3060 Ti, 8 GB
VRAM). Almost all of that is the 32-run hyperparameter sweep; the classical baselines
and table generation take minutes. A GPU is required for the transformer sweep. If you
only have a CPU, you can still run the classical baselines and, if someone else's
`results/` and `probs/` folders are copied in, regenerate the tables without a GPU at
all (`--skip-sweep --skip-robustness`).

## It is safe to stop and restart

This is the most important thing to know before running it. **Every finished piece of
work is written to disk before the next one starts, and the script checks what is
already done before repeating it.** If your laptop sleeps, the terminal closes, the
power goes out, or you just want to stop and resume tomorrow: press Ctrl+C (or just
let it die) and run the exact same command again. It will:

- skip every training run that already has both a results file and a probability file
  saved (instant, no GPU time spent re-checking),
- resume a run that was interrupted mid-training from its last completed epoch rather
  than starting that one run over,
- never leave a half-written result file behind that could be mistaken for a finished
  one (every save is atomic: written to a temporary file first, then renamed into
  place only once complete).

Nothing about restarting requires you to remember which run you were on. The
checkpoint and results folders track that for you.

## Command-line options

```bash
python run_full_pipeline.py --skip-sweep         # skip the 32-run sweep
python run_full_pipeline.py --skip-robustness    # skip the 3-seed robustness runs
python run_full_pipeline.py --skip-classical     # skip Naive Bayes / LogReg / SVM
```

Useful if you want to split the work across sessions, or if someone else has already
run part of it and shared their `results/` and `probs/` folders with you -- just drop
those two folders in next to this script before running, and anything already present
will be skipped automatically.

## Why the duplicate-aware split

HC3 was audited at full scale and found to contain 7.16% duplicate or near-duplicate
rows (repeated answers, largely from the `reddit_eli5` source). Under a plain random
80/20 split, this leaks 11.2-11.3% of the test set: those test rows have a
byte-identical or near-identical twin sitting in the training data, which inflates
whatever accuracy gets reported. DAIGT does not have this problem (0.01% duplication),
but the same group-aware split is applied to both datasets for consistency.

This script's split guarantees that no group of duplicate-content rows is ever split
across train, validation, and test -- an entire duplicate group lands on one side of
the boundary or the other, never both. This is verified by an assertion in the code
that will halt the script immediately if it is ever violated.

## Important: the transformers version is pinned on purpose

`requirements.txt` pins `transformers==4.57.6`. **Do not upgrade this.** Version 5.x of
transformers removed the `warmup_ratio` argument that the project specification
requires (Warmup Ratio = 0.1). If you `pip install --upgrade transformers`, the script
will fail with a `TypeError` about an unexpected keyword argument.

## What you get at the end

| File | What |
|---|---|
| `table1_experiments_full.csv` | the 17-row experiment table (8 configs x 2 models, plus ENSEMBLE), full-dataset scale |
| `table2_combined_full.csv` | the 6-row combined table (classical models + BERT + DeBERTa + ENSEMBLE), full-dataset scale |
| `results/` | one JSON per run: exact hyperparameters, accuracy/precision/recall/F1, confusion matrix, training time |
| `probs/` | one NPZ per transformer run: predicted probabilities on validation and test, used to build the ensemble |
| `models/` | the seed-42 winning BERT and DeBERTa checkpoints, saved for inference |
| `work/` | the fixed, duplicate-aware train/val/test split for each dataset |

## Disk and memory

**Two different kinds of memory can run out, and this script protects against both.**

**System RAM (the 16 GB most laptops ship with today):** the classical baselines use
a multi-core preprocessing step (parallel NLTK tokenising and lemmatising, since doing
this on one core for 85,000+ HC3 documents would be the slowest part of the whole
run). The number of worker processes is auto-detected from your CPU core count but
capped at 8, specifically so this step cannot balloon in memory use on a machine with
many cores. 16 GB of system RAM is comfortably enough with the default settings. If
you ever do see a `MemoryError` (not the same thing as a CUDA error below), set
`NLP_WORKERS=1` or `NLP_WORKERS=2` before running to use less parallelism:

```bash
NLP_WORKERS=2 python run_full_pipeline.py
```

**GPU VRAM (the memory on your graphics card, separate from system RAM):** this is
what "CUDA out of memory" errors refer to, and it is handled automatically. Every
training run starts at its assigned batch size and, if that does not fit in your
GPU's memory, automatically retries at half the batch size with twice the gradient
accumulation steps -- repeating down to a floor of batch size 4 -- so the effective
batch size (what the hyperparameter actually means for training) stays exactly what
the configuration specifies, only spread across more, smaller steps. The same
automatic fallback applies separately to the evaluation step after training, which
uses its own batch size and can hit a different memory ceiling than training does.
You do not need to configure anything for this -- if your GPU has less VRAM than the
one this was built on (an RTX 3060 Ti, 8 GB), the script will simply take a bit
longer per run rather than crashing.

Roughly 15-20 GB of disk is needed for HuggingFace model caches plus intermediate
checkpoints (checkpoints are deleted automatically after each run finishes, so this
does not grow unbounded).
