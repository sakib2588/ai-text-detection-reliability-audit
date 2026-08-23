# NLP Final Term Project -- Your Half of the BERT/DeBERTa Sweep

Group 02, Section B. The full-dataset hyperparameter sweep (32 configurations total)
is split across two machines to finish faster. **This package contains your assigned
14 configurations.** Sakib's machine is running the other 18 (4 already-identified
winning configs, plus his own half of the remaining 28).

## Install torch FIRST -- this is the step that broke last time

Do not reuse a `requirements.txt` that has `torch` pinned with a long list of
`nvidia-*` sub-packages copied from someone else's `pip freeze`. That is exactly
what caused the `nvidia-cufile-cu12` error before -- that specific package is
Linux-only and has no Windows wheel at all, so it can never install on your
machine no matter what you do.

Instead, install torch from PyTorch's own index, which resolves the correct
Windows build automatically:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then install everything else:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python run_my_half.py
```

## What you need

- An NVIDIA GPU (you mentioned an RTX 4060 -- that has 8-12 GB of VRAM depending on
  the exact model, which is enough; the script automatically reduces its batch size
  if it ever isn't).
- A reasonably recent NVIDIA driver. If the torch install command above can't find a
  matching wheel, your driver may be too old for CUDA 12.8 -- update it from
  nvidia.com/drivers and try again.
- Python 3.12 (you already have this from the classical-baseline package).

## What you do NOT need

You do **not** need `daigt.csv` or `hc3.jsonl`. This package includes the exact
balanced, duplicate-aware-split data already prepared:

| File | What |
|---|---|
| `data_D1.parquet`, `data_D2.parquet` | the balanced full-dataset text |
| `split_D1.npz`, `split_D2.npz` | the exact train/validation/test split |

These are the same files used for the classical baselines you already ran, and the
same ones Sakib's machine is using for its half of the sweep. This is what makes your
14 results combinable with everyone else's into one table -- if you rebuilt the split
from the raw datasets yourself, even a small difference in library versions could
shuffle it slightly differently, and your results would then not belong in the same
table as everyone else's.

## Expected time

Roughly **6-8 hours** for all 14 configurations on an RTX 4060. Leave it running --
you do not need to watch it.

## Safe to stop and restart

Exactly like the classical-baseline script: if you close the terminal, the power goes
out, or you just want to pause and resume later, run `python run_my_half.py` again.
It checks what is already finished (both a results file and a probability file must
exist) and skips it, and if a run was interrupted partway through training it resumes
from its last completed epoch rather than starting over.

## Do not touch these two things

**Do not upgrade `transformers` past 4.57.6.** It is pinned exactly on purpose --
version 5.x removed the `warmup_ratio` argument this project's specification
requires. If pip tries to install a newer version anyway, force it back:

```bash
pip install transformers==4.57.6
```

**Do not run more than your assigned 14 configurations.** `MY_HALF` in
`run_my_half.py` is deliberately exactly the configurations nobody else is running --
running extra ones wastes your GPU time on work that already exists elsewhere.

## What to send back

The entire `results/` folder and the entire `probs/` folder. Both are small (a few
megabytes total, not gigabytes) -- these are just JSON and NPZ files, not model
weights, so they zip up and transfer easily over Discord, email, a USB drive, or
whatever is convenient.

## How this merges -- no conflict is possible, and here is how to check

Every single result file across all three sources (your 14 configs, Sakib's 26,
and the classical baselines you already ran) has a unique filename by construction:
it always encodes the exact dataset, model, and hyperparameters, so two different
runs can never produce the same filename. Merging is not a manual reconciliation --
it is copying files into the same folder:

```bash
# from the folder where Sakib's results/ and probs/ live:
cp path/to/your_results/*.json  results/
cp path/to/your_probs/*.npz     probs/
```

**To confirm nothing collided,** count files before and after copying. If the count
after equals the count before plus exactly the number of files you copied in, nothing
was overwritten:

```bash
ls results/*.json | wc -l   # before copying, note this number
# ... copy your files in ...
ls results/*.json | wc -l   # after: should be exactly (before + 14)
```

If that final count is anything less than `before + 14`, something did overwrite a
file and that needs investigating before trusting the combined table -- but given the
filenames are built from an exact configuration that only one machine was ever
assigned, this should not happen.
