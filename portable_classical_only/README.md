# NLP Final Term Project -- Classical Baselines (the midterm part)

Group 02, Section B. **This is only the midterm's three models** -- Naive Bayes,
Logistic Regression, and Support Vector Machine -- re-run at full dataset scale.
BERT, DeBERTa, and the ensemble are a separate job running on a different machine;
you do not need to touch those, and this package cannot run them even if you wanted
to (no torch is installed by this `requirements.txt` on purpose).

## Why this is a much smaller, simpler package

No GPU. No CUDA. No torch. No transformers. Just `pandas`, `numpy`, `scikit-learn`,
`nltk`, and `pyarrow` -- ordinary Python packages that install cleanly on any machine,
any recent Python version, Windows/Mac/Linux, no version gymnastics required.

**Expected time: 15-30 minutes on a normal laptop CPU.** Not hours.

## How to run it

```bash
pip install -r requirements.txt
python run_classical_full.py
```

That is genuinely the whole procedure.

## What's included and why

| File | What |
|---|---|
| `run_classical_full.py` | the script you run |
| `requirements.txt` | five ordinary packages, no exact version pins needed |
| `data_D1.parquet`, `data_D2.parquet` | the balanced full-dataset text, already prepared |
| `split_D1.npz`, `split_D2.npz` | the exact train/validation/test split |

**The split files are the important part.** They are included rather than rebuilt from
scratch specifically so your results land on the *exact same* split as the BERT/DeBERTa
run happening elsewhere. If you rebuilt the split independently, even a tiny library
version difference could shuffle it slightly differently, and then your three rows and
their two rows would not belong in the same table together. Using the same split files
is what makes that combination valid.

The split is also duplicate-aware: HC3 was found to contain 7.16% duplicate or
near-duplicate rows, which would otherwise leak from training into the test set. This
split already accounts for that -- nothing you need to do, it is just how the included
files were built.

## What you get at the end

- `results/*.json` -- one file per model, with accuracy/precision/recall/F1
- `table2_classical_rows_full.csv` -- the three classical rows, ready to slot into
  the final combined table (Naive Bayes, Logistic Regression, Support Vector Machine)

**Send back `table2_classical_rows_full.csv` and the `results/` folder.** Those three
rows plug directly into the same Table 2 as the BERT/DeBERTa/ENSEMBLE rows coming from
the other machine.

## Safe to stop and restart

If you close the terminal partway through, just run `python run_classical_full.py`
again -- it checks what is already finished and skips it, so you never lose progress
or redo work that already completed.
