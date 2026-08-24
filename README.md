# Detecting Machine-Generated Text: a Surface-Content Decomposition

Detectors of AI-generated text routinely report 99%-plus accuracy on the DAIGT V2
and HC3 benchmarks. This project asks what that number is actually made of. It
decomposes reported accuracy into three separable components, measured at full
corpus scale with five models (Naive Bayes, Logistic Regression, SVM, and
fine-tuned BERT and DeBERTa):

1. **Cross-dataset generalization** — strict one-way transfer, three seeds.
2. **Dependence on surface artifacts** — an artifact-cleaning ablation, run both
   zero-shot and with full retraining.
3. **Adversarial robustness** — typo injection, homoglyph substitution, and
   back-translation.

The headline finding is dataset-dependent rather than universal: HC3's reported
accuracy substantially overstates real-world reliability, while DAIGT V2's does so
only modestly. Three independent experiments converge on that split. The full
statement of results, with every number traced to a file on disk, is in
`paper/draft/NUMBERS_SSOT.md`.

Two write-ups exist: a six-page ICCIT conference paper in `paper/iccit/` and a
longer markdown manuscript in `paper/draft/`.

---

## Layout

```
README.md                      this file
requirements.txt               Python dependencies

docs/                          prose that is not the paper
  REPORT.md                    the course final-term report
  REPORT_OUTLINE.md            its outline, with the source file behind each table
  ENSEMBLE_EXPLAINED.md        why the ensemble underperformed its members
  REVIEW_2026-08-24.md         a dated record of the last review round
  PROJECT_EXPLAINER.md/.pdf    plain-language walkthrough of the whole project
  literature/                  the literature review and the research-gap report

paper/
  draft/                       markdown manuscript, section per file, plus
                               NUMBERS_SSOT.md -- the single source of truth for
                               every number quoted anywhere
  iccit/                       LaTeX conference paper (main.tex, sections/, bib/,
                               figures/); build.sh builds it, check.sh gates it

notebooks/                     the submission notebooks and the analysis notebook
  figures/                     17 plots the analysis notebook emits
  tables/                      4 CSVs the analysis notebook emits
  builders/                    the scripts that generate those notebooks

experiments/
  midterm/                     the original small-scale sweep (6k rows)
    work/ results/ probs/ figures/ models/
  paper_scale/                 the full-scale runs behind the paper
    work/ results/ probs/ models/ figures/ logs/
  audit/                       contamination, decomposition and claim-verification
                               scripts, and the JSON they produce
  portable/                    self-contained bundles handed to collaborators
    full_dataset/ classical_only/ gpu_half/

tables/                        the result tables the paper cites, plus run_summary.json

archive/                       kept for provenance, not used by anything live
  backup_results_1102/         byte-identical duplicate of an earlier midterm run
  quick_validation/            an early quick-validation sweep
  demo_full_dataset/           the demo notebook's outputs
  submissions/                 submission zip bundles
  logs/                        stray logs
```

## Reproducing the full-scale results

Everything below runs from `experiments/paper_scale/` and resolves its own paths,
so it does not matter which directory you launch it from. The source corpora
(`daigt.csv`, `hc3.jsonl`) live outside this repository and are not tracked.

```bash
PY="/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Project /.venv/bin/python"
cd experiments/paper_scale

"$PY" build_full_splits.py            # one fixed split per dataset, duplicate-group-aware
"$PY" run_full_scale.py               # the 4 winning transformer configs, 3 seeds
"$PY" run_full_grid.py                # the remaining hyperparameter cells, seed 42
"$PY" classical_full.py               # Naive Bayes / Logistic Regression / SVM at full scale
"$PY" full_model_evaluation.py        # unified evaluation across all five model families

"$PY" retrain_seeds_for_cross.py      # seeds 123/456 with weight saving on
"$PY" cross_dataset_eval_multiseed.py # strict one-way cross-dataset transfer

"$PY" run_artifact_cleaning_zeroshot.py   # artifact ablation, existing checkpoints
"$PY" run_artifact_cleaning_full.py       # artifact ablation, full retraining
"$PY" run_adversarial_robustness.py       # typo, homoglyph and back-translation attacks

"$PY" build_naive_splits.py           # the leaky random split, for contrast
"$PY" run_naive_comparison.py         # the accuracy cost of that leakage

"$PY" make_paper_figures.py           # writes PDFs into paper/iccit/figures/
```

Then verify, from the repository root:

```bash
"$PY" experiments/audit/verify_paper_claims.py
```

That script re-derives every claim in the paper from the result files and writes
`experiments/audit/paper_claim_verification.json`. It is the fastest way to
confirm a working tree is intact.

The last two stages are wrapped by
`experiments/paper_scale/run_seed_robust_cross_dataset_overnight.sh`, and
`experiments/paper_scale/_guardian_nlp.sh` is a cron-driven recovery guardian that relaunches any of the
long jobs that die. It acts only on jobs that are neither alive nor already
complete, and `touch experiments/paper_scale/.nlp.stop` halts it.

## Where the numbers live

- `paper/draft/NUMBERS_SSOT.md` is the source of truth. Every figure quoted in
  either manuscript traces to a row there, and each row names the file it came from.
- `tables/` holds the cited result tables as CSV.
- `experiments/*/results/` holds the raw per-run JSON those tables aggregate.

Never quote a number from memory or from a notebook cell. If it is not in
NUMBERS_SSOT, it is not established.

## Things worth knowing

- **Model checkpoints are not in git.** `experiments/midterm/models/` (2.3 GB) and
  `experiments/paper_scale/models/` (6.7 GB) are ignored, as are the raw corpora
  and the regenerable prediction arrays under the live `probs/` and `work/`
  directories. The archived prediction arrays under `archive/` and the audit
  arrays under `experiments/audit/` are tracked on purpose: the checkpoints that
  produced them no longer exist, so they cannot be regenerated.
- **The `*_executed.ipynb` notebooks are records, not programs.** They were run
  under the pre-reorganization layout and are deliberately left byte-identical, so
  the paths inside them point at directories that have since moved. The runnable
  notebooks beside them have been updated.
- **`archive/` is inert.** Nothing in the live pipeline reads from it.
