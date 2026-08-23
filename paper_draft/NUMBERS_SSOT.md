# Numbers SSOT — Paper Track (Full-Scale Only)

Every number below traces to a file on disk. Do not type a number into prose
without checking it here first, and do not add a row here without citing its
source file. Course-track (6k-sample) numbers are OUT OF SCOPE for this paper
and are not listed — see `REPORT.md` if the course submission needs them.

## 1. Headline 5-model comparison (Table 2, full-corpus)

Source: `Final/table2_combined_full.csv`

| Model | D1 (DAIGT) F1 | D2 (HC3) F1 |
|---|---|---|
| Naive Bayes | 0.9591 | 0.8713 |
| Logistic Regression | 0.9893 | 0.9551 |
| SVM | 0.9910 | 0.9449 |
| BERT (best config) | 0.9916 | 0.9945 |
| DeBERTa (best config) | 0.9949 | 0.9980 |
| Ensemble (soft-vote) | 0.9949 | 0.9982 |

Note: review recommends demoting the ensemble to a discussion point, not a
headline result (see `paper_review/LITERATURE_REVIEW_CANONICAL.md` Section 5.3).

## 2. Full hyperparameter sweep (Table 1, full-corpus, BERT/DeBERTa only)

Source: `Final/table1_experiments_full.csv` — 16 configs (2 models x 2 lr x 2
bs x 2 wd) x 2 datasets. Use for the methodology/sweep-design section; do not
re-list all 16 rows in the main paper, reference the CSV.

## 3. Cross-dataset transfer — DONE, 3 seeds (42/123/456)

Source: `Final/table_cross_dataset_generalization_3seed.csv`. Supersedes the
single-seed table (`table_cross_dataset_generalization.csv`, kept for
reference — seed-42-only values match this table's per-seed data).

| Trained on | Model | In-domain F1 (mean) | Cross-domain F1 (mean) | Cross-domain F1 (min-max) | Gap (mean) | Gap (min-max) |
|---|---|---|---|---|---|---|
| DAIGT V2 | BERT | 0.9921 | 0.7902 | 0.7696-0.8103 | 0.2019 | 0.1817-0.2220 |
| DAIGT V2 | DeBERTa | 0.9930 | 0.9096 | 0.8907-0.9259 | 0.0833 | 0.0672-0.1010 |
| HC3 | BERT | 0.9930 | 0.8311 | 0.8130-0.8523 | 0.1619 | 0.1400-0.1822 |
| HC3 | DeBERTa | 0.9970 | 0.8512 | 0.8304-0.8874 | 0.1458 | 0.1098-0.1668 |

The single-seed pattern holds under 3-seed averaging: DeBERTa's gap stays
roughly half BERT's in both directions. BERT's asymmetry (DAIGT-V2-to-HC3
harder than the reverse) also holds at the mean level (0.2019 vs 0.1619),
though the two directions' min-max ranges are close and nearly touch
(0.1817-0.2220 vs 0.1400-0.1822) — with n=3 per cell this project's protocol
(bootstrap CI, no parametric tests at n<=30) means the asymmetry should be
reported as a mean-level pattern that survives 3-seed averaging, not as a
statistically confirmed effect; do not overclaim non-overlap.

## 4. Scale-dependent BERT seed instability

- Small-scale (6k-sample) D1 BERT, config `lr=3e-05 bs=32 wd=0.1`: F1 values
  `[0.9908, 0.985, 0.9641]` across seeds 42/123/456, **spread = 0.0267**.
  Source: `Final/work/seed_robustness.csv` (verified identical to
  `Final/models/manifest.json`'s seed-42 entry, confirming this file is
  course-track/small-scale, not full-scale — do not confuse with row 5 below).
- Full-scale D1 BERT, same config `lr=3e-05 bs=32 wd=0.1`: test F1
  `[0.9916 (s42), 0.9927 (s123), 0.9920 (s456)]`, **spread = 0.0011**.
  Source: `Final/paper_scale/results/full_D1_BERT_lr3e-05_bs32_wd0.1_s{42,123,456}.json`
  (`test.f1` field). **Shrinkage factor: 0.0267 / 0.0011 ≈ 24.3x.**
  **CORRECTED 2026-08-23** — the 2026-08-22 value (spread 0.0016, ratio
  16.7x) was verified correctly at the time but the seeds-123/456 files
  were subsequently overwritten by a post-session-crash relaunch of
  `retrain_seeds_for_cross.py` (`force=True`, re-trains rather than
  skipping) — this project's own `REPORT.md` already documents that GPU
  training is not bitwise-deterministic and best-epoch selection can pick
  a materially different checkpoint run-to-run, so a different spread on
  retraining the same seeds is expected behavior, not an error. Any prose
  citing 0.0016 / 16.7x is stale and must be updated to 0.0011 / 24.3x.

## 5. Contamination audit — full corpus

Source: `Final/audit/daigt_full_audit.json`, `Final/audit/hc3_full_audit.json`

| | DAIGT V2 (n=44,868) | HC3 (n=85,449) |
|---|---|---|
| Duplicate rows | 6 (0.01%) | 6,118 (7.16%) |
| Duplicate groups | 6 | 5,986 |
| Cross-label texts | 0 | 0 |
| Artefact rows (e.g. API-error text stored as answer) | n/a | 476 ChatGPT-side, 7 human-side |

**OPEN — do not cite yet:** the "11.2–11.3% test-set leakage under naive
80/20 split" figure used in earlier session summaries has **no located source
file**. `hc3_full_audit.json` has no leakage-rate field; `daigt_full_audit.json`'s
`leakage_by_seed` is null for all 3 seeds (DAIGT has ~zero leakage, consistent
with its 0.01% dup rate). The only located leakage number is the **small-scale**
one in `Final/work/contamination_audit.json`: HC3 test set, 8/1200 rows leaked
= **0.67%** (this is the number `REPORT.md` and the canonical lit review both
actually cite — "eight leaked rows... 0.67%"). Treat 11.2–11.3% as
unconfirmed/possibly-wrong until a full-corpus leakage script and its output
file are located or re-run; do not put it in the paper.

## 6. Artifact-cleaning ablation (Gap 4) — DONE, both stages

Source: `Final/table_artifact_cleaning_zeroshot.csv` (test-time-only cleaning,
existing checkpoints, no retraining) and `Final/table_artifact_cleaning_full.csv`
(retrain from scratch on cleaned data, same winning config, seed 42 only).
The full retrain-based numbers are the headline; zero-shot is the
secondary/discussion figure per the review's own framing.

| Dataset | Model | Raw F1 | Cleaned F1 (zero-shot) | Cleaned F1 (full retrain) |
|---|---|---|---|---|
| DAIGT V2 | BERT | 0.9916 | 0.9916 | 0.9936 |
| DAIGT V2 | DeBERTa | 0.9917 | 0.9917 | 0.9943 |
| HC3 | BERT | 0.9917 | 0.9917 | 0.9916 |
| HC3 | DeBERTa | 0.9973 | 0.9906 | 0.9962 |

Note the "Raw F1" column here is the WINNERS-config checkpoint's own score
(see Section 7 below for why this sometimes differs from Section 1's
table2-reported number), not table2's per-cell grid maximum — this table is
an internally consistent raw-vs-cleaned comparison on a fixed checkpoint,
which is what an ablation requires.

## 7. Adversarial robustness (Gap 2) — DONE, 28/28

Source: `Final/table_adversarial_robustness.csv`. Headline: DAIGT V2 robust
across all attacks (worst case BERT/10%-typo: 0.9916 -> 0.7966); HC3 collapses
under typo/homoglyph (worst case BERT/5%-typo: 0.9945 -> 0.4390, near-chance)
but stays comparatively robust to back-translation (worst case DeBERTa:
0.9980 -> 0.9773). The HC3-vs-DAIGT-V2 fragility asymmetry corroborates the
Section 6 artifact-cleaning finding (HC3's stronger, more character-level
surface artifact predicts its greater vulnerability to character-level
attack). Full prose: Results 5.5.

**Root-cause note for reproducibility:** the first attempt at this table
silently ran back-translation on CPU for ~3 hours despite `NLP_CROSS_DEVICE
=cuda` being set, because `perturb_texts()` didn't thread the `device`
parameter through to `backtranslate()` (`run_adversarial_robustness.py`,
fixed 2026-08-23). Verified fixed via `nvidia-smi pmon` showing real
per-process GPU utilization after the patch.

## 8. Known methodological inconsistency — checkpoint config vs. table2's reported "best"

**Discovered 2026-08-23, does not affect internal consistency of Sections
6-7 above, but affects how Section 1 (table2) is described in prose.**
`table2_combined_full.csv` reports, per (dataset, model) cell, the best F1
found anywhere in the 8-config hyperparameter grid (`table1_experiments_full.csv`),
selected independently per cell. The actual saved model **weights** (used
for cross-dataset transfer, adversarial robustness, and both cleaning
ablations) are fixed to the `WINNERS` dict in `run_full_scale.py` — the
configs that won at the smaller, course-matched 6k-row scale, carried over
rather than re-selected against the full-scale grid. For D1 BERT and D2
DeBERTa these happen to coincide with table2's reported best. For **D1
DeBERTa and D2 BERT they do not**:

| | table2's reported "best" | actual checkpoint's config | actual checkpoint's F1 |
|---|---|---|---|
| D1 DeBERTa | 0.9949 (lr3e-5/bs32/wd0.1) | lr3e-5/bs16/wd0.01 | 0.9917 |
| D2 BERT | 0.9945 (lr2e-5/bs32/wd0.1) | lr2e-5/bs16/wd0.1 | 0.9916 |

Decision (2026-08-23): document as a methods note rather than retrain the
2 true-best checkpoints and redo downstream experiments for them. Section
5.1 (in-distribution comparison) should keep citing table2's true grid-max
numbers as "the best config found in the sweep," while Sections 5.3-5.5
(everything using saved weights) should explicitly note they use the
carried-over WINNERS config, which is not always identical to 5.1's cited
number. This is now the methods-note text to add to `04_methodology.md`.

## 9. DAIGT V2 token-length asymmetry under 128-token truncation

Source: `Final/audit/daigt_token_length_audit.py` (output:
`Final/audit/daigt_token_length_audit.json`), 2,000-row seeded sample
(seed=42), `bert-base-uncased` tokenizer.

| | overall | human | AI |
|---|---|---|---|
| median token length | 415 | 448.5 | 398.5 |
| mean token length | 442.67 | 487.73 | 395.77 |
| % exceeding 128 tokens | 99.7% | 100.0% | 99.4% |

Median % of essay kept at 128-token truncation: 30.8%. Human essays run
12.55% longer (median) than AI essays in this sample, so truncation
discards proportionally more human content. Previously an in-session ad-hoc
measurement; now persisted as a proper audit script following the
`daigt_full_audit.py` / `hc3_full_audit.py` convention.

## Files explicitly excluded from this SSOT (superseded/course-track/duplicate)

- `Final/table1_experiments.csv`, `Final/table2_combined.csv` — small-scale (6k), course-track only
- `Final/models/manifest.json` — small-scale manifest; its `matches_recorded: false` for D1_BERT is
  the *documented* seed-instability finding (see row 4), not a data bug
- `Final/_backup_results_1102/` — byte-identical duplicate of `results/`+`probs/`
- `Final/paper_review/LITERATURE_REVIEW.md`/`.pdf` (21 Aug) — superseded by
  `LITERATURE_REVIEW_CANONICAL.md` per user decision 2026-08-22
