# Numbers SSOT — Paper Track (Full-Scale Only)

---

## RETRACTIONS AND CORRECTIONS, 2026-08-23

Three audits plus direct measurement invalidated several numbers previously recorded here.
**Nothing in the sections below marked RETRACTED may be quoted in the paper.**

**R1. The artifact-cleaning ablation is VOID.** `run_artifact_cleaning_full.py:94-105` compares
`raw_f1` (best validation F1 across the *entire grid*) against `cleaned_f1` (one fixed config),
so three of four cells compare different hyperparameters. Checkpoint-matched, the deltas are
HC3/BERT **0.0000**, HC3/DeBERTa **0.0010**, DAIGT/DeBERTa **-0.0026**. It is additionally
confounded because `build_cleaned_data` applies `length_match(unit='words')` to the cleaned arm
only. Section 6's table is retracted in full.

**R2. Every adversarial `drop` is inflated by the same bug.** `table_adversarial_robustness.csv`
takes `in_domain_f1` from the grid maximum rather than the attacked checkpoint. Deployed
checkpoints are D2_BERT **0.9916** (not 0.9945), D2_DeBERTa **0.9972** (not 0.9980), D1_DeBERTa
**0.9917** (not 0.9949). Corrected typo-10 drops: D1 BERT 0.1950, D1 DeBERTa 0.0327, D2 BERT
0.6170, D2 DeBERTa 0.6328.

**R3. The noise band 0.0011 was cherry-picked** — smallest of four measured seed spreads. The
band for the relevant config is **0.0036**, and it is a *range over n=3*, not an SD, whose own
sampling error exceeds every delta previously tested against it. Use paired McNemar on the
existing `.npz` predictions instead.

**R4. "Below chance" is a misreading.** Weighted F1 floors at 0.333 for an all-one-class
predictor on a 50/50 split; guessing is ~0.50. The observed 0.3644 is 1.4% above total
degeneracy, not below chance.

**R5. "DAIGT V2 is at chance on the whitespace cue" is retracted.** That was an instrument
artifact of a single >=1-occurrence threshold applied to a sparse cue. A fitted 47-feature
surface model reaches **0.9214** on DAIGT V2. Both datasets carry substantial surface signal.

**R6. The adversarial-robustness interpretation is withdrawn** (see Section 11).

---

Every number below traces to a file on disk. Do not type a number into prose
without checking it here first, and do not add a row here without citing its
source file. Course-track (6k-sample) numbers are OUT OF SCOPE for this paper
and are not listed — see `REPORT.md` if the course submission needs them.

## 1. Headline 5-model comparison — **SUPERSEDED by Section 13 (used grid maxima)**

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

## 4. Scale-dependent BERT seed instability — **DEMOTED, not a contribution (n=3 estimate moved 31% on rerun)**

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

## 6. Artifact-cleaning ablation (Gap 4) — **RETRACTED, see R1. DO NOT QUOTE.**

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

## 7. Adversarial robustness (Gap 2) — **drops RETRACTED (R2); interpretation WITHDRAWN, see Section 11**

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

## 10. Surface/content separability decomposition (THE PAPER'S LEAD RESULT)

Source: `audit/surface_content_decomposition.py` -> `audit/surface_content_decomposition.json`.
Logistic regression for both arms so they are directly comparable; transformer read from the
deployed checkpoint's `run_info.json`, never from a grid maximum.

| dataset | surface-only (47 orthographic feats, no lexical content) | content-only (BoW, surface stripped) | full transformer (reference) |
|---|---|---|---|
| DAIGT V2 | 0.9214 wF1 = mF1, err 7.86%, FPR 0.0881 | **0.9901, err 0.99%, FPR 0.0080** | 0.9917 DeBERTa / 0.9916 BERT |
| HC3 | **0.9680 wF1 = mF1, err 3.20%, FPR 0.0562** | 0.9674, err 3.26%, FPR 0.0303 | 0.9972 DeBERTa / 0.9916 BERT |

Macro equals weighted in every arm, so no arm is exploiting class skew.

- **HC3: orthography matches content** (3.20% vs 3.26% error). 47 features that never read a word
  do as well as bag-of-words.
- **DAIGT V2: content wins 7.9x on error** (0.99% vs 7.86%).
- On DAIGT V2 the transformer adds nothing over content-only (0.9917 vs 0.9901); on HC3 it adds
  ~0.03 over either arm.

## 11. Adversarial collapse: interpretation WITHDRAWN

Source: `audit/collapse_probe.py` -> `audit/collapse_probe.json`. Percent classified **human** on
inputs carrying no label information, with mean max-softmax:

| model | random chars | token-shuffled | punctuation-only | repeated token | empty |
|---|---|---|---|---|---|
| D2_BERT | 51.0% (0.71) | 98.5% (1.00) | 100% (1.00) | 100% (0.98) | 0% (1.00) |
| D2_DeBERTa | **100% (0.98)** | 100% (1.00) | 100% (0.99) | 0% (0.84) | 0% (0.99) |
| D1_BERT | 0% (0.91) | 94.8% (0.99) | 97.8% (0.71) | 0% (0.83) | 0% (0.82) |
| D1_DeBERTa | 100% (0.99) | 100% (1.00) | 100% (1.00) | 100% (1.00) | 0% (0.94) |

The model that collapsed to 0.3644 under typo attack classifies **random character strings as
human at 0.98 confidence**. All four models do so for token-shuffled and punctuation-only text.
The one-directional flip is therefore not a targeted vulnerability and not a learned cue — it is a
degenerate response to unreadable input, and it is not HC3-specific.

Mechanisms tested and refuted, for the record: majority-prior collapse (training is balanced,
D1 0.4981/0.5019, D2 0.4999/0.5001); cue injection (typo-10 moves machine-text artifact count
0.000 -> 0.005, homoglyph 0.000); "messy implies human" (subword fertility runs the wrong way,
HC3 human 1.1192 vs machine 1.1980).

**Paper position:** report the label-free control as a methodological requirement, note that our
models fail it, and decline to read the attack numbers as robustness evidence. The same caution
applies to prior work reporting comparable collapses without such a control.

## 12. Tokenizer blindness to the whitespace cue

Verified on the deployed checkpoints. BERT (`paper_scale/models/D2_BERT`, WordPiece) yields
**identical token ids for 3/3** artifact pairs — `"the answer is simple ."` and
`"the answer is simple."` both give `['the','answer','is','simple','.']`. DeBERTa
(`D2_DeBERTa`, SentencePiece) yields **0/3 identical**, encoding the difference as `▁.` versus `.`.

BERT cannot represent the cue and still reaches **0.9916** on HC3, so the cue is not necessary for
~0.99 performance. **Do not attribute the BERT/DeBERTa difference to the cue** — the two differ in
architecture, pretraining corpus, and size.

Pipeline-fires control (required, else the bit-identical result is uninterpretable): SHA-256 over
`paper_scale/probs/` shows D2/BERT raw and cleaned **bit-identical**, while D1/BERT raw and
cleaned **differ** — emoji removal is visible to WordPiece. The cleaning pipeline does fire; the
blindness is specific to whitespace.

## 13. Complete model evaluation, all 8 configurations (SUPERSEDES Section 1)

Source: `paper_scale/full_model_evaluation.py` -> `audit/full_model_evaluation.json`
and `audit/full_model_scores.npz`. Transformer rows read from the **deployed
checkpoint's** `run_info.json`, never a grid maximum. Classical rows refit with
`LinearSVC(max_iter=20000)` because the sklearn default fails to converge on HC3.
Section 1's table used grid maxima for the transformers and is superseded by this one.

| Model | Rep. | DAIGT V2 F1 / err% / AUC | HC3 F1 / err% / AUC |
|---|---|---|---|
| Naive Bayes | BoW | 0.9591 / 4.09 / 0.991804 | 0.8713 / 12.87 / 0.930489 |
| Naive Bayes | TF-IDF | 0.9577 / 4.23 / 0.991469 | 0.8670 / 13.28 / 0.946474 |
| Logistic Regression | BoW | 0.9893 / 1.07 / 0.998809 | **0.9551** / 4.49 / 0.986545 |
| Logistic Regression | TF-IDF | 0.9857 / 1.43 / 0.998354 | 0.9365 / 6.35 / 0.983501 |
| SVM | BoW | 0.9870 / 1.30 / 0.998368 | 0.9475 / 5.25 / 0.981816 |
| SVM | TF-IDF | **0.9910** / 0.90 / 0.999192 | 0.9449 / 5.51 / 0.986309 |
| BERT | raw | 0.9916 / 0.84 / 0.999641 | 0.9916 / 0.84 / 0.999632 |
| DeBERTa | raw | **0.9917** / 0.83 / 0.999144 | **0.9972** / 0.28 / 0.999977 |

Macro F1 equals weighted F1 to four decimals in every row. Two observations:
- **DAIGT V2**: best classical (SVM/TF-IDF 0.9910) is 0.0007 below best transformer
  (0.9917), inside the 0.0036 seed range. Transformers add nothing measurable.
- **HC3**: best classical 0.9551 vs 0.9972, an error-rate ratio of 16 to one.
- AUC ordering differs from F1 ordering on DAIGT V2 (BERT 0.999641 > DeBERTa
  0.999144 by AUC, reversed by F1). Report both.

## Files explicitly excluded from this SSOT (superseded/course-track/duplicate)

- `Final/table1_experiments.csv`, `Final/table2_combined.csv` — small-scale (6k), course-track only
- `Final/models/manifest.json` — small-scale manifest; its `matches_recorded: false` for D1_BERT is
  the *documented* seed-instability finding (see row 4), not a data bug
- `Final/_backup_results_1102/` — byte-identical duplicate of `results/`+`probs/`
- `Final/paper_review/LITERATURE_REVIEW.md`/`.pdf` (21 Aug) — superseded by
  `LITERATURE_REVIEW_CANONICAL.md` per user decision 2026-08-22
