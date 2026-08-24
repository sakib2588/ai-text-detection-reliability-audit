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

**R2. Every adversarial `drop` is inflated by the same bug.** `tables/table_adversarial_robustness.csv`
takes `in_domain_f1` from the grid maximum rather than the attacked checkpoint. Deployed
checkpoints are D2_BERT **0.9916** (not 0.9945), D2_DeBERTa **0.9972** (not 0.9980), D1_DeBERTa
**0.9917** (not 0.9949). Corrected typo-10 drops: D1 BERT 0.1950, D1 DeBERTa 0.0327, D2 BERT
0.6170, D2 DeBERTa 0.6328.

**R3. The noise band 0.0011 was cherry-picked** — smallest of four measured seed spreads.
**CORRECTED 2026-08-24: the replacement, 0.0036, was cherry-picked the other way.** It is the
spread of `full_D2_BERT_lr2e-05_bs16_wd0.1`, i.e. HC3/BERT, and the ICCIT paper was applying it
as a noise floor to a DAIGT-V2 DeBERTa-vs-SVM comparison. Never borrow a spread across a dataset
or a model family. All four measured spreads, from `experiments/audit/paper_claim_verification.json`:

| config | n | spread |
|---|---|---|
| `full_D1_BERT_lr3e-05_bs32_wd0.1` | 3 | 0.0011 |
| `full_D1_DeBERTa_lr3e-05_bs16_wd0.01` | 3 | 0.0024 |
| `full_D2_BERT_lr2e-05_bs16_wd0.1` | 3 | 0.0036 |
| `full_D2_DeBERTa_lr3e-05_bs16_wd0.1` | 3 | 0.0005 |

A range over n=3 is not a test statistic in any case. Use paired McNemar and a paired bootstrap
on the existing `.npz` predictions instead. This is now done for every comparison the ICCIT
paper draws.

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

Source: `Final/tables/table2_combined_full.csv`

| Model | D1 (DAIGT) F1 | D2 (HC3) F1 |
|---|---|---|
| Naive Bayes | 0.9591 | 0.8713 |
| Logistic Regression | 0.9893 | 0.9551 |
| SVM | 0.9910 | 0.9449 |
| BERT (best config) | 0.9916 | 0.9945 |
| DeBERTa (best config) | 0.9949 | 0.9980 |
| Ensemble (soft-vote) | 0.9949 | 0.9982 |

Note: review recommends demoting the ensemble to a discussion point, not a
headline result (see `docs/literature/LITERATURE_REVIEW_CANONICAL.md` Section 5.3).

## 2. Full hyperparameter sweep (Table 1, full-corpus, BERT/DeBERTa only)

Source: `Final/tables/table1_experiments_full.csv` — 16 configs (2 models x 2 lr x 2
bs x 2 wd) x 2 datasets. Use for the methodology/sweep-design section; do not
re-list all 16 rows in the main paper, reference the CSV.

## 3. Cross-dataset transfer — DONE, 3 seeds (42/123/456)

Source: `Final/tables/table_cross_dataset_generalization_3seed.csv`. Supersedes the
single-seed table (`tables/table_cross_dataset_generalization.csv`, kept for
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
  Source: `Final/experiments/paper_scale/results/full_D1_BERT_lr3e-05_bs32_wd0.1_s{42,123,456}.json`
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

Source: `Final/experiments/audit/daigt_full_audit.json`, `Final/experiments/audit/hc3_full_audit.json`

| | DAIGT V2 (n=44,868) | HC3 (n=85,449) |
|---|---|---|
| Duplicate rows | 6 (0.01%) | 6,118 (7.16%) |
| Duplicate groups | 6 | 5,986 |
| Cross-label texts | 0 | 0 |
| Artefact rows (e.g. API-error text stored as answer) | n/a | 476 ChatGPT-side, 7 human-side |

**CLOSED 2026-08-24 — the 11.2—11.3% figure is WRONG, the measured value is 5.30%.**
`experiments/audit/verify_paper_claims.py` measures exact-text leakage on both split variants directly from
`experiments/paper_scale/work/`. Group-aware split (the one every reported number uses): **0 of 10,732 HC3
test rows, 0.00%**, and DAIGT V2 **0 of 6,998**. Naive stratified split of the same balanced
sample: HC3 **570 of 10,762, 5.30%**, DAIGT V2 **0**. The wrong figure has been removed from
`build_full_splits.py`'s docstring, where it originated. Original note kept below for the record.

**(superseded)** **OPEN — do not cite yet:** the "11.2–11.3% test-set leakage under naive
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

Source: `Final/tables/table_artifact_cleaning_zeroshot.csv` (test-time-only cleaning,
existing checkpoints, no retraining) and `Final/tables/table_artifact_cleaning_full.csv`
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

Source: `Final/tables/table_adversarial_robustness.csv`. Headline: DAIGT V2 robust
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
`tables/table2_combined_full.csv` reports, per (dataset, model) cell, the best F1
found anywhere in the 8-config hyperparameter grid (`tables/table1_experiments_full.csv`),
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

Source: `Final/experiments/audit/daigt_token_length_audit.py` (output:
`Final/experiments/audit/daigt_token_length_audit.json`), 2,000-row seeded sample
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

Source: `experiments/audit/surface_content_decomposition.py` -> `experiments/audit/surface_content_decomposition.json`.
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

**PREPROCESSING CORRECTION 2026-08-24.** The content arm applies **no stopword removal and no
lemmatisation**. Those steps live in `experiments/paper_scale/classical_full.py:preprocess`, which feeds the
Table 1 classical models, not in `surface_content_decomposition.py`, whose content arm is
`content_normalise()` followed by a bare `CountVectorizer()`. The arm is therefore an *unfiltered*
bag-of-words, a stronger content model than a filtered one. Any prose describing it as reduced by
stopword removal is wrong. The script now records this in its own JSON output.

**LENGTH CONTROL 2026-08-24.** The two primary arms are not disjoint. The surface arm reads
character, word and sentence counts, and raw CountVectorizer rows sum to document length. Closing
the channel on both sides (`length_controlled` block in the same JSON):

| arm | DAIGT V2 err | HC3 err |
|---|---|---|
| length-only (chars, words, sentences) | 24.59% | 15.23% |
| surface-only, 5 document-size feats dropped | 9.93% | 3.37% |
| content-only, rows L1-normalised then rescaled | **0.94%** | 6.28% |
| content-only, rows L1-normalised, NOT rescaled | 9.69% | 14.25% |

Both findings survive and strengthen. DAIGT V2's content advantage widens from 7.9x to 10.6x, and
HC3's parity becomes a 2.91-point *advantage for orthography*, because removing length costs the
content arm 3.02 points against the surface arm's 0.18.

**Do not quote the un-rescaled L1 row.** L1 rows sum to 1, shrinking every feature by roughly the
mean document length, so at fixed `C` the penalty is far heavier and the arm collapses for reasons
of scale, not length. It converges in 16-24 iterations against 121-192 for the raw-count arm. Same
class of instrument artifact as R5.

## 11. Adversarial collapse: interpretation WITHDRAWN

Source: `experiments/audit/collapse_probe.py` -> `experiments/audit/collapse_probe.json`. Percent classified **human** on
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

Verified on the deployed checkpoints. BERT (`experiments/paper_scale/models/D2_BERT`, WordPiece) yields
**identical token ids for 3/3** artifact pairs — `"the answer is simple ."` and
`"the answer is simple."` both give `['the','answer','is','simple','.']`. DeBERTa
(`D2_DeBERTa`, SentencePiece) yields **0/3 identical**, encoding the difference as `▁.` versus `.`.

BERT cannot represent the cue and still reaches **0.9916** on HC3, so the cue is not necessary for
~0.99 performance. **Do not attribute the BERT/DeBERTa difference to the cue** — the two differ in
architecture, pretraining corpus, and size.

Pipeline-fires control (required, else the bit-identical result is uninterpretable): SHA-256 over
`experiments/paper_scale/probs/` shows D2/BERT raw and cleaned **bit-identical**, while D1/BERT raw and
cleaned **differ** — emoji removal is visible to WordPiece. The cleaning pipeline does fire; the
blindness is specific to whitespace.

## 13. Complete model evaluation, all 8 configurations (SUPERSEDES Section 1)

Source: `experiments/paper_scale/full_model_evaluation.py` -> `experiments/audit/full_model_evaluation.json`
and `experiments/audit/full_model_scores.npz`. Transformer rows read from the **deployed
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

## 14. Split geometry and paired statistics (added 2026-08-24)

Source: `experiments/audit/verify_paper_claims.py` -> `experiments/audit/paper_claim_verification.json`. Re-runnable on
CPU in under a minute; every figure below is read from artifacts already on disk.

**The split is 72/8/20, not 80/10/10.** `experiments/paper_scale/build_full_splits.py:group_split` takes 20%
for test, then 10% of the remaining 80% for validation. Both docstrings said 80/10/10 and the
ICCIT paper inherited it. Corrected in the paper and in both scripts on 2026-08-24.

| dataset | balanced n | train | val | test |
|---|---|---|---|---|
| DAIGT V2 | 34,994 | 25,196 (72%) | 2,800 (8%) | 6,998 (20%) |
| HC3 | 53,806 | 38,785 (72%) | 4,289 (8%) | 10,732 (20%) |

**Whitespace cue, now sourced.** The 10.745 / 0.013 pair quoted in prose had no source file. It
is the HC3 **test split**, mean space-before-punctuation count per document: human 10.7452,
machine 0.0134. Balanced-corpus values are 10.7345 and 0.0092, and the cue is present in 88.74%
of HC3 human documents against 0.28% of machine ones. DAIGT V2 carries the same cue weakly,
0.5475 against 0.0044 per document, 17.08% against 0.22% of documents.

**Table 1 paired tests.** All sixteen cells carry a bootstrap 95%-CI on error rate (10,000
resamples). Derived predictions reproduce every recorded error rate to within 5e-4, which is the
self-check that makes the rest of this block trustworthy.

| comparison | discordant (b:c) | McNemar exact p | error diff | 95% CI |
|---|---|---|---|---|
| DAIGT V2, SVM/TF-IDF vs DeBERTa | 47:52 | 0.688 | +0.07 pp | [-0.21, +0.34] |
| DAIGT V2, BERT vs DeBERTa | 37:38 | 1.000 | +0.01 pp | [-0.23, +0.26] |
| HC3, LogReg/BoW vs DeBERTa | 16:468 | <1e-6 | +4.21 pp | [+3.82, +4.61] |
| HC3, BERT vs DeBERTa | 15:75 | <1e-6 | +0.56 pp | [+0.39, +0.74] |

So "transformers add nothing measurable on DAIGT V2" is now a tested claim, and the AUC-vs-F1
rank reversal on DAIGT V2 is noise, not an ordering.

**Decomposition paired tests** (predictions from `experiments/audit/surface_content_predictions.npz`):

| comparison | discordant (b:c) | McNemar exact p | error diff | 95% CI |
|---|---|---|---|---|
| HC3, surface vs content | 316:309 | 0.810 | -0.07 pp | [-0.52, +0.40] |
| DAIGT V2, surface vs content | 44:525 | <1e-6 | +6.87 pp | [+6.23, +7.52] |
| HC3, length-free surface vs length-free content | 610:298 | <1e-6 | -2.91 pp | [-3.44, -2.37] |
| DAIGT V2, length-free surface vs length-free content | 43:672 | <1e-6 | +8.99 pp | [+8.26, +9.70] |
| DAIGT V2, content vs length-free content | 13:16 | 0.711 | +0.04 pp | [-0.11, +0.19] |
| HC3, content vs length-free content | 468:144 | <1e-6 | -3.02 pp | [-3.47, -2.57] |

HC3 parity is now a tested null, not an eyeballed one.

## 15. Truncation-matched comparison and split-level variance (added 2026-08-24)

### 15.1 The DAIGT V2 "classical matches transformer" result was an input-budget artefact

Source: `experiments/audit/truncation_matched_comparison.py` -> `experiments/audit/truncation_matched_comparison.json`.

The classical models read the WHOLE document. The transformers read 128 tokens. Refitting the
full six-cell classical grid on the exact character span each deployed checkpoint's own tokenizer
kept (offset mapping, no decode round-trip) removes the asymmetry:

| dataset | window | chars kept | best classical, full text | best classical, 128-tok window | transformer | McNemar p |
|---|---|---|---|---|---|---|
| DAIGT V2 | BERT | 0.335 | 0.90% | **2.10%** | 0.84% | <1e-6 |
| DAIGT V2 | DeBERTa | 0.341 | 0.90% | **2.09%** | 0.83% | <1e-6 |
| HC3 | BERT | 0.746 | 4.49% | 5.66% | 0.84% | <1e-6 |
| HC3 | DeBERTa | 0.756 | 4.49% | 5.72% | 0.28% | <1e-6 |

DAIGT V2 paired result at matched input: error difference +1.257 pp, 95% CI [+0.900, +1.629].
The DeBERTa window returns +1.257 as well, so the conclusion does not depend on which tokenizer
defines the window.

**Do not write "transformers add nothing on DAIGT V2" again.** Held to one text budget the
transformers lead on BOTH corpora. The defensible claim is narrower: a bag-of-words model reading
a whole essay matches a transformer reading its opening. That is a statement about information
access, not architecture, and it is arguably a more interesting one.

Caveat on record: loading the DeBERTa tokenizer under transformers 4.57.6 emits a Mistral-regex
warning. The two windows agree to 0.6 percentage points of characters kept and to 0.01 pp of
error, so it did not affect the conclusion, but re-check if that tokenizer is reused.

### 15.2 Split-level variance

Source: `experiments/audit/multisplit_decomposition.py` -> `experiments/audit/multisplit_decomposition.json`. Balanced
sample held fixed, only the partition seed varies over 42/123/456/789/1337. Seed 42 reproduces
the single-split numbers exactly, which is the harness self-check.

| arm | DAIGT V2 mean err (range) | HC3 mean err (range) |
|---|---|---|
| surface-only | 7.15% (6.74-7.86) | 3.16% (3.07-3.33) |
| content-only | 0.82% (0.66-0.99) | 3.24% (3.06-3.41) |
| surface-only, no length | 9.76% (9.47-9.93) | 3.26% (3.14-3.37) |
| content-only, length-normalised | 0.84% (0.77-0.94) | 6.02% (5.82-6.28) |
| length-only | 24.31% (23.56-24.85) | 15.32% (15.23-15.49) |

Paired tests per split:

- **DAIGT V2 surface vs content**: +6.00 to +6.87 pp, p<1e-6 on all five, sign consistent.
- **HC3 surface vs content**: p = 0.81, 0.27, 0.34, 0.23, 0.78. Significant on NONE. Differences
  -0.30 to +0.27 pp, **sign changes across splits**. That sign instability is the strong form of
  the null, since an underpowered real difference keeps its sign.
- **HC3 length-free surface vs content**: -2.67 to -2.91 pp, p<1e-6 on all five, sign consistent.

Split variance is small (0.17-1.29 pp), so the single-split figures were not misleading, but the
multi-split evidence is what the parity claim should now rest on.

## Files explicitly excluded from this SSOT (superseded/course-track/duplicate)

- `Final/tables/table1_experiments.csv`, `Final/tables/table2_combined.csv` — small-scale (6k), course-track only
- `Final/models/manifest.json` — small-scale manifest; its `matches_recorded: false` for D1_BERT is
  the *documented* seed-instability finding (see row 4), not a data bug
- `Final/archive/backup_results_1102/` — byte-identical duplicate of `results/`+`probs/`
- `Final/docs/literature/LITERATURE_REVIEW.md`/`.pdf` (21 Aug) — superseded by
  `LITERATURE_REVIEW_CANONICAL.md` per user decision 2026-08-22
