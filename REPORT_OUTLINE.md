# Final Term Report — Outline

Follows `NLP Report Template with Cover Page - Midterm Summer 25-26.docx`, whose required order is:
Cover Page, Project Contributions, Dataset Description, Project Implementation Detail, Project Code.

Numbers marked `{{...}}` are filled from files on disk. Nothing is typed by hand.
Sources: `table1_experiments.csv`, `table2_combined.csv`, `run_summary.json`, `work/sweep_results.csv`,
`work/seed_robustness.csv`, `work/length_stats.csv`, `work/ensemble_detail.json`.

---

## 1. Project Contributions

Table of four members with ID and the specific parts each contributed.
Members (from the midterm report): NILOY PAUL 23-51773-2, ARNOB SARKER SUPTA 23-52080-2,
NAZMUS SAKIB 23-52638-2, AFNAN UR RAYAN 23-51992-2.

**Left for the group to fill in.** Do not invent a division of labour.

## 2. Dataset Description

Reusable from the midterm report, with two corrections that must be made rather than copied:

- HC3's identifier field is `index`, not `id`.
- The "147 MB" figure quoted for HC3 is the HuggingFace `size_in_bytes` value, which includes the
  download; the file on disk is 70 MB.

Add for the final term: the balanced sample (3,000 per class), the 4,320 / 480 / 1,200 split, and the
token-length statistics from `length_stats.csv`, including that {{pct_truncated}} of documents exceed
the 128-token limit.

Licences: DAIGT MIT, HC3 CC-BY-SA-4.0. HC3 paper: arXiv 2301.07597.

## 3. Project Implementation Detail

Per the template, each task is written as Title, Description, Code, Sample output, Code description.

- **Task 1 — Split reconstruction and midterm verification.** The point that carries the whole
  report: the transformers are scored on the identical 1,200-row test set the classical models were
  scored on, and this is proven by re-deriving all twenty-four midterm numbers from scratch.
- **Task 2 — Tokenisation and sequence-length diagnostics.** Why 128 tokens is a real limitation on
  DAIGT and what fraction of each document survives truncation.
- **Task 3 — Fine-tuning harness.** Hyperparameter grid, early stopping on validation, and the three
  checkpointing mechanisms (run-level skip, within-run resume, atomic writes).
- **Task 4 — BERT sweep.** Eight configurations per dataset.
- **Task 5 — DeBERTa sweep.** Eight configurations per dataset. Note the batch-32 runs use a
  per-device batch of 16 with two gradient accumulation steps, which preserves the specified
  effective batch size within the 8 GB VRAM budget.
- **Task 6 — Ensemble.** Validation-tuned soft vote; report the chosen weight {{weight_bert}}.
- **Task 7 — Seed robustness.** Three seeds on the winning configuration; widest spread
  {{max_spread}} F1, which bounds what differences in Table 1 can be read as real.

## 4. Results

**Table 1** — the specification's experiment table, 17 rows, from `table1_experiments.csv`.

**Table 2** — the combined table, 6 rows, from `table2_combined.csv`. State beneath it that the
classical rows use each model's stronger representation (Naive Bayes and Logistic Regression at
Bag-of-Words, Support Vector Machine at TF-IDF) and that the transformer rows use the configuration
with the best validation F1.

Figures: `token_length_distribution.png`, `validation_f1_heatmap.png`, `ensemble_weight_sweep.png`,
`ensemble_confusion.png`.

## 5. Discussion — points that must be made honestly

- **Different inputs per model family.** The classical models consume the midterm cleaned text;
  the transformers consume raw text. This is deliberate and justified, and a reader comparing rows
  in Table 2 is entitled to know it.
- **Dataset 1 has almost no headroom.** LinearSVC with TF-IDF already reaches 0.9875 accuracy. If the
  transformers do not beat that, say so and attribute it to 128-token truncation of essays that
  average roughly 450 tokens. Do not tune toward the test set to close the gap.
- **Seed noise.** Differences smaller than {{max_spread}} F1 between configurations are not
  interpretable at one seed.
- **Single run per configuration.** The sweep is one seed; only the winning configuration was
  repeated. This is a compute limitation, stated rather than hidden.

## 6. Project Code

Entire notebook, per the template.

---

## Submission checklist

- [ ] ZIP named `nlp_final_project_group_02.zip`
- [ ] `nlp_final_project_group_02.ipynb` with all outputs saved
- [ ] `nlp_final_project_group_02.pdf` report
- [ ] `sample_dataset1_daigt.csv` and `sample_dataset2_hc3.jsonl` (100 rows each, already exist)
- [ ] Full `daigt.csv` and `hc3.jsonl` kept beside the notebook for the VIVA re-run
- [ ] Banned-glyph audit returns zero
- [ ] Confirm with the instructor that torch and transformers are permitted, since the midterm
      restricted libraries to nltk, spaCy, matplotlib, numpy, pandas, and scikit-learn
