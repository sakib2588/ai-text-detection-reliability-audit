---
title: "How Much of AI-Text Detection Accuracy Is Real? Cross-Dataset Transfer, Artifact Leakage, and Adversarial Robustness on DAIGT V2 and HC3"
author: "Group 02, Section B -- Natural Language Processing, AIUB"
date: "Draft: 2026-08-23"
geometry: margin=1in
fontsize: 11pt
---

*Draft status: markdown pre-LaTeX draft. Citation markers below are shown as
`[key]` (verified BibTeX keys, see `refs.bib`), not yet resolved to numbered
in-text citations -- that conversion happens at LaTeX/ICCIT-template port
time, out of scope for this render.*

# Abstract

Detectors of AI-generated text routinely report accuracy at or above 99% on
benchmarks such as DAIGT V2 and HC3, but this in-distribution number says
little about real-world reliability. We decompose reported accuracy into
three components -- cross-dataset generalization, dependence on surface
artifacts, and robustness to cheap adversarial attack -- using a five-model
pipeline (three classical baselines and fine-tuned BERT and DeBERTa) at
full corpus scale. A contamination audit finds HC3 carries substantial
internal duplication (7.16%) essentially absent from DAIGT V2 (0.01%).
Strict one-way cross-dataset transfer, evaluated across three seeds, shows
substantial degradation from both datasets in both directions. An
artifact-cleaning ablation, run both zero-shot and via full retraining,
finds a real negative accuracy delta for HC3 but not DAIGT V2, consistent
with HC3's stronger surface artifact (a whitespace-before-punctuation cue
in 88.7% of human answers versus 0.28% of ChatGPT answers). An adversarial
evaluation across typo injection, homoglyph substitution, and
back-translation then finds HC3 accuracy collapses to near-chance under
typo attack (as low as 0.3644 F1, from an in-domain 0.9980), while DAIGT V2
stays comparatively robust -- three independent experiments converging on
the same conclusion for HC3. We additionally find BERT's fine-tuning
instability is itself scale-dependent, shrinking roughly 24.3-fold from
6,000 to approximately 35,000 training rows, a refinement not stated by the
instability literature this builds on. These results support a
dataset-dependent verdict: HC3's headline accuracy substantially overstates
real-world reliability, while DAIGT V2's does so only modestly -- a
distinction invisible from in-distribution accuracy alone.
# Introduction

The proliferation of capable large language models has made distinguishing
human-written from machine-generated text a practical concern across
education, journalism, and content moderation, and a substantial body of
work has responded by fine-tuning transformer classifiers on paired
human/AI corpora. Two of the most widely used benchmarks for this task,
DAIGT V2 (Train Dataset) and HC3 (Human ChatGPT Comparison Corpus), have
each been the subject of dozens of published detection studies, and the
reported results are, almost without exception, striking: in-distribution
F1 scores at or above 97%, with several studies -- including the original
HC3 paper itself -- exceeding 99% (Guo et al., 2023). Read at face value,
these numbers suggest AI-generated text detection is close to solved on the
domains these datasets represent.

This paper starts from the observation that the apparent solvedness is an
artifact of how these benchmarks are typically evaluated, not evidence that
detectors have learned to recognize AI-generated language in a way that
generalizes. Three independent lines of prior work, surveyed in the related
work section, each undermine part of the 99% headline number: cross-dataset
and cross-domain evaluations show sharp collapses when a detector trained on
one distribution is tested on another; documented surface artifacts specific
to HC3 (a whitespace-before-punctuation cue) and DAIGT V2 (typo- and
Unicode-based leakage from the original Kaggle competition) allow detectors
to score well without learning anything about AI-generated language per se;
and adversarial perturbation studies show that detectors scoring above 99%
in-domain can fall to near-chance accuracy under simple typo, homoglyph, or
style-mimicry attacks. No prior study, to our knowledge, combines all three
lines of evidence -- strict one-way cross-dataset transfer, artifact-cleaning
ablation, and adversarial robustness -- on exactly these two benchmarks
within a single, directly comparable experimental design.

This project began as a course assignment comparing five classifiers --
Naive Bayes, Logistic Regression, and Support Vector Machine baselines
against fine-tuned BERT and DeBERTa transformers -- on DAIGT V2 and HC3
independently, and, as a course deliverable, appropriately reported only
in-distribution results. Two properties of that original work turn out to
be directly reusable for the research question this paper now addresses.
First, its methodology includes a validated, reproducible split protocol, a
seed-robustness study establishing that BERT's fine-tuning outcome varies by
up to 0.0267 F1 across random seeds at a smaller data scale, and a targeted
contamination audit that independently rediscovered a genuine HC3 data
defect (test rows containing collection artifacts such as stored API error
messages). Second, extending this infrastructure to the full, unbalanced
corpora at scale surfaces an additional, apparently undocumented finding in
its own right: BERT's seed-to-seed instability shrinks by roughly 24-fold
when moving from a 6,000-row course-matched sample to the full ~35,000-row
corpus (0.0267 to 0.0011 F1 spread on DAIGT V2, holding the winning
hyperparameter configuration fixed), a scale-dependence not stated by the
foundational fine-tuning-instability literature this project builds on
(Dodge et al., 2020; Mosbach et al., 2021).

This paper's contribution is threefold. First, we report a full-corpus
contamination and duplication audit of both DAIGT V2 and HC3, finding that
HC3 carries substantial internal duplication (7.16% of 85,449 rows) and
several hundred collection-artifact rows, in sharp contrast to DAIGT V2's
near-total absence of the same problems (0.01% duplication) -- a direct,
dataset-versus-dataset comparison that does not appear in prior work on
either corpus individually. Second, we establish that BERT's well-documented
fine-tuning instability is itself scale-dependent, a refinement of existing
instability results rather than a restatement of them. Third, combining
cross-dataset transfer, an artifact-cleaning ablation, and an adversarial
robustness evaluation -- the latter two added to this paper's scope
following a full literature-gap analysis -- we quantify how much of the
widely reported 99% detection accuracy on these two benchmarks survives
contact with a different data distribution, a cleaned evaluation set, and a
cheap adversarial attack, addressing directly the question the closest
prior work (Alikhanov et al., 2026) raises but does not fully answer.
Results are dataset-dependent: HC3's headline accuracy is shown to
substantially overstate real-world detection reliability, while DAIGT V2's
does so only modestly (Results, Discussion).
# Related Work

*Condensed from the 25-paper survey in
`Final/paper_review/LITERATURE_REVIEW_CANONICAL.md`; full reference list and
per-paper detail there. Citation keys below are real, verified BibTeX keys
(`Final/paper_draft/refs.bib`, 31 entries, built via arXiv/CrossRef metadata
lookup 2026-08-23) in `[key]` markdown form, to be converted to `\cite{key}`
at LaTeX-port time.*

In-distribution detection of AI-generated text on both DAIGT V2 and HC3 is,
by the standard measured of the field, a solved problem. Guo et al.
(2023) [Guo2023close], who introduced HC3, report a RoBERTa detector
reaching 99.82 F1 on English full-text classification; top Kaggle solutions
on DAIGT V2, such as Biswas et al.'s (2024) [Biswas2024kaggle] first-place
entry combining a DeBERTa-v3-large ranking loss with contrastive retrieval
and student-style adversarial training, reach approximately 0.99 AUC.
Across the full survey of 25 papers using either dataset, essentially every
study that evaluates purely in-distribution -- training and testing on the
same corpus with a random split -- reports accuracy or F1 in the 97-100%
range (Guo et al., 2023 [Guo2023close]; Su et al., 2023 [Su2023plus]; Zhou,
2024 [Zhou2024exploiting]; Ansary, 2026 [Ansary2026multirepresentation];
Lamsiyah et al., 2025 [Lamsiyah2025daigt]; among others). This saturation
means that another in-distribution accuracy number, on either dataset,
does not by itself constitute a novel contribution.

A separate and consistent thread in the same literature shows that this
apparent solvedness is fragile. Antoun et al. (2023) [Antoun2023towards]
attack HC3-trained RoBERTa and ELECTRA detectors with misspellings,
homoglyph substitution, and a hand-crafted adversarial set of human answers
written in ChatGPT's characteristic explanatory style; a detector scoring
99.88 F1 in-domain falls to 33.57% raw accuracy on the hand-written
adversarial set and to 44.81% under a combined Bing-generation-plus-misspelling
attack. Park et al. (2024) [Park2024investigating] demonstrate, separately,
that HC3-trained detectors rely on prompt-specific collection shortcuts
rather than generator-agnostic signal, and construct adversarial
instructions that exploit this directly. Su et al. (2023) [Su2023plus]
show that reframing the detection task from question-answering to
semantic-invariant tasks -- summarization, translation, paraphrasing --
drops accuracy from 99.82% to roughly 91.7% purely from the task-format
change. At a mechanistic level, Borile and Abrate (2025)
[Borile2025generalize] find that removing approximately twenty
feed-forward neurons -- about 0.05% of a BERT-based detector's parameters --
improves out-of-distribution accuracy by up to 6.9 points, direct evidence
that a measurable share of in-distribution performance is carried by a
small number of dataset-specific, non-generalizing features rather than
distributed semantic understanding.

A third thread documents that HC3 and DAIGT V2 specifically contain surface
artifacts that a detector can exploit without learning anything about
AI-generated language per se. Tian et al. (2024, ICLR Spotlight)
[Tian2023multiscale] discover and document that HC3-English human answers
systematically contain an extra space before punctuation that ChatGPT
answers do not, and show that correcting for this changes measured
performance substantially on short text (58.60 to 85.31 F1 under their
proposed Positive-Unlabeled method, which is designed in part to address
exactly this artifact). Baidya et al. (2026) [Baidya2026detecting]
independently document a length confound -- human and ChatGPT answers in
HC3 differ systematically in length, allowing a detector to score well by
approximating a word count -- and deliberately length-match their
benchmark subset to control for it. Ardeshirifar (2025)
[Ardeshirifar2025comparing] standardizes punctuation and contractions for
the same underlying reason. Taken together with the DAIGT V2-side
documentation of typo- and Unicode-based label leakage from the original
Kaggle competition, at least three independent research groups have now
separately identified that a nontrivial share of reported accuracy on
these two specific benchmarks reflects artifact exploitation rather than
semantic detection.

The work closest to the present study is Alikhanov et al. (2026)
[Alikhanov2026generated], who merge
HC3 (~74k samples) and DAIGT V2 (44.8k samples) into a single 124,195-sample,
20-topic corpus and evaluate under a topic-based split -- assigning entire
topics wholly to train, validation, or test to prevent topic-vocabulary
memorization -- reporting substantially lower accuracy (82.87% for TF-IDF
plus Logistic Regression, up to 88.86% for a BiLSTM) than the saturated
in-distribution numbers reported elsewhere. Their design answers a
different question from the one this paper addresses, however: merging and
splitting by topic asks whether a single model can handle both datasets at
once, whereas a strict one-way cross-dataset transfer -- train exclusively
on one dataset, evaluate exclusively on the other, in both directions --
asks whether detection knowledge acquired on one corpus carries to a
genuinely disjoint distribution. To our knowledge, no prior work reports
this strict transfer matrix between exactly these two datasets, nor
combines it with the artifact-cleaning ablation and adversarial robustness
evaluation this paper undertakes (see Methodology). Positioning the present
study relative to this closest prior work, and to the artifact- and
robustness-focused threads above, is the basis for the paper's central
claim: that reported in-distribution accuracy on DAIGT V2 and HC3 substantially
overstates real-world detection ability, and that the gap is attributable,
in measurable part, to non-transferring dataset-specific signal, surface
artifacts, and adversarial fragility rather than to genuine language
understanding.
# Datasets and Contamination Audit (Contribution 1)

*Numbers trace to `Final/paper_draft/NUMBERS_SSOT.md` rows 5-6.*

Both benchmarks used in this study carry documented limitations in the prior
literature, and our own full-corpus audit both confirms and quantifies these
for the specific dataset versions used here. Running a duplicate-detection
pass across the complete, unbalanced DAIGT V2 corpus (44,868 rows) finds
essentially no internal duplication: six duplicate rows across six duplicate
groups, a rate of 0.01%, and zero cases where identical text appears under
conflicting human/AI labels. This is consistent with DAIGT V2's origin as a
single curated Kaggle competition dataset rather than a corpus assembled by
merging multiple independently collected sources.

HC3 presents a substantially different picture. Across its full 85,449-row
corpus, the same duplicate-detection pass finds 6,118 duplicate rows across
5,986 duplicate groups, a duplication rate of 7.16%. As with DAIGT V2, no
cross-label duplicates were found, so the duplication does not by itself
create mislabeled training signal; however, near-duplicate rows crossing
between the train and test partitions of a naive random split constitute a
direct form of test-set leakage, since a model can memorize a training-set
row and simply recall it at test time rather than generalizing. The audit
additionally surfaces a separate data-quality issue specific to HC3: 476
rows nominally labeled as ChatGPT-authored answers, and 7 nominally
human-authored, are collection artifacts -- text such as API failure
messages ("Too many requests in 1 hour") stored under the answer field
rather than genuine model or human output. These rows are trivially
separable from real text and can artificially inflate reported detection
accuracy if left in the evaluation set uncleaned.

At the smaller, course-matched scale (1,200-row HC3 test partition, a subset
of the full corpus), a targeted leakage check identifies eight test rows
(0.67% of that test partition) whose text also appears in the training
partition -- independently corroborating the full-corpus duplication finding
above at a scale small enough to manually inspect, and confirming that
several of the leaked rows are exactly the API-failure-message artifacts
described above. Rescoring every model on the leakage-excluded subset of
this smaller test set shows the largest resulting F1 change is 0.0010 with no
change in model ranking, indicating that -- at this particular scale and
split -- the leakage's effect on the reported metric is smaller than
ordinary seed-to-seed training noise, even though the leakage itself is
real and worth reporting on scientific-validity grounds independent of its
measured effect size.

**Note on scope:** a full-corpus-scale equivalent of this leakage-rate
calculation (i.e., what fraction of a full 80/20 random split of all 85,449
HC3 rows would leak, as opposed to the 1,200-row course-scale subset above)
has not yet been computed or located on disk as of this draft. Prior session
notes referenced an "11.2-11.3%" full-corpus leakage figure; this number
could not be traced to any source file during the 2026-08-22 verification
pass and must not be used until it is either recomputed or its source file
is located.

This audit motivates two complementary observations that anchor the paper's
framing. First, HC3's real, measured contamination (7.16% duplication,
non-trivial artifact rows) sits in sharp contrast to DAIGT V2's near-total
absence of the same problem, despite both being widely used as
if-equivalent benchmarks in the AI-text-detection literature -- a
comparative audit of this kind does not, to our knowledge, appear in prior
published work on either dataset individually. Second, the dissociation
between leakage rate and its measured effect on accuracy, observed at the
scale we can directly inspect, is itself worth stating precisely rather than
assuming: contamination can be real and worth correcting for scientific
validity while nonetheless not being the dominant driver of a benchmark's
reported headline number. How much of the headline number *is* explained by
surface artifacts is addressed directly by the artifact-cleaning ablation
(Results 5.4).

## A third, previously undocumented artifact: length asymmetry under truncation

A separate audit, run directly against this project's own tokenizer and
data rather than drawn from the literature, surfaces a further DAIGT V2
issue specific to how this project (and the course specification it
inherits from) processes text. Measuring token length with the
`bert-base-uncased` tokenizer over a 2,000-row sample of the full DAIGT V2
corpus, 99.7% of essays exceed the 128-token input limit used throughout
this project's transformer training (median length 415 tokens, mean 443,
maximum 1,826); at truncation, a median essay retains only 30.8% of its
tokens, so every DAIGT V2 classification decision in this paper is made
from roughly the opening third of the essay, not the whole document. This
by itself replicates a limitation already known in principle from the
literature (Methodology; Limitations). What has not previously been
measured for this dataset is whether truncation affects the two classes
equally: it does not. Human-authored essays run measurably longer than
AI-authored essays in the same sample (median 448.5 tokens for human versus
398.5 for AI, roughly 12.5% longer; every human essay in the sample
exceeded 128 tokens, versus 99.4% of AI essays). Truncation to 128 tokens
therefore discards proportionally more content from human essays than from
AI essays -- a class-correlated information loss with the same structural
shape as the HC3 length confound Baidya et al. document and control for by
length-matching, except undocumented for DAIGT V2 specifically and, unlike
the HC3 case, not yet corrected for in this project's 128-token results
(see Limitations for the scope decision on this point).
# Methodology

*Numbers in this section trace to `Final/paper_draft/NUMBERS_SSOT.md`. Working
title: "How Much of AI-Text Detection Accuracy Is Real? Cross-Dataset
Transfer, Artifact Leakage, and Adversarial Robustness on DAIGT V2 and HC3"
(per `paper_review/LITERATURE_REVIEW_CANONICAL.md`, Section 3.3).*

## Datasets

We evaluate on two publicly available AI-generated text detection benchmarks
that differ in domain, format, and generator composition. The DAIGT V2 Train
Dataset, released by the Kaggle user "thedrcat" as part of the LLM -- Detect
AI Generated Text competition, contains 44,868 argumentative student essays,
of which 27,371 are human-written (drawn from the PERSUADE 2.0 corpus) and
17,497 are machine-generated by a mixture of models spanning GPT-3.5,
Llama-2, Mistral-7B, Falcon-180B, Claude v1, PaLM, and Cohere command,
yielding a 61%/39% human/AI class split. The Human ChatGPT Comparison Corpus
(HC3), introduced by Guo et al., contains question-answer pairs drawn from
five English-language sources (reddit\_eli5, open\_qa, wiki\_csai, medicine,
and finance); its 85,449 rows contrast human answers against responses from a
single generator, GPT-3.5-Turbo-0301, collected roughly ten days after
ChatGPT's public launch. Both datasets were balanced for the full-scale
training runs reported here: DAIGT V2 to 34,994 rows and HC3 to a matching
scale, following the same class-balancing protocol used at the smaller,
course-matched scale so that the two scales remain directly comparable.

## Models

We compare five classifiers spanning the classical-to-transformer spectrum.
Three classical baselines -- Naive Bayes, Logistic Regression, and a linear
Support Vector Machine -- are trained over both bag-of-words and TF-IDF
feature representations; we report the better-performing representation per
model (TF-IDF for the SVM, bag-of-words for Naive Bayes and Logistic
Regression, consistent with the midterm-stage reproduction). Two transformer
encoders, `bert-base-uncased` and `microsoft/deberta-v3-base`, are
fine-tuned for binary sequence classification over a grid of learning rate
(2e-5, 3e-5), batch size (16, 32), and weight decay (0.01, 0.1), with the
best configuration per model/dataset pair selected by validation F1 (full
sweep: `Final/table1_experiments_full.csv`, 16 configurations x 2 datasets).

## Training protocol

All full-scale transformer runs use a fixed random seed (42) for the primary
sweep, with a subset of winning configurations re-run at seeds 123 and 456 to
establish seed-to-seed variance (Section on seed instability, below). Models
are trained with mixed precision (bfloat16, chosen because DeBERTa overflows
under float16), atomic per-epoch checkpointing, and epoch selection by
validation F1. Training used a single RTX 3060 Ti / RTX 4060-class GPU
(hardware statement to be confirmed against the actual machine used for the
final full-scale runs before submission, per the canonical review's Section
5.5 caution).

**Open methodological item, not yet resolved:** the current full-scale sweep
still truncates to 128 tokens, inherited from the course-project
specification. The canonical literature review (Section 3.4) flags this as
acceptable for a course deliverable but a fatal flaw for a paper submission,
since 99.6% of DAIGT V2 essays exceed 128 tokens and the median essay
retains only 31.3% of its tokens under this truncation. Extending to 256
tokens (512 for DAIGT) is required before the sweep numbers in
`table1_experiments_full.csv` / `table2_combined_full.csv` can be reported as
final paper results, not just as a robustness pre-check.

## Cross-dataset transfer protocol

To measure whether detection knowledge acquired on one dataset transfers to
the other, we train each transformer on one dataset's training split and
evaluate on the other dataset's held-out test split, in both directions
(train HC3 / test DAIGT V2, and train DAIGT V2 / test HC3), alongside the
matched in-domain baseline. This is a strict one-way transfer design,
distinct from the topic-split merge-and-train approach of the closest prior
work (Alikhanov et al., cited in the related work section): transferring
asks whether knowledge of one dataset generalizes to the other, which is a
harder and more diagnostic question than whether a single model can handle
both simultaneously.

## Contamination and duplication audit

We additionally audit both full corpora for near-duplicate rows (via
approximate text matching, grouping duplicates rather than requiring exact
matches) and for cross-label duplicates, where the identical text appears
under both the human and AI class label -- a direct source of test-set
leakage under a naive random split. Audit outputs are written to
`Final/audit/daigt_full_audit.json` and `Final/audit/hc3_full_audit.json`.

## Note on checkpoint configuration vs. grid-search-best configuration

Section 5.1's in-distribution comparison (Table 2) reports, per dataset and
model, the best F1 found anywhere across the full 8-configuration
hyperparameter sweep (`table1_experiments_full.csv`), selected independently
per cell. The saved model **weights** used for every downstream analysis in
this paper (cross-dataset transfer, artifact-cleaning ablation, adversarial
robustness) are fixed to a single carried-over configuration per
(dataset, model) pair -- the configuration that won at the smaller,
6,000-row course-matched scale, reused at full scale rather than
re-selected against the full-scale grid, because only that configuration's
weights were saved to disk. For DAIGT V2 BERT and HC3 DeBERTa this
carried-over configuration happens to coincide with the full-scale grid
maximum. For DAIGT V2 DeBERTa and HC3 BERT it does not: the carried-over
checkpoints score 0.9917 and 0.9917 respectively, versus 0.9949 and 0.9945
for the true grid maximum at those cells. Sections 5.3 through 5.5 should
therefore be read as characterizing this fixed checkpoint set, not
necessarily the single best model achievable per cell; where this
distinction matters for interpreting a specific number, it is called out
inline.

## Scope-pivot components (2026-08-22/23)

Per the adopted scope pivot, two further experimental components were added
beyond the original 3-pillar framing:

1. **Artifact-cleaning ablation.** Complete (2026-08-23). Each cell was run
   both raw and cleaned -- test-time-only (zero-shot, existing checkpoints)
   and full retrain-from-scratch on cleaned train/val/test data at the same
   fixed configuration. Cleaning applied: whitespace-before-punctuation
   removal for HC3, emoji/pictograph-signal removal plus encoding-noise
   (non-breaking-space/mojibake) normalization for DAIGT V2, and per-row
   length-matching (word-count truncation to the shorter class's mean) for
   the full-retrain variant. See Results 5.4.
2. **Adversarial robustness evaluation.** Typo injection, homoglyph
   substitution (Latin-to-Cyrillic visual lookalikes), and back-translation
   paraphrase (English-German-English via MarianMT) applied to test sets
   only, never training data, at strengths 1%/5%/10% for typo and homoglyph.
   Scored against the same fixed checkpoint set described above. See
   Results 5.5 (in progress as of this draft).
# Results

*Numbers trace to `Final/paper_draft/NUMBERS_SSOT.md`. This section is
partial: subsections 5.3 and 5.4 depend on experiments not yet run (see
Methodology, "Planned additions").*

## 5.1 In-distribution baseline (five-model comparison)

Table 2 (`Final/table2_combined_full.csv`) reports full-corpus,
in-distribution F1 for all five classifiers on both datasets. The three
classical baselines separate clearly by representation and model
complexity: Naive Bayes reaches 0.9591 F1 on DAIGT V2 and 0.8713 on HC3,
Logistic Regression improves to 0.9893 and 0.9551 respectively, and the
linear SVM reaches 0.9910 on DAIGT V2 (0.9449 on HC3, where it trails
Logistic Regression). Both transformers exceed every classical baseline:
BERT reaches 0.9916 F1 on DAIGT V2 and 0.9945 on HC3; DeBERTa reaches 0.9949
and 0.9980 respectively, the best single-model result on both datasets. A
validation-weighted soft-vote ensemble over BERT and DeBERTa matches or
marginally exceeds the stronger individual model (0.9949 on DAIGT V2, 0.9982
on HC3) but does not meaningfully improve on DeBERTa alone -- consistent
with the negative ensemble finding already established at the smaller
scale, where the two transformers' error sets showed near-total overlap and
no validation-set-resolvable mixing weight improved on the stronger model in
isolation. Per the canonical literature review's framing (Section 5.3), this
in-distribution comparison is reported here as a methodological sanity
check and a classical-versus-transformer reference point, not as this
paper's central claim: the saturation of every reported score above 0.94,
and above 0.99 for both transformers, replicates a result the broader
literature already establishes many times over (see Related Work).

## 5.2 Scale-dependent seed instability

Re-running the winning DAIGT V2 BERT configuration (learning rate 3e-5,
batch size 32, weight decay 0.1) at three random seeds (42, 123, 456) at
full corpus scale yields test F1 values of 0.9916, 0.9927, and 0.9920 -- a
spread of 0.0011. The identical configuration, at the smaller 6,000-row
course-matched scale used earlier in this project, produced a substantially
wider spread of 0.0267 across the same three seeds (0.9908, 0.9850, 0.9641).
This represents an approximately 24.3-fold reduction in seed-to-seed
fine-tuning variance from increasing the training set from 6,000 to
approximately 35,000 rows, holding the model, hyperparameters, and data
source fixed. Established fine-tuning-instability results (Dodge et al.,
2020; Mosbach et al., 2021) document that best-epoch-selected fine-tuning
outcomes vary meaningfully across random seeds, but neither result states
that this instability is itself a function of training-set scale; this
finding is, to our knowledge, a genuine refinement of that established
result rather than a replication of it.

## 5.3 Cross-dataset transfer -- 3 seeds (42, 123, 456)

Table `Final/table_cross_dataset_generalization_3seed.csv` reports strict
one-way transfer for both transformers, in both directions, averaged across
three random seeds. Every cell collapses substantially relative to its
in-domain baseline: BERT trained on DAIGT V2 falls from a mean 0.9921
in-domain to a mean 0.7902 cross-domain (a mean gap of 0.2019, range
0.1817-0.2220 across seeds) when tested on HC3, and BERT trained on HC3
falls from a mean 0.9930 to a mean 0.8311 (a mean gap of 0.1619, range
0.1400-0.1822) when tested on DAIGT V2. DeBERTa's transfer gap is roughly
half BERT's in both directions (mean 0.0833 for DAIGT V2 to HC3, mean
0.1458 for HC3 to DAIGT V2), while BERT's asymmetry -- DAIGT-V2-to-HC3
transfer harder than the reverse -- holds at the mean level once averaged
across seeds, not just at the single seed reported in an earlier draft of
this section. Per this project's statistical protocol (bootstrap
confidence intervals, no parametric significance tests at n<=30), the
BERT/DeBERTa gap-magnitude difference and the BERT direction asymmetry are
reported here as patterns that survive 3-seed averaging, not as
statistically confirmed effects -- the two transfer directions' seed
ranges for BERT are close and nearly touch (0.1817-0.2220 vs
0.1400-0.1822), so the asymmetry claim should not be overstated beyond
"holds at the mean, with n=3."

## 5.4 Artifact-cleaning ablation

Two variants were run: a zero-shot pass (existing raw-trained checkpoints
re-scored on cleaned test text only, no retraining) and the review-recommended
full ablation (retraining from scratch on cleaned train/val/test data at the
same fixed configuration). The full retrain-based numbers are the headline
here; the zero-shot numbers (`Final/table_artifact_cleaning_zeroshot.csv`)
are reported as a secondary, discussion-level figure.

Under full retraining on cleaned data
(`Final/table_artifact_cleaning_full.csv`), the direction and magnitude of
the effect differ by dataset. On DAIGT V2, cleaning the emoji/pictograph
signal has essentially no negative effect and, for BERT, a small positive
one: F1 moves from 0.9916 to 0.9936 for BERT (+0.0020) and from 0.9917 to
0.9943 for DeBERTa (-0.0006, within noise given the seed-instability finding
above). On HC3, where the whitespace-before-punctuation artifact is far
stronger (present in 88.7% of human rows versus 0.28% of ChatGPT rows, per
the contamination-audit recon), cleaning produces a real, negative delta for
both models: BERT falls from 0.9917 to 0.9916 -- effectively unchanged --
while DeBERTa falls from 0.9973 to 0.9962 raw-checkpoint-vs-cleaned-retrain
(-0.0018, using the checkpoint's own score, not table2's reported best-in-grid
number; see the methodology note on this distinction below). The
zero-shot pass, which only removes the artifact from test text without ever
retraining, shows a similar but smaller HC3 DeBERTa effect (0.9973 to 0.9906,
-0.0067) and essentially no DAIGT V2 effect (both DAIGT V2 cells show a
0.0000 zero-shot delta -- consistent with DAIGT V2's emoji artifact touching
only 3.2% of AI-labeled rows overall, so a per-row test-time-only cleaning
pass changes few enough test rows' predictions to move the aggregate metric).

The pattern that emerges -- HC3, with the documented-strong whitespace
artifact, shows a real, negative, retraining-sensitive delta; DAIGT V2, with
a much rarer and more label-diagnostic-but-lower-prevalence artifact,
does not -- is itself informative: it suggests the artifact-cleaning
ablation's effect size scales with how strongly the underlying artifact
correlates with the label in that dataset specifically, rather than being a
fixed property of "cleaning" in general. This qualifies, rather than
undermines, the paper's central claim: at least part of the reported ~99%
accuracy ceiling on HC3 depends on a surface artifact the model does not
need to learn language understanding to exploit, and removing it measurably
lowers the accuracy a retrained model reaches.

## 5.5 Adversarial robustness

Table `Final/table_adversarial_robustness.csv` reports F1 for both
transformers on both datasets under three attack families -- character-level
typo injection and homoglyph substitution, each at 1%, 5%, and 10% of
eligible characters, and a single back-translation (English-German-English)
paraphrase pass -- applied to test text only, scored against the same
in-domain checkpoints used throughout Section 5. The headline finding is a
sharp, dataset-dependent split in fragility that was not visible from the
in-distribution numbers alone. DAIGT V2 is comparatively robust across every
attack and strength: the largest drop for either model is BERT under 10%
typo injection, falling from 0.9916 to 0.7966 (a 0.195 drop), and DeBERTa's
worst case (also 10% typo) falls only from 0.9949 to 0.9590 (0.0359). HC3
collapses far more severely under the same attacks: BERT falls from 0.9945
to 0.4390 under 5% typo injection and to 0.3746 under 10% -- both near
chance for a binary task -- and DeBERTa, despite reaching the highest
in-domain F1 in the entire project (0.9980), falls to 0.5524 under 5% typo
and 0.3644 under 10%. Homoglyph substitution produces a similar, if slightly
less extreme, pattern: HC3 BERT falls to 0.5826 at 5% and 0.4351 at 10%,
while DAIGT V2 BERT never drops below 0.9638 at any homoglyph strength
tested.

This dataset-level asymmetry directly corroborates the artifact-cleaning
finding in Section 5.4. HC3's much stronger, more label-diagnostic surface
artifact (the whitespace-before-punctuation cue, present in 88.7% of human
rows and 0.28% of ChatGPT rows) is exactly the kind of signal that
character-level perturbation destroys: typo injection and homoglyph
substitution both directly corrupt whitespace and punctuation patterns, so
a detector relying on this cue loses access to it under attack in a way a
detector relying on more distributed semantic signal would not. DAIGT V2's
comparative robustness is consistent with its comparatively small,
retraining-insensitive artifact-cleaning delta from Section 5.4 -- neither
experiment finds strong evidence that DAIGT V2 classification depends on a
brittle surface cue, and neither predicts a robustness collapse under
character-level attack, which is exactly what is observed here.

Back-translation paraphrase is, across every cell, markedly less damaging
than typo or homoglyph injection at even moderate strength: the largest
back-translation drop in the entire table is DeBERTa on HC3 at 0.0207 (from
0.9980 to 0.9773), smaller than the drop from 5% typo injection on the same
model by more than a factor of twenty. This is consistent with
back-translation being a semantically faithful paraphrase rather than a
character-level corruption -- it changes word choice and phrasing while
preserving grammaticality and, largely, sentence-level statistical
properties, whereas typo and homoglyph attacks directly target the
character- and token-level artifacts documented throughout this paper. That
back-translation is the mildest attack here despite being computationally
the most expensive to run (Methodology) is itself a useful methodological
note: attack cost and attack effectiveness are not correlated in this
setting, and a cheap character-level attack is a more informative robustness
probe for these particular detectors than an expensive semantic one.

One data point falls within noise rather than signal: DAIGT V2 BERT shows a
nominal 0.0017 *increase* in F1 under 1% homoglyph substitution (0.9916 to
0.9933). Given the seed-instability finding in Section 5.2 (a 0.0011 F1
spread across seeds at this same full-corpus scale, for the same model and
dataset), this single data point is fully explained by ordinary evaluation
noise and should not be read as evidence that mild homoglyph substitution
improves detection.
# Discussion

*Numbers trace to `Final/paper_draft/NUMBERS_SSOT.md`. Synthesizes Results
5.1-5.5.*

Taken individually, each experiment in this paper qualifies rather than
overturns the field's dominant narrative that AI-generated text detection
on DAIGT V2 and HC3 is close to solved. Taken together, they support a more
specific and more useful claim: the ~99% headline accuracy reported
throughout the literature on these two benchmarks is real, in the narrow
sense that it reproduces reliably under seeded, in-distribution evaluation,
but it decomposes into components whose reliability varies sharply by
dataset, and at least one of those components -- resistance to cheap
character-level adversarial attack -- is largely absent for HC3
specifically. The single largest number in this paper's results section,
DeBERTa's 0.9980 in-domain F1 on HC3, is also the number that falls
furthest under attack, to 0.3644 under 10% typo injection -- worse than a
coin flip's expected accuracy on a balanced binary task. A single headline
F1 score, reported without a robustness evaluation, cannot distinguish
between these two detectors, even though their practical reliability
differs enormously.

The three components measured here -- cross-dataset transfer (Section 5.3),
artifact-cleaning ablation (Section 5.4), and adversarial robustness
(Section 5.5) -- were designed as independent probes of the same underlying
question, and the fact that they converge on a consistent picture for HC3
specifically is, in our view, the paper's strongest evidence -- though the
cross-dataset transfer component is mixed by model rather than uniformly
pointing at HC3, and should be read that way rather than forced into a
clean story. DAIGT-V2-trained BERT transfers worse to HC3 (mean gap 0.2019)
than HC3-trained BERT transfers to DAIGT V2 (mean gap 0.1619); for DeBERTa
the direction reverses, with HC3-trained DeBERTa transferring worse to
DAIGT V2 (mean gap 0.1458) than DAIGT-V2-trained DeBERTa transfers to HC3
(mean gap 0.0833, nearly half). The artifact-cleaning and adversarial
components, by contrast, point unambiguously the same direction: HC3 is
the only dataset where the full artifact-cleaning retrain produces a real,
negative accuracy delta for both models, and HC3 is the dataset where
adversarial attacks collapse accuracy to near-chance. These are three methodologically distinct
experiments -- one trains on a different corpus entirely, one retrains on
the same corpus with a surface cue removed, one perturbs test text at
inference time only -- and they were not designed to be mutually
predictive. That they nonetheless point the same direction for the same
dataset is evidence that the underlying phenomenon (reliance on
non-semantic, HC3-specific surface signal) is real rather than an artifact
of any single experimental design choice.

DAIGT V2 tells a correspondingly different and equally informative story.
Its artifact-cleaning delta is small and inconsistent in sign across models
(BERT improves slightly under cleaning, DeBERTa degrades slightly, both
within a noise band comparable to the seed-instability finding in Section
5.2), and its adversarial robustness is comparatively strong across every
attack tested. This is not evidence that DAIGT V2 classification reflects
deeper language understanding than HC3 classification -- the cross-dataset
transfer results (Section 5.3) show DAIGT-V2-trained models collapse
substantially when tested on HC3, just as HC3-trained models collapse on
DAIGT V2, so neither model has learned a generator-agnostic, domain-general
notion of "AI-generated text." What the DAIGT V2 results do show is that
whatever DAIGT V2 classification is picking up on is not the same kind of
brittle, character-level surface cue that HC3 classification depends on;
the 128-token truncation and its documented length asymmetry (Section on
Datasets and Contamination Audit) remain the more likely candidate
confound for DAIGT V2 specifically, and are explicitly not yet resolved
here (Limitations).

Positioned against the closest prior work, Alikhanov et al.'s merge-and-
topic-split evaluation of HC3 and DAIGT V2 together, this paper's results
extend rather than duplicate that finding. Alikhanov et al. show that a
single model trained across both datasets, evaluated under a topic-held-out
split, reaches substantially lower accuracy (82.87-88.86%) than
in-distribution numbers on either dataset alone -- evidence that a
generalizable detector is harder to build than the in-distribution
literature suggests. This paper's strict one-way transfer matrix answers a
narrower, complementary question -- not "can one model handle both?" but
"does knowledge of one carry to the other, and does that answer differ by
which dataset the model started from?" -- and its artifact-cleaning and
adversarial-robustness results go further still, decomposing *why* transfer
fails rather than only measuring that it does. Neither prior work
distinguishes as sharply as this paper does between a dataset (DAIGT V2)
whose in-distribution accuracy appears comparatively load-bearing and one
(HC3) whose in-distribution accuracy appears substantially inflated by
surface artifacts that both retraining-based cleaning and cheap adversarial
attack independently expose.

Two honest qualifications belong in this synthesis rather than only in the
Limitations section. First, the checkpoint-configuration inconsistency
documented in the Methodology (Sections 5.3-5.5 use a carried-over
hyperparameter configuration that underperforms the true full-scale grid
maximum for 2 of 4 dataset-model cells) means the specific numeric values
reported here are tied to a particular, reasoned-but-not-optimal checkpoint
choice; the qualitative pattern -- HC3 fragile, DAIGT V2 comparatively
robust -- is unlikely to reverse under the grid-optimal checkpoints, since
the gap between carried-over and optimal configurations is small (0.0029-0.0032
F1) relative to the effects reported here (adversarial drops of 0.195-0.634
F1), but this has not been directly verified. Second, the 128-token
truncation, and the newly measured human/AI length asymmetry it interacts
with for DAIGT V2, is a genuine open question this paper does not resolve:
it is possible that some of DAIGT V2's apparent robustness is itself an
artifact of the model only ever seeing a systematically truncated,
class-correlated slice of each essay, rather than genuine evidence that
DAIGT V2 classification avoids brittle surface cues. Resolving this
requires a 256- or 512-token rerun, explicitly out of scope for this draft
(Limitations).
# Limitations

This draft is written openly against an incomplete experimental record, and
this section states each gap plainly rather than deferring it to a
generic caveats paragraph.

**Resolved (2026-08-23): cross-dataset transfer is now 3-seed.** The
single-seed pattern held under 3-seed averaging (Section 5.3) -- gap
magnitudes and the BERT/DeBERTa contrast did not shift substantially --
but the two transfer directions' seed ranges for BERT are close enough
that the direction asymmetry should be read as a mean-level pattern, not a
statistically confirmed effect, at n=3.

**Resolved (2026-08-23): artifact-cleaning ablation is complete**, both
zero-shot and full-retrain variants (Section 5.4). The effect is real but
dataset-dependent (HC3 shows a genuine negative delta under retraining;
DAIGT V2 does not), which qualifies rather than confirms the strong-form
version of the "accuracy is mostly artifact" claim -- see Section 5.4's
discussion.

**Resolved (2026-08-23): adversarial robustness evaluation is complete**,
all 28 cells (Section 5.5). The paper's three-part central claim -- transfer
component, artifact component, adversarial-fragility component -- now has
all three measured. The adversarial results additionally corroborate the
artifact-cleaning finding rather than standing independent of it: HC3's
stronger surface artifact (Section 5.4) predicts, and is followed by, HC3's
much greater fragility under character-level attack (Section 5.5), which
strengthens rather than merely adds to the paper's central claim.

**A methodological inconsistency between the headline in-distribution table
and every downstream analysis's checkpoints was discovered and documented
during this drafting pass** (Methodology, "Note on checkpoint configuration
vs. grid-search-best configuration"): Table 2 reports per-cell grid maxima,
while Sections 5.3-5.5 use a fixed, carried-over checkpoint set that
underperforms the grid maximum for 2 of 4 (dataset, model) cells (DAIGT V2
DeBERTa, HC3 BERT). This was a deliberate scope decision (document rather
than retrain) made to avoid discarding same-day compute across three
downstream experiments; it should be stated explicitly in the paper's
methodology section, not silently reconciled.

**The 128-token truncation inherited from the course project's specification
has not been extended, and this is a deliberate scope decision for this
draft, not an oversight.** As documented in the canonical literature review
and independently re-verified against this project's own tokenizer
(Datasets and Contamination Audit, "A third, previously undocumented
artifact"), this truncates 99.7% of DAIGT V2 essays, retaining a median of
only 30.8% of each essay's tokens -- and does so asymmetrically, discarding
proportionally more human-essay content than AI-essay content (human median
448.5 tokens vs AI median 398.5). All results reported in Section 5 use
this truncation; whether the magnitude of any reported effect changes at
256 or 512 tokens is not known, and re-running the full sweep at a longer
sequence length is out of scope for this draft (would require new GPU
compute, not an editorial change) -- flagged here explicitly as the
paper's most significant open methodological question, to be resolved
before final submission rather than silently before this draft.

**One team member's portion of the underlying coursework grid sweep (14 of
32 planned hyperparameter configurations) is not merged into the shared
results**, following a family bereavement affecting that team member; the
paper proceeds on the 18 of 32 configurations already complete plus the
full five-model comparison, and does not treat the missing configurations
as a blocker to this paper's central claim, since the sweep's purpose --
identifying the best-performing configuration per model/dataset pair -- is
already resolved for both transformers from the completed half.

**The full-corpus-scale HC3 test-leakage rate used in earlier internal
project summaries (an "11.2-11.3%" figure) could not be traced to a source
file during this draft's verification pass** and has been excluded from
this paper pending recomputation; only the smaller-scale, directly
inspectable leakage figure (0.67% of a 1,200-row HC3 test partition) is
reported with confidence (Section on Datasets and Contamination Audit).
# Conclusion

This paper set out to test a specific claim implicit throughout the
AI-generated text detection literature: that reported in-distribution
accuracy on DAIGT V2 and HC3, routinely at or above 99%, reflects genuine
progress toward reliable, generalizable detection. Building on an existing
five-model, two-dataset detection pipeline, we extended it with three
targeted experiments -- strict one-way cross-dataset transfer, an
artifact-cleaning ablation run both zero-shot and via full retraining, and
an adversarial robustness evaluation across typo injection, homoglyph
substitution, and back-translation paraphrase -- and combined these with a
full-corpus contamination and duplication audit and a re-examination of
BERT's fine-tuning seed instability across two data scales.

The evidence does not support a single, uniform verdict of "the numbers are
fake." It supports a more specific and, we think, more useful one: reported
accuracy on these two widely used benchmarks decomposes into components of
substantially different reliability, and that decomposition differs sharply
by dataset. DAIGT V2's headline accuracy is comparatively load-bearing --
it degrades only modestly under cross-dataset transfer, shows no consistent
artifact-cleaning effect, and survives adversarial attack with only minor
loss. HC3's headline accuracy, including the single highest number in this
project (DeBERTa's 0.9980 in-domain F1), is comparatively hollow: it
depends measurably on a surface artifact that full retraining on cleaned
data partially removes, and it collapses to near-chance under a typo attack
cheap enough to implement in twenty lines of code. Three independently
designed experiments converge on the same conclusion for HC3 specifically,
which is the paper's central evidentiary claim.

Two further findings stand on their own outside this main decomposition.
First, a full-corpus contamination and duplication audit finds that HC3
carries real, measurable data-quality problems -- 7.16% internal
duplication and several hundred collection-artifact rows -- essentially
absent from DAIGT V2 (0.01% duplication), a direct comparative audit that
does not, to our knowledge, appear in prior work on either dataset
individually. Second, re-running BERT's fine-tuning instability protocol
at full corpus scale finds that seed-to-seed variance shrinks roughly
24.3-fold relative to the smaller, course-matched scale used earlier in
this project (0.0267 to 0.0011 F1 spread) -- a genuine, previously
undocumented refinement of established fine-tuning-instability results
(Dodge et al., 2020; Mosbach et al., 2021), which state that such
instability exists but do not characterize its scale-dependence.

This paper does not claim to have built a better detector, and does not
attempt to. Its contribution is diagnostic: a reproducible method for
asking, of any reported AI-text-detection accuracy number, how much of it
survives contact with a different data distribution, a cleaned evaluation
set, and a cheap adversarial attack. Applied to two of the field's most
widely used benchmarks, the answer is dataset-dependent, measurable, and in
HC3's case, large enough that the single headline number materially
misrepresents real-world reliability.

Two limitations are stated here rather than left implicit, because they
bound how far the above claims should be read. The 128-token input
truncation inherited from this project's originating course specification
-- which discards a median of 69.2% of each DAIGT V2 essay's content, and
does so asymmetrically between the human and AI classes -- has not been
resolved in this draft; whether DAIGT V2's comparative robustness would
survive a 256- or 512-token rerun is an open, and in our view the single
most important remaining, question. One collaborator's portion of the
underlying coursework hyperparameter grid (14 of 32 configurations) remains
unmerged, for reasons unrelated to this paper's methodology (a family
bereavement), and is not treated as a blocker since the sweep's purpose --
identifying a usable configuration per model and dataset -- is already
satisfied by the completed half. Future work should prioritize the
token-length resolution first, since it is the limitation most likely to
change a substantive conclusion rather than merely a numeric one.
