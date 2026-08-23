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
