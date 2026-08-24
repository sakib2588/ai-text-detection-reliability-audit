# Related Work

*Condensed from the 25-paper survey in
`Final/docs/literature/LITERATURE_REVIEW_CANONICAL.md`; full reference list and
per-paper detail there. Citation keys below are real, verified BibTeX keys
(`Final/paper/draft/refs.bib`, 31 entries, built via arXiv/CrossRef metadata
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
