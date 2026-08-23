---
title: "Literature Review and Gap Analysis: HC3 and DAIGT V2 in AI-Generated Text Detection"
author: "Group 02, Section B -- Natural Language Processing, AIUB"
date: "21 August 2026"
---

# Literature Review and Gap Analysis: HC3 and DAIGT V2 in AI-Generated Text Detection

## Purpose and Method

This document surveys published research that uses the two datasets our course project is built on -- the Human ChatGPT Comparison Corpus (HC3) and the DAIGT V2 Train Dataset -- extracts what each study did, what it found lacking, and what it proposed for future work, then asks a direct question: is there an unaddressed gap our own project's findings could fill, and is a paper worth writing.

Searching was done through arXiv, the ACL Anthology, Semantic Scholar, ResearchGate, GitHub, and Kaggle via web search and direct fetch of abstract and full-text pages, not through a licensed academic database with citation-count filtering. Every paper below was read directly rather than assumed from a title, and every dataset-usage claim was checked against the paper's own Datasets or Experimental Setup section rather than taken from a secondary summary. Two consequences of this method are stated plainly rather than hidden. First, HC3 turned out to have far more traceable academic reuse than DAIGT V2, which is a finding in itself and is discussed in Section 4. Second, where a paper's dataset usage could not be confirmed from primary text, it is excluded from the count of confirmed users and, if included at all, is explicitly marked as unconfirmed or as grey literature (a Kaggle write-up, blog post, or course project) rather than presented as peer-reviewed evidence. No citation below was accepted without a source page resolving to real content.

---

## 1. Papers Using HC3 (Human ChatGPT Comparison Corpus)

### 1.1 Guo, B., Zhang, X., Wang, Z., Jiang, M., Nie, J., Ding, Y., Yue, J., & Wu, Y. (2023). *How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection.*
**arXiv:2301.07597** -- https://arxiv.org/abs/2301.07597

This is the paper that introduced HC3. The authors collected tens of thousands of paired responses from human domain experts and ChatGPT across open-domain, financial, medical, legal, and psychological questions, then built three detection systems (a logistic-regression baseline, a fine-tuned RoBERTa classifier, and a GPT-Zero-style perplexity detector) to distinguish the two. They report that human evaluators and simple linguistic features can both separate ChatGPT text from expert text with reasonably high accuracy in-domain.

**Limitations stated:** the paper frames its concern around downstream risks -- fake news, plagiarism, and academic integrity -- rather than around the corpus's own construction. It does not audit HC3 for duplication or label noise.
**Future work suggested:** further study of the qualitative gap between ChatGPT and human experts, and continued development of detection methods as language models evolve.

### 1.2 Su, Z., Wu, X., Zhou, W., Ma, G., & Hu, S. (2023). *HC3 Plus: A Semantic-Invariant Human ChatGPT Comparison Corpus.*
**arXiv:2309.02731** (CIKM 2023 workshop) -- https://arxiv.org/abs/2309.02731

This paper directly extends HC3. Its stated motivation is that HC3 and comparable corpora "primarily focus on question-answering tasks, often overlooking tasks with semantic-invariant properties, such as summarization, translation, and paraphrasing," and that detecting machine-generated text in those semantic-invariant tasks is materially harder than in open QA. HC3 Plus adds those task types and explores instruction-tuned detectors on the expanded corpus.

**Limitation of HC3 identified:** narrow task coverage, restricted to QA-style generation.
**Future work suggested:** continued exploration of instruction-tuned detection models across the expanded task set.

### 1.3 Wang, R., Chen, H., Zhou, R., Ma, H., Duan, Y., Kang, Y., Yang, S., Fan, B., & Tan, T. (2024). *LLM-Detector: Improving AI-Generated Chinese Text Detection with Open-Source LLM Instruction Tuning.*
**arXiv:2402.01158** -- https://arxiv.org/abs/2402.01158

This paper uses HC3's seed questions and human expert responses as the human-text half of its training data, then has nine additional LLMs (including ChatGPT and GPT-4) answer the same 12,853 sub-questions to build the machine-text half, at both document and sentence granularity (151.7k samples total). The resulting instruction-tuned LLM-Detector outperforms baseline detectors and shows strong out-of-domain generalisation.

**Limitations stated:** the paper does not include a dedicated limitations section; its impact statement instead flags the general risk of false positives and negatives and of biases inherited from pretrained LLMs.
**Future work suggested:** none stated explicitly beyond continued research as models evolve.

### 1.4 Huang, G., Zhang, Y., Li, Z., You, Y., Wang, M., & Yang, Z. (2024). *Are AI-Generated Text Detectors Robust to Adversarial Perturbations?*
**arXiv:2406.01179** (ACL 2024) -- https://arxiv.org/abs/2406.01179

The authors train and test their Siamese Calibrated Reconstruction Network (SCRN) detector on four public datasets, one of which is confirmed in the paper's Section 4.1 to be HC3 (question-answer pairs spanning media, wiki, medicine, and finance). SCRN adds noise, reconstructs the clean representation, and calibrates predictions across noise levels, improving robustness by 6.5-18.25 absolute accuracy points over the best baseline under adversarial attack.

**Limitations stated verbatim:** "We did not consider the text paraphrasing attack... Our focus was primarily on adversarial perturbations with minor modifications," and "our experiments mainly focused on English corpora... we did not explore its performance on multilingual corpora."
**Future work implied:** paraphrasing-attack robustness and multilingual evaluation.

### 1.5 Yadagiri, A., Shree, L., Parween, S., Raj, A., Maurya, S., & Pakray, P. (2024). *Detecting AI-Generated Text with Pre-Trained Models Using Linguistic Features.*
**ACL Anthology 2024.icon-1.21** (ICON 2024, pp. 188-196) -- https://aclanthology.org/2024.icon-1.21/

Trained and evaluated on the English portion of HC3. The authors fuse linguistic and structural features (part-of-speech distribution, vocabulary size, word density, active/passive voice ratio, Flesch Reading Ease, Gunning Fog Index) with contextual embeddings from CNN-BiLSTM, RNN, BERT, GPT-2, and RoBERTa. Their fine-tuned RoBERTa variant reaches 99.73 percent accuracy on HC3.

**Limitations and future work:** not explicitly stated in the available text; the near-ceiling accuracy on a single-generator, single-snapshot corpus is not discussed as a possible sign of an easy or saturated benchmark, which is itself worth noting given our own finding that HC3 duplication inflates apparent performance.

### 1.6 Mady, M., Reschke, J., & Schuller, B. (2026). *Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators.*
**arXiv:2605.03969** -- https://arxiv.org/abs/2605.03969

Trains a DeBERTa-v3-base detector with attention-based linguistic feature fusion on HC3 Plus (the extended HC3 corpus from 1.2 above), then evaluates cross-dataset transfer on the M4 benchmark and the AI-Text-Detection-Pile. In-domain balanced accuracy reaches up to 99.5 percent, but the paper explicitly demonstrates that this collapses under distribution shift, reaching 85.9 percent on M4 with a still-notable 7.22-point gap over zero-shot baselines.

**Limitations stated verbatim:** "performance under shift is brittle and strongly model-dependent," despite strong in-domain numbers.
**Future work:** not explicitly stated in the available abstract; the brittleness finding implies continued cross-dataset robustness work is needed.

### 1.7 Alikhanov, A., Amangeldi, A., Demeubay, D., Akhmetzhan, D., Moldakhmetov, N., Polat, O., & Zharas, G. (2026). *AI Generated Text Detection.*
**arXiv:2601.03812** -- https://arxiv.org/abs/2601.03812

This is the paper closest in spirit to our own project, and the single most relevant find in this review. It combines HC3 and DAIGT V2 into one benchmark, comparing TF-IDF with logistic regression (82.87 percent accuracy), a BiLSTM classifier (88.86 percent), and DistilBERT (88.11 percent accuracy, 0.96 ROC-AUC, best overall). Critically, the authors report that an initial random 80/20 split caused their models to "learn only topic-specific vocabulary" rather than genuine AI-writing-style signal, producing "overly optimistic accuracies" -- a textbook case of the split-methodology risk our project addressed by rebuilding and asserting the midterm's split rather than assuming it was safe. Their fix was a **topic-based split** rather than a random one, to prevent the same essay topic (and its vocabulary) from appearing in both train and test.

**Limitations stated verbatim:** "primarily related to dataset diversity and computational constraints."
**Future work suggested:** expand dataset diversity, apply parameter-efficient fine-tuning (LoRA), explore distilled models, and use more efficient optimisation strategies.

**Why this paper matters most to our gap analysis:** it identifies topic leakage through naive random splitting as a real problem and fixes it by splitting on topic. It does **not**, however, check for exact or near-duplicate document leakage -- the failure mode our own audit of the full 85,449-row HC3 corpus found to affect 7.16 percent of rows and to leak 11.2-11.3 percent of a randomly split test set. Topic-based splitting and duplicate-aware splitting solve two different problems, and this paper solves only the first.

### 1.8 Detecting ChatGPT: A Survey of the State of Detecting ChatGPT-Generated Text (2023).
**arXiv:2309.07689** -- https://arxiv.org/abs/2309.07689

Included as the eighth entry with an explicit caveat: this is a **survey**, not a paper that trains on HC3 itself. It is included because it systematically catalogues the detection methods that were trained and evaluated on HC3 in 2023, and because a defensible literature review should distinguish primary dataset users from the secondary literature that maps the field around them. Its stated contribution is a taxonomy of ChatGPT-detection approaches (statistical, classifier-based, human-in-the-loop) and a discussion of their comparative robustness; its stated limitation is that the field was moving too fast for any snapshot survey to stay current, and it calls for continued benchmarking as new model versions are released.

---

## 2. Papers Using DAIGT V2 Train Dataset

DAIGT V2 (thedrcat, Kaggle, November 2023 -- https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset) was assembled for the "LLM -- Detect AI Generated Text" Kaggle competition (Vanderbilt University and The Learning Agency Lab, October 2023 to January 2024, over 4,300 participants). This is stated honestly rather than smoothed over: the peer-reviewed academic footprint of DAIGT V2 specifically is much thinner than HC3's. Most of the several hundred practitioner solutions that use it live on Kaggle and GitHub as competition write-ups and notebooks, which are useful engineering references but are not peer-reviewed evidence, and are labelled as such below.

### 2.1 The DAIGT V2 Train Dataset itself.
**Kaggle, thedrcat** -- https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset

44,868 short argumentative essays on topics such as distance learning, driverless cars, and community service, aggregating human writing (predominantly the PERSUADE corpus) against text from seventeen distinct generators (GPT-3.5/4-family models, Llama 2, Mistral, Falcon, Claude, PaLM, and others), natively imbalanced 61 percent human to 39 percent machine. There is no accompanying academic paper describing its construction in detail; the dataset card is the primary documentation, which is itself a limitation worth naming (Section 3).

### 2.2 Lai, Z., Zhang, X., & Chen, S. (2024). *Adaptive Ensembles of Fine-Tuned Transformers for LLM-Generated Text Detection.*
**arXiv:2403.13335** -- https://arxiv.org/abs/2403.13335

Confirmed DAIGT usage: the paper's Dataset section states a DAIGT training set of 35,364 samples with an 80/20 split, tested in-distribution on DAIGT and out-of-distribution on a separate "Deepfake" text corpus. Five transformer-based classifiers are combined with an adaptive ensembling algorithm, raising in-distribution accuracy from 91.8 to 99.2 percent and out-of-distribution accuracy from 62.9 to 72.5 percent.

**Limitation demonstrated (not explicitly labelled as such by the authors, but clear from their own numbers):** an over 25-percentage-point gap between in-distribution and out-of-distribution accuracy, even after ensembling, meaning DAIGT-trained detectors generalise poorly beyond the essay domain and generator set they were built on.
**Future work:** not explicitly stated; the paper's own results argue implicitly for further cross-domain robustness work.

### 2.3 Alikhanov, A. et al. (2026). *AI Generated Text Detection.*
**arXiv:2601.03812** -- https://arxiv.org/abs/2601.03812

Already described in full at 1.7 above; listed again here because it is a confirmed direct user of DAIGT V2 as well as HC3, combined into one topic-split benchmark. Repeated rather than duplicated in full to avoid double documentation, but it counts toward both dataset totals because it genuinely uses both.

### 2.4-2.8 Grey-literature and adjacent sources (explicitly not peer-reviewed academic papers).

To be transparent about the actual state of DAIGT V2's traceable usage rather than pad the list with weakly-verified academic citations, the following are named honestly as competition solutions, course projects, and closely related but distinct datasets, not as confirmed peer-reviewed DAIGT papers:

- **Kaggle competition top-solution write-ups** (e.g. "6th place solution, entropy-based text detector," github.com/chg0901; multiple public notebooks scoring 0.96-0.99 public leaderboard AUC using TF-IDF, LightGBM, and DeBERTa ensembles) -- useful engineering reference, not peer reviewed.
- **Socolof, G. Z. & Kacholia, R. (Stanford CS224N course project).** *Fast, Interpretable AI-Generated Text Detection Using Style Embeddings.* Confirmed to use the DAIGT dataset (210,000 training triplets, 768-dimensional style embeddings via triplet loss). A genuine research report, but a graduate course project rather than a peer-reviewed publication.
- **The AIDE dataset** (Vanderbilt University and The Learning Agency Lab, released via Kaggle) is a **related but distinct** dataset -- student essays against multiple modern LLM generators (PaLM 2, Gemini, GPT-4) -- built by the same institutions as the original DAIGT competition but not derived from DAIGT V2 itself. It is named here to prevent it being mistaken for a DAIGT-user paper in future searches.
- **M-DAIGT** (Lamsiyah, Ezzini, El Mahdaouy, Alami, Benlahbib, El Amrany, Chafik, & Hammouchi, 2026, arXiv:2511.11340) shares the "DAIGT" naming convention and the general detection task but is confirmed, on reading its own text, to be an **independently constructed** dataset of CNN news articles and arXiv abstracts, unrelated to the original Kaggle DAIGT V2 essays. Its stated limitations -- a static snapshot of LLM capability, binary framing that misses human-AI collaborative writing, no adversarial evaluation track, English-only -- are recorded here because they apply equally well to DAIGT V2 itself, even though the two datasets are not the same object.

This honest accounting is itself part of the gap analysis: **DAIGT V2 has seen far less peer-reviewed academic scrutiny than HC3**, despite being at least as widely used in practice through the Kaggle competition. That asymmetry is one of the clearest openings for new work, discussed next.

---

## 3. Known Limitations of Each Dataset

### HC3

| Limitation | Evidence |
|---|---|
| Single generator (ChatGPT only) | Stated directly as HC3 Plus's motivating gap (1.2) |
| Narrow task coverage (QA only, no summarisation/translation/paraphrasing) | Stated directly as HC3 Plus's motivating gap (1.2) |
| Static snapshot, collected January 2023 | Noted in comparison with the later M4 benchmark's May-2023 ChatGPT prompting (1.6) |
| Domain skew | Our own full-corpus audit: 67,996 of 85,449 answers (79.6 percent) come from `reddit_eli5` alone |
| **Exact and near-duplicate rows** | **Our own full-corpus audit: 6,118 redundant rows, 7.16 percent of the corpus, concentrated in `reddit_eli5` (8.69 percent duplicate rate); a standard random 80/20 split leaks 11.1-11.3 percent of test rows across three seeds** |
| Collection artefacts labelled as AI text | Our own audit: 476 ChatGPT-labelled "answers" are rate-limit errors, auth failures, or refusal boilerplate (1.77 percent of the ChatGPT class); capped at 0.88 percentage points of possible accuracy inflation on a balanced test set |
| Topic-vocabulary leakage under random split | Demonstrated on the HC3+DAIGT combined benchmark (1.7/2.3) |

### DAIGT V2

| Limitation | Evidence |
|---|---|
| Highly uneven per-generator sample counts (17 generators, ranging from 25,996 down to 200 samples) | Verifiable directly from the dataset's `source` column |
| Native class imbalance, 61 percent human / 39 percent machine | Dataset card and independently confirmed in our own project |
| Narrow domain (argumentative student essays only, a handful of prompts) | Dataset card |
| Topic-vocabulary leakage under naive random splitting | Directly demonstrated by Alikhanov et al. (1.7/2.3), and structurally likely given the essay-topic design |
| Outdated generator roster relative to 2026 frontier models | The generator list (GPT-3.5-era, Llama 2, Mistral-v1, PaLM) predates GPT-4o, Claude 3.5+, and Llama 3+, none of which are represented |
| Essay length far exceeds common detector context limits | Our own project's finding: median 409 WordPiece tokens per essay, 99.6 percent of essays exceed a 128-token limit, median essay contributes only 31.3 percent of its tokens to a truncated model |
| Sparse peer-reviewed documentation of dataset construction | No academic paper found describing DAIGT V2's assembly process in the way Guo et al. (1.1) describe HC3's; the dataset card is the only primary source |

---

## 4. Research Gaps

Four gaps stand out after reading these fourteen sources as a set, ranked by how directly our own project's evidence speaks to them.

**Gap 1: no paper audits HC3 for exact-duplicate test leakage, as opposed to topic leakage.** Alikhanov et al. (1.7) solve topic leakage with a topic-based split, which is a real and useful fix, but it is a different failure mode from duplicate rows. Our own full-corpus audit is, as far as this review can establish, the only quantification of HC3's 7.16 percent duplication rate and the resulting 11.2-11.3 percent test-set leakage under a standard random split. Every HC3 paper surveyed above that reports a random or stratified split (Guo et al., Wang et al., Huang et al., Yadagiri et al.) is silent on whether that split checked for duplicate content. This is the strongest, most concretely evidenced gap in this review.

**Gap 2: DAIGT V2 has no equivalent contamination audit at all.** Given that DAIGT V2 aggregates seventeen separate generators and a `persuade_corpus` human baseline of unclear internal duplication, and given that our own HC3 audit found meaningful duplication in a dataset built from a much smaller, more curated set of sources, the same audit applied to DAIGT V2's 44,868 essays is unrun territory. No paper surveyed above reports it.

**Gap 3: no paper examines *why* ensembling two AI-text detectors can fail, specifically the case of asymmetric member strength.** Both DAIGT-domain ensemble papers found (2.2, and the general ensembling literature this review's authors are aware of from the broader field) report that ensembling **helps**, and stop there. None report or explain a case where it does not. Our own project measured a concrete instance: on HC3, BERT and DeBERTa-v3 had **zero overlapping errors** (a theoretically ideal condition for ensembling) yet weighted soft voting still underperformed the stronger single model, because the weaker model was confidently wrong in cases the stronger model was tentatively right. This is a specific, reproducible negative result that the surveyed ensembling papers do not address, because they only report configurations where ensembling worked.

**Gap 4: no paper cross-evaluates a detector trained on one of these datasets against the other as an out-of-domain test**, distinct from Alikhanov et al.'s approach of merging the two into one training pool. Training on HC3 and testing on DAIGT (or the reverse) would show how much of a detector's accuracy is genuine AI-versus-human signal as opposed to signal specific to HC3's QA format or DAIGT's essay format. M4 and MAGE (cited in the course of this search but confirmed, on reading their own dataset-construction sections, **not** to include HC3 or DAIGT as source material) run this kind of cross-domain test within their own multi-source benchmarks, but that is not the same experiment as HC3-versus-DAIGT specifically.

A fifth, softer observation: **BERT fine-tuning instability under best-epoch selection**, which our seed-robustness runs demonstrated concretely (BERT varying 0.0267 F1 across three seeds on DAIGT, against DeBERTa's 0.0033), is a documented general phenomenon in the wider NLP literature (Dodge et al. 2020, Mosbach et al. 2021, neither of which is HC3- or DAIGT-specific) rather than a gap unique to these two datasets. It strengthens a paper's methodology section but is a replication, not a novel finding, and is presented as such rather than oversold.

---

## 5. Assessment: Can Our Own Project Become a Paper?

Our NLP final-term project fine-tuned BERT and DeBERTa on balanced 6,000-row samples of DAIGT V2 and HC3, reproduced all midterm classical-model numbers exactly to validate the data split, ran a 32-configuration hyperparameter sweep plus 3-seed robustness checks, built a validation-weighted soft-vote ensemble, and separately ran a full-corpus contamination audit of HC3. Measured against the four gaps above:

- **The headline accuracy result (DeBERTa reaching 0.9933 F1 on DAIGT and 0.9992 on HC3) is not publishable on its own.** Both numbers sit within the range other papers in this review already report (Yadagiri et al. reach 99.73 percent on HC3 with RoBERTa alone; Alikhanov et al.'s DistilBERT reaches 88.11 percent on the harder combined benchmark). A reviewer's first question would be what is new, and "we fine-tuned two standard transformers on two standard datasets" does not answer it.

- **The full-corpus HC3 contamination audit (Gap 1) is genuinely new and is the strongest candidate for a paper.** No surveyed paper quantifies HC3's duplication rate or its effect on random-split leakage the way our audit does. This would need to be extended from our project's 6,000-row sample to the full 85,449-row corpus with a systematic per-source breakdown (already partially done: `reddit_eli5` alone carries 8.69 percent duplication versus 0.49-4.22 percent for the other four sources), and ideally paired with a demonstration of how much a real detector's reported accuracy changes once duplicates are removed at full scale, not just the 0.0010 F1 maximum effect observed on our small balanced sample.

- **The BERT-versus-DeBERTa instability finding (Gap 5) is real but is a replication of Dodge et al. and Mosbach et al., applied to this specific task.** It strengthens a paper as a secondary finding but cannot carry one alone.

- **The ensemble asymmetry finding (Gap 3) is a genuinely interesting negative result with a clean mechanism** (zero overlapping errors, yet the weaker model's confident-wrong predictions still degrade the blend), but on its own it is closer to a instructive case study than a full paper's worth of contribution, since the underlying principle -- ensembling needs comparably strong members -- is textbook.

- **Gap 2 and Gap 4 are entirely open and neither would require new GPU training beyond what this project already ran.** A DAIGT-side duplication audit is a few hours of CPU work analogous to the HC3 audit already completed. Cross-dataset evaluation (train on HC3, test on DAIGT, and the reverse) reuses the checkpoints and evaluation code already written for this project directly.

**Overall verdict:** not a paper as it stands, but a real one is reachable with a scoped, honest reframe. The strongest single contribution would be **"contamination and evaluation pitfalls in AI-generated-text detection benchmarks,"** built around: (a) the full-scale HC3 duplication audit, extended and quantified precisely, (b) the equivalent audit run on DAIGT V2 for the first time, (c) a train-on-one-test-on-other cross-dataset evaluation using the already-fine-tuned BERT and DeBERTa checkpoints, and (d) the ensemble-asymmetry case study as a secondary, illustrative finding. That framing is honest about what is and is not novel, directly extends the one closest paper found (Alikhanov et al.'s topic-leakage fix, by addressing the duplicate-leakage failure mode it does not cover), and does not overclaim the raw accuracy numbers as an advance. A realistic venue given this scope is a conference such as ICCIT or a detection-focused workshop, not a top-tier NLP venue, and a realistic timeline is three to four weeks of additional work: a few hours for the DAIGT audit, one to two days for cross-dataset evaluation using existing checkpoints, and the remainder for writing.

---

## 6. Full Reference List

1. Guo, B., Zhang, X., Wang, Z., Jiang, M., Nie, J., Ding, Y., Yue, J., & Wu, Y. (2023). *How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection.* arXiv:2301.07597. https://arxiv.org/abs/2301.07597
2. Su, Z., Wu, X., Zhou, W., Ma, G., & Hu, S. (2023). *HC3 Plus: A Semantic-Invariant Human ChatGPT Comparison Corpus.* arXiv:2309.02731. https://arxiv.org/abs/2309.02731
3. Wang, R., Chen, H., Zhou, R., Ma, H., Duan, Y., Kang, Y., Yang, S., Fan, B., & Tan, T. (2024). *LLM-Detector: Improving AI-Generated Chinese Text Detection with Open-Source LLM Instruction Tuning.* arXiv:2402.01158. https://arxiv.org/abs/2402.01158
4. Huang, G., Zhang, Y., Li, Z., You, Y., Wang, M., & Yang, Z. (2024). *Are AI-Generated Text Detectors Robust to Adversarial Perturbations?* arXiv:2406.01179. https://arxiv.org/abs/2406.01179
5. Yadagiri, A., Shree, L., Parween, S., Raj, A., Maurya, S., & Pakray, P. (2024). *Detecting AI-Generated Text with Pre-Trained Models Using Linguistic Features.* ACL Anthology 2024.icon-1.21. https://aclanthology.org/2024.icon-1.21/
6. Mady, M., Reschke, J., & Schuller, B. (2026). *Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators.* arXiv:2605.03969. https://arxiv.org/abs/2605.03969
7. Alikhanov, A., Amangeldi, A., Demeubay, D., Akhmetzhan, D., Moldakhmetov, N., Polat, O., & Zharas, G. (2026). *AI Generated Text Detection.* arXiv:2601.03812. https://arxiv.org/abs/2601.03812
8. *Detecting ChatGPT: A Survey of the State of Detecting ChatGPT-Generated Text* (2023). arXiv:2309.07689. https://arxiv.org/abs/2309.07689 -- survey, included with that caveat stated.
9. thedrcat (2023). *DAIGT V2 Train Dataset.* Kaggle. https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset -- dataset card, not a peer-reviewed paper.
10. Lai, Z., Zhang, X., & Chen, S. (2024). *Adaptive Ensembles of Fine-Tuned Transformers for LLM-Generated Text Detection.* arXiv:2403.13335. https://arxiv.org/abs/2403.13335
11. Alikhanov, A. et al. (2026). *AI Generated Text Detection.* arXiv:2601.03812. (Repeated from #7; a confirmed direct user of both datasets.)
12. Socolof, G. Z. & Kacholia, R. *Fast, Interpretable AI-Generated Text Detection Using Style Embeddings.* Stanford CS224N course project. https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/GiuliaZoeSocolofRitikaKacholia.pdf -- course project, not peer-reviewed.
13. chg0901 (2024). *6th-place solution, entropy-based text detector, "LLM -- Detect AI Generated Text" Kaggle competition.* GitHub. https://github.com/chg0901/6th-kaggle-DAIGT-entropy-based-text-detector -- grey literature, not peer-reviewed.
14. Vanderbilt University & The Learning Agency Lab. *AIDE: AI Detection for Essays Dataset.* Kaggle. https://www.kaggle.com/datasets/lburleigh/tla-lab-ai-detection-for-essays-aide-dataset -- related but distinct dataset, named to prevent confusion, not a DAIGT-V2 user.
15. Lamsiyah, S., Ezzini, S., El Mahdaouy, A., Alami, H., Benlahbib, A., El Amrany, S., Chafik, S., & Hammouchi, H. (2026). *M-DAIGT: A Shared Task on Multi-Domain Detection of AI-Generated Text.* arXiv:2511.11340. https://arxiv.org/abs/2511.11340 -- independently constructed dataset, shares naming convention only, not a DAIGT-V2 user.
16. Wang, Y., Mansurov, J., Ivanov, P., Su, J., Shelmanov, A., Tsvigun, A., Whitehouse, C., Afzal, O. M., Mahmoud, T., Sasaki, T., Arnold, T., Aji, A. F., Habash, N., Gurevych, I., & Nakov, P. (2023). *M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection.* arXiv:2305.14902. https://arxiv.org/abs/2305.14902 -- confirmed, on reading the paper's own dataset-construction section, **not** to use HC3 or DAIGT as source material; cited only in Related Work by that paper. Included here for completeness of the search trail, explicitly marked as a non-user.
17. Li, Y., Li, Q., Cui, L., Bi, W., Wang, Z., Wang, L., Yang, L., Shi, S., & Zhang, Y. (2024). *MAGE: Machine-generated Text Detection in the Wild.* arXiv:2305.13242 (ACL 2024). https://arxiv.org/abs/2305.13242 -- likewise confirmed **not** to use HC3 or DAIGT as source material; listed for the same reason as #16.

Entries 16 and 17 are included deliberately even though they turned out not to use either dataset, because the search process initially suggested they might, and the honest correction -- verified by reading the papers' own text rather than a secondary summary -- is itself informative for anyone continuing this search: HC3 and DAIGT are cited widely as related work across the field, but that citation count should not be mistaken for a count of papers that actually train or evaluate on them.
