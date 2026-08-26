---
title: "A Surface-Content Decomposition of AI-Text Detection Benchmarks -- Project Explainer and Related-Work Gap Table"
author: "Group 02, Section B -- Natural Language Processing, AIUB"
date: "2026-08-23"
geometry: margin=1in
fontsize: 11pt
---

# A Surface-Content Decomposition of AI-Text Detection Benchmarks

*One place to understand this project top to bottom: what it is, why it exists, what it found, what it had to retract, and exactly which prior papers it fills a gap against.*

Every number in this document comes from a committed script that writes a JSON file. Nothing here is quoted from memory. The scripts are `experiments/audit/surface_content_decomposition.py`, `experiments/audit/collapse_probe.py`, and `experiments/paper_scale/full_model_evaluation.py`.

---

## 1. The one-sentence version

AI-text detectors report 99% accuracy on the two benchmarks everyone uses, and we built a measurement that asks how much of that score a detector could get **without reading the language at all** -- on HC3 the answer is essentially all of it, because a model using only punctuation and spacing statistics matches a model using only words, while on DAIGT V2 content wins by a factor of eight.

---

## 2. Explain like I'm 5

Imagine a teacher who wants to catch homework written by a robot. She collects 100 essays from students and 100 from a robot, and trains a helper to tell them apart. The helper gets 99% right. Everyone is impressed.

But there are two very different reasons the helper could be doing so well.

Maybe it genuinely learned how robot writing *sounds* -- the word choices, the phrasing, the ideas.

Or maybe it noticed something much dumber. Maybe when the school collected the student essays, they were typed in a word processor that puts a space before every comma, and the robot essays were copy-pasted from a chat window that does not. Now the helper does not need to understand writing at all. It just counts spaces before commas. It still gets 99%.

Both stories produce the exact same 99%. The score cannot tell them apart.

So we built a test. We make **two** helpers. The first is only allowed to look at punctuation, spacing, capital letters, and how long things are -- it is forbidden from seeing a single word. The second is only allowed to see the words, after we delete all punctuation and capitals so it cannot cheat. Then we compare.

On one benchmark (HC3), the helper who cannot read any words does **just as well** as the one who can. That means roughly half the test could be passed by counting punctuation. On the other benchmark (DAIGT V2), the word-reader wins easily. Same task, same method, opposite answer -- and no accuracy score would ever have told you.

---

## 3. How this project came to life

This started as our NLP course project: train five models on two AI-text datasets and compare them. That part worked. Every model scored between 0.87 and 0.997 F1.

The problem is that being good at the task is not a research contribution, because it is already solved in the literature many times over. Guo et al. reported 99.82 F1 on HC3 back in 2023. Adding another 99% number to that pile is not publishable.

What *is* still open is a question the literature keeps circling without answering directly. Several papers have found individual flaws in these benchmarks:

- Tian et al. (ICLR 2024) found that HC3's human answers contain a space before punctuation and the ChatGPT answers do not, and released a cleaning kit.
- Baidya et al. found that human and ChatGPT answers differ systematically in length, and length-matched their benchmark to control for it.
- Ardeshirifar standardised punctuation and contractions before comparing models, treating it as necessary hygiene.
- Park et al. showed detectors latch onto prompt-specific collection shortcuts.

Each of those finds **one** cue and either fixes it or warns about it. Nobody asks the aggregate question:

> *For a given benchmark, how much of its total human-versus-machine separability is carried by surface form rather than by content -- and does that answer differ between benchmarks?*

That is the gap. It matters because it is the number a person actually needs when choosing which benchmark to trust, and no individual artifact paper provides it.

### The measurement protocol that answers it

Three models per benchmark, on identical train/test splits.

1. **Surface-only.** 47 hand-built features covering punctuation rates, whitespace behaviour, capitalisation, document and sentence length, non-ASCII and emoji rates, digit rates. It never reads word identity, so it cannot access content *by construction*.
2. **Content-only.** Text is lowercased, every character outside `[a-z\s]` is replaced with a space, stopwords removed, lemmatised, then bag-of-words. Punctuation, capitals and emoji cannot survive this, so it cannot access surface cues *by construction*.
3. **Full.** The fine-tuned transformer on raw text, as a reference point.

Arms 1 and 2 both use logistic regression, so they are directly comparable to each other. The transformer uses a different model family, so we report it as a reference and never draw a conclusion from a transformer-versus-classical gap alone.

---

## 4. What you actually found

**Current title:** *"A Surface-Content Decomposition of AI-Generated Text Detection Benchmarks"* (`paper/iccit/main.tex`, 13 pages development draft; `paper/iccit6/main.tex`, the 6-page ICCIT cut).

**Datasets and models.** DAIGT V2 (44,868 argumentative student essays, balanced to 34,994, test set 6,998) and HC3 (85,449 question-answer pairs, balanced to 53,806, test set 10,732). Five model families: Naive Bayes, logistic regression and linear SVM each under bag-of-words and TF-IDF (six classical configurations), plus fine-tuned BERT and DeBERTa. Eight configurations per dataset, sixteen total.

### 4.1 The headline: the two benchmarks decompose differently

| Arm | DAIGT V2 F1 / error | HC3 F1 / error |
|---|---|---|
| surface-only (47 features, no words) | 0.9214 / 7.86% | **0.9680 / 3.20%** |
| content-only (words, surface stripped) | **0.9901 / 0.99%** | 0.9674 / 3.26% |
| full transformer (reference) | 0.9917 / 0.83% | 0.9972 / 0.28% |

**On HC3 the two arms are indistinguishable**, 3.20% against 3.26% error. Forty-seven features that never read a word do as well as a bag-of-words model over the entire vocabulary. **On DAIGT V2 content wins by a factor of 7.9 in error.**

Macro F1 equals weighted F1 in every cell, so none of this is being produced by class imbalance.

### 4.2 The model comparison shows the same split independently

On **DAIGT V2**, the best classical model (SVM over TF-IDF, 0.9910) lands **0.0007 F1 below** the best transformer (0.9917). That is well inside our measured seed range of 0.0036, so the transformers add nothing measurable. A bag-of-words model is as good as DeBERTa there.

On **HC3**, the same comparison is 0.9551 against 0.9972 -- an **error-rate ratio of 16 to one**. The transformers win decisively.

This is a completely separate line of evidence from the decomposition, using different models, and it points the same way.

### 4.3 The cue everyone discusses is sufficient but unnecessary, and tokenisation proves it

HC3's famous whitespace artifact is real and strong in our copy: **10.745 spaces-before-punctuation per human document against 0.013 per machine document**. A classifier using only that one feature reaches **0.9413 F1** on HC3.

So detectors must be exploiting it, right? No -- and the proof is mechanical rather than statistical.

BERT's WordPiece tokeniser splits on punctuation regardless of what precedes it. For `"the answer is simple ."` and `"the answer is simple."` it emits **identical token identifiers**, on 3 of 3 pairs we tested. DeBERTa's SentencePiece encodes the leading space and distinguishes all 3.

**BERT literally cannot see this cue, and still reaches 0.9916 F1 on HC3.** The cue is sufficient on its own and unnecessary in practice, because there is enough other signal to saturate without it.

This also explains a null we could not interpret earlier: cleaning the whitespace from HC3 produces predictions that are *bit-identical* for BERT. That is not a broken pipeline. It is exactly what WordPiece predicts.

### 4.4 An adversarial result we withdrew

We ran typo and homoglyph attacks. Corrupting 10% of characters drives DeBERTa on HC3 from 0.9972 down to 0.3644 F1, with a striking one-directional confusion matrix -- human documents still perfect, machine documents all reassigned to human. This looked like a headline finding about HC3 fragility.

It is not, and the control that killed it is cheap. We fed the same model inputs carrying **no** human-versus-machine information at all:

| input | predicted "human" | confidence |
|---|---|---|
| uniform random character strings | 100% | 0.98 |
| token-shuffled text | 100% | 1.00 |
| punctuation-only strings | 100% | 0.99 |

A model that calls random noise "human" with 98% confidence is not displaying a targeted vulnerability. It is displaying a degenerate response to unreadable input. All four of our models do this. It is not specific to HC3 and not specific to the attack.

We tested three explanations and refuted all three. Majority-prior collapse is out, because training is balanced to within 0.002. Cue injection is out, because typos do not create the whitespace artifact (machine documents move from 0.000 to 0.005 per document). "The model learned that messy text is human" is unsupported, because subword fertility runs the wrong way -- HC3 human text is 1.1192 against machine 1.1980, so human text is the *cleaner* of the two.

**So we report the requirement for a label-free control instead of reporting a vulnerability.** That matters beyond us, because published papers report comparable collapses without running this control.

### 4.5 What we had to retract, and why you should know

An earlier version of this analysis, which you may have seen, made claims that did not survive audit. They are listed here because the retractions are recorded in `paper/draft/NUMBERS_SSOT.md` and you should not quote the old numbers.

- **The artifact-cleaning experiment was void.** The code compared a validation maximum taken across the whole hyperparameter grid against a single fixed configuration, so three of four cells compared *different hyperparameters*. Corrected, the HC3/BERT delta is exactly **0.0000**, not the 0.0029 originally reported.
- **The same bug inflated every adversarial drop.** Real deployed checkpoints are 0.9916 and 0.9972, not the 0.9945 and 0.9980 reported.
- **The noise band was cherry-picked** -- 0.0011 was the smallest of four measured seed spreads. The honest figure is 0.0036, against which no cleaning delta clears noise.
- **"Below chance" was a misreading.** Weighted F1 has a floor of 0.333 for a single-class predictor on a balanced split, so 0.3644 is 1.4% *above* total degeneracy, not below chance.
- **"DAIGT V2 is at chance on the whitespace cue" was wrong.** That came from one hand-picked threshold, which is the wrong instrument for a cue that is real but sparse there (0.551 vs 0.003 per document). A properly fitted 47-feature model reaches 0.9214 on DAIGT V2. **Neither dataset is clean.**

### 4.6 Supporting results

**Contamination audit.** HC3 carries 6,118 duplicate rows in 5,986 groups (7.16%), plus 476 rows labelled machine-generated that are actually stored API failure messages. DAIGT V2 has six duplicate rows total (0.01%). We report this as data quality, not as an explanation of any accuracy number, because at the scale where every leaked row can be inspected by hand the effect on F1 is at most 0.0010.

**Cross-dataset transfer.** Training on one benchmark and testing on the other costs between 0.0833 and 0.2019 mean F1 over three seeds. The asymmetry reverses between BERT and DeBERTa, so we report magnitudes and draw no conclusion from direction.

---

## 5. Real-world scenario

A university is choosing a tool to flag AI-written coursework. Two vendors both cite 99% accuracy on public benchmarks. The university picks one and deploys it.

If the vendor's number came from a benchmark like HC3, roughly half that separability is available from punctuation and spacing alone. That is fine inside the benchmark and useless in deployment, because a student pasting text through a different editor, or a word processor with different autoformatting, changes the punctuation spacing without changing a single word. The detector's accuracy was partly measuring which text editor produced the file.

Worse, a false positive here is a plagiarism accusation against a real student. Our HC3 surface-only model has a **5.62% false-positive rate** on human text, 302 of 5,377 -- roughly one in eighteen honest students flagged by a model that never reads a word of what they wrote. On DAIGT V2 the same arm sits at 8.81%.

The decomposition is the check that catches this before deployment. It runs in minutes, needs no new labelling, and returns one number per benchmark that the headline accuracy cannot express: how much of this test could be passed without reading the language.

---

## 6. Literature review / research-gap table

| Paper | What they did | Dataset | Result | Limitation we address |
|---|---|---|---|---|
| Guo et al. 2023 | Introduced HC3, RoBERTa detector | HC3 | 99.82 F1 | In-distribution only; establishes the accuracy regime but not what carries it |
| Tian et al. 2024 (ICLR) | Found HC3 whitespace artifact, released cleaning kit | HC3 | PU method 58.6 to 85.3 F1 on short text | Identifies one cue; never measures total surface-carried separability, and never tests whether a given detector can even represent the cue |
| Baidya et al. 2026 | Documented length confound, length-matched | HC3 + ELI5 | Near-perfect in-domain, drops OOD | One artifact in isolation |
| Ardeshirifar 2025 | Standardised punctuation/contractions | HC3 + GPT-2 | DL wins in-domain, both drop cross-model | Establishes cleaning as necessary without measuring what it removes |
| Park et al. 2024 | Prompt-specific shortcut probing (FAILOpt) | HC3 | Detectors rely on collection shortcuts | Shortcut existence, not magnitude |
| Antoun et al. 2023 | Typo/homoglyph attacks | HC3 (En+Fr) | 99.88 F1 falls to 33.57% | **No label-free control**, so the collapse cannot be separated from degenerate response |
| Huang et al. 2024 (ACL) | Systematic character perturbation | HC3 + TruthfulQA | Large degradation | Same missing control |
| Borile & Abrate 2025 (EMNLP) | Neuron ablation for OOD | DAIGT + HC3 + others | +6.9% OOD from removing ~20 neurons | Mechanistic, at neuron level; complementary to our input-level decomposition |
| Alikhanov et al. 2026 | Merged both corpora, topic-held-out split | HC3 + DAIGT V2 | 82.87 to 88.86% | Asks whether one model serves both; not what each corpus is separable *by* |
| Zhou 2024 | Classical models, character n-grams | DAIGT V2 | Ensemble beats singles | Competitive without transformers, consistent with our DAIGT finding, but no surface/content separation to explain it |
| Kubrusly et al. 2025 | BoW/TF-IDF/doc2vec classical | HC3 | High in-domain | Does not isolate what the features encode |
| Annepaka et al. 2026 | DistilBERT + explicit linguistic features | HC3 + M4GT | 99.45% on HC3 | *Mixes* the two information sources our decomposition separates |

**Foundational anchors (cited as lens, not competitors)**

- **Dodge et al. 2020** and **Mosbach et al. 2020** -- fine-tuning instability under initialisation and data order; why we report seed spread as a range and refuse to build claims on variance ratios.
- **Thedrcat (DAIGT V2)**, **Crossley et al. (PERSUADE)**, **Hello-SimpleAI (HC3)** -- the corpora themselves.

---

## 7. Novelty summary -- gap to contribution, one line each

| Gap left by prior work | What this project did about it |
|---|---|
| Individual benchmark artifacts are documented one at a time; nobody measures the aggregate | Built a surface-only versus content-only decomposition giving one number per benchmark for total surface-carried separability |
| No prior work compares two AI-text benchmarks on *how* they are separable, only on how well models score | Applied the decomposition to DAIGT V2 and HC3 and found they differ qualitatively: parity on HC3, 7.9x content advantage on DAIGT V2 |
| The HC3 whitespace cue is assumed to be exploited by detectors because it exists and is strong | Tokenisation control shows BERT emits identical ids for strings differing only by that cue, so it cannot represent it, yet reaches 0.9916 -- sufficient in isolation, unnecessary in practice |
| A cleaning experiment that reports "no change" is uninterpretable, because it looks identical to a cleaning step that silently failed | Pipeline-fires control: the same cleaning *does* change BERT's DAIGT V2 predictions (emoji, which WordPiece encodes), establishing the HC3 null is real |
| Adversarial collapses on text detectors are reported without checking what the model does on inputs carrying no label information | Label-free control on random strings, shuffled tokens, punctuation-only and empty input; our models answer "human" at 100% and 0.98 confidence, so we withdrew our own adversarial claim rather than publish it |
| Classical baselines are usually reported for one representation only, hiding the representation choice | Full 3-model by 2-representation grid at full scale, plus ROC, AUC and confusion matrices for all eight configurations |
| Papers rarely say which of their own earlier claims did not survive audit | Five retractions recorded explicitly in `NUMBERS_SSOT.md` and acknowledged in the paper text, including a hyperparameter-matching bug we wrote ourselves |

---

## 8. Three-minute pitch

> "Everyone reports 99% accuracy on AI-text detection benchmarks. That number is real, and it tells you almost nothing, because it cannot distinguish a detector that learned how machine writing sounds from one that learned how one particular dataset was punctuated.
>
> Those two stories predict the same 99%. So we built the measurement that separates them. We train two models on the same data. The first sees only punctuation, spacing, capitalisation and length -- 47 features, and it is forbidden from seeing a single word. The second sees only words, after we strip out every punctuation mark and capital letter so it cannot cheat. Whichever wins tells you what the benchmark is actually made of.
>
> On HC3, the model that cannot read any words does just as well as the one that can -- 3.20% error against 3.26%. Roughly half that benchmark is passable on orthography alone. On DAIGT V2, the word-reader wins by a factor of eight. Same method, same metric, opposite answer, and no accuracy score would have told you.
>
> Three controls kept us honest, and each one killed something we believed. The famous HC3 whitespace artifact turns out to be sufficient on its own -- one feature gets 0.9413 F1 -- but unnecessary in practice, because BERT's tokeniser provably cannot represent it and BERT still scores 0.9916. And we had an adversarial result showing accuracy collapsing from 0.997 to 0.364 under typos, which looked like our best finding, until we fed the same model random character strings and it called them human 100% of the time at 0.98 confidence. So we withdrew it, and report the missing control instead -- which published papers reporting the same kind of collapse also do not run.
>
> This is not a paper about building a better detector. It is a measurement that takes minutes, needs no new labelling, and tells you before you trust a 99% figure how much of that benchmark a model could pass without reading the language at all."

---

*Last updated: 2026-08-23. Paper source: `paper/iccit/`. Numbers source of truth: `paper/draft/NUMBERS_SSOT.md`.*
