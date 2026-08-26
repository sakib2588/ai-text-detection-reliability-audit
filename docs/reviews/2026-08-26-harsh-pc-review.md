# Harsh PC Review — Surface-Content Decomposition paper
Reviewed 2026-08-26 against `paper/iccit/main.pdf` (11 pp, 43 refs).
Protocol: `Research Paper skills /harsh-paper-review.md`.

## Summary
The paper proposes a decomposition that measures how much of a machine-generated-text
benchmark's separability is available from orthographic surface form alone versus lexical
content alone. Two matched logistic-regression arms are fitted on identical splits, a
fine-tuned transformer is reported as a reference point, and a document-length control
closes the one channel both arms share. Applied to DAIGT V2 and HC3 across five model
families, with a SHAP attribution and three methodological controls.

## Recommendation
**Major Revision.** Confidence 4/5.
Calibration: above bar for a regional IEEE venue on science, below bar on format. At an
ACL/EMNLP-tier venue the missing related work in W1 and W4 would likely sink it.

## Scores
| | |
|---|---|
| Novelty | 3/5 |
| Technical soundness | 4/5 |
| Evaluation rigor | 3/5 |
| Clarity | 4/5 |
| Reproducibility | 5/5 |
| Significance | 3/5 |

## Strengths
- Statistical discipline well above the norm for this literature. Every comparison carries
  exact McNemar plus a paired bootstrap interval, and the central null is replicated over
  five partitions with a sign-instability argument rather than asserted from one p-value.
- Leakage is measured, not assumed: 0 of 10,732 HC3 test rows under the group-aware split
  against 570 under a naive one.
- Baselines are read from deployed checkpoints, never grid maxima, and the paper says why.
- Three controls that overturn the authors' own earlier conclusions. Reporting a
  requirement instead of the finding it dissolved is rare and creditable.

## Fatal Flaws
1. **Length: 11 pages against ICCIT's 6-page cap including references.** Desk reject before
   any reviewer reads it. Fatal for submission, not for the science.

No scientific fatal flaw identified.

## Major Weaknesses

**W1. Related work materially understates the closest prior result.**
Tian et al. (Multiscale PU, Appendix B) already report that a detector consisting of the
single logical test "does token id 479 appear" reaches **F1 82.12 on HC3 sentence-level,
beating finetuned RoBERTa-base's officially reported 81.89**. Section II says they "do not
quantify how much of the corpus's total separability surface form carries". They partly
did, and with a stronger-sounding headline than this paper's. A reviewer who knows that
appendix will read the gap statement as overclaimed.
*Fix:* quote their number explicitly, then state precisely what is new here — document
level rather than sentence level, a matched content arm rather than an unmatched
transformer, two corpora on one axis, and a five-partition null.

**W2. The 16-to-1 HC3 ratio is computed against a handicapped baseline, and the paper
contains the evidence.**
Table III's best classical on HC3 is logistic regression over bag-of-words at 4.49% error,
fitted on stopword-removed, lemmatised text. The paper's own content-only arm is also
logistic regression over bag-of-words, but unfiltered, and reaches **3.26%**. The paper
therefore reports a classical model in Section IV-D that beats its own "best classical" in
Section IV-A. Against the stronger model the ratio is **11.6:1, not 16:1**.
*Fix:* add the unfiltered bag-of-words model to Table III, or state the ratio against the
stronger arm and explain why the Table III pipeline is filtered.

**W3. No published detector is benchmarked.** Guo et al.'s 99.82 F1 on HC3 is quoted in
Section II but never reproduced on these splits. Every number compared is the authors' own.
*Fix:* run one published detector on the same partitions, or state plainly in Section I
that this measures corpus properties and is not a detector comparison.

**W4. Missing state of the art.** Not cited: RAID (Dugan et al., ACL 2024), the shared
robustness benchmark whose stated motivation is the same 99%-plus problem this paper opens
with; M4 and SemEval-2024 Task 8 for multi-generator evaluation; DetectRL. Most damaging
methodologically, **Feng et al., "Misleading Failures of Partial-input Baselines" (ACL
2019)** — the paper's entire instrument is a partial-input baseline, and that paper is
specifically about when such baselines mislead.

**W5. Headline transformer numbers rest on a single training seed.** Seed spread appears
only as a range for a few cells, and the paper admits one such estimate moved 31% on a
rerun. Tables III to V are one seed each.
*Fix:* three seeds for the four deployed checkpoints, reported as mean and range.

**W6. Two corpora support a comparative demonstration, not a general method claim.** The
conclusion recommends reporting this alongside "any new detection benchmark". N=2.
*Fix:* soften to a demonstration, or run the decomposition over RAID's domains, which is
cheap since the arms are logistic regressions.

**W7. The parity is between two weak arms, and the abstract does not say so.** On HC3 both
arms sit near 3.2% error while DeBERTa reaches 0.28%. "Orthography matches content" is
true and reads as "orthography is enough" unless the 4x gap to the transformer is stated
in the same breath.

**W8. Terminology collision.** "Eight configurations per corpus" (abstract, conclusion)
against "sixteen configurations per dataset" (Section III-C). Both are true of different
things and a reader cannot tell which.

## Minor Issues
- ROC AUC quoted to six decimals (0.999641 against 0.999144) from a single seed. Spurious
  precision, and the paper itself then says the ordering is not real.
- "The transformer improves on either arm by roughly 0.03" — that is DeBERTa. BERT gives
  0.024. Name the model.
- Table III bolds both "best classical" and "best overall" with no visual distinction.
- Four figures are built and committed but unused: `fig_pipeline`, `fig_confusion`,
  `fig_roc`, `fig_decomposition`. Either cite them or drop them from the repo.
- Contribution 4 (paired statistics) is a methodological choice, not a contribution.
- The Metrics subsection ended in a dangling comma with the weighted and macro F1 equation
  missing entirely. Fixed on 2026-08-26. The gate suite did not catch it, which is worth
  knowing about the gates.

## Questions for the Authors
1. Given Tian et al.'s 82.12 single-token result, what precisely is new in the HC3 finding?
2. Why does Table III's best classical on HC3 underperform your own content-only arm, and
   should the 16:1 ratio be restated?
3. Would the HC3 null survive a content arm with stopwords removed, matching Table III?
4. What does the decomposition give on a corpus with no known artefact, as a negative
   control for the instrument itself?
5. Do the four deployed checkpoints hold their numbers across three seeds?
6. Does the surface arm's advantage on HC3 survive the cleaning kit Tian et al. released?

## Verdict in one line
Accept once the paper fits the page limit, credits and delimits Tian et al.'s prior result,
and stops quoting a 16:1 ratio that its own content arm contradicts.
