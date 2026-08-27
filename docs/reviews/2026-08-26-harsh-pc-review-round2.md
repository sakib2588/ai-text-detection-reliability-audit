# Harsh PC Review, round 2 --- Surface-Content Decomposition paper

Reviewed 2026-08-26 against `paper/iccit6/main.pdf`, the six-page ICCIT submission cut
(6 pp, 25 printed references, A4, double-blind, all gates passing).
Protocol: `Semester 10/Research Paper skills /harsh-paper-review.md`.
Round 1 reviewed the eleven-page draft and is at `docs/reviews/2026-08-26-harsh-pc-review.md`.
Under the protocol's multi-round rule, prior objections are tracked first and new objections
that were visible in round 1 are demoted rather than raised as fresh majors.

## Summary

The paper measures how much of a machine-generated-text benchmark's separability is
reachable from orthographic surface form alone against lexical content alone. Two logistic
regressions sharing a classifier family and a regularisation setting are fitted on identical
group-aware splits, a fine-tuned transformer is reported as a reference point, and a
document-length control closes the one channel the two arms share. The instrument is applied
to DAIGT V2 and HC3, then to their sub-corpora, with a cleaning-kit ablation, a tokenisation
control, a matched text budget, cross-corpus transfer and one published detector run out of
domain.

## Recommendation

**Minor Revision.** Confidence 4/5.

Calibration is to ICCIT, a regional IEEE conference. At that bar the paper is above the line
on evidence and comfortably above it on statistical discipline. The one item that could still
move a headline number is M1, and even if M1 goes against the authors it moves the HC3 claim
from parity to near-parity rather than overturning the localisation result that the paper now
leads with.

## Scores

| | |
|---|---|
| Novelty | 3/5 |
| Technical soundness | 4/5 |
| Evaluation rigor | 4/5 |
| Clarity | 4/5 |
| Reproducibility | 5/5 |
| Significance | 3/5 |

Evaluation rigor rises from 3 to 4 against round 1. The five-partition replication of the
null, three seeds on every deployed checkpoint, paired intervals on the subgroup reversals
and the matched text budget together answer what round 1 said was missing.

## Round 1 objections, tracked

| | Objection | Status |
|---|---|---|
| Fatal | Eleven pages against a six-page cap | **Resolved.** 6 pp on A4, references end on page 6. |
| W1 | Tian et al.'s single-token result understated | **Resolved.** Section II, "The closest prior result", quotes their 82.12 against the 81.89 they cite for RoBERTa and states what this paper adds. |
| W2 | The 16-to-1 HC3 ratio used a handicapped baseline | **Resolved.** Section IV-B reports 11.6 to one against the stronger unfiltered arm and explains why the filtered pipeline differs. |
| W3 | No published detector benchmarked | **Resolved.** Section IV-F runs Hello-SimpleAI's HC3 RoBERTa on the same partitions, 0.8230 weighted F1 out of domain. See M2 for what it does not settle. |
| W4 | Missing state of the art | **Resolved.** RAID, M4, SemEval-2024 Task 8, Feng et al. on partial-input baselines, plus Torralba and Efros, Gururangan et al., Poliak et al. and Geirhos et al. |
| W5 | Headline transformer numbers on one seed | **Resolved for the deployed checkpoints.** Section IV-A gives three-seed means and ranges for all four. The selection grid behind them is still single-seed and Section V says so. |
| W6 | Two corpora support a demonstration, not a method claim | **Partially resolved.** The abstract and Section V are scoped, but the closing recommendation still generalises to "any new detection benchmark" from n=2. See m4. |
| W7 | Parity is between two weak arms and the abstract hid it | **Resolved.** The abstract names DeBERTa's 0.28% in the same breath as the parity. |
| W8 | Eight configurations against sixteen | **Resolved.** The grid is described as sixteen runs, eight per model, and the table is eight configurations. |

No round 1 objection was ignored.

## Strengths

- The central claim is a null, and it is defended the way a null has to be. Five group-aware
  partitions, McNemar with an exact binomial, paired bootstrap intervals, and a sign that
  changes between partitions rather than a single p-value above 0.05.
- The paper argues against its own earlier conclusions in public. Section IV-C shows the HC3
  parity is 74.8% one sub-domain and that Tian et al.'s cleaning kit costs the surface arm
  10.03 points, which is the strongest available evidence against the reading its own title
  invites.
- Leakage is measured rather than assumed, 0 of 10,732 test rows against 570 under a naive
  split, and the group rule is stated precisely enough to reimplement.
- Section IV-D is a genuine methodological contribution independent of the corpora. A
  cleaning experiment reporting no change is uninterpretable without evidence the pipeline
  ran and evidence the model could read what was removed.

## Fatal Flaws

None. The page-limit flaw from round 1 is gone and no scientific fatal flaw is present.

## Major Weaknesses

**M1. The central null is a comparison between two untuned models.**
Section III-C fixes the inverse regularisation strength at the library default for both arms
and argues that sharing it is what makes them comparable. Sharing it makes them comparable to
each other, not to their own best. The two arms are not symmetric in dimension, 47 features
against roughly 47,000, so a fixed penalty does not bite them equally, and the HC3 gap being
defended is 0.06 points. A reviewer will ask whether a validation sweep over that parameter,
run independently for each arm, leaves the null standing.
*Fix:* sweep the regularisation on validation for each arm separately, report the selected
value for both, and re-run the five-partition null at the selected values. State the outcome
either way. If content pulls ahead, the honest revision is that surface reaches near-parity
rather than parity, which the localisation result in Section IV-C survives unchanged.

**M2. There is still no clean comparison against a published detector.**
Section IV-F is a real improvement on round 1, but read carefully it leaves the gap open. The
HC3 number is contaminated by the detector's own training data and the paper says so. The
DAIGT V2 number is out of domain against an in-domain model, at the detector's native decision
boundary with no recalibration. So no published detector has been evaluated on either corpus
under conditions that would let a reader compare it to the models here.
*Fix:* fine-tune one published detector architecture on these splits, or state in Section I
that the paper measures corpus properties and offers no detector comparison. The second costs
nothing and is defensible.

**M3. The instrument is 47 hand-built features and the paper never says which.**
Section III-C names categories, punctuation rates, whitespace behaviour, casing, length,
non-ASCII and digit rates, and Section V states that the arm is an upper bound over that set.
A reader cannot judge whether the set is fair or whether it smuggles in a proxy for content,
which is the first thing a sceptical reviewer will suspect of a model that reaches 0.9214
weighted F1 without reading a word. Six pages leaves no room for a table.
*Fix:* give per-category counts in one sentence, which fits, and point to the released
extractor for the enumeration.

## Minor Issues

- **m1.** Section IV-A moves between weighted F1 and error points inside one subsection. The
  three-seed ranges are quoted as F1, the paired comparisons as points of error. One unit per
  subsection would read better.
- **m2.** Section IV-E reports in-domain and cross-corpus numbers as weighted F1 while the rest
  of Section IV is error percentage. Same issue, different subsection.
- **m3.** Twenty-five printed references sits exactly on the protocol's floor for a conference
  paper, and nine of the twenty-five are preprints or web pages. Two are load-bearing. Tian
  et al., now the closest prior result, is arXiv only, and Socolof et al., cited beside
  refereed methods in Section II, is an unrefereed Stanford course project. Neither is
  avoidable, but a PC member will notice.
- **m4.** The closing sentence of Section V recommends reporting the decomposition alongside
  "any new detection benchmark". That is a recommendation from n=2 corpora. The hypothesis-only
  precedent it cites earns the analogy, but the sentence still reaches further than the
  evidence.
- **m5.** Figure 1 panel (e) is titled "Cross-dataset transfer" while the caption and Section
  IV-E call the same experiment cross-corpus.
- **m6.** Section IV-C compresses the whole DAIGT V2 subgroup analysis into one paragraph with
  no table. The sixteenfold spread and the two generator pairs are stated with their
  qualifications, which is honest, but thirteen generators are summarised by three numbers and
  a reader cannot check the pattern. Either point to the long version or accept that this
  paragraph carries less weight than the HC3 half it is paired with.

## Questions for the Authors

1. Does the HC3 null survive a validation sweep over the regularisation parameter, run
   separately for each arm?
2. What are the 47 surface features by category, and does any of them proxy for word identity?
3. Would a published detector fine-tuned on these splits close the gap to your DeBERTa, and if
   it would, what does Section IV-F establish beyond the cost of domain shift?
4. Does the cross-corpus ordering in Section IV-E, DeBERTa's gap at roughly half BERT's, hold
   beyond three seeds, given that the two BERT ranges nearly touch?
5. The surface arm reaches 0.01% error on one sub-corpus. What rules out a residual identifier
   in that sub-corpus that the 47 features happen to capture?
6. On a corpus with no known collection artefact, what does the decomposition return, and is
   that negative control available?

## Verdict in one line

Accept once the regularisation of both arms is chosen on validation rather than left at a
default, the surface feature set is characterised in the text, and Section I says plainly that
no detector comparison is being offered.

---

## Actions taken after this review, 2026-08-26

**M1 answered, and the null survives.** `experiments/audit/regularisation_sweep.py` sweeps the
inverse regularisation strength over five values, selects on validation weighted F1
independently per arm and per corpus, refits on the same training partition, and re-runs all
five group-aware partitions at the selected values. Selection lands on 100 for surface and 0.1
for content on both corpora. On HC3 the arms move to 3.15% and 3.47% error and stay
indistinguishable on five of five partitions, p from 0.19 to 0.75, every interval containing
zero. On DAIGT V2 content keeps its advantage on all five at p below 1e-6.

One thing did change and the paper now says so. At the tuned setting the surface arm is
nominally ahead on all five partitions, by 0.08 to 0.32 points, so the sign does not flip. The
round-1 "strong form of the null" argument, that an underpowered real difference would keep
its sign, therefore belongs to the default fit. The tuned fit rests on the intervals alone.
Both are reported.

**M2 answered by the cheap route.** Section I now states that the object of measurement is a
corpus and not a detector, and that no detector comparison is offered because a fair one would
need matched training data on both corpora.

**M3 remains open.** The 47 surface features are still not characterised in the text.

Artefacts written: `experiments/audit/regularisation_sweep.json`,
`experiments/audit/single_rule_baseline.json`.
