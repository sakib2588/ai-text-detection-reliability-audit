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
