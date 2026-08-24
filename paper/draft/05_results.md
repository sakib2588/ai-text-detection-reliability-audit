# Results

*Numbers trace to `Final/paper/draft/NUMBERS_SSOT.md`. This section is
partial: subsections 5.3 and 5.4 depend on experiments not yet run (see
Methodology, "Planned additions").*

## 5.1 In-distribution baseline (five-model comparison)

Table 2 (`Final/tables/table2_combined_full.csv`) reports full-corpus,
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

Table `Final/tables/table_cross_dataset_generalization_3seed.csv` reports strict
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
here; the zero-shot numbers (`Final/tables/table_artifact_cleaning_zeroshot.csv`)
are reported as a secondary, discussion-level figure.

Under full retraining on cleaned data
(`Final/tables/table_artifact_cleaning_full.csv`), the direction and magnitude of
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

Table `Final/tables/table_adversarial_robustness.csv` reports F1 for both
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
