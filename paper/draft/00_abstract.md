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
