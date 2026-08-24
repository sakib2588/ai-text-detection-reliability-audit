# Discussion

*Numbers trace to `Final/paper/draft/NUMBERS_SSOT.md`. Synthesizes Results
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
