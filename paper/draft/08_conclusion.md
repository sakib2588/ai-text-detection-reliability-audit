# Conclusion

This paper set out to test a specific claim implicit throughout the
AI-generated text detection literature: that reported in-distribution
accuracy on DAIGT V2 and HC3, routinely at or above 99%, reflects genuine
progress toward reliable, generalizable detection. Building on an existing
five-model, two-dataset detection pipeline, we extended it with three
targeted experiments -- strict one-way cross-dataset transfer, an
artifact-cleaning ablation run both zero-shot and via full retraining, and
an adversarial robustness evaluation across typo injection, homoglyph
substitution, and back-translation paraphrase -- and combined these with a
full-corpus contamination and duplication audit and a re-examination of
BERT's fine-tuning seed instability across two data scales.

The evidence does not support a single, uniform verdict of "the numbers are
fake." It supports a more specific and, we think, more useful one: reported
accuracy on these two widely used benchmarks decomposes into components of
substantially different reliability, and that decomposition differs sharply
by dataset. DAIGT V2's headline accuracy is comparatively load-bearing --
it degrades only modestly under cross-dataset transfer, shows no consistent
artifact-cleaning effect, and survives adversarial attack with only minor
loss. HC3's headline accuracy, including the single highest number in this
project (DeBERTa's 0.9980 in-domain F1), is comparatively hollow: it
depends measurably on a surface artifact that full retraining on cleaned
data partially removes, and it collapses to near-chance under a typo attack
cheap enough to implement in twenty lines of code. Three independently
designed experiments converge on the same conclusion for HC3 specifically,
which is the paper's central evidentiary claim.

Two further findings stand on their own outside this main decomposition.
First, a full-corpus contamination and duplication audit finds that HC3
carries real, measurable data-quality problems -- 7.16% internal
duplication and several hundred collection-artifact rows -- essentially
absent from DAIGT V2 (0.01% duplication), a direct comparative audit that
does not, to our knowledge, appear in prior work on either dataset
individually. Second, re-running BERT's fine-tuning instability protocol
at full corpus scale finds that seed-to-seed variance shrinks roughly
24.3-fold relative to the smaller, course-matched scale used earlier in
this project (0.0267 to 0.0011 F1 spread) -- a genuine, previously
undocumented refinement of established fine-tuning-instability results
(Dodge et al., 2020; Mosbach et al., 2021), which state that such
instability exists but do not characterize its scale-dependence.

This paper does not claim to have built a better detector, and does not
attempt to. Its contribution is diagnostic: a reproducible method for
asking, of any reported AI-text-detection accuracy number, how much of it
survives contact with a different data distribution, a cleaned evaluation
set, and a cheap adversarial attack. Applied to two of the field's most
widely used benchmarks, the answer is dataset-dependent, measurable, and in
HC3's case, large enough that the single headline number materially
misrepresents real-world reliability.

Two limitations are stated here rather than left implicit, because they
bound how far the above claims should be read. The 128-token input
truncation inherited from this project's originating course specification
-- which discards a median of 69.2% of each DAIGT V2 essay's content, and
does so asymmetrically between the human and AI classes -- has not been
resolved in this draft; whether DAIGT V2's comparative robustness would
survive a 256- or 512-token rerun is an open, and in our view the single
most important remaining, question. One collaborator's portion of the
underlying coursework hyperparameter grid (14 of 32 configurations) remains
unmerged, for reasons unrelated to this paper's methodology (a family
bereavement), and is not treated as a blocker since the sweep's purpose --
identifying a usable configuration per model and dataset -- is already
satisfied by the completed half. Future work should prioritize the
token-length resolution first, since it is the limitation most likely to
change a substantive conclusion rather than merely a numeric one.
