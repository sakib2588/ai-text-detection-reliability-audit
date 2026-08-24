# Datasets and Contamination Audit (Contribution 1)

*Numbers trace to `Final/paper/draft/NUMBERS_SSOT.md` rows 5-6.*

Both benchmarks used in this study carry documented limitations in the prior
literature, and our own full-corpus audit both confirms and quantifies these
for the specific dataset versions used here. Running a duplicate-detection
pass across the complete, unbalanced DAIGT V2 corpus (44,868 rows) finds
essentially no internal duplication: six duplicate rows across six duplicate
groups, a rate of 0.01%, and zero cases where identical text appears under
conflicting human/AI labels. This is consistent with DAIGT V2's origin as a
single curated Kaggle competition dataset rather than a corpus assembled by
merging multiple independently collected sources.

HC3 presents a substantially different picture. Across its full 85,449-row
corpus, the same duplicate-detection pass finds 6,118 duplicate rows across
5,986 duplicate groups, a duplication rate of 7.16%. As with DAIGT V2, no
cross-label duplicates were found, so the duplication does not by itself
create mislabeled training signal; however, near-duplicate rows crossing
between the train and test partitions of a naive random split constitute a
direct form of test-set leakage, since a model can memorize a training-set
row and simply recall it at test time rather than generalizing. The audit
additionally surfaces a separate data-quality issue specific to HC3: 476
rows nominally labeled as ChatGPT-authored answers, and 7 nominally
human-authored, are collection artifacts -- text such as API failure
messages ("Too many requests in 1 hour") stored under the answer field
rather than genuine model or human output. These rows are trivially
separable from real text and can artificially inflate reported detection
accuracy if left in the evaluation set uncleaned.

At the smaller, course-matched scale (1,200-row HC3 test partition, a subset
of the full corpus), a targeted leakage check identifies eight test rows
(0.67% of that test partition) whose text also appears in the training
partition -- independently corroborating the full-corpus duplication finding
above at a scale small enough to manually inspect, and confirming that
several of the leaked rows are exactly the API-failure-message artifacts
described above. Rescoring every model on the leakage-excluded subset of
this smaller test set shows the largest resulting F1 change is 0.0010 with no
change in model ranking, indicating that -- at this particular scale and
split -- the leakage's effect on the reported metric is smaller than
ordinary seed-to-seed training noise, even though the leakage itself is
real and worth reporting on scientific-validity grounds independent of its
measured effect size.

**Note on scope:** a full-corpus-scale equivalent of this leakage-rate
calculation (i.e., what fraction of a full 80/20 random split of all 85,449
HC3 rows would leak, as opposed to the 1,200-row course-scale subset above)
has not yet been computed or located on disk as of this draft. Prior session
notes referenced an "11.2-11.3%" full-corpus leakage figure; this number
could not be traced to any source file during the 2026-08-22 verification
pass and must not be used until it is either recomputed or its source file
is located.

This audit motivates two complementary observations that anchor the paper's
framing. First, HC3's real, measured contamination (7.16% duplication,
non-trivial artifact rows) sits in sharp contrast to DAIGT V2's near-total
absence of the same problem, despite both being widely used as
if-equivalent benchmarks in the AI-text-detection literature -- a
comparative audit of this kind does not, to our knowledge, appear in prior
published work on either dataset individually. Second, the dissociation
between leakage rate and its measured effect on accuracy, observed at the
scale we can directly inspect, is itself worth stating precisely rather than
assuming: contamination can be real and worth correcting for scientific
validity while nonetheless not being the dominant driver of a benchmark's
reported headline number. How much of the headline number *is* explained by
surface artifacts is addressed directly by the artifact-cleaning ablation
(Results 5.4).

## A third, previously undocumented artifact: length asymmetry under truncation

A separate audit, run directly against this project's own tokenizer and
data rather than drawn from the literature, surfaces a further DAIGT V2
issue specific to how this project (and the course specification it
inherits from) processes text. Measuring token length with the
`bert-base-uncased` tokenizer over a 2,000-row sample of the full DAIGT V2
corpus, 99.7% of essays exceed the 128-token input limit used throughout
this project's transformer training (median length 415 tokens, mean 443,
maximum 1,826); at truncation, a median essay retains only 30.8% of its
tokens, so every DAIGT V2 classification decision in this paper is made
from roughly the opening third of the essay, not the whole document. This
by itself replicates a limitation already known in principle from the
literature (Methodology; Limitations). What has not previously been
measured for this dataset is whether truncation affects the two classes
equally: it does not. Human-authored essays run measurably longer than
AI-authored essays in the same sample (median 448.5 tokens for human versus
398.5 for AI, roughly 12.5% longer; every human essay in the sample
exceeded 128 tokens, versus 99.4% of AI essays). Truncation to 128 tokens
therefore discards proportionally more content from human essays than from
AI essays -- a class-correlated information loss with the same structural
shape as the HC3 length confound Baidya et al. document and control for by
length-matching, except undocumented for DAIGT V2 specifically and, unlike
the HC3 case, not yet corrected for in this project's 128-token results
(see Limitations for the scope decision on this point).
