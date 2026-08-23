# Ensembling — what we did, why it underperformed, and what would have worked

NLP Final Term Project, Group 02. Companion note to the notebook and the two result tables.
Every number here is measured from files in `results/` and `probs/`, not estimated.

---

## 1. What an ensemble is

Combining several models so their mistakes cancel out. It only works under two conditions,
and both matter:

1. Members are **roughly comparable in strength**.
2. Members make **different mistakes**.

If one member is far weaker, letting it vote drags the result down. If all members make the
same mistakes, combining them changes nothing. Our project satisfies condition 2 and fails
condition 1, which is the whole story of this document.

---

## 2. What we used

**Weighted soft voting.** Each model outputs a probability rather than just a label, and we
average those probabilities:

```
P_ensemble = w * P_BERT + (1 - w) * P_DeBERTa
```

Procedure:

1. Pick the best BERT configuration and the best DeBERTa configuration, each chosen by
   **validation** F1.
2. Sweep `w` from 0 to 1 in steps of 0.05, keep whichever maximises **validation** F1.
3. Apply that single fixed `w` to the test set exactly once.

The test set is never used to choose anything. That is what makes the reported number honest.

It is *soft* voting because we average probabilities. *Hard* voting, which takes a majority of
predicted labels, is impossible with two models because there is no way to break a 1-1 tie.

### Result

| | weight BERT | weight DeBERTa | BERT F1 | DeBERTa F1 | Ensemble F1 |
|---|---|---|---|---|---|
| Dataset 1 (DAIGT) | 0.35 | 0.65 | 0.9908 | 0.9933 | 0.9925 |
| Dataset 2 (HC3) | 0.00 | 1.00 | 0.9875 | 0.9992 | 0.9992 |

---

## 3. Why Dataset 2 shows a weight of 0.00

**This does not mean BERT was rejected, and the report must not say the ensemble "collapsed".**

DeBERTa already gets 1199 of 1200 test rows correct on Dataset 2. Mixing in BERT changes
almost nothing, so a wide range of weights produces exactly the same predictions. The full
validation sweep:

| w (BERT) | validation F1 | test F1 |
|---|---|---|
| 0.00 | 0.9937 | 0.9992 |
| 0.10 | 0.9937 | 0.9992 |
| 0.20 | 0.9937 | 0.9992 |
| 0.30 | 0.9937 | 0.9992 |
| 0.40 | 0.9937 | 0.9992 |
| 0.50 | 0.9937 | 0.9992 |
| 0.60 | 0.9833 | 0.9892 |
| 0.80 | 0.9833 | 0.9875 |
| 1.00 | 0.9854 | 0.9875 |

Every weight from 0.00 to 0.50 is tied at validation F1 0.9937 and test F1 0.9992. That is an
eleven-way tie. `numpy.argmax` returns the *first* maximum it encounters, which is 0.00.

**Correct wording for the report:** the optimum is a flat plateau over w in [0, 0.5], and a
validation set of 480 rows cannot distinguish between weights inside it. The reported 0.00 is an
arbitrary tie-break, not evidence that BERT contributes nothing.

---

## 4. Why the ensemble did not beat DeBERTa alone

On Dataset 1 the ensemble scores 0.9925 against DeBERTa's 0.9933. It is slightly worse.

### Error analysis, Dataset 2 test set (1,200 rows)

| | count |
|---|---|
| BERT wrong | 15 |
| DeBERTa wrong | 1 |
| **Both wrong** | **0** |
| Only BERT wrong | 15 |
| Only DeBERTa wrong | 1 |
| The two models disagree | 16 (1.33%) |

The "both wrong = 0" line is remarkable: **their mistakes do not overlap at all.** Condition 2
from Section 1 is satisfied perfectly. In principle a flawless combiner would score 1200/1200,
which is an oracle accuracy of 1.0000.

The problem is condition 1. BERT makes fifteen times as many errors as DeBERTa. Any weight large
enough to let BERT fix DeBERTa's single error is also large enough to let BERT's fifteen errors
leak through. Simple averaging cannot separate the two cases.

### A second reason: BERT is unstable

BERT is not just weaker, it is inconsistent, which makes it a poor ensemble member:

| | spread across seeds 42/123/456 (Dataset 1) | reproduced on retraining? |
|---|---|---|
| BERT | 0.0267 | no, up to 0.0167 F1 lower |
| DeBERTa | 0.0033 | yes, exactly, both datasets |

Averaging a stable strong model with an unstable weak one imports the instability without
gaining accuracy.

---

## 5. Techniques we tested against ours

All selection done on validation, all numbers reported on test.

| Method | Dataset 1 | Dataset 2 |
|---|---|---|
| Stacking, logistic-regression meta-learner, all 5 models | **0.9950** | 0.9992 |
| Hard majority vote (BERT + DeBERTa + SVM) | **0.9950** | 0.9975 |
| DeBERTa 3-seed average | 0.9958 | 0.9983 |
| DeBERTa top-5 configuration average | 0.9958 | 0.9975 |
| DeBERTa alone | 0.9933 | **0.9992** |
| Soft vote, equal weights (BERT + DeBERTa) | 0.9925 | 0.9992 |
| **Weighted soft vote — what we shipped** | 0.9925 | 0.9992 |
| Stacking (BERT + DeBERTa only) | 0.9925 | 0.9992 |
| Confidence routing | 0.9917 | 0.9992 |
| Soft vote, all 5 models equal weights | 0.9900 | 0.9783 |
| Hard majority vote, all 5 models | 0.9892 | 0.9700 |
| BERT alone | 0.9908 | 0.9875 |
| Oracle upper bound (not achievable) | 0.9975 | 1.0000 |

Two lessons from this table:

- **A meta-learner beats naive averaging when members are unequal.** Stacking over all five
  models reaches 0.9950 on Dataset 1, because the meta-learner can *learn* to distrust Naive
  Bayes. Equal-weight voting over the same five models drops to 0.9900 on Dataset 1 and 0.9783 on
  Dataset 2, because it lets Naive Bayes outvote DeBERTa.
- **Same-architecture averaging removes run-to-run noise.** Averaging three DeBERTa seeds reaches
  0.9958 on Dataset 1. Since we measured run-to-run variance as the dominant noise source, the
  technique that targets that noise is the one that helps.

Caveats to state if this table goes in the report: the stacking meta-learner is fitted on only
480 validation rows, so it may be overfitting; the Dataset 1 gain of 0.0025 is three test rows;
and the classical models here were refitted on the 4,320-row training split so their individual
scores differ slightly from the main table, which used the full 4,800.

---

## 6. Common ensemble techniques, for reference

### Averaging and voting, no meta-learner
- **Hard voting** — majority of predicted labels. Needs an odd number of members.
- **Soft voting** — average of predicted probabilities.
- **Weighted soft voting** — as above with tuned weights. What we used.
- **Rank averaging** — average the rank rather than the probability. Robust to miscalibration.

### Resampling-based
- **Bagging** — members trained on bootstrap samples. Random Forest is the classic example.
- **Pasting** — the same idea, sampling without replacement.
- **Random subspace** — members see different feature subsets.

### Sequential
- **Boosting** — AdaBoost, Gradient Boosting, XGBoost, LightGBM. Each member corrects the errors
  of the one before it.

### Meta-learning
- **Stacking** — a meta-learner trained on the base models' outputs, ideally using out-of-fold
  predictions.
- **Blending** — stacking with a single holdout split instead of k-fold. Strictly, this is what
  we did when we used the validation split.

### Specific to deep learning
- **Multi-seed ensemble** — same architecture, different random seeds.
- **Snapshot or checkpoint ensemble** — combine predictions from several epochs of one run.
- **Weight averaging** — SWA or "model soups" average the *weights* rather than the predictions,
  giving one model at single-model inference cost.
- **Monte Carlo dropout and test-time augmentation** — several stochastic forward passes.
- **Knowledge distillation** — train one student to imitate the ensemble, keeping accuracy at a
  fraction of the cost.

### Selection-based
- **Mixture of experts** — a gating network routes each input to a specialist.
- **Cascading or confidence routing** — use the cheap model unless its confidence is low.
- **Dynamic classifier selection** — choose whichever member performs best near each test point.

### Which do not fit this project
Bagging and boosting assume cheap, high-variance base learners such as decision trees. Each
member here is a full transformer fine-tune costing minutes of GPU time, and boosting's
sequential dependency prevents parallel training. They are the right tools for the wrong model
class.

---

## 7. What to write in the report

State plainly:

1. The ensemble is a validation-weighted soft vote of the best BERT and the best DeBERTa.
2. It did **not** beat the best single model. On Dataset 1 it scores 0.9925 against DeBERTa's
   0.9933; on Dataset 2 it ties at 0.9992.
3. The reason is unequal member strength, not a coding error. BERT makes fifteen times more
   errors than DeBERTa on Dataset 2, and BERT is additionally unstable across seeds.
4. The Dataset 2 weight of 0.00 is an arbitrary tie-break within a plateau, not a rejection of
   BERT.
5. Alternatives were evaluated. Stacking over all five models reaches 0.9950 and multi-seed
   DeBERTa averaging reaches 0.9958 on Dataset 1, both above the shipped ensemble.

A negative result reported with its mechanism is worth more than a positive result reported
without one. The interesting finding here is not that the ensemble won, because it did not. It is
that the two models' errors are perfectly disjoint yet simple averaging still cannot exploit
that, because one member is far stronger than the other.
