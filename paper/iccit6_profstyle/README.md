# Six-page ICCIT submission cut

`main.pdf` here is 6 pages including references and passes every gate with the page cap
re-armed. The full-length manuscript stays in `../iccit/` at 13 pages and is the version
to develop.

Both build from the same figures and the same `bib/refs.bib`, and every number in this cut
was re-verified against `experiments/audit/*.json` after the trim, 38 of 38 matching.

## What was dropped to reach six pages

Recoverable in full from `../iccit/`.

- ROC curves and the area-against-F1 rank reversal
- The soft-vote ensemble
- The label-free perturbation control and the adversarial numbers
- The surface-feature correlation matrix and the SHAP beeswarm figure
- The per-generator DAIGT table, kept as one prose paragraph

The matched-text-budget refit and cross-corpus transfer were restored on 2026-08-26 as
prose, since the dashboard already carried their panels and the cut had shown the results
without describing them.

## What was kept, and why

The decomposition, its localisation to one sub-domain and one cue, the cleaning-kit
ablation, and the tokenisation control. Those four carry the paper's argument. The
consolidated dashboard survives as the single figure because it shows six results at once,
including several whose own sections were cut.
