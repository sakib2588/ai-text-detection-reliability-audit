# Conference-paper corpus study, 11 six-page papers against our ICCIT draft

**How to read the columns.** *ours* is `paper/iccit6/main.pdf`, our six-page ICCIT submission. *min*, *median*, *max* are the band over the eleven reference papers; our paper is never a member of its own band, so every band here is n=11. *pct* is our percentile inside that band, printed on every row because a min..max band over eleven papers passes at the extreme. A flag of OUT-HIGH or OUT-LOW means outside the band entirely.

**What the corpus is, and what it is not.** Eleven papers of six pages or fewer, pulled from four of our other project reference folders (SAR drone, ML paper, XAI NIDS). They are not a sample of ICCIT and not a sample of any single venue: ten of the eleven were read from arXiv-hosted copies, and only one (Okabe, Interspeech 2018) carries a printed conference venue line. Topic spread is wide. Read a band as "the norm among these eleven short papers", never as "the ICCIT norm".

**What was changed in the instrument.** The measurement code is `scripts/78_corpus_style_study.py` and `src/markdown_corpus.py`, copied from the ML_Paper repository, where it was written for twelve-to-twenty page NDSS papers. Four changes were made and no measurement rule beyond them. Three are recalibrations for six-page papers, namely the body-word sanity window from 5,000..20,000 down to 1,500..20,000, the minimum detected sections from five to four, and the references-heading pattern now also accepts arabic numbering (`6. References`, the Interspeech style). Image markdown is stripped before measurement because this corpus was converted with figure extraction on, unlike the NDSS bundle. The fourth is a parsing fix found by reading the output rather than the page count, since the caption pattern allowed one page-anchor span before the label where marker sometimes emits two, which silently cost a figure. It now accepts any number of anchors, and every count in this report, corpus and ours alike, was recomputed under that rule.


## 1. The corpus roster

| paper                                        | pp | body words | sent | sec | tab | fig | med w/sent |
|----------------------------------------------|----|------------|------|-----|-----|-----|------------|
| OURS 00_OURS_iccit6                          | 6  | 3470       | 167  | 7   | 5   | 3   | 20         |
| AlvarezMelis2018_robustness-interpretability | 6  | 2464       | 126  | 4   | 0   | 8   | 18.0       |
| Anon2025_uav-vlrr-nmpc                       | 6  | 2654       | 140  | 8   | 3   | 7   | 19.0       |
| Batool2026_humandiffusion-sar                | 5  | 1873       | 102  | 6   | 0   | 4   | 18.0       |
| Boddu2025_quantized-yolov4tiny-pi            | 6  | 2874       | 158  | 6   | 1   | 4   | 18.0       |
| Manzini2023b_differentiable-boustrophedon    | 6  | 3671       | 181  | 9   | 1   | 5   | 18         |
| Yang2022_rgb-ir-fusion-drone                 | 6  | 3341       | 137  | 6   | 6   | 7   | 21         |
| ref17_gueriani_resnet_bigru_arxiv2026        | 6  | 2744       | 146  | 6   | 3   | 4   | 18.0       |
| ref18_gueriani_vit_bilstm_arxiv2026          | 6  | 2292       | 113  | 5   | 4   | 4   | 20         |
| ref28_okabe_attentivestats_interspeech2018   | 5  | 2563       | 128  | 5   | 3   | 2   | 17.5       |
| ref32_yang_lnetskd_arxiv2023                 | 6  | 2930       | 158  | 5   | 4   | 4   | 18.0       |
| ref34_debicha_advtrain_arxiv2021             | 5  | 3022       | 130  | 5   | 1   | 4   | 20.0       |

**Body words means first numbered heading to REFERENCES.** Front matter and the bibliography are outside it, so a body-word figure is not a page count in words.


## 2. Paper-level metrics, ours against the eleven-paper band

| metric                   | ours | min  | median | max  | pct | flag |
|--------------------------|------|------|--------|------|-----|------|
| body words               | 3470 | 1873 | 2744   | 3671 | 91  | high |
| sentences                | 167  | 102  | 137    | 181  | 91  | high |
| mean words / sentence    | 20.8 | 18.2 | 19.6   | 24.4 | 82  | -    |
| median words / sentence  | 20   | 17.5 | 18.0   | 21   | 91  | high |
| % sentences over 35w     | 11.4 | 1.9  | 8.0    | 16.2 | 73  | -    |
| longest sentence         | 52   | 38   | 53     | 103  | 45  | -    |
| paragraphs               | 61   | 37   | 49     | 94   | 73  | -    |
| mean words / paragraph   | 56.9 | 30.6 | 50.6   | 90.3 | 64  | -    |
| citations per 1k         | 12.4 | 0.0  | 10.6   | 18.2 | 73  | -    |
| first person per 1k      | 4.3  | 0.0  | 3.0    | 16.2 | 55  | -    |
| passive per 1k [proxy]   | 9.2  | 8.5  | 15.7   | 30.0 | 9   | low  |
| % sentences past [proxy] | 6.0  | 0.6  | 12.3   | 30.4 | 27  | -    |
| % no tense anchor        | 37.1 | 17.7 | 38.4   | 54.0 | 36  | -    |
| sections                 | 7    | 4    | 6      | 9    | 82  | -    |
| sections (numbered)      | 7    | 4    | 5      | 8    | 91  | high |
| tables (total)           | 5    | 0    | 3      | 6    | 91  | high |
| figures (total)          | 3    | 2    | 4      | 8    | 9   | low  |


## 3. Per-section word budget

| role         | n papers | ours   | min | median | max  | verdict                                            |
|--------------|----------|--------|-----|--------|------|----------------------------------------------------|
| introduction | 11       | 514    | 301 | 493    | 671  | in band, median 493                                |
| background   | 1        | ABSENT | 818 | 818    | 818  | no band, only 1 of 11 papers                       |
| related_work | 5        | 459    | 342 | 372    | 725  | in band, median 372                                |
| threat_model | 2        | ABSENT | 287 | 822    | 1357 | no band, only 2 of 11 papers                       |
| method       | 4        | 771    | 532 | 743    | 973  | in band, median 743                                |
| evaluation   | 10       | 1202   | 462 | 874    | 1575 | in band, median 874                                |
| discussion   | 2        | 414    | 405 | 503    | 601  | no band, only 2 of 11 papers, ours inside the pair |
| conclusion   | 9        | 110    | 74  | 121    | 325  | in band, median 121                                |

**A role carried by fewer than three of the eleven papers has no band.** Background appears in one reference paper and a threat model in two, so neither is a convention this corpus can be said to have, and their absence from our draft is not a violation. Every role that can be banded, and every role our draft carries, is inside its band.

**The conclusion is now its own section.** The draft previously ran one Discussion and Conclusion section, so every word of it was filed under discussion and the conclusion role read ABSENT against nine of eleven reference papers that carry a standalone one. Splitting the two puts both inside their bands and raises the section count to six, which is the corpus median.

**Roles are classified from section titles**, by the same regexes for every paper, ours included. A paper whose method section is titled after its artifact rather than after its function classifies elsewhere or nowhere, which is why the n column varies by role and why an ABSENT is a statement about titles, not about content.


## 4. Structural conventions

| convention                            | corpus | ours |
|---------------------------------------|--------|------|
| Related Work as its own section       | 5/11   | yes  |
| roadmap paragraph in the introduction | 2/11   | yes  |
| any back matter before REFERENCES     | 5/11   | no   |
| marker page anchors present           | 9/11   | yes  |


## 5. Sentence-length distribution

| paper                                    | 1-10 | 11-20 | 21-30 | 31-40 | 41-50 | 51+ |
|------------------------------------------|------|-------|-------|-------|-------|-----|
| OURS 00_OURS_iccit6                      | 20%  | 34%   | 27%   | 14%   | 5%    | 1%  |
| AlvarezMelis2018_robustness-interpretabi | 19%  | 43%   | 21%   | 13%   | 2%    | 2%  |
| Anon2025_uav-vlrr-nmpc                   | 19%  | 41%   | 30%   | 8%    | 1%    | 1%  |
| Batool2026_humandiffusion-sar            | 20%  | 40%   | 31%   | 8%    | 1%    | 0%  |
| Boddu2025_quantized-yolov4tiny-pi        | 22%  | 40%   | 28%   | 10%   | 0%    | 0%  |
| Manzini2023b_differentiable-boustrophedo | 12%  | 49%   | 24%   | 12%   | 3%    | 0%  |
| Yang2022_rgb-ir-fusion-drone             | 10%  | 37%   | 31%   | 11%   | 3%    | 8%  |
| ref17_gueriani_resnet_bigru_arxiv2026    | 14%  | 47%   | 34%   | 4%    | 1%    | 1%  |
| ref18_gueriani_vit_bilstm_arxiv2026      | 18%  | 35%   | 35%   | 8%    | 4%    | 1%  |
| ref28_okabe_attentivestats_interspeech20 | 20%  | 42%   | 22%   | 10%   | 5%    | 2%  |
| ref32_yang_lnetskd_arxiv2023             | 15%  | 45%   | 32%   | 6%    | 1%    | 0%  |
| ref34_debicha_advtrain_arxiv2021         | 11%  | 42%   | 29%   | 7%    | 5%    | 6%  |

**Percentages of that paper's own sentences.** The short-sentence column is where our draft separates from the corpus, and it is the same fact the mean and median words-per-sentence rows in Section 2 report.


## 6. Read before quoting any number above

**Figure and table counts are caption counts.** They come from markdown produced by marker, so a caption lost in conversion is an undercount with no error raised. Numeral gaps are the only detector and are reported per paper in the manifest.

**The tense and passive columns are regex proxies, not parses.** No part-of-speech tagger was used. Past passive increments both the past and the passive counter, so the two are not independent and must never be summed. Between 18 and 54 percent of sentences carry no tense anchor at all in this corpus; read every tense figure against that share.

**Calibration was not done.** The proxies have not been hand-validated against labelled sentences here any more than they were in the NDSS study.

**Provenance.** Manifest `report/corpus_style_study.json`, normalisation v1, 17 ordered cleaning steps. Every source markdown file is sha256-pinned in the manifest, as is the PDF it was converted from. Regenerate with `python3 scripts/corpus_style_study.py --md-dir ../md --write-manifest` then `python3 scripts/build_report.py`.
