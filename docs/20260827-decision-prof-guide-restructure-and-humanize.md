---
id: 20260827-decision-prof-guide-restructure-and-humanize
title: "iccit6 cut restructured to the professor's IEEE guide, humanized, six overclaims fixed (PR #12)"
type: decision
tags: [iccit, editorial, structure, humanize, overclaim, benchmarks, surface-content]
status: open (PR #12 not yet merged)
date: 2026-08-27
stakeholders: [sakib2588, course professor]
---

# Prof-guide restructure and humanize pass on `paper/iccit6/`, 2026-08-27

## Why

The course professor's writing guide,
`Semester 10/Research Paper skills /writing-ieee-security-paper.md`, is now the house structure
for every conference paper, with the owner's two additions: sentences of normal human length and
no overclaiming. The guide's research-type matrix has no exact row for this paper; it was treated
as Analysis/Measurement, a finding about benchmarks in which the models are instruments, and the
paper now says so.

## What changed (PR #12, base `paper/iccit6-round2-2026-08-26`, commits `263a3d1` and `6c27fdf`)

Files: `paper/iccit6/sections/*.tex`, `main.pdf`, `main.bbl`, `.build-manifest`. The base branch
was local-only and is now pushed.

- Introduction is five paragraphs: a new "three lines of prior work stop short" paragraph, three
  contribution bullets with `Section~\ref` links, an organisation paragraph.
- Related Work cites by number only (zero author names), is grouped by theme, and closes with
  "The gap. ... Unlike [4] ... and [7], [8] ..., this paper ...".
- Methods gains a Problem Setting subsection. The long version's labelled pipeline diagram
  (`figures/fig_pipeline.pdf`, previously unused in this cut) is wired in as Fig. 1; no new script.
- Table II now bolds the lower-error arm on every row (it was only some).
- Conclusion ends with three specific extensions.
- References: 25 before, 25 after. Nothing added, nothing needed.

## Overclaims fixed

1. "BERT provably cannot represent it" -> "BERT cannot represent it". Three of three pairs tested
   is evidence, not a proof.
2. "Surface separability there tracks collection provenance more closely than model identity" ->
   "On this evidence, ... appears to track ...". Rests on two pairs below the 200-row floor.
3. "The diffuse stylistic signal DAIGT V2's surface arm reads cannot [be cleaned]" -> "has no
   equivalent single cleaning step in our experiments".
4. "neither is standard practice" -> "neither is routinely reported".
5. "it runs in minutes" dropped. No timing is reported anywhere.
6. "[Tian et al.] go further than they are usually credited for" dropped. Opinion.

## Not done, and why

- "Alarming statistic" hook: none verified; the existing cited 97% and 99% accuracy figures stay as
  the hook.
- "At least two published baselines in a table": the paper deliberately makes no detector
  comparison. The one published detector is reported in prose with its contamination caveat.

## Verification

Six pages. `check.sh` passes every gate: A4, `\blindtrue`, 0 undefined references, 0 overfull
boxes, 0 bibtex warnings, banned glyphs 0 in source and rendered PDF, prose colons 0, semicolons 0.
Sentence length (mean / max): intro 15.1/31, methods 16.4/34, results 17.6/35, related 17.8/36,
discussion 18.1/36; the two at 36 are the guide-form bridge sentence and a three-item future-work
list. Numeric diff new-vs-old: `0.6` only (a figure width). Every result literal unchanged.

## Owner to do, and things to know

- Merge PR #12.
- Both double-column figures are at 0.6 textwidth to hold six pages; Fig. 1 text is small. 0.7 is
  more legible at a cost of about eight lines elsewhere.
- `experiments/paper_scale/logs/guardian_nlp.log` was being modified by a running process during
  the pass; left untouched and uncommitted.
- The long version in `paper/iccit/` (13 pages) was not touched and is now stylistically out of
  sync with this cut.
