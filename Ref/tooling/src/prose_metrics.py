r"""Sentence splitting and prose metrics, shared by every prose measurement here.

Hoisted VERBATIM from scripts/73_prose_style_audit.py on 2026-08-10 (CannotMeasure,
WORD, sentences, metrics). Script 73 imports them back and re-exports, so anything
loading that file by path is unaffected. Same pattern as the two hoists on 2026-08-09:
build_ladder_trigger out of scripts/28 (ec43b55) and the MDE machinery out of
scripts/58 (1acce84), both for the same reason -- a digit-prefixed module cannot be
imported with a normal `import` statement.

The mover here is scripts/78_corpus_style_study.py, which measures the marker-converted
markdown corpus. Why hoist rather than copy: script 73's splitter produced the numbers
frozen in notes/ndss_style_bands.json. A copy that drifted by one character would make
the two manifests incommensurable while both were still labelled "mean words per
sentence" -- exactly the failure the _normalisation_version guard exists to catch, but
one level below where that guard can see it. Script 78's cross-check against script 73's
committed PDF-path numbers is only meaningful if the splitter is literally this object.

KNOWN DEFECT, deliberately preserved. The single-initial guard
`\b([A-Z])\.\s(?=[A-Z])` misfires on constructions like "Section V-A. We test...",
merging two sentences into one. It is real, and it inflates the odd sentence length.
It is NOT fixed here because notes/ndss_style_bands.json already embeds this behaviour;
fixing it inside a hoist would silently invalidate a committed band while the commit
claimed to be behaviour-preserving. Fix it as its own commit that bumps
_normalisation_version on every manifest that depends on it.
"""
from __future__ import annotations

import re
import statistics


class CannotMeasure(Exception):
    pass


WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def sentences(text: str) -> list[str]:
    t = re.sub(r"(\d)\.(\d)", r"\1<D>\2", text)
    t = re.sub(r"\bet al\.", "et al<D>", t)
    t = re.sub(r"\b(Fig|Eq|No|cf|vs|Sec|Tab|St|Dr|approx|Refs?|Inc|Ltd)\.",
               lambda m: m.group(1) + "<D>", t)
    t = re.sub(r"\b(e\.g|i\.e)\.", lambda m: m.group(1).replace(".", "<D>") + "<D>", t)
    t = re.sub(r"\b([A-Z])\.\s(?=[A-Z])", r"\1<D> ", t)
    out = []
    for p in re.split(r"(?<=[.!?])\s+(?=[A-Z\"(\[])", t):
        p = p.replace("<D>", ".").strip()
        if len(re.findall(r"[A-Za-z]", p)) > 3:
            out.append(p)
    return out


def metrics(text: str, over: int = 35) -> dict:
    ss = sentences(text)
    if not ss:
        raise CannotMeasure("no sentences found after normalisation")
    lens = [len(WORD.findall(s)) for s in ss]
    words = sum(lens)
    fp = len(re.findall(r"\b(we|our|ours|us)\b", text, re.I))
    return {
        "words": words,
        "sentences": len(ss),
        "mean_words_per_sentence": round(statistics.mean(lens), 1),
        "median_words_per_sentence": round(statistics.median(lens), 1),
        f"pct_over_{over}w": round(100 * sum(1 for x in lens if x > over) / len(lens), 1),
        f"n_over_{over}w": sum(1 for x in lens if x > over),
        "first_person_per_1k": round(1000 * fp / words, 1) if words else 0.0,
    }
