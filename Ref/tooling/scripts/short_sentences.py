#!/usr/bin/env python3
"""Print the short-sentence tail of our paper, which is what the mean and median
words-per-sentence rows are made of. Read-only: measures, never edits.

  python3 scripts/short_sentences.py [--under 12] [--paper 00_OURS_iccit6]
"""
from __future__ import annotations
import argparse, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import src.markdown_corpus as mc  # noqa: E402

CAPTION_PROBES = ("bold marks", "eight configurations", "headline result", "decomposition on each",
                  "deployed checkpoints, never", "measurement pipeline")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--under", type=int, default=12)
    ap.add_argument("--paper", default="00_OURS_iccit6")
    ap.add_argument("--md-dir", default="../md")
    a = ap.parse_args()

    path = pathlib.Path(a.md_dir) / a.paper / f"{a.paper}.md"
    p = mc.read_paper(path)
    lines = p.raw.splitlines()

    total_short = 0
    for s in p.sections:
        cleaned = mc.finish_clean(mc.strip_noise(s.text))
        try:
            ss = mc.split_sentences(cleaned, [], s.title)
        except mc.CannotMeasure:
            continue
        short = [x for x in ss if len(mc.WORD.findall(x)) < a.under]
        total_short += len(short)
        print(f"\n== {s.numeral or '-'} {s.title}  ({len(ss)} sentences, {len(short)} under {a.under}w)")
        for x in short:
            print(f"   {len(mc.WORD.findall(x)):>3}w | {x[:96]}")

    body = "\n".join(lines[p.body_start_line - 1: p.refs_line - 1])
    ss = mc.split_sentences(mc.finish_clean(mc.strip_noise(body)), [], "body")
    leak = [x for x in ss if any(q in x.lower() for q in CAPTION_PROBES)]
    print(f"\nbody sentences {len(ss)}, under {a.under}w {total_short}, "
          f"caption text counted as prose {len(leak)}")
    for x in leak:
        print(f"   LEAK {len(mc.WORD.findall(x)):>3}w | {x[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
