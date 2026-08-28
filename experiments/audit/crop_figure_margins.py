"""Crop the white border out of the paper's figure PDFs.

Every figure exported from the analysis notebook carries a white background
rectangle that covers the whole canvas, so Ghostscript's bbox device reports the
page itself as ink and pdfcrop has nothing to trim. The border is real even so,
between one and six percent of each figure's height, and at full column width it
reads as a gap between the figure and the text around it.

This measures the true extent of the drawn content by rasterising the page and
finding the non-white pixels, then rewrites the PDF with the media box set to
that extent plus a small uniform pad. The content stays vector, nothing is
resampled, and the operation is idempotent, since a second run finds no margin
left to remove.

Run from anywhere. Pass --check to report the margins without rewriting.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

FINAL = Path(__file__).resolve().parents[2]
FIGDIR = FINAL / 'paper' / 'iccit_profstyle'

RASTER_DPI = 150
WHITE = 248          # a pixel lighter than this counts as background
PAD_PT = 1.5         # uniform pad kept around the ink, in PDF points


def page_size(pdf):
    out = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith('Page size:'):
            parts = line.split()
            return float(parts[2]), float(parts[4])
    raise RuntimeError(f'no page size for {pdf}')


def ink_box(pdf):
    """Bounding box of the drawn content in PDF points, origin bottom-left."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(['pdftoppm', '-r', str(RASTER_DPI), '-gray', '-png',
                        '-singlefile', str(pdf), f'{d}/p'], check=True)
        a = np.asarray(Image.open(f'{d}/p.png').convert('L'))
    ink = a < WHITE
    if not ink.any():
        return None
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    h_px, w_px = a.shape
    w_pt, h_pt = page_size(pdf)
    sx, sy = w_pt / w_px, h_pt / h_px
    x0 = cols[0] * sx
    x1 = (cols[-1] + 1) * sx
    y0 = h_pt - (rows[-1] + 1) * sy
    y1 = h_pt - rows[0] * sy
    return x0, y0, x1, y1, w_pt, h_pt


def crop(pdf, dry_run=False):
    box = ink_box(pdf)
    if box is None:
        return None
    x0, y0, x1, y1, w_pt, h_pt = box
    x0 = max(0.0, x0 - PAD_PT)
    y0 = max(0.0, y0 - PAD_PT)
    x1 = min(w_pt, x1 + PAD_PT)
    y1 = min(h_pt, y1 + PAD_PT)
    new_w, new_h = x1 - x0, y1 - y0
    saved_h = 1 - new_h / h_pt
    saved_w = 1 - new_w / w_pt
    if dry_run or (saved_h < 0.002 and saved_w < 0.002):
        return saved_w, saved_h, False
    out = pdf.with_suffix('.crop.pdf')
    cmd = ['gs', '-q', '-o', str(out), '-sDEVICE=pdfwrite',
           '-dFIXEDMEDIA', f'-dDEVICEWIDTHPOINTS={new_w:.4f}',
           f'-dDEVICEHEIGHTPOINTS={new_h:.4f}',
           '-c', f'<</PageOffset [{-x0:.4f} {-y0:.4f}]>> setpagedevice',
           '-f', str(pdf)]
    subprocess.run(cmd, check=True, capture_output=True)
    if out.stat().st_size < 1000:
        out.unlink(missing_ok=True)
        raise RuntimeError(f'ghostscript produced an empty crop for {pdf}')
    out.replace(pdf)   # same directory, so this is an atomic rename
    return saved_w, saved_h, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='report margins without rewriting anything')
    args = ap.parse_args()

    pdfs = sorted(p for p in FIGDIR.glob('*.pdf') if p.name != 'main.pdf')
    if not pdfs:
        sys.exit(f'no figures found in {FIGDIR}')
    total_h = []
    for p in pdfs:
        res = crop(p, dry_run=args.check)
        if res is None:
            print(f'{p.name:38s} blank, skipped')
            continue
        sw, sh, wrote = res
        total_h.append(sh)
        verb = 'cropped' if wrote else 'already tight'
        print(f'{p.name:38s} width -{sw:5.1%}  height -{sh:5.1%}  {verb}')
    if total_h:
        print(f'\nmean height removed {np.mean(total_h):.1%} over {len(total_h)} figures')


if __name__ == '__main__':
    main()
