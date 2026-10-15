#!/usr/bin/env python
"""Figure 4 on the published page, with the early-stim sensory prior swapped in.

  a  overall table (region, prior_choice, type_choice, prior_stim, prior_80ms, type_stim)
  b  header strip: prior_80ms FDR@0.01 ∩ type_stim sensory (stim / stim early)
  c  one example region trajectory and the cell-weighted sensory combined,
     the plots already in alyx manifold/figs/earlystim80_sensory_prior/

The integrator / choice-motor example grid and the schematics stay crops of
the original PDF. Nothing under manifold/res is rewritten.

  conda activate iblenv   # outside the sandbox
  python scripts/paper_figs/fig4/fig4_plot.py
  python scripts/paper_figs/fig4/fig4_plot.py --example VISam
"""
import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import fig4_common as fc

Image.MAX_IMAGE_PIXELS = None
plt.rcParams.update({'font.family': 'Helvetica', 'svg.fonttype': 'none'})

# Boxes are fractions of the page, y from the top: (x0, y0, x1, y1).
A_CLEAR = (0.055, 0.010, 0.995, 0.148)   # old strip, bracket, and circuit remnant
A_BOX = (0.012, 0.028, 0.988, 0.145)
B_CLEAR = (0.08, 0.148, 0.92, 0.288)
B_BOX = (0.10, 0.158, 0.90, 0.282)
C_CLEAR = (0.325, 0.792, 0.998, 0.995)   # old averages, including their axis labels
C_EXAMPLE = (0.335, 0.800, 0.655, 0.978)
C_OVERALL = (0.665, 0.800, 0.992, 0.978)
RENDER_DPI = 140


def region_order():
    order = [l.strip() for l in fc.REGION_ORDER.read_text().splitlines() if l.strip()]
    present = {r['region'] for r in csv.DictReader(fc.SC_TABLE.open())}
    regs = [r for r in order if r in present]
    if len(regs) != 208:
        raise RuntimeError(f'expected 208 table regions, got {len(regs)}')
    return regs


def sensory_prior_overlap(regs):
    """prior_80ms FDR ≤ α ∩ type_stim sensory, in the table's region order."""
    sens = set()
    for row in csv.DictReader(fc.SC_TABLE.open()):
        raw = row['sc_duringstim_regtype']
        if raw in ('', 'nan', 'None'):
            continue
        if float(raw) in fc.SENSORY_REGTYPES:
            sens.add(row['region'])
    early = np.load(fc.EARLYSTIM_NPY, allow_pickle=True).item()
    hit = set()
    detail = {}
    for reg, rec in early.items():
        p = float(rec.get('p_mean_c', 1.0) or 1.0)
        if p <= fc.ALPHA and reg in sens:
            hit.add(reg)
            detail[reg] = {
                'p_mean_c': p,
                'amp_euc': float(rec.get('amp_euc') or 0.0),
                'p_offset': float(rec.get('p_offset') or 1.0),
                'p_gain': float(rec.get('p_gain') or 1.0),
            }
    ordered = [r for r in regs if r in hit]
    missing = sorted(hit - set(regs))
    if missing:
        raise RuntimeError(f'overlap regions missing from the table order: {missing}')
    return ordered, detail


def table_columns(n_reg):
    """x ranges of the 208 region columns on the overall-table png."""
    im = np.asarray(Image.open(fc.TABLE_PNG).convert('RGB'))
    data = im[200:1200]
    white = (data > 245).all(axis=2)
    on = white.mean(axis=0) < 0.85
    runs, i = [], 0
    while i < on.size:
        if on[i]:
            j = i
            while j < on.size and on[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    cols = runs[1:]
    if len(cols) != n_reg:
        raise RuntimeError(f'table columns {len(cols)} != regions {n_reg}')
    return im, cols


def crop_overall_table(im):
    """Drop the black bars above and below the strip."""
    return Image.fromarray(im[fc.TABLE_Y0:fc.TABLE_Y1])


def cell_color(im, cols, col, y0, y1):
    x0, x1 = cols[col]
    patch = im[y0 + 8:y1 - 8, x0 + 6:x1 - 6]
    return tuple(int(v) for v in np.median(patch.reshape(-1, 3), axis=0))


def draw_overlap_table(im, cols, regs, overlap):
    """Compact strip, same cell colors as the overall table, overlap columns only.

    Sized to the header band so a later 1:1 paste does not spill onto the
    example trajectories under it.
    """
    labels = [name for name, _, _ in fc.TABLE_ROWS]
    n_rows = len(labels)
    dpi = 120
    fig = plt.figure(figsize=(11.2, 1.85), dpi=dpi)
    ax = fig.add_axes([0.13, 0.04, 0.86, 0.92])
    ax.set_xlim(0, len(overlap))
    ax.set_ylim(0, n_rows)
    ax.axis('off')
    idx = {r: i for i, r in enumerate(regs)}
    for j, reg in enumerate(overlap):
        for i, (name, y0, y1) in enumerate(fc.TABLE_ROWS):
            rgb = np.array(cell_color(im, cols, idx[reg], y0, y1)) / 255
            ax.add_patch(plt.Rectangle((j, n_rows - 1 - i), 1, 1,
                                       facecolor=rgb, edgecolor='white', lw=0.4,
                                       clip_on=True))
            if name == 'region':
                lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                ax.text(j + 0.5, n_rows - 1 - i + 0.5, reg, rotation=90,
                        ha='center', va='center', fontsize=6, clip_on=True,
                        color='white' if lum < 0.45 else 'black')
    for i, name in enumerate(labels):
        if name == 'region':
            continue
        ax.text(-0.12, n_rows - 1 - i + 0.5, name, ha='right', va='center',
                fontsize=7, clip_on=False)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(buf)


def on_white(im):
    """Flatten an RGBA render onto white and trim the transparent margin."""
    im = im.convert('RGBA')
    arr = np.asarray(im)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0:
        return Image.new('RGB', im.size, 'white')
    pad = 8
    x0, x1 = max(0, xs.min() - pad), min(arr.shape[1], xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(arr.shape[0], ys.max() + pad)
    crop = arr[y0:y1, x0:x1]
    out = np.full(crop.shape, 255, dtype=np.uint8)
    a = crop[:, :, 3:4] / 255.0
    out[:, :, :3] = (crop[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    out[:, :, 3] = 255
    return Image.fromarray(out).convert('RGB')


def traj_image(region):
    if region == 'combined':
        path = fc.TRAJ_DIR / (
            'combined_sensory_act_block_duringstim_unsplit_p_mean_c_dist_avg.png'
        )
    else:
        path = fc.TRAJ_DIR / f'{region}_act_block_duringstim_unsplit_p_mean_c_dist.png'
    if not path.exists():
        raise FileNotFoundError(path)
    return on_white(Image.open(path))


def render_original(dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['gs', '-q', '-dNOPAUSE', '-dBATCH', '-sDEVICE=png16m', f'-r{RENDER_DPI}',
         f'-sOutputFile={dest}', str(fc.ORIGINAL_PDF)],
        check=True,
    )
    return Image.open(dest).convert('RGB')


def paste_fit(page, image, box):
    """Paste ``image`` inside a top-origin fractional box, aspect preserved."""
    W, H = page.size
    x0, y0, x1, y1 = box
    bw, bh = max(1, int((x1 - x0) * W)), max(1, int((y1 - y0) * H))
    ox, oy = int(x0 * W), int(y0 * H)
    scale = min(bw / image.size[0], bh / image.size[1])
    nw, nh = max(1, int(image.size[0] * scale)), max(1, int(image.size[1] * scale))
    image = image.resize((nw, nh), Image.Resampling.LANCZOS)
    page.paste(image, (ox + (bw - nw) // 2, oy + (bh - nh) // 2))


def white_out(page, box):
    W, H = page.size
    x0, y0, x1, y1 = box
    page.paste(
        Image.new('RGB', (int((x1 - x0) * W), int((y1 - y0) * H)), 'white'),
        (int(x0 * W), int(y0 * H)),
    )


def save_page(page, out_dir):
    w_in, h_in = fc.PAGE_WH[0] / 72, fc.PAGE_WH[1] / 72
    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(page)
    ax.axis('off')
    fig.savefig(out_dir / 'figure4.pdf')
    fig.savefig(out_dir / 'figure4.png', dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--example', default=fc.EXAMPLE,
                    help='sensory overlap region for the panel-c example trajectory')
    args = ap.parse_args()

    out = fc.FIG_DIR
    (out / 'original').mkdir(parents=True, exist_ok=True)
    (out / 'panels').mkdir(parents=True, exist_ok=True)
    shutil.copy2(fc.ORIGINAL_PDF, out / 'original' / 'figure4.pdf')

    regs = region_order()
    overlap, detail = sensory_prior_overlap(regs)
    if args.example not in overlap:
        raise SystemExit(f'{args.example} is not in the sensory ∩ prior_80ms set {overlap}')

    im, cols = table_columns(len(regs))
    table = crop_overall_table(im)
    small = draw_overlap_table(im, cols, regs, overlap)
    example = traj_image(args.example)
    overall = traj_image('combined')
    table.save(out / 'panels' / 'panel_a_table.png')
    small.save(out / 'panels' / 'panel_b_overlap.png')
    example.save(out / 'panels' / f'panel_c_example_{args.example}.png')
    overall.save(out / 'panels' / 'panel_c_overall.png')

    page = render_original(out / 'original' / 'figure4_page.png')
    white_out(page, A_CLEAR)
    white_out(page, (0.78, 0.0, 1.0, 0.045))  # top of the old prior-node
    white_out(page, B_CLEAR)
    white_out(page, C_CLEAR)
    paste_fit(page, table, A_BOX)
    paste_fit(page, small, B_BOX)
    paste_fit(page, example, C_EXAMPLE)
    paste_fit(page, overall, C_OVERALL)
    save_page(page, out)

    summary = {
        'example': args.example,
        'overlap': overlap,
        'n_overlap': len(overlap),
        'detail': {r: detail[r] for r in overlap},
        'sources': {
            'table': str(fc.TABLE_PNG),
            'earlystim': str(fc.EARLYSTIM_NPY),
            'sc_table': str(fc.SC_TABLE),
            'trajectories': str(fc.TRAJ_DIR),
            'original': str(fc.ORIGINAL_PDF),
        },
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'overlap ({len(overlap)}): {", ".join(overlap)}')
    print(f'example: {args.example}')
    print(f'wrote {out / "figure4.pdf"}')


if __name__ == '__main__':
    main()
