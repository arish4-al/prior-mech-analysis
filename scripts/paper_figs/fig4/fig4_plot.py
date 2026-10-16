#!/usr/bin/env python
"""Figure 4 on the published page, redrawn (not pasted diagnostic rasters).

  a  overall table PNG (region, prior_choice, type_choice, prior_stim,
     prior_80ms, type_stim) — alyx meta, cropped
  b  one-row strip: prior_80ms FDR@0.01 ∩ type_stim sensory, plus LGd
  c  cell-weighted sensory combined, 0–80 ms; gain/offset schematic empty
  d–g  C / M examples and averages from current openalyx 4-split act-block
  h  offset / gain_s / gain_m strip from the same maps
  i  shuffle-subtracted C vs M (stim- and move-aligned)

Schematics in c, e, g are left blank. Nothing under manifold/res is rewritten.

  conda activate iblenv   # outside the sandbox
  python scripts/paper_figs/fig4/fig4_plot.py
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image
from statsmodels.stats.multitest import multipletests

import fig4_common as fc

Image.MAX_IMAGE_PIXELS = None
plt.rcParams.update({
    'font.family': 'Helvetica',
    'svg.fonttype': 'none',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.labelsize': 11,
    'pdf.fonttype': 42,
})

FIG_W, FIG_H = fc.PAGE_WH[0] / 72, fc.PAGE_WH[1] / 72
STIM_TF = 'act_block_duringstim'
CHOICE_TF = 'act_block_duringchoice'
UNSPLIT_TF = 'act_block_duringstim_unsplit'


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class Data:
    def __init__(self):
        self.af = fc.load_af(fc.OA_RES)
        self.sc = pd.read_csv(fc.SC_TABLE)
        self.order = [l.strip() for l in fc.REGION_ORDER.read_text().splitlines()
                      if l.strip()]
        present = set(self.sc['region'])
        self.regs = [r for r in self.order if r in present]
        self.sc = self.sc.set_index('region').loc[self.regs]
        self.beryl = {r: self.af.swanson_to_beryl_hex(r, self.af.br)
                      for r in self.regs}
        self.cmap_type = self.af.get_cmap_('sc_duringstim_regtype')
        self.cmap_prior = self.af.get_cmap_('act_block_duringstim')

        self.early = _load_dict(fc.EARLYSTIM_NPY)
        self.stim_d = _load_dict(_stem(self.af, STIM_TF, fc.OA_RES, False))
        self.choice_d = _load_dict(_stem(self.af, CHOICE_TF, fc.OA_RES, False))
        self._fill_choice_fdr()

        self.type_stim = self.sc['sc_duringstim_regtype']
        self.type_choice = self.sc['sc_duringchoice_regtype']
        self.sensory = set(self.type_stim.index[self.type_stim.isin(fc.SENSORY_REGTYPES)])
        self.integ_s = set(self.type_stim.index[self.type_stim == 0.5])
        self.integ_c = set(self.type_choice.index[self.type_choice == 0.5])
        self.move_s = set(self.type_stim.index[self.type_stim == 1.0])
        self.move_c = set(self.type_choice.index[self.type_choice == 1.0])
        self.prior80 = {r for r, rec in self.early.items()
                        if isinstance(rec, dict) and _p(rec, 'p_mean_c') <= fc.ALPHA}
        self.prior_stim = {r for r, rec in self.stim_d.items()
                           if isinstance(rec, dict) and _p(rec, 'p_mean_c') <= fc.ALPHA}
        self.prior_choice = {r for r, rec in self.choice_d.items()
                             if isinstance(rec, dict) and _p(rec, 'p_mean_c') <= fc.ALPHA}
        self.overlap = [r for r in self.regs if r in self.prior80 and r in self.sensory]
        self.c_strip = [r for r in self.regs if r in self.integ_s and r in self.prior_stim]
        self.m_strip = [r for r in self.regs
                        if (r in self.move_s or r in self.move_c)
                        and (r in self.prior_stim or r in self.prior_choice)]

        self._regde = {}
        self._nclus = {}

    def nclus(self, reg, timeframe=UNSPLIT_TF):
        if timeframe not in self._nclus:
            split = self.af.run_align[timeframe][0]
            res = fc.ALYX_RES if 'unsplit' in timeframe else fc.OA_RES
            d = _load_dict(res / f'{split}.npy')
            self._nclus[timeframe] = {
                r: int(rec.get('nclus') or 0)
                for r, rec in d.items() if isinstance(rec, dict)
            }
        return int(self._nclus[timeframe].get(reg, 0))

    def curves(self, timeframe, reg, res_dir=None):
        r_all = self._regde_all(timeframe, res_dir)
        r = r_all[reg]
        r = np.concatenate([np.asarray(r[0]).reshape(1, -1), np.asarray(r[1])], 0)
        r = r / len(self.af.run_align[timeframe])
        t = self.af._time_axis_for_plot(timeframe, r.shape[1])
        return t, r

    def unsplit80(self, reg):
        t, r = self.curves(UNSPLIT_TF, reg, fc.ALYX_RES)
        n = fc.n_bins_le_tmax(r.shape[1])
        return t[:n], r[:, :n]

    def rec(self, timeframe, reg):
        if timeframe == UNSPLIT_TF or 'earlystim' in timeframe:
            return self.early.get(reg, {})
        if 'duringchoice' in timeframe:
            return self.choice_d.get(reg, {})
        return self.stim_d.get(reg, {})

    def _regde_all(self, timeframe, res_dir=None):
        key = (timeframe, str(res_dir or fc.OA_RES))
        if key not in self._regde:
            path = _stem(self.af, timeframe, Path(res_dir or fc.OA_RES), True)
            self._regde[key] = _load_dict(path)
        return self._regde[key]

    def _fill_choice_fdr(self):
        """The Aug-24 duringchoice combine has p_mean but no p_mean_c. Fill in memory."""
        regs = [r for r in self.choice_d if isinstance(self.choice_d[r], dict)]
        sample = self.choice_d[regs[0]]
        if sample.get('p_mean_c') is not None:
            return
        pvals = [float(self.choice_d[r].get('p_mean', 1.0) or 1.0) for r in regs]
        _, pc, _, _ = multipletests(pvals, fc.ALPHA, method='fdr_bh')
        for r, p_c in zip(regs, pc):
            self.choice_d[r]['p_mean_c'] = float(p_c)

    def weighted_average(self, regs, timeframe, res_dir=None, t_max=None):
        r_sum, n_cells, used = None, 0, []
        for reg in regs:
            try:
                t, r = self.curves(timeframe, reg, res_dir)
            except (KeyError, TypeError):
                continue
            if t_max is not None:
                n = fc.n_bins_le_tmax(r.shape[1], t_max_ms=t_max)
                t, r = t[:n], r[:, :n]
            nclus = self.nclus(reg, timeframe)
            if nclus <= 0:
                continue
            r_sum = r * nclus if r_sum is None else r_sum + r * nclus
            n_cells += nclus
            used.append(reg)
        if r_sum is None:
            raise RuntimeError(f'no cells for {timeframe} in {list(regs)[:5]}…')
        return t, r_sum / n_cells, used, n_cells

    def load_group_sub(self, regs, timeframe):
        """Shuffle-mean subtracted cell-weighted mean (panel i). Includes all regs."""
        _, r, _, _ = self.weighted_average(regs, timeframe)
        return r[0] - np.mean(r[1:], axis=0)


def _stem(af, timeframe, res_dir, regde):
    name, rd = fc.combined_stems(af, timeframe)
    return Path(res_dir) / f'{rd if regde else name}.npy'


def _load_dict(path):
    d = np.load(path, allow_pickle=True)
    return d.item() if hasattr(d, 'item') else d.flat[0]


def _p(rec, key, default=1.0):
    v = rec.get(key, default)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        v = rec.get('p_mean', default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def offset_gain(r, alpha=fc.ALPHA, r_stim=None):
    """p_offset / p_gain / shifted observed curve. r is (1+U, T)."""
    src = r_stim if r_stim is not None else r
    mean5 = np.mean(src[:, :5], axis=1)
    p_off = float(np.mean(mean5 >= mean5[0]))
    if p_off <= alpha:
        off = mean5[0] - np.mean(mean5[1:])
        shifted = r[0] - off
    else:
        shifted = r[0]
    p_gain = float(np.mean(np.mean(r[1:, 4:], axis=1) >= np.mean(shifted[4:])))
    return p_off, p_gain, shifted


def nice_ylim(rs, pad=0.04):
    lo = min(np.nanmin(r[:fc.N_SHOW + 1]) for r in rs)
    hi = max(np.nanmax(r[:fc.N_SHOW + 1]) for r in rs)
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    step = 0.05 if hi - lo < 0.4 else 0.1 if hi - lo < 1.0 else 0.2
    lo, hi = np.floor(lo / step) * step, np.ceil(hi / step) * step
    ticks = np.arange(lo, hi + 0.5 * step, step)
    if len(ticks) > 5:
        ticks = np.linspace(lo, hi, 4)
    return (float(lo), float(hi)), [float(v) for v in ticks]


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def letter(fig, x_pt, y_pt, s):
    fig.text(x_pt / fc.PAGE_WH[0], 1 - y_pt / fc.PAGE_WH[1], s,
             fontsize=26, va='center', ha='left', fontweight='regular')


def badge_ax(fig):
    ax = fig.add_axes([0, 0, 1, 1], zorder=5)
    ax.set_xlim(0, fc.PAGE_WH[0])
    ax.set_ylim(fc.PAGE_WH[1], 0)
    ax.axis('off')
    ax.patch.set_alpha(0)
    return ax


def badge(ax, x_pt, y_pt, letter_, color, radius_pt=11):
    ax.add_patch(Circle((x_pt, y_pt), radius_pt, facecolor=color,
                        edgecolor='none', clip_on=False))
    ax.text(x_pt, y_pt, letter_, fontsize=12, ha='center', va='center',
            color='white' if letter_ == 'M' else 'k', zorder=6)


def _cell(ax, x, y, w, h, face):
    """Filled cell. Edge matches the fill so neighbours meet without a white seam."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=face,
                           lw=0.35, joinstyle='miter', clip_on=True))


def _scaled_prior(recs, regs, zero_before_scale):
    """Amp × FDR, then 0–1, matching plot_combined_table_summary.

    150 ms columns scale every region, then zero the non-significant ones.
    The earlystim column zeros first, then scales, then drops the 1e-4 floor.
    """
    raw, sig = [], []
    for reg in regs:
        rec = recs.get(reg) or {}
        a = rec.get('amp_euc')
        try:
            a = float(a)
        except (TypeError, ValueError):
            a = np.nan
        raw.append(a)
        sig.append(_p(rec, 'p_mean_c') <= fc.ALPHA)
    raw = np.asarray(raw, dtype=float)
    sig = np.asarray(sig, dtype=bool)
    if zero_before_scale:
        raw = np.where(sig, np.nan_to_num(raw, nan=0.0), 0.0)
    finite = raw[np.isfinite(raw)]
    lo = float(finite.min()) if len(finite) else 0.0
    hi = float(finite.max()) if len(finite) else 1.0
    scaled = np.zeros(len(regs), dtype=float)
    if hi > lo:
        scaled = (np.nan_to_num(raw, nan=lo) - lo) / (hi - lo) + 1e-4
    if zero_before_scale:
        scaled = np.where(scaled > 1e-4, scaled, 0.0)
    else:
        scaled = np.where(sig, scaled, 0.0)
        scaled = np.nan_to_num(scaled, nan=0.0)
    return scaled


def _type_face(val, cmap):
    if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
        return fc.C_GREY
    return to_hex(cmap(float(np.clip(val, 0.0, 1.0))))


def _prior_face(v, cmap):
    if v <= 0:
        return fc.C_GREY
    return to_hex(cmap(float(np.clip(v, 0.0, 1.0))))


def draw_a(ax, data):
    """Overall table as vector cells (same encoding as the earlystim PNG).

    Top to bottom: type_stim, prior_80ms, prior_stim, type_choice, prior_choice, region.
    """
    regs = data.regs
    n = len(regs)
    type_s = [_type_face(data.type_stim.get(r), data.cmap_type) for r in regs]
    type_c = [_type_face(data.type_choice.get(r), data.cmap_type) for r in regs]
    prior_s = [_prior_face(v, data.cmap_prior)
               for v in _scaled_prior(data.stim_d, regs, zero_before_scale=False)]
    prior_c = [_prior_face(v, data.cmap_prior)
               for v in _scaled_prior(data.choice_d, regs, zero_before_scale=False)]
    prior_80 = [_prior_face(v, data.cmap_prior)
                for v in _scaled_prior(data.early, regs, zero_before_scale=True)]
    rows = [type_s, prior_80, prior_s, type_c, prior_c]  # top → bottom
    h_data, h_reg = 1.0, 1.45
    total = len(rows) * h_data + h_reg
    y = total
    for faces in rows:
        y -= h_data
        for i, face in enumerate(faces):
            _cell(ax, i, y, 1, h_data, face)
    for i, reg in enumerate(regs):
        _cell(ax, i, 0, 1, h_reg, data.beryl.get(reg, '#ffffff'))
        ax.text(i + 0.5, h_reg * 0.97, reg, rotation=90, ha='center', va='top',
                fontsize=3.0, clip_on=True)
    ax.set_xlim(0, n)
    ax.set_ylim(0, total)
    ax.axis('off')
    for i, name in enumerate(fc.A_ROW_LABELS):
        yc = total - (i + 0.5) * h_data
        ax.text(n + 1.5, yc, name, ha='left', va='center', fontsize=8,
                clip_on=False, color='#333333',
                style='italic' if name in ('s', 'm') else 'normal',
                fontfamily='serif' if name in ('s', 'm') else 'Helvetica')


def draw_onetype_strip(ax, data, hits, show_combined=True):
    """Hit row over a thin Beryl bar. No region names, no gaps between cells."""
    n = len(data.regs)
    # Region bar the same height as the hit row above it.
    h_reg = 0.5 if show_combined else 1.0
    h_comb = 0.5
    cmap = data.af.get_cmap_('intertrial')
    for i, reg in enumerate(data.regs):
        _cell(ax, i, 0, 1, h_reg, data.beryl.get(reg, '#ffffff'))
        if show_combined:
            v = 1.0 if reg in hits else 0.0
            face = to_hex(cmap(v)) if v else fc.C_GREY
            _cell(ax, i, h_reg, 1, h_comb, face)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis('off')


def _style_traj_ax(ax, t, ylim, yticks, align, show_x, ylabel=None):
    ev = fc.C_STIM_ON if align == 'stim' else fc.C_MOVE_ON
    t0 = 0
    ax.axvline(t0, color=ev, lw=3.0, zorder=0, solid_capstyle='butt')
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{v:g}' if abs(v) >= 0.05 or v == 0 else f'{v:.2f}'
                        for v in np.round(yticks, 2)])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, fontfamily='Times New Roman', labelpad=0)
    if align == 'stim':
        ax.set_xlim(t[0] - 0.02 * (t[-1] - t[0]), t[-1])
        xt = [0, 40, 80] if t[-1] <= 85 else [0, 50, 100, 150]
    else:
        ax.set_xlim(t[0], t[-1] + 0.02 * (t[-1] - t[0]))
        xt = [-150, -100, -50, 0]
    ax.set_xticks(xt)
    if align == 'move':
        ax.set_xticklabels(['\u2212150', '\u2212100', '\u221250', '0'])
    if not show_x:
        ax.tick_params(axis='x', labelbottom=False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8)


def draw_traj(ax, hax, t, r, rec, align, ylim, yticks, color='k',
              plot_offset=True, plot_gain=True, ylabel=None, show_x=True,
              r_stim=None):
    p_off, p_gain, shifted = offset_gain(r, r_stim=r_stim)
    p_mean = _p(rec, 'p_mean_c', _p(rec, 'p_mean'))
    n_show = min(fc.N_SHOW, r.shape[0] - 1)
    for i in range(1, n_show + 1):
        if plot_offset and align == 'stim' and r.shape[1] >= 5:
            ax.plot(t[:5], r[i, :5], c=fc.C_OFFSET, alpha=0.45, lw=0.45)
            ax.plot(t[4:], r[i, 4:], c='gray', alpha=0.22, lw=0.45)
        else:
            ax.plot(t, r[i], c='gray', alpha=0.22, lw=0.45)
    ls = '--' if plot_gain else '-'
    ax.plot(t, r[0], c=color, lw=1.15, ls=ls, zorder=3)
    if plot_gain and p_off <= fc.ALPHA:
        ax.plot(t, shifted, c=color, lw=1.15, ls='-', zorder=3)
    _style_traj_ax(ax, t, ylim, yticks, align, show_x, ylabel)

    if hax is None:
        return dict(p_mean=p_mean, p_offset=p_off, p_gain=p_gain)
    hax.hist(np.mean(r[1:], axis=1), bins=18, density=True, color='silver',
             orientation='horizontal')
    hax.axhline(np.mean(r[0]), color=color, ls=ls, lw=1.1)
    if plot_offset and align == 'stim':
        hax.hist(np.mean(r[1:, :5], axis=1), bins=18, density=True,
                 color=fc.C_OFFSET, alpha=0.45, orientation='horizontal')
        hax.axhline(np.mean(r[0, :5]), color='#2B6EA8', ls='--', lw=1.0)
    if plot_gain and p_off <= fc.ALPHA:
        hax.axhline(np.mean(shifted[4:]), color=color, ls='-', lw=1.1)
    hax.set_ylim(ylim)
    hax.tick_params(axis='y', left=False, labelleft=False)
    hax.tick_params(axis='x', bottom=False, labelbottom=False)
    hax.spines['bottom'].set_visible(False)
    hax.spines['left'].set_visible(False)
    hax.set_xlim(0, max(hax.get_xlim()[1], 1e-6) * 1.05)
    # significance stars, right of the hist
    stars = []
    if p_mean <= fc.ALPHA:
        stars.append(np.mean(r[0]))
    if plot_offset and align == 'stim' and p_off <= fc.ALPHA:
        stars.append(np.mean(r[0, :5]))
    if plot_gain and p_off <= fc.ALPHA and p_gain <= fc.ALPHA:
        stars.append(np.mean(shifted[4:]))
    for y in stars:
        hax.text(0.78, y, '*', fontsize=10, ha='left', va='center',
                 transform=hax.get_yaxis_transform(), clip_on=True)
    return dict(p_mean=p_mean, p_offset=p_off, p_gain=p_gain)


def draw_average(ax, t, r, align, ylim, yticks, color, ylabel, show_x=True,
                 inset=True):
    p_off, p_gain, shifted = offset_gain(r)
    n_show = min(fc.N_SHOW, r.shape[0] - 1)
    for i in range(1, n_show + 1):
        ax.plot(t, r[i], c='gray', alpha=0.22, lw=0.45)
    ax.plot(t, r[0], c=color, lw=1.6, ls='--', zorder=3)
    if p_off <= fc.ALPHA:
        ax.plot(t, shifted, c=color, lw=1.6, ls='-', zorder=3)
    _style_traj_ax(ax, t, ylim, yticks, align, show_x, ylabel)
    if inset:
        diff = r[0] - np.mean(r[1:], axis=0)
        sh = r[1:] - np.mean(r[1:], axis=0, keepdims=True)
        lo = np.percentile(sh, 100 * (fc.ALPHA / 2), axis=0)
        hi = np.percentile(sh, 100 * (1 - fc.ALPHA / 2), axis=0)
        ax_ins = inset_axes(ax, width='22%', height='22%', loc='upper left',
                            borderpad=1.4)
        ax_ins.fill_between(t, lo, hi, color='gray', alpha=0.3, lw=0)
        ax_ins.plot(t, diff, color=color, lw=1.0)
        ax_ins.set_xticks([])
        ax_ins.set_ylim(-0.05, 0.2)
        ax_ins.set_yticks([0.0, 0.1, 0.2])
        ax_ins.tick_params(labelsize=4.8, length=2)
        ax_ins.set_facecolor('none')
        for sp in ('top', 'right'):
            ax_ins.spines[sp].set_visible(False)
    return dict(p_offset=p_off, p_gain=p_gain,
                p_mean=float(np.mean(np.mean(r, axis=1) >= np.mean(r[0]))))


def ylabel_example(align, reg):
    a = 's' if align == 'stim' else 'm'
    return rf'$d^{{\mathrm{{prior}},{a}}}_{{\mathrm{{{reg}}}}}(t)$'


def ylabel_avg(align, group):
    a = 's' if align == 'stim' else 'm'
    return rf'$d^{{\mathrm{{prior}},{a}}}_{{\mathrm{{{group}}}}}(t)$'


def _choice_gain(data, reg):
    """duringchoice p_gain / effect, using stim-aligned offset (plot_regional_distance)."""
    dc = data.choice_d.get(reg, {})
    stored = dc.get('p_gain_effect')
    pgc = dc.get('p_gain')
    if stored not in (None, 0) and pgc is not None and not (isinstance(pgc, float) and np.isnan(pgc)):
        return float(pgc), float(stored)
    try:
        _, r = data.curves(CHOICE_TF, reg)
        _, r_s = data.curves(STIM_TF, reg)
    except (KeyError, TypeError):
        return 1.0, 0.0
    p_off, p_gain, shifted = offset_gain(r, r_stim=r_s)
    late = shifted[4:]
    max_idx = int(np.argmax(late))
    effect = float(late[max_idx] - np.mean(r[1:, 4:][:, max_idx]))
    return p_gain, max(effect, 0.0)


def _gated_effect(rec, kind):
    """Effect size if the prior FDR and this component both pass α, else 0."""
    if _p(rec, 'p_mean_c') > fc.ALPHA:
        return 0.0
    if _p(rec, f'p_{kind}') >= fc.ALPHA:
        return 0.0
    try:
        return max(float(rec.get(f'p_{kind}_effect') or 0.0), 0.0)
    except (TypeError, ValueError):
        return 0.0


def draw_h(ax, data):
    """offset / gain_s / gain_m.

    offset and gain s take the larger effect of the 150 ms 4-split prior and
    the 80 ms stim-side prior, so a prior_80ms hit in panel a shows up here
    when its offset or gain is significant. gain m stays move-aligned.
    """
    rows = []
    for reg in data.regs:
        ds = data.stim_d.get(reg, {})
        es = data.early.get(reg, {})
        dc = data.choice_d.get(reg, {})
        off = max(_gated_effect(ds, 'offset'), _gated_effect(es, 'offset'))
        gs = max(_gated_effect(ds, 'gain'), _gated_effect(es, 'gain'))
        sig_c = float(_p(dc, 'p_mean_c') <= fc.ALPHA)
        pgc, ge = _choice_gain(data, reg)
        gm = ge * sig_c * float(pgc < fc.ALPHA)
        rows.append((off, gs, gm))
    arr = np.array(rows, dtype=float)
    cmap = data.cmap_prior
    names = ('offset', 'gain s', 'gain m')
    # Stacked with no gap: region, then gain m, gain s, offset.
    n = len(data.regs)
    h_reg, h_row = 1.15, 0.72
    for i, reg in enumerate(data.regs):
        _cell(ax, i, 0, 1, h_reg, data.beryl.get(reg, '#fff'))
        ax.text(i + 0.5, h_reg * 0.97, reg, rotation=90,
                ha='center', va='top', fontsize=3.0, clip_on=True)
        for k, name in enumerate(reversed(names)):
            v = arr[i, 2 - k]
            face = fc.C_GREY if v == 0 else to_hex(cmap(min(max(float(v), 0), 1)))
            y = h_reg + k * h_row
            _cell(ax, i, y, 1, h_row, face)
    total = h_reg + 3 * h_row
    ax.set_xlim(0, n)
    ax.set_ylim(0, total)
    ax.axis('off')
    for k, name in enumerate(reversed(names)):
        ax.text(n + 1.5, h_reg + (k + 0.5) * h_row, name,
                ha='left', va='center', fontsize=8, clip_on=False)
    return names


def draw_i(ax, t, y_c, y_m, align, ylim, yticks, ylabel, show_y):
    ax.plot(t, y_c, color='gold', lw=1.5)
    ax.plot(t, y_m, color='tomato', lw=1.5)
    _style_traj_ax(ax, t, ylim, yticks, align, True, ylabel if show_y else None)
    if align == 'move':
        ax.set_xticklabels(['\u2212150', '\u2212100', '\u221250', '0'])
    if not show_y:
        ax.tick_params(axis='y', labelleft=False)


# ---------------------------------------------------------------------------
# composite
# ---------------------------------------------------------------------------
def composite(data, out):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor('white')
    info = {}

    # --- a (vector cells, same encoding as the earlystim table PNG) ---
    ax_a = fig.add_axes(fc.fig_box(16, 26, 700, 90))
    draw_a(ax_a, data)
    letter(fig, 2, 18, 'a')

    badges = [
        (110, 104, 'S', fc.C_STIM, 11),
        (370, 104, 'C', '#F3D37A', 11),
        (630, 104, 'M', fc.C_MOVE, 11),
    ]

    # --- strips b/d/f ---
    ax_b = fig.add_axes(fc.fig_box(25, 111.5, 194, 131))
    draw_onetype_strip(ax_b, data, set(data.overlap))
    ax_d = fig.add_axes(fc.fig_box(250, 111.5, 441, 131))
    draw_onetype_strip(ax_d, data, set(data.c_strip))
    ax_f = fig.add_axes(fc.fig_box(539, 111.5, 729, 131))
    draw_onetype_strip(ax_f, data, set(data.m_strip))
    letter(fig, 2, 108, 'b')
    letter(fig, 205, 108, 'd')
    letter(fig, 499, 108, 'f')

    # --- b example (LGd, 80 ms) ---
    t_s, r_s = data.unsplit80(fc.EXAMPLE_S)
    ylim_s, yt_s = nice_ylim([r_s])
    ax = fig.add_axes(fc.fig_box(48, 132, 154, 222))
    hax = fig.add_axes(fc.fig_box(156, 132, 174, 222), sharey=ax)
    rec = data.early.get(fc.EXAMPLE_S, {})
    info['b'] = draw_traj(ax, hax, t_s, r_s, rec, 'stim', ylim_s, yt_s,
                          ylabel=ylabel_example('stim', fc.EXAMPLE_S), show_x=False)

    # --- d examples PAG, PRNc ---
    t_pag, r_pag = data.curves(STIM_TF, fc.EXAMPLE_C[0])
    t_prn, r_prn = data.curves(STIM_TF, fc.EXAMPLE_C[1])
    ylim_d1, yt_d1 = nice_ylim([r_pag])
    ylim_d2, yt_d2 = nice_ylim([r_prn])
    ax = fig.add_axes(fc.fig_box(248, 132, 322, 222))
    hax = fig.add_axes(fc.fig_box(324, 132, 342, 222), sharey=ax)
    info['d_PAG'] = draw_traj(ax, hax, t_pag, r_pag, data.rec(STIM_TF, fc.EXAMPLE_C[0]),
                              'stim', ylim_d1, yt_d1,
                              ylabel=ylabel_example('stim', fc.EXAMPLE_C[0]), show_x=False)
    ax = fig.add_axes(fc.fig_box(392, 132, 466, 222))
    hax = fig.add_axes(fc.fig_box(468, 132, 486, 222), sharey=ax)
    info['d_PRNc'] = draw_traj(ax, hax, t_prn, r_prn, data.rec(STIM_TF, fc.EXAMPLE_C[1]),
                               'stim', ylim_d2, yt_d2,
                               ylabel=ylabel_example('stim', fc.EXAMPLE_C[1]), show_x=False)

    # --- f examples IRN stim / move ---
    t_irn_s, r_irn_s = data.curves(STIM_TF, fc.EXAMPLE_M)
    t_irn_m, r_irn_m = data.curves(CHOICE_TF, fc.EXAMPLE_M)
    ylim_f1, yt_f1 = nice_ylim([r_irn_s])
    ylim_f2, yt_f2 = nice_ylim([r_irn_m])
    ax = fig.add_axes(fc.fig_box(548, 132, 618, 222))
    hax = fig.add_axes(fc.fig_box(620, 132, 638, 222), sharey=ax)
    info['f_IRN_s'] = draw_traj(ax, hax, t_irn_s, r_irn_s, data.rec(STIM_TF, fc.EXAMPLE_M),
                                'stim', ylim_f1, yt_f1,
                                ylabel=ylabel_example('stim', fc.EXAMPLE_M), show_x=False)
    ax = fig.add_axes(fc.fig_box(666, 132, 736, 222))
    hax = fig.add_axes(fc.fig_box(738, 132, 756, 222), sharey=ax)
    info['f_IRN_m'] = draw_traj(ax, hax, t_irn_m, r_irn_m, data.rec(CHOICE_TF, fc.EXAMPLE_M),
                                'move', ylim_f2, yt_f2, plot_offset=False,
                                ylabel=ylabel_example('move', fc.EXAMPLE_M), show_x=False,
                                r_stim=r_irn_s)

    # --- c sensory combined ---
    t_c, r_c, used_s, n_s = data.weighted_average(
        data.overlap, UNSPLIT_TF, fc.ALYX_RES, t_max=fc.T_MAX_MS)
    ylim_c, yt_c = nice_ylim([r_c])
    ax = fig.add_axes(fc.fig_box(50, 246, 168, 340))
    info['c'] = draw_average(ax, t_c, r_c, 'stim', ylim_c, yt_c, fc.C_STIM,
                             ylabel_avg('stim', 'S'), show_x=True)
    info['c']['n_reg'] = len(used_s)
    info['c']['n_cells'] = n_s
    letter(fig, 2, 228, 'c')
    badges.append((155, 268, 'S', fc.C_STIM, 9))
    fig.text(105 / fc.PAGE_WH[0], 1 - 358 / fc.PAGE_WH[1], 'time (ms)',
             fontsize=14, ha='center', va='top')

    # --- e C average ---
    c_regs = sorted(data.integ_s | data.integ_c)
    t_e, r_e, used_c, n_c = data.weighted_average(c_regs, STIM_TF)
    ylim_e, yt_e = nice_ylim([r_e])
    ax = fig.add_axes(fc.fig_box(300, 246, 428, 340))
    info['e'] = draw_average(ax, t_e, r_e, 'stim', ylim_e, yt_e, 'gold',
                             ylabel_avg('stim', 'C'), show_x=True)
    info['e']['n_reg'] = len(used_c)
    info['e']['n_cells'] = n_c
    letter(fig, 238, 248, 'e')
    badges.append((400, 300, 'C', '#F3D37A', 9))
    fig.text(360 / fc.PAGE_WH[0], 1 - 358 / fc.PAGE_WH[1], 'time (ms)',
             fontsize=14, ha='center', va='top')

    # --- g M average (move-aligned) ---
    m_regs = sorted(data.move_s | data.move_c)
    t_g, r_g, used_m, n_m = data.weighted_average(m_regs, CHOICE_TF)
    ylim_g, yt_g = nice_ylim([r_g])
    ax = fig.add_axes(fc.fig_box(548, 243, 682, 338))
    info['g'] = draw_average(ax, t_g, r_g, 'move', ylim_g, yt_g, 'tomato',
                             ylabel_avg('move', 'M'), show_x=True)
    info['g']['n_reg'] = len(used_m)
    info['g']['n_cells'] = n_m
    letter(fig, 497, 248, 'g')
    badges.append((612, 268, 'M', fc.C_MOVE, 9))
    fig.text(611 / fc.PAGE_WH[0], 1 - 358 / fc.PAGE_WH[1], 'time (ms)',
             fontsize=14, ha='center', va='top')

    # c/e/g schematics: intentionally empty (gain / offset cartoons)

    # --- h ---
    ax = fig.add_axes(fc.fig_box(18, 422, 710, 468))
    draw_h(ax, data)
    letter(fig, 2, 419, 'h')

    # --- i ---
    data.af.pth_res = fc.OA_RES
    y_c_s = data.load_group_sub(c_regs, STIM_TF)
    y_m_s = data.load_group_sub(m_regs, STIM_TF)
    y_c_m = data.load_group_sub(c_regs, CHOICE_TF)
    y_m_m = data.load_group_sub(m_regs, CHOICE_TF)
    t_is = np.linspace(0, 150, len(y_c_s))
    t_im = np.linspace(-150, 0, len(y_c_m))
    ylim_i, yt_i = nice_ylim([y_c_s[None], y_m_s[None], y_c_m[None], y_m_m[None]],
                             pad=0.08)
    # original i is a small 0–0.15 range (already subtracted)
    lo = min(y_c_s.min(), y_m_s.min(), y_c_m.min(), y_m_m.min())
    hi = max(y_c_s.max(), y_m_s.max(), y_c_m.max(), y_m_m.max())
    ylim_i = (min(0.03, lo - 0.01), max(0.16, hi + 0.01))
    yt_i = [0.06, 0.09, 0.12, 0.15]
    ax = fig.add_axes(fc.fig_box(192, 492, 346, 586))
    draw_i(ax, t_is, y_c_s, y_m_s, 'stim', ylim_i, yt_i,
           r'$d^{\mathrm{prior},s}_{\mathrm{\{C,M\}}}(t)$', True)
    ax = fig.add_axes(fc.fig_box(378, 492, 532, 586))
    draw_i(ax, t_im, y_c_m, y_m_m, 'move', ylim_i, yt_i,
           r'$d^{\mathrm{prior},m}_{\mathrm{\{C,M\}}}(t)$', True)
    letter(fig, 124, 501, 'i')
    badges.append((232, 508, 'C', '#F3D37A', 8))
    badges.append((258, 508, 'M', fc.C_MOVE, 8))
    bax = badge_ax(fig)
    for x, y, L, c, rad in badges:
        badge(bax, x, y, L, c, radius_pt=rad)
    fig.text(263 / fc.PAGE_WH[0], 1 - 615 / fc.PAGE_WH[1], 'time (ms)',
             fontsize=14, ha='center', va='top')
    fig.text(453 / fc.PAGE_WH[0], 1 - 615 / fc.PAGE_WH[1], 'time (ms)',
             fontsize=14, ha='center', va='top')
    info['i'] = {
        'ylim': ylim_i,
        'c_n': len(c_regs), 'm_n': len(m_regs),
        'amp_s_c': float(np.ptp(y_c_s)), 'amp_s_m': float(np.ptp(y_m_s)),
        'amp_m_c': float(np.ptp(y_c_m)), 'amp_m_m': float(np.ptp(y_m_m)),
    }

    fc.savefig_svg_png_pdf(fig, out / 'figure4', dpi=200)
    plt.close(fig)
    return info


def write_panel_strips(data, pdir):
    fig, ax = plt.subplots(figsize=(10, 1.1))
    draw_a(ax, data)
    fig.savefig(pdir / 'panel_a_table.png', dpi=300, facecolor='white',
                bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 0.45))
    draw_onetype_strip(ax, data, set(data.overlap))
    fig.savefig(pdir / 'panel_b_overlap.png', dpi=200, facecolor='white',
                bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 0.45))
    draw_onetype_strip(ax, data, set(data.c_strip))
    fig.savefig(pdir / 'panel_d_strip.png', dpi=200, facecolor='white',
                bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 0.45))
    draw_onetype_strip(ax, data, set(data.m_strip))
    fig.savefig(pdir / 'panel_f_strip.png', dpi=200, facecolor='white',
                bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    out = fc.FIG_DIR
    (out / 'original').mkdir(parents=True, exist_ok=True)
    (out / 'panels').mkdir(parents=True, exist_ok=True)
    shutil.copy2(fc.ORIGINAL_PDF, out / 'original' / 'figure4.pdf')

    data = Data()
    if fc.EXAMPLE_S not in data.overlap:
        raise SystemExit(f'{fc.EXAMPLE_S} is not in sensory ∩ prior_80ms: {data.overlap}')
    info = composite(data, out)
    write_panel_strips(data, out / 'panels')

    summary = {
        'example_s': fc.EXAMPLE_S,
        'example_c': list(fc.EXAMPLE_C),
        'example_m': fc.EXAMPLE_M,
        'overlap': data.overlap,
        'n_overlap': len(data.overlap),
        'c_strip': data.c_strip,
        'm_strip': data.m_strip,
        'n_prior_stim': len(data.prior_stim),
        'n_prior_choice': len(data.prior_choice),
        'n_prior_80': len(data.prior80),
        'type_stim': data.type_stim.value_counts(dropna=False).to_dict(),
        'type_choice': data.type_choice.value_counts(dropna=False).to_dict(),
        'panels': info,
        'sources': {
            'table': str(fc.TABLE_PNG),
            'earlystim': str(fc.EARLYSTIM_NPY),
            'sc_table': str(fc.SC_TABLE),
            'oa_res': str(fc.OA_RES),
            'alyx_res': str(fc.ALYX_RES),
            'original': str(fc.ORIGINAL_PDF),
        },
        'notes': {
            'a': 'cropped overall_earlystim80 PNG (not a diagnostic raster paste)',
            'b': 'plot_combined_onetype-style strip of prior_80ms ∩ type_stim sensory',
            'c': 'cell-weighted unsplit 80 ms average over the 13 overlap regions',
            'd_g': 'replotted from current openalyx 4-split act_block; '
                   'examples PAG/PRNc/IRN kept (all still FDR ≤ 0.01 on prior_stim)',
            'h': 'offset and gain s: max of 150 ms 4-split and prior_80ms '
                 'gated effects; gain_m from duringchoice curves',
            'schematics': 'c/e/g gain-offset cartoons left empty',
        },
    }
    # json-friendly
    def _conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, dict):
            return {str(k): _conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_conv(v) for v in o]
        return o
    (out / 'summary.json').write_text(json.dumps(_conv(summary), indent=2))
    print(f'overlap ({len(data.overlap)}): {", ".join(data.overlap)}')
    print(f'C strip ({len(data.c_strip)}): {", ".join(data.c_strip)}')
    print(f'M strip ({len(data.m_strip)}): {", ".join(data.m_strip)}')
    print(f'wrote {out / "figure4.pdf"}')


if __name__ == '__main__':
    main()
