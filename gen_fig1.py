#!/usr/bin/env python3
"""Generate the two-part Figure 1 for the curvature paper.

Part (a): Illustrative submodular functions h(|S|) at different curvatures.
Part (b): Guarantee curves (1-e^{-c})/c and 1/e.
"""
from release_paths import RESULTS, FIGURES, ORIGINAL

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# NeurIPS-compatible style
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'axes.labelsize': 15,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
    'text.usetex': False,
})


def submodular_shape(t, c):
    """Model submodular function h(|S|) parameterized by curvature c.

    At curvature c, the marginal gain decays linearly:
      h'(t) ~ 1 - c*t

    Integrating: h(t) = t - c*t^2/2

    This gives:
      c=0: h(t) = t (modular/linear)
      c=0.5: mild concavity
      c=1: h(t) = t - t^2/2 (monotone boundary)
      c=2: h(t) = t - t^2, peaks at t=0.5, h(1)=0 (symmetric)
      c=3: h(t) = t - 3t^2/2, goes negative for t > 2/3
    """
    return t - c * t**2 / 2


def guarantee(c):
    """Curvature guarantee (1 - e^{-c}) / c."""
    return (1 - np.exp(-c)) / c


def make_figure(outpath=None):
    FIGURES.mkdir(parents=True, exist_ok=True)
    outpath = outpath or FIGURES / 'fig1_curvature_spectrum.pdf'
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 4.5))

    # --- Part (a): Illustrative submodular functions h(|S|) ---
    t = np.linspace(0, 1, 200)

    # Plot curves from highest curvature to lowest so low-c draws on top
    curves = [
        (3,   '#9C27B0', '--', 1.8),
        (2,   '#F44336', '-',  1.8),
        (1,   '#FF9800', '-',  1.8),
        (0.5, '#4CAF50', '-',  1.8),
        (0,   '#2196F3', '-',  1.8),
    ]

    label_map = {
        0:   '$c=0$ (modular)',
        0.5: '$c=0.5$',
        1:   '$c=1$ (monotone boundary)',
        2:   '$c=2$ (symmetric)',
        3:   '$c=3$',
    }

    for c_val, color, ls, lw in curves:
        f_vals = submodular_shape(t, c_val)
        ax_a.plot(t, f_vals, color=color, linestyle=ls,
                  label=label_map[c_val], linewidth=lw)

    # Mark the peak for c=2 at t=0.5
    ax_a.plot(0.5, submodular_shape(0.5, 2), 'o', color='#F44336',
              markersize=4, zorder=5)

    # Small endpoint markers for all curves
    for c_val, color, ls, lw in curves:
        f_end = submodular_shape(1, c_val)
        ax_a.plot(1, f_end, 'o', color=color, markersize=3, zorder=6)

    ax_a.axhline(0, color='gray', linewidth=0.5, linestyle='-')
    ax_a.set_xlabel('Set size $|S|/n$')
    ax_a.set_ylabel('$h(|S|)$')
    ax_a.set_xlim(0, 1.02)
    ax_a.set_ylim(-0.65, 1.15)
    ax_a.legend(loc='lower left', fontsize=12, framealpha=0.7, ncol=1)
    ax_a.set_title('(a) Submodular functions at different curvatures', fontsize=11)

    # --- Part (b): Guarantee curves ---
    c = np.linspace(0.01, 6, 500)
    curv_guarantee = (1 - np.exp(-c)) / c
    # Best known uniform ratio for non-negative f: alpha_0 of Aggarwal-Du,
    # arXiv:2610.21072v2 (Oct 25, 2026), ~0.46338 (was 0.401, BF '24).
    bf_baseline = 54440750000 / 117486474169
    c_star = brentq(lambda x: (1 - np.exp(-x)) / x - bf_baseline, 1.0, 5.0)

    # Shade c <= 1 region (classical)
    ax_b.axvspan(0, 1, alpha=0.10, color='blue', label='Classical ($c \\leq 1$)')

    # Shade 1 < c < c* region (new: beats best known)
    ax_b.axvspan(1, c_star, alpha=0.10, color='green',
                 label='Beats $0.463$ ($1 < c < c^*$)')

    # Main guarantee curve — extends to all c
    ax_b.plot(c, curv_guarantee, 'b-', label='Curvature: $(1-e^{-c})/c$', linewidth=2.2)

    # Best known uniform ratio (Aggarwal-Du 2026) — only up to c*, beyond which
    # hard instances are negative so non-negative algorithms need not apply
    c_nonneg = c[c <= c_star + 0.01]
    bf_line = np.full_like(c_nonneg, bf_baseline)
    ax_b.plot(c_nonneg, bf_line, 'r--',
              label='Best known: $0.463$ [AD \'26] (requires $f \\geq 0$)', linewidth=1.5)
    # Small open circle at the truncation point to show it ends
    ax_b.plot(c_star, bf_baseline, 'o', color='red', markersize=4,
              markerfacecolor='white', markeredgecolor='red', markeredgewidth=1.5, zorder=5)

    # Mark c* crossover
    ax_b.axvline(c_star, color='#444444', linestyle=':', alpha=0.6)
    ax_b.annotate(f'$c^* \\approx {c_star:.1f}$',
                  xy=(c_star, 0.55), xytext=(c_star + 0.5, 0.63),
                  arrowprops=dict(arrowstyle='->', color='#444444'),
                  fontsize=11, color='#444444')

    # Mark c=1 (monotone boundary)
    ax_b.plot(1, 1 - 1/np.e, 'ko', markersize=5)
    ax_b.annotate('$c=1$ (monotone)', xy=(1, 1-1/np.e),
                  xytext=(1.4, 0.76), fontsize=10, color='black',
                  arrowprops=dict(arrowstyle='->', color='black'))

    # Mark c=2 (symmetric/maxcut)
    c2_val = (1 - np.exp(-2)) / 2
    ax_b.plot(2, c2_val, 's', color='#F44336', markersize=5, zorder=5)
    ax_b.annotate('$c=2$ (max-cut)', xy=(2, c2_val),
                  xytext=(2.8, 0.53), fontsize=10, color='#F44336',
                  arrowprops=dict(arrowstyle='->', color='#F44336'))

    # Annotate the c > c* region — still a contribution (new ratios for negative f)
    ax_b.annotate('New ratios\n(negative $f$)',
                  xy=(4.2, curv_guarantee[np.argmin(np.abs(c - 4.2))]),
                  xytext=(4.8, 0.45), fontsize=9, color='#E65100', ha='center',
                  arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.0))

    ax_b.set_xlabel('Curvature $c$')
    ax_b.set_ylabel('Approximation ratio')
    ax_b.set_xlim(0, 6)
    ax_b.set_ylim(0, 1.05)
    ax_b.legend(loc='upper right', fontsize=8)
    ax_b.set_title('(b) Approximation guarantees vs. curvature', fontsize=11)

    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f'Saved: {outpath}')
    return c_star


if __name__ == '__main__':
    c_star = make_figure()
    print(f'Crossover at c* = {c_star:.4f}')
