"""Plotting utilities for the curvature experiments.

Generates publication-quality figures for the NeurIPS paper.

Figures:
  - fig:crossover — (1-e^{-c})/c vs 1/e crossover
  - fig:cost_sweep — cost sweep: algorithms + guarantees vs cost level
  - fig:tightness — curvature-vs-ratio scatter: empirical c_g vs singleton bound
  - fig:guarantee_comparison — combined guarantee comparison across apps
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# NeurIPS-compatible style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.figsize': (5.5, 3.5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
    'text.usetex': False,
})

COLORS = {
    'gp': '#2196F3',       # blue — Greedy+Pruning
    'dg': '#F44336',       # red — Distorted Greedy
    'rg': '#4CAF50',       # green — Random Greedy
    'sg': '#FF9800',       # orange — Standard Greedy
    'curv': '#1565C0',     # dark blue — curvature guarantee
    'harshaw': '#C62828',  # dark red — HFWK guarantee
    'opt_free': '#7B1FA2', # purple — singleton diagnostic
}


def plot_crossover(outpath='results/fig_crossover.pdf'):
    """Plot (1-e^{-c})/c vs 1/e — the curvature-BFNS crossover."""
    c = np.linspace(0.01, 6, 500)
    curv_guarantee = (1 - np.exp(-np.maximum(1, c))) / np.maximum(1, c)
    bfns_guarantee = np.full_like(c, 1.0 / np.e)

    from scipy.optimize import brentq
    c_star = brentq(lambda x: (1 - np.exp(-x)) / x - 1 / np.e, 1.0, 5.0)

    fig, ax = plt.subplots()
    ax.plot(c, curv_guarantee, 'b-', label=r'Curvature: $(1-e^{-\bar c})/\bar c$')
    ax.plot(c, bfns_guarantee, 'r--', label=r'BFNS: $1/e$')

    ax.axvline(c_star, color='gray', linestyle=':', alpha=0.5)
    ax.annotate(f'$c^* \\approx {c_star:.2f}$',
                xy=(c_star, 1 / np.e), xytext=(c_star + 0.3, 1 / np.e + 0.06),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, color='gray')

    ax.plot(1, 1 - 1 / np.e, 'ko', markersize=5)
    ax.annotate('$c=1$ (monotone)', xy=(1, 1 - 1 / np.e),
                xytext=(1.3, 0.7), fontsize=8, color='black',
                arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_xlabel('Curvature $c$')
    ax.set_ylabel('Approximation ratio')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')

    fig.savefig(outpath)
    plt.close(fig)
    print(f'  Saved: {outpath}')
    return c_star


def plot_cost_sweep(results, outpath, title=''):
    """Plot Fig 2: cost sweep comparison.

    X-axis: cost ratio ρ = ℓ(OPT)/g(OPT)
    Lines: GP ratio, DG ratio (empirical), curvature guarantee, Harshaw guarantee
    Shaded: ± 1 std across seeds
    Vertical line: where Harshaw goes vacuous

    Args:
        results: list of aggregated dicts (one per cost level)
        outpath: path for PDF
        title: figure title
    """
    rhos = [r.get('rho_opt_mean', r['cost_scale']) for r in results]
    gp_means = [r['greedy_pruning']['ratio_mean'] for r in results]
    gp_stds = [r['greedy_pruning'].get('ratio_std', 0) for r in results]
    dg_means = [r.get('distorted_greedy', {}).get('ratio_mean', float('nan'))
                for r in results]
    dg_stds = [r.get('distorted_greedy', {}).get('ratio_std', 0) for r in results]
    from curvature import guarantee_curvature
    curv_means = [float(np.mean([guarantee_curvature(s['empirical_c_g'])
                    for s in r.get('per_seed', []) if 'empirical_c_g' in s]))
                  for r in results]
    curv_stds = [r.get('curv_guarantee_std', 0) for r in results]
    harshaw_means = [r.get('harshaw_guarantee_mean', float('nan')) for r in results]
    ofb_means = [r.get('guarantee_opt_free_mean', float('nan')) for r in results]

    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Empirical ratios (solid lines with error bands)
    rhos_arr = np.array(rhos)
    gp_arr = np.array(gp_means)
    gp_std_arr = np.array(gp_stds)
    dg_arr = np.array(dg_means)
    dg_std_arr = np.array(dg_stds)

    ax.plot(rhos_arr, gp_arr, '-o', color=COLORS['gp'], markersize=4,
            label='Greedy+Pruning', zorder=5)
    ax.fill_between(rhos_arr, gp_arr - gp_std_arr, gp_arr + gp_std_arr,
                    color=COLORS['gp'], alpha=0.15)

    # Filter out NaN for DG
    valid_dg = ~np.isnan(dg_arr)
    if valid_dg.any():
        ax.plot(rhos_arr[valid_dg], dg_arr[valid_dg], '--s', color=COLORS['dg'],
                markersize=4, label='Distorted Greedy', zorder=4)
        ax.fill_between(rhos_arr[valid_dg],
                        (dg_arr - dg_std_arr)[valid_dg],
                        (dg_arr + dg_std_arr)[valid_dg],
                        color=COLORS['dg'], alpha=0.1)

    # Guarantees (dashed lines, thinner)
    curv_arr = np.array(curv_means)
    valid_curv = ~np.isnan(curv_arr)
    if valid_curv.any():
        ax.plot(rhos_arr[valid_curv], curv_arr[valid_curv], '-', linewidth=1.2,
                color=COLORS['curv'], alpha=0.7, label='Curvature guarantee')

    harshaw_arr = np.array(harshaw_means)
    valid_h = ~np.isnan(harshaw_arr)
    if valid_h.any():
        ax.plot(rhos_arr[valid_h], harshaw_arr[valid_h], '--', linewidth=1.2,
                color=COLORS['harshaw'], alpha=0.7, label='HFWK guarantee')

    # Singleton trajectory diagnostic
    ofb_arr = np.array(ofb_means)
    valid_ofb = ~np.isnan(ofb_arr)
    if valid_ofb.any():
        ax.plot(rhos_arr[valid_ofb], ofb_arr[valid_ofb], ':', linewidth=1.2,
                color=COLORS['opt_free'], alpha=0.7, label='Removal certificate')

    # Mark where Harshaw goes vacuous
    vacuous_rho = 1.0 - 1.0 / np.e  # ≈ 0.632
    if rhos_arr.max() > vacuous_rho * 0.8:
        ax.axvline(vacuous_rho, color=COLORS['harshaw'], linestyle=':', alpha=0.4)
        ax.annotate('HFWK\nvacuous', xy=(vacuous_rho, 0.02),
                    fontsize=7, color=COLORS['harshaw'], alpha=0.7,
                    ha='center')

    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel(r'Cost ratio $\rho = \ell(O^*)/g(O^*)$')
    ax.set_ylabel(r'$f(S)/f(O^*)$')
    # Keep informative positive ratios readable; negative HFWK values remain
    # unchanged in the data and are explicitly identified as off scale.
    ax.set_ylim(-0.5, 1.1)
    ax.legend(loc='best', fontsize=7)
    if title:
        ax.set_title(title)

    fig.savefig(outpath)
    plt.close(fig)
    print(f'  Saved: {outpath}')


def plot_curvature_tightness(results, outpath, title=''):
    """Plot Fig 3: curvature-vs-ratio scatter.

    For each (cost_level, seed), plot:
    - X = empirical c_g (exact, uses OPT)
    - Y = actual f(S)/f(OPT) ratio
    Overlay the curvature guarantee curve.

    Args:
        results: list of aggregated dicts (with per_seed data)
        outpath: path for PDF
        title: figure title
    """
    # Collect all per-seed points
    c_gs = []
    ratios = []
    c_g_bounds = []
    for r in results:
        for sr in r.get('per_seed', []):
            if 'empirical_c_g' in sr and sr.get('f_opt', 0) > 0:
                c_gs.append(sr['empirical_c_g'])
                ratios.append(sr['greedy_pruning']['ratio'])
                if 'c_g_bound_opt_free' in sr:
                    c_g_bounds.append(sr['c_g_bound_opt_free'])
                else:
                    c_g_bounds.append(None)

    if not c_gs:
        return

    c_gs = np.array(c_gs)
    ratios = np.array(ratios)

    fig, ax = plt.subplots(figsize=(5, 4))

    # Curvature guarantee curve
    c_range = np.linspace(0.01, max(c_gs.max() + 0.5, 3), 200)
    c_bar = np.maximum(1.0, c_range)
    curv_curve = (1 - np.exp(-c_bar)) / c_bar
    ax.plot(c_range, curv_curve, '-', color=COLORS['curv'], linewidth=2,
            label=r'$(1-e^{-\bar c_g})/\bar c_g$', zorder=3)

    # Scatter of actual (c_g, ratio) points
    ax.scatter(c_gs, ratios, s=20, c=COLORS['gp'], alpha=0.6,
               edgecolors='white', linewidth=0.5, zorder=4,
               label='GP instances')

    # Singleton bound points (horizontal arrows from bound to empirical c_g)
    for i, cb in enumerate(c_g_bounds):
        if cb is not None and cb <= max(c_gs.max() + 0.5, 3):
            ax.annotate('', xy=(c_gs[i], ratios[i]),
                        xytext=(cb, ratios[i]),
                        arrowprops=dict(arrowstyle='<-', color=COLORS['opt_free'],
                                       alpha=0.3, linewidth=0.5))

    ax.set_xlabel(r'Greedy curvature $c_g$')
    ax.set_ylabel(r'$f(S)/f(O^*)$')
    ax.legend(loc='upper right', fontsize=8)
    if title:
        ax.set_title(f'{title}: curvature vs ratio')
    ax.set_ylim(0, 1.1)

    fig.savefig(outpath)
    plt.close(fig)
    print(f'  Saved: {outpath}')


def plot_guarantee_comparison(all_data, outpath):
    """Plot combined guarantee comparison across applications.

    For each application at moderate cost, show a bar group:
    [GP ratio, DG ratio, curvature guarantee, Harshaw guarantee]

    Args:
        all_data: dict mapping app_name -> list of aggregated results
        outpath: path for PDF
    """
    # Pick a moderate cost level for each application (middle of the range)
    entries = []
    for app_name, results in all_data.items():
        mid = len(results) // 2
        r = results[mid]
        entries.append({
            'app': app_name,
            'rho': r.get('rho_opt_mean', r['cost_scale']),
            'gp': r['greedy_pruning']['ratio_mean'],
            'dg': r.get('distorted_greedy', {}).get('ratio_mean', 0),
            'curv': r.get('curv_guarantee_mean', 0),
            'harshaw': max(r.get('harshaw_guarantee_mean', 0), 0),
            'opt_free': r.get('guarantee_opt_free_mean', 0),
        })

    apps = [e['app'] for e in entries]
    x = np.arange(len(apps))
    w = 0.15

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(x - 2*w, [e['gp'] for e in entries], w,
           color=COLORS['gp'], label='GP (empirical)')
    ax.bar(x - w, [e['dg'] for e in entries], w,
           color=COLORS['dg'], label='DG (empirical)')
    ax.bar(x, [e['curv'] for e in entries], w,
           color=COLORS['curv'], label='Curvature guarantee')
    ax.bar(x + w, [e['opt_free'] for e in entries], w,
           color=COLORS['opt_free'], label='Removal certificate')
    ax.bar(x + 2*w, [e['harshaw'] for e in entries], w,
           color=COLORS['harshaw'], label='HFWK guarantee')

    # Annotate ρ values
    for i, e in enumerate(entries):
        ax.text(x[i], -0.08, f'ρ={e["rho"]:.2f}', ha='center', fontsize=7,
                color='gray')

    ax.set_ylabel(r'$f(S)/f(O^*)$')
    ax.set_xticks(x)
    ax.set_xticklabels(apps, fontsize=9)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_ylim(-0.15, 1.1)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)

    fig.savefig(outpath)
    plt.close(fig)
    print(f'  Saved: {outpath}')
