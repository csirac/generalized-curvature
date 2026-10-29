#!/usr/bin/env python3
"""Tier 2 experiments: large instances with singleton diagnostics.

Matches instance sizes from the additive-ratio literature:
  - Experimental design: n=200, k=20
  - Coverage with costs: n=300, k=30
  - Feature selection: n=100, k=15

No exact OPT — reports raw f(S) values and singleton trajectory diagnostics.
Uses heuristic "best known" for context but does NOT report ratios
as ground truth.

Usage:
    python run_tier2.py                   # all experiments
    python run_tier2.py --exp design      # single experiment
    python run_tier2.py --seeds 3         # fewer seeds
"""
from release_paths import RESULTS, FIGURES, ORIGINAL

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms import greedy_pruning, distorted_greedy, standard_greedy
from algorithms.lazy_greedy_pruning import lazy_greedy_pruning
from objectives import ExperimentalDesign, CoverageCost, FeatureSelection
from objectives.base import DecomposableObjective
from curvature import (opt_free_curvature_bound, guarantee_curvature,
                       guarantee_harshaw)
from opt_finder import find_best_solution

RESULTS_DIR = str(RESULTS)
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SEEDS = 5  # fewer seeds since instances are larger

# Tier 2 parameters
DESIGN_N, DESIGN_K = 200, 20
COVERAGE_N, COVERAGE_K = 300, 30
FEATURES_N, FEATURES_K = 100, 15

COST_SCALES_DESIGN = [0.0, 0.02, 0.05, 0.10, 0.15, 0.22, 0.30]
COST_SCALES_COVERAGE = [0.0, 0.5, 1.5, 3.0, 5.0, 8.0, 12.0]
COST_SCALES_FEATURES = [0.0, 0.03, 0.08, 0.15, 0.25, 0.4, 0.6]


def run_tier2_instance(f, k, name):
    """Run algorithms on a large instance with a singleton diagnostic."""
    t0 = time.time()

    results = {
        'instance': name,
        'n': f.n,
        'k': k,
    }

    # --- Greedy + Pruning (lazy for speed) ---
    gp = greedy_pruning(f, k, trace=True)
    results['greedy_pruning'] = {
        'value': gp['value'],
        'size': len(gp['solution']),
        'queries': gp.get('query_count', -1),
    }

    # Removal-marginal certificate (design remains diagnostic: not submodular).
    if isinstance(f, DecomposableObjective) and gp.get('trajectory'):
        from run_camera_ready import alpha_for_certificate
        from compute_removal_cert import compute_removal_marginal_cert
        alpha_g, alpha_source = alpha_for_certificate(f)
        cert = compute_removal_marginal_cert(f, gp['trajectory'], alpha_g)
        ofb = dict(alpha_g=alpha_g, s_hat=cert['r_hat'], c_g_bound=cert['c_g_cert'], guarantee=cert['guarantee'])
        results['certificate'] = cert
        results['gp_trajectory'] = gp['trajectory']
        results['alpha_g'] = ofb['alpha_g']
        results['s_hat'] = ofb['s_hat']
        results['c_g_certificate'] = ofb['c_g_bound']
        results['guarantee_certificate'] = ofb['guarantee']

        # g and ℓ values of GP solution
        gp_set = gp['solution']
        results['greedy_pruning']['g_value'] = f.g_evaluate(gp_set)
        results['greedy_pruning']['ell_value'] = f.ell_evaluate(gp_set)

    # --- Distorted Greedy ---
    if isinstance(f, DecomposableObjective):
        dg = distorted_greedy(f, k, trace=True)
        results['distorted_greedy'] = {
            'value': dg['value'],
            'size': len(dg['solution']),
            'g_value': dg['g_value'],
            'ell_value': dg['ell_value'],
        }

    # --- Standard Greedy ---
    sg = standard_greedy(f, k, trace=True)
    results['standard_greedy'] = {
        'value': sg['value'],
        'size': len(sg['solution']),
    }

    # BFNS14 fixed-k/dummy RG. Submitted Tier-1 shrinking-pool ratios are
    # retained separately and must not be relabelled as this implementation.
    from algorithms.random_greedy import random_greedy
    rg_runs = []
    for sub_seed in range(10):
        f.reset_queries()
        rg = random_greedy(f, k, seed=sub_seed * 1000, trace=True)
        rg_runs.append(dict(value=rg['value'], queries=f.query_counts,
                            seed=sub_seed * 1000))
    results['random_greedy'] = dict(value=float(np.mean([r['value'] for r in rg_runs])),
                                     runs=rg_runs)

    # --- Best-known solution (heuristic) for context ---
    best = find_best_solution(f, k, exact_threshold=0)  # force heuristic
    results['best_known'] = {
        'value': f.evaluate(best),
        'size': len(best),
    }
    if isinstance(f, DecomposableObjective):
        g_best = f.g_evaluate(best)
        ell_best = f.ell_evaluate(best)
        results['best_known']['g_value'] = g_best
        results['best_known']['ell_value'] = ell_best
        results['best_known']['rho'] = ell_best / max(g_best, 1e-12)
        # HFWK bound relative to best known
        results['harshaw_vs_best'] = guarantee_harshaw(results['best_known']['rho'])

    results['wall_time'] = time.time() - t0
    return results


def run_tier2_multi(instance_factory, cost_scales, k, app_name, n_seeds):
    """Run Tier 2 experiments across cost levels and seeds."""
    all_results = []

    for cs in cost_scales:
        seed_results = []
        for seed in range(n_seeds):
            f = instance_factory(cs, seed)
            name = f'{app_name}(cost={cs}, seed={seed})'
            r = run_tier2_instance(f, k, name)
            r['cost_scale'] = cs
            r['seed'] = seed
            seed_results.append(r)

        agg = aggregate_tier2(seed_results, cs, app_name)
        all_results.append(agg)

        # Print summary
        gp_val = agg['greedy_pruning']['value_mean']
        dg_val = agg.get('distorted_greedy', {}).get('value_mean', float('nan'))
        cert = agg.get('guarantee_certificate_mean', float('nan'))
        c_g = agg.get('c_g_certificate_mean', float('nan'))
        hrsh = agg.get('harshaw_vs_best_mean', float('nan'))
        wt = agg.get('wall_time_mean', 0)
        print(f'  cost={cs:<6.2f}  GP={gp_val:>9.3f}  DG={dg_val:>9.3f}  '
              f'ĉ_g={c_g:.3f}  cert={cert:.3f}  '
              f'hrshw={hrsh:.3f}  time={wt:.1f}s')

    return all_results


def aggregate_tier2(seed_results, cost_scale, app_name):
    """Aggregate Tier 2 results."""
    agg = {
        'application': app_name,
        'cost_scale': cost_scale,
        'n': seed_results[0]['n'],
        'k': seed_results[0]['k'],
        'n_seeds': len(seed_results),
    }

    # GP values
    gp_vals = [r['greedy_pruning']['value'] for r in seed_results]
    agg['greedy_pruning'] = {
        'value_mean': float(np.mean(gp_vals)),
        'value_std': float(np.std(gp_vals)),
    }

    # Curvature certificate
    certs = [r.get('guarantee_certificate', None) for r in seed_results]
    certs = [c for c in certs if c is not None]
    if certs:
        agg['guarantee_certificate_mean'] = float(np.mean(certs))
        agg['guarantee_certificate_std'] = float(np.std(certs))

    c_gs = [r.get('c_g_certificate', None) for r in seed_results]
    c_gs = [c for c in c_gs if c is not None and c < float('inf')]
    if c_gs:
        agg['c_g_certificate_mean'] = float(np.mean(c_gs))
        agg['c_g_certificate_std'] = float(np.std(c_gs))

    alphas = [r.get('alpha_g', None) for r in seed_results]
    alphas = [a for a in alphas if a is not None]
    if alphas:
        agg['alpha_g_mean'] = float(np.mean(alphas))

    s_hats = [r.get('s_hat', None) for r in seed_results]
    s_hats = [s for s in s_hats if s is not None]
    if s_hats:
        agg['s_hat_mean'] = float(np.mean(s_hats))

    # DG values
    dg_vals = [r['distorted_greedy']['value'] for r in seed_results
               if 'distorted_greedy' in r]
    if dg_vals:
        agg['distorted_greedy'] = {
            'value_mean': float(np.mean(dg_vals)),
            'value_std': float(np.std(dg_vals)),
        }

    # SG values
    sg_vals = [r['standard_greedy']['value'] for r in seed_results]
    agg['standard_greedy'] = {
        'value_mean': float(np.mean(sg_vals)),
        'value_std': float(np.std(sg_vals)),
    }

    # Best known
    bk_vals = [r['best_known']['value'] for r in seed_results]
    agg['best_known'] = {
        'value_mean': float(np.mean(bk_vals)),
        'value_std': float(np.std(bk_vals)),
    }

    # HFWK guarantee (relative to best known)
    h_vals = [r.get('harshaw_vs_best', None) for r in seed_results]
    h_vals = [h for h in h_vals if h is not None]
    if h_vals:
        agg['harshaw_vs_best_mean'] = float(np.mean(h_vals))

    # Best-known rho
    rhos = [r['best_known'].get('rho', None) for r in seed_results]
    rhos = [r for r in rhos if r is not None]
    if rhos:
        agg['rho_best_mean'] = float(np.mean(rhos))

    # Wall time
    wts = [r.get('wall_time', 0) for r in seed_results]
    agg['wall_time_mean'] = float(np.mean(wts))

    # Per-seed for detailed analysis
    agg['per_seed'] = seed_results

    return agg


# Instance factories — Tier 2 sizes

def make_design_t2(cost_scale, seed):
    return ExperimentalDesign.synthetic(
        n_exp=DESIGN_N, d=10, condition_number=5.0,
        cost_scale=cost_scale, noise_var=1.0, seed=seed
    )

def make_coverage_t2(cost_scale, seed):
    return CoverageCost.random_bipartite(
        n_vertices=COVERAGE_N, n_items=600,
        edge_prob=0.05, cost_scale=cost_scale, seed=seed
    )

def make_features_t2(cost_scale, seed):
    return FeatureSelection.synthetic(
        p=FEATURES_N, n_groups=10, within_corr=0.7,
        cost_scale=cost_scale, noise_var=1.0, seed=seed
    )


def save_results(results, filename):
    """Save results to JSON."""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, frozenset):
            return sorted(obj)
        if isinstance(obj, set):
            return sorted(obj)
        if obj is float('nan') or (isinstance(obj, float) and np.isnan(obj)):
            return None
        return obj

    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w') as fp:
        json.dump(results, fp, indent=2, default=convert)
    print(f'  Saved: {path}')


def print_tier2_summary(all_data):
    """Print Tier 2 summary table."""
    print('\n' + '=' * 100)
    print('TIER 2 SUMMARY (OPT-free certificates, large instances)')
    print('=' * 100)
    print(f'{"Application":<12} {"n":>4} {"k":>3} {"Cost":>5} '
          f'{"GP":>10} {"DG":>10} {"Best":>10} '
          f'{"α_g":>6} {"ĉ_g":>6} {"Cert":>6} {"Hrshw":>6}')
    print('-' * 100)

    for app_name, results in all_data.items():
        for r in results:
            cs = r['cost_scale']
            n = r['n']
            k = r['k']
            gp = r['greedy_pruning']['value_mean']
            dg = r.get('distorted_greedy', {}).get('value_mean', float('nan'))
            bk = r['best_known']['value_mean']
            alpha = r.get('alpha_g_mean', float('nan'))
            c_g = r.get('c_g_certificate_mean', float('nan'))
            cert = r.get('guarantee_certificate_mean', float('nan'))
            hrsh = r.get('harshaw_vs_best_mean', float('nan'))

            print(f'{app_name:<12} {n:>4} {k:>3} {cs:>5.2f} '
                  f'{gp:>10.3f} {dg:>10.3f} {bk:>10.3f} '
                  f'{alpha:>6.3f} {c_g:>6.3f} {cert:>6.3f} {hrsh:>6.3f}')
        print()


def generate_tier2_figures(all_data):
    """Generate Tier 2 figures: GP vs DG values + diagnostic comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 10, 'font.family': 'serif',
        'axes.labelsize': 11, 'legend.fontsize': 8,
        'figure.figsize': (5.5, 4), 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'lines.linewidth': 1.5,
    })

    COLORS = {
        'gp': '#2196F3', 'dg': '#F44336', 'sg': '#FF9800',
        'cert': '#7B1FA2', 'harshaw': '#C62828',
    }

    for app_name, results in all_data.items():
        prefix = app_name.lower().replace(' ', '_')
        costs = [r['cost_scale'] for r in results]

        gp_means = [r['greedy_pruning']['value_mean'] for r in results]
        gp_stds = [r['greedy_pruning']['value_std'] for r in results]
        dg_means = [r.get('distorted_greedy', {}).get('value_mean', 0) for r in results]
        dg_stds = [r.get('distorted_greedy', {}).get('value_std', 0) for r in results]
        bk_means = [r['best_known']['value_mean'] for r in results]

        # Figure: raw f(S) values
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        costs_arr = np.array(costs)
        ax1.plot(costs_arr, gp_means, '-o', color=COLORS['gp'], markersize=4,
                label='Greedy+Pruning')
        ax1.fill_between(costs_arr,
                         np.array(gp_means) - np.array(gp_stds),
                         np.array(gp_means) + np.array(gp_stds),
                         color=COLORS['gp'], alpha=0.15)
        ax1.plot(costs_arr, dg_means, '--s', color=COLORS['dg'], markersize=4,
                label='Distorted Greedy')
        ax1.fill_between(costs_arr,
                         np.array(dg_means) - np.array(dg_stds),
                         np.array(dg_means) + np.array(dg_stds),
                         color=COLORS['dg'], alpha=0.1)
        ax1.plot(costs_arr, bk_means, ':^', color='gray', markersize=3,
                label='Best known', alpha=0.5)
        ax1.set_xlabel('Cost scale')
        ax1.set_ylabel('$f(S)$')
        ax1.set_title(f'{app_name} (n={results[0]["n"]}, k={results[0]["k"]})')
        ax1.legend(fontsize=7)

        # Figure: singleton diagnostic vs HFWK
        certs = [r.get('guarantee_certificate_mean', 0) for r in results]
        hrshws = [r.get('harshaw_vs_best_mean', 0) for r in results]

        ax2.plot(costs_arr, certs, '-o', color=COLORS['cert'], markersize=4,
                label='Singleton diagnostic')
        ax2.plot(costs_arr, hrshws, '--s', color=COLORS['harshaw'], markersize=4,
                label='HFWK guarantee')
        ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax2.set_xlabel('Cost scale')
        ax2.set_ylabel('Diagnostic / guarantee')
        ax2.set_title(f'{app_name}: diagnostics')
        ax2.legend(fontsize=7)

        fig.tight_layout()
        outpath = os.path.join(RESULTS_DIR, f'fig_tier2_{prefix}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'  Saved: {outpath}')

    # Combined certificate comparison
    if len(all_data) > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (app_name, results) in enumerate(all_data.items()):
            costs = [r['cost_scale'] for r in results]
            certs = [r.get('guarantee_certificate_mean', 0) for r in results]
            hrshws = [r.get('harshaw_vs_best_mean', 0) for r in results]
            linestyle = ['-', '--', ':'][i % 3]
            color_c = ['#2196F3', '#4CAF50', '#FF9800'][i % 3]
            color_h = ['#C62828', '#880E4F', '#E65100'][i % 3]
            ax.plot(costs, certs, linestyle + 'o', color=color_c, markersize=3,
                    label=f'{app_name} (curvature)')
            ax.plot(costs, hrshws, linestyle + 's', color=color_h, markersize=3,
                    alpha=0.5, label=f'{app_name} (HFWK)')
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Cost scale')
        ax.set_ylabel('Diagnostic / guarantee')
        ax.set_title('Moderate-scale: Singleton diagnostics vs HFWK')
        ax.legend(fontsize=6, ncol=2)
        outpath = os.path.join(RESULTS_DIR, 'fig_tier2_combined.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'  Saved: {outpath}')


def main():
    global DESIGN_N, DESIGN_K, COVERAGE_N, COVERAGE_K, FEATURES_N, FEATURES_K
    global COST_SCALES_DESIGN, COST_SCALES_COVERAGE, COST_SCALES_FEATURES
    parser = argparse.ArgumentParser(description='Run curvature experiments (Tier 2)')
    parser.add_argument('--exp', choices=['design', 'coverage', 'features', 'all'],
                        default='all')
    parser.add_argument('--seeds', type=int, default=N_SEEDS)
    parser.add_argument('--camera-ready', action='store_true', help='New runner: all six baselines, counts, provenance and separate files')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--smoke', action='store_true', help='Small synthetic test, not paper settings')
    args = parser.parse_args()

    if args.camera_ready:
        import concurrent.futures
        from run_camera_ready import cells, run_instance
        jobs = [p for p in cells('tier2') if p['seed'] < args.seeds and (args.exp == 'all' or p['app'] == args.exp)]
        if not 1 <= args.workers <= 15:
            parser.error('workers must be 1..15')
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(run_instance, jobs):
                print(result, flush=True)
        return

    if args.smoke:
        DESIGN_N = COVERAGE_N = FEATURES_N = 12
        DESIGN_K = COVERAGE_K = FEATURES_K = 2
        COST_SCALES_DESIGN = COST_SCALES_COVERAGE = COST_SCALES_FEATURES = [0.0, 0.1]

    n_seeds = args.seeds
    print('Curvature — Tier 2 Experiments (large instances, singleton diagnostics)')
    print(f'Results directory: {RESULTS_DIR}')

    all_data = {}

    if args.exp in ('all', 'design'):
        print(f'\n{"#" * 60}')
        print(f'TIER 2 — Experimental Design (n={DESIGN_N}, k={DESIGN_K}, d=10)')
        print(f'{"#" * 60}')
        results = run_tier2_multi(make_design_t2, COST_SCALES_DESIGN,
                                  DESIGN_K, 'ExpDesign', n_seeds)
        save_results(results, 'tier2_experimental_design.json')
        all_data['Exp. Design'] = results

    if args.exp in ('all', 'coverage'):
        print(f'\n{"#" * 60}')
        print(f'TIER 2 — Coverage with Costs (n={COVERAGE_N}, k={COVERAGE_K})')
        print(f'{"#" * 60}')
        results = run_tier2_multi(make_coverage_t2, COST_SCALES_COVERAGE,
                                  COVERAGE_K, 'Coverage', n_seeds)
        save_results(results, 'tier2_coverage_cost.json')
        all_data['Coverage'] = results

    if args.exp in ('all', 'features'):
        print(f'\n{"#" * 60}')
        print(f'TIER 2 — Feature Selection (n={FEATURES_N}, k={FEATURES_K})')
        print(f'{"#" * 60}')
        results = run_tier2_multi(make_features_t2, COST_SCALES_FEATURES,
                                  FEATURES_K, 'FeatureSel', n_seeds)
        save_results(results, 'tier2_feature_selection.json')
        all_data['Feature Sel.'] = results

    if all_data:
        print_tier2_summary(all_data)
        generate_tier2_figures(all_data)

    print('\nDone.')


if __name__ == '__main__':
    main()
