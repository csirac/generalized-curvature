#!/usr/bin/env python3
"""Main experiment runner — Tier 1 (n ≤ 20, exact OPT).

Runs all experiments with multiple seeds, computes exact and OPT-free
curvature bounds, and generates publication figures.

Usage:
    python run_all.py                   # all experiments
    python run_all.py --exp design      # experimental design only
    python run_all.py --exp coverage    # coverage only
    python run_all.py --exp features    # feature selection only
    python run_all.py --exp crossover   # crossover figure only
    python run_all.py --seeds 5         # fewer seeds for quick test

Results are saved to experiments/results/.
"""
from release_paths import RESULTS, FIGURES, ORIGINAL

import argparse
import json
import os
import sys
import time

import numpy as np

# Add parent dir so imports work when running from experiments/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms import greedy_pruning, distorted_greedy, random_greedy, standard_greedy
from algorithms.lazy_greedy_pruning import lazy_greedy_pruning
from objectives import ExperimentalDesign, CoverageCost, FeatureSelection
from objectives.base import DecomposableObjective
from curvature import (empirical_greedy_curvature, decomposable_curvature_bound,
                       opt_free_curvature_bound,
                       guarantee_curvature, guarantee_harshaw)
from plotting import plot_crossover
from opt_finder import find_best_solution


RESULTS_DIR = str(RESULTS)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Tier 1 parameters: exact OPT via enumeration
N = 20       # ground set size (C(20,5) = 15504 subsets — fast)
K = 5        # cardinality constraint
N_SEEDS = 10 # seeds per configuration
COST_SCALES_DESIGN = [0.0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.28]
COST_SCALES_COVERAGE = [0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0]
COST_SCALES_FEATURES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]


def run_on_instance(f, k, name, exact_opt=True):
    """Run all algorithms on a single instance and collect results.

    Args:
        f: objective function
        k: cardinality constraint
        name: instance name for reporting
        exact_opt: if True, use exact enumeration (n must be ≤ 22)

    Returns:
        dict with all results
    """
    # Find OPT
    if exact_opt:
        opt_set = find_best_solution(f, k, exact_threshold=f.n + 1)
    else:
        opt_set = find_best_solution(f, k)
    f_opt = f.evaluate(opt_set)

    results = {
        'instance': name,
        'n': f.n,
        'k': k,
        'f_opt': f_opt,
        'opt_size': len(opt_set),
        'opt_exact': exact_opt,
    }

    # --- Greedy + Pruning ---
    gp = greedy_pruning(f, k, trace=True)
    results['greedy_pruning'] = {
        'value': gp['value'],
        'ratio': gp['value'] / max(f_opt, 1e-12) if f_opt > 0 else float('nan'),
        'size': len(gp['solution']),
    }

    # Empirical c_g (exact, uses OPT)
    if gp.get('trajectory') and f_opt > 0:
        curv = empirical_greedy_curvature(f, gp['trajectory'], opt_set)
        results['empirical_c_g'] = curv['c_g']
        results['curv_guarantee'] = guarantee_curvature(curv['c_g'])

    # Decomposable bounds (if applicable)
    if isinstance(f, DecomposableObjective) and gp.get('trajectory'):
        # OPT-aware bound
        dcb = decomposable_curvature_bound(f, gp['trajectory'], opt_set)
        results['alpha_g_sampled'] = dcb['alpha_g_approx']
        results['r_max'] = dcb['r_max']
        results['c_g_bound_opt_aware'] = dcb['c_g_bound']

        # Analytic α_g
        if hasattr(f, 'alpha_g_exact'):
            alpha_g = f.alpha_g_exact
            results['alpha_g_exact'] = alpha_g
        else:
            alpha_g = dcb['alpha_g_approx']
            results['alpha_g_exact'] = None

        # OPT-free bound (uses analytic α_g + trajectory only)
        ofb = opt_free_curvature_bound(f, gp['trajectory'], alpha_g=alpha_g)
        results['s_hat'] = ofb['s_hat']
        results['c_g_bound_opt_free'] = ofb['c_g_bound']
        results['guarantee_opt_free'] = ofb['guarantee']

        # Cost ratio at OPT: ρ = ℓ(OPT) / g(OPT)
        g_opt = f.g_evaluate(opt_set)
        ell_opt = f.ell_evaluate(opt_set)
        results['rho_opt'] = ell_opt / max(g_opt, 1e-12)
        results['harshaw_guarantee'] = guarantee_harshaw(results['rho_opt'])

        # Does f go negative on this instance?
        results['f_goes_negative'] = any(
            f.evaluate(frozenset([e])) < 0 for e in range(f.n)
        )

    # --- Distorted Greedy ---
    if isinstance(f, DecomposableObjective):
        dg = distorted_greedy(f, k, trace=True)
        results['distorted_greedy'] = {
            'value': dg['value'],
            'ratio': dg['value'] / max(f_opt, 1e-12) if f_opt > 0 else float('nan'),
            'size': len(dg['solution']),
            'g_value': dg['g_value'],
            'ell_value': dg['ell_value'],
        }

    # --- Random Greedy (10 sub-seeds) ---
    rg_values = []
    for sub_seed in range(10):
        rg = random_greedy(f, k, seed=sub_seed * 1000)
        rg_values.append(rg['value'])
    rg_mean = np.mean(rg_values)
    results['random_greedy'] = {
        'value_mean': rg_mean,
        'value_std': float(np.std(rg_values)),
        'ratio_mean': rg_mean / max(f_opt, 1e-12) if f_opt > 0 else float('nan'),
    }

    # --- Standard Greedy ---
    sg = standard_greedy(f, k, trace=True)
    results['standard_greedy'] = {
        'value': sg['value'],
        'ratio': sg['value'] / max(f_opt, 1e-12) if f_opt > 0 else float('nan'),
        'size': len(sg['solution']),
    }

    return results


def run_multi_seed(instance_factory, cost_scales, k, app_name, n_seeds):
    """Run an experiment across cost levels and seeds.

    Args:
        instance_factory: callable(cost_scale, seed) -> ObjectiveFunction
        cost_scales: list of cost scale parameters
        k: cardinality constraint
        app_name: application name for reporting
        n_seeds: number of random seeds

    Returns:
        list of aggregated results (one per cost level)
    """
    all_results = []

    for cs in cost_scales:
        seed_results = []
        for seed in range(n_seeds):
            f = instance_factory(cs, seed)
            name = f'{app_name}(cost={cs}, seed={seed})'
            r = run_on_instance(f, k, name, exact_opt=True)
            r['cost_scale'] = cs
            r['seed'] = seed
            seed_results.append(r)

        # Aggregate across seeds
        agg = aggregate_seeds(seed_results, cs, app_name)
        all_results.append(agg)

        # Print summary line
        gp_ratio = agg['greedy_pruning']['ratio_mean']
        dg_ratio = agg.get('distorted_greedy', {}).get('ratio_mean', float('nan'))
        c_g = agg.get('empirical_c_g_mean', float('nan'))
        rho = agg.get('rho_opt_mean', 0)
        print(f'  cost={cs:<6.2f}  ρ={rho:.3f}  '
              f'GP={gp_ratio:.3f}  DG={dg_ratio:.3f}  '
              f'c_g={c_g:.3f}  '
              f'curv_guar={agg.get("curv_guarantee_mean", 0):.3f}  '
              f'harshaw={agg.get("harshaw_guarantee_mean", 0):.3f}')

    return all_results


def aggregate_seeds(seed_results, cost_scale, app_name):
    """Aggregate results across seeds into mean ± std."""
    agg = {
        'application': app_name,
        'cost_scale': cost_scale,
        'n': seed_results[0]['n'],
        'k': seed_results[0]['k'],
        'n_seeds': len(seed_results),
    }

    # f(OPT)
    f_opts = [r['f_opt'] for r in seed_results]
    agg['f_opt_mean'] = float(np.mean(f_opts))
    agg['f_opt_std'] = float(np.std(f_opts))

    # Greedy+Pruning
    gp_vals = [r['greedy_pruning']['value'] for r in seed_results]
    gp_ratios = [r['greedy_pruning']['ratio'] for r in seed_results
                 if not np.isnan(r['greedy_pruning']['ratio'])]
    agg['greedy_pruning'] = {
        'value_mean': float(np.mean(gp_vals)),
        'value_std': float(np.std(gp_vals)),
        'ratio_mean': float(np.mean(gp_ratios)) if gp_ratios else float('nan'),
        'ratio_std': float(np.std(gp_ratios)) if gp_ratios else float('nan'),
    }

    # Empirical c_g
    c_gs = [r['empirical_c_g'] for r in seed_results if 'empirical_c_g' in r]
    if c_gs:
        agg['empirical_c_g_mean'] = float(np.mean(c_gs))
        agg['empirical_c_g_std'] = float(np.std(c_gs))
        curv_gs = [r['curv_guarantee'] for r in seed_results if 'curv_guarantee' in r]
        agg['curv_guarantee_mean'] = float(np.mean(curv_gs))
        agg['curv_guarantee_std'] = float(np.std(curv_gs))

    # Alpha_g (should be the same across seeds for same n, but average anyway)
    alphas = [r['alpha_g_exact'] for r in seed_results
              if r.get('alpha_g_exact') is not None]
    if alphas:
        agg['alpha_g_exact_mean'] = float(np.mean(alphas))

    # OPT-free bound
    ofb_cgs = [r['c_g_bound_opt_free'] for r in seed_results
               if 'c_g_bound_opt_free' in r and r['c_g_bound_opt_free'] < float('inf')]
    if ofb_cgs:
        agg['c_g_bound_opt_free_mean'] = float(np.mean(ofb_cgs))
        agg['c_g_bound_opt_free_std'] = float(np.std(ofb_cgs))
        guar_ofb = [r['guarantee_opt_free'] for r in seed_results
                    if 'guarantee_opt_free' in r]
        agg['guarantee_opt_free_mean'] = float(np.mean(guar_ofb))

    # Harshaw guarantee
    rhos = [r['rho_opt'] for r in seed_results if 'rho_opt' in r]
    if rhos:
        agg['rho_opt_mean'] = float(np.mean(rhos))
        agg['rho_opt_std'] = float(np.std(rhos))
        harshaw_gs = [r['harshaw_guarantee'] for r in seed_results
                      if 'harshaw_guarantee' in r]
        agg['harshaw_guarantee_mean'] = float(np.mean(harshaw_gs))
        agg['harshaw_guarantee_std'] = float(np.std(harshaw_gs))

    # Distorted Greedy
    dg_vals = [r['distorted_greedy']['value'] for r in seed_results
               if 'distorted_greedy' in r]
    dg_ratios = [r['distorted_greedy']['ratio'] for r in seed_results
                 if 'distorted_greedy' in r
                 and not np.isnan(r['distorted_greedy']['ratio'])]
    if dg_vals:
        agg['distorted_greedy'] = {
            'value_mean': float(np.mean(dg_vals)),
            'value_std': float(np.std(dg_vals)),
            'ratio_mean': float(np.mean(dg_ratios)) if dg_ratios else float('nan'),
            'ratio_std': float(np.std(dg_ratios)) if dg_ratios else float('nan'),
        }

    # Standard Greedy
    sg_vals = [r['standard_greedy']['value'] for r in seed_results]
    sg_ratios = [r['standard_greedy']['ratio'] for r in seed_results
                 if not np.isnan(r['standard_greedy']['ratio'])]
    agg['standard_greedy'] = {
        'value_mean': float(np.mean(sg_vals)),
        'ratio_mean': float(np.mean(sg_ratios)) if sg_ratios else float('nan'),
    }

    # Random Greedy
    rg_means = [r['random_greedy']['value_mean'] for r in seed_results]
    rg_ratios = [r['random_greedy']['ratio_mean'] for r in seed_results
                 if not np.isnan(r['random_greedy']['ratio_mean'])]
    agg['random_greedy'] = {
        'value_mean': float(np.mean(rg_means)),
        'ratio_mean': float(np.mean(rg_ratios)) if rg_ratios else float('nan'),
    }

    # f goes negative?
    agg['f_goes_negative'] = any(
        r.get('f_goes_negative', False) for r in seed_results
    )

    # Keep per-seed data for scatter plots
    agg['per_seed'] = seed_results

    return agg


# ---- Instance factories ----

def make_design_instance(cost_scale, seed):
    return ExperimentalDesign.synthetic(
        n_exp=N, d=5, condition_number=5.0,
        cost_scale=cost_scale, noise_var=1.0, seed=seed
    )

def make_coverage_instance(cost_scale, seed):
    return CoverageCost.random_bipartite(
        n_vertices=N, n_items=40,
        edge_prob=0.2, cost_scale=cost_scale, seed=seed
    )

def make_features_instance(cost_scale, seed):
    return FeatureSelection.synthetic(
        p=N, n_groups=4, within_corr=0.7,
        cost_scale=cost_scale, noise_var=1.0, seed=seed
    )


# ---- Experiment runners ----

def experiment_design(n_seeds):
    """Experiment 1: Bayesian A-optimality with varying cost levels."""
    print('\n' + '#' * 60)
    print('EXPERIMENT 1: Experimental Design (Bayesian A-optimality)')
    print(f'  n={N}, k={K}, d=5, κ=5, {n_seeds} seeds')
    print('#' * 60)

    results = run_multi_seed(make_design_instance, COST_SCALES_DESIGN,
                             K, 'ExpDesign', n_seeds)
    save_results(results, 'experimental_design.json')
    return results


def experiment_coverage(n_seeds):
    """Experiment 2: Directed vertex cover with costs."""
    print('\n' + '#' * 60)
    print('EXPERIMENT 2: Coverage with Costs (Directed Vertex Cover)')
    print(f'  n={N}, k={K}, m=40, {n_seeds} seeds')
    print('#' * 60)

    results = run_multi_seed(make_coverage_instance, COST_SCALES_COVERAGE,
                             K, 'Coverage', n_seeds)
    save_results(results, 'coverage_cost.json')
    return results


def experiment_features(n_seeds):
    """Experiment 3: Feature selection with MI and ℓ₁ cost."""
    print('\n' + '#' * 60)
    print('EXPERIMENT 3: Feature Selection (MI - ℓ₁)')
    print(f'  n={N}, k={K}, groups=4, ρ=0.7, {n_seeds} seeds')
    print('#' * 60)

    results = run_multi_seed(make_features_instance, COST_SCALES_FEATURES,
                             K, 'FeatureSel', n_seeds)
    save_results(results, 'feature_selection.json')
    return results


def experiment_crossover():
    """Generate the crossover figure (fig:crossover in the paper)."""
    print('\n' + '#' * 60)
    print('Generating crossover figure')
    print('#' * 60)
    c_star = plot_crossover(os.path.join(RESULTS_DIR, 'fig_crossover.pdf'))
    print(f'Crossover at c* = {c_star:.4f}')
    return c_star


def generate_figures(all_data):
    """Generate all publication figures from experiment data.

    Args:
        all_data: dict mapping app_name -> list of aggregated results
    """
    from plotting import (plot_cost_sweep, plot_curvature_tightness,
                          plot_guarantee_comparison)

    for app_name, results in all_data.items():
        prefix = app_name.lower().replace(' ', '_')

        # Fig 2-style: cost sweep comparison
        plot_cost_sweep(results,
                        os.path.join(RESULTS_DIR, f'fig_{prefix}_sweep.pdf'),
                        title=app_name)

        # Fig 3-style: curvature tightness scatter
        plot_curvature_tightness(results,
                                  os.path.join(RESULTS_DIR, f'fig_{prefix}_tightness.pdf'),
                                  title=app_name)

    # Combined guarantee comparison across all applications
    if len(all_data) > 1:
        plot_guarantee_comparison(all_data,
                                  os.path.join(RESULTS_DIR, 'fig_guarantee_comparison.pdf'))


def save_results(results, filename):
    """Save results to JSON (convert non-serializable types)."""
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


def print_summary_table(all_data):
    """Print a summary table across all experiments."""
    print('\n' + '=' * 90)
    print('SUMMARY TABLE')
    print('=' * 90)
    print(f'{"Application":<14} {"Cost":>5} {"ρ":>6} {"α_g":>6} '
          f'{"c_g":>6} {"ĉ_g":>6} '
          f'{"GP":>6} {"DG":>6} {"Curv":>6} {"Hrshw":>6} {"f<0":>4}')
    print('-' * 90)

    for app_name, results in all_data.items():
        for r in results:
            cs = r['cost_scale']
            rho = r.get('rho_opt_mean', 0)
            alpha = r.get('alpha_g_exact_mean', float('nan'))
            c_g = r.get('empirical_c_g_mean', float('nan'))
            c_g_of = r.get('c_g_bound_opt_free_mean', float('nan'))
            gp = r['greedy_pruning']['ratio_mean']
            dg = r.get('distorted_greedy', {}).get('ratio_mean', float('nan'))
            curv_g = r.get('curv_guarantee_mean', float('nan'))
            h_g = r.get('harshaw_guarantee_mean', float('nan'))
            neg = 'Y' if r.get('f_goes_negative', False) else 'N'

            alpha_s = f'{alpha:.3f}' if not np.isnan(alpha) else '  N/A'
            c_g_s = f'{c_g:.3f}' if not np.isnan(c_g) else '  N/A'
            c_g_of_s = f'{c_g_of:.3f}' if not np.isnan(c_g_of) else '  N/A'
            gp_s = f'{gp:.3f}' if not np.isnan(gp) else '  N/A'
            dg_s = f'{dg:.3f}' if not np.isnan(dg) else '  N/A'
            curv_s = f'{curv_g:.3f}' if not np.isnan(curv_g) else '  N/A'
            h_s = f'{h_g:.3f}' if not np.isnan(h_g) else '  N/A'

            print(f'{app_name:<14} {cs:>5.2f} {rho:>6.3f} {alpha_s:>6} '
                  f'{c_g_s:>6} {c_g_of_s:>6} '
                  f'{gp_s:>6} {dg_s:>6} {curv_s:>6} {h_s:>6} {neg:>4}')
        print()


def main():
    global N, K, COST_SCALES_DESIGN, COST_SCALES_COVERAGE, COST_SCALES_FEATURES, random_greedy
    parser = argparse.ArgumentParser(description='Run curvature experiments (Tier 1)')
    parser.add_argument('--exp', choices=['design', 'coverage', 'features',
                                          'crossover', 'all'],
                        default='all', help='Which experiment to run')
    parser.add_argument('--seeds', type=int, default=N_SEEDS,
                        help=f'Number of seeds per config (default: {N_SEEDS})')
    parser.add_argument('--legacy', action='store_true', help='Submitted shrinking-pool heuristic, for Table 9 only')
    parser.add_argument('--smoke', action='store_true', help='Small synthetic test, not paper settings')
    args = parser.parse_args()

    if args.smoke:
        N, K = 8, 2
        COST_SCALES_DESIGN = COST_SCALES_COVERAGE = COST_SCALES_FEATURES = [0.0, 0.1]

    if args.legacy:
        from algorithms.random_greedy import legacy_random_greedy
        random_greedy = legacy_random_greedy

    n_seeds = args.seeds
    print('Curvature for Non-Monotone Submodular Maximization — Tier 1 Experiments')
    print(f'n={N}, k={K}, seeds={n_seeds}')
    print(f'Results directory: {RESULTS_DIR}')

    all_data = {}

    if args.exp in ('all', 'crossover'):
        experiment_crossover()

    if args.exp in ('all', 'design'):
        all_data['ExpDesign'] = experiment_design(n_seeds)

    if args.exp in ('all', 'coverage'):
        all_data['Coverage'] = experiment_coverage(n_seeds)

    if args.exp in ('all', 'features'):
        all_data['FeatureSel'] = experiment_features(n_seeds)

    if all_data:
        print_summary_table(all_data)
        generate_figures(all_data)

    print('\nDone.')


if __name__ == '__main__':
    main()
