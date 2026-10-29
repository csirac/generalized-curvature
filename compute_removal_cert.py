#!/usr/bin/env python3
r"""Compute the removal-marginal certificate r_hat for Tier 1 experiments.

r_hat = max_i max_{e in A_i} ell(e) / Delta_g(e | A_i \ {e})
c_g_cert = alpha_g / (1 - r_hat)

This is the Yixin certificate (Proposition opt-free in the paper).
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms import greedy_pruning
from objectives import ExperimentalDesign, CoverageCost, FeatureSelection
from objectives.base import DecomposableObjective
from curvature import empirical_greedy_curvature, guarantee_curvature
from opt_finder import find_best_solution

N = 20
K = 5
N_SEEDS = 10

# Cost scales matching the 4 rows per application in Table 1
COST_SCALES_DESIGN = [0.03, 0.10, 0.20, 0.28]
COST_SCALES_COVERAGE = [0.5, 2.0, 3.5, 5.0]
COST_SCALES_FEATURES = [0.05, 0.2, 0.3, 0.5]


def compute_removal_marginal_cert(f, trajectory, alpha_g):
    """Compute r_hat from the greedy trajectory."""
    r_hat = 0.0

    for step_info in trajectory:
        A_i = step_info['set']
        if not A_i:
            continue

        g_A = f.g_evaluate(A_i)
        for e in A_i:
            A_minus_e = A_i - {e}
            g_A_minus_e = f.g_evaluate(A_minus_e)
            delta_g = g_A - g_A_minus_e
            ell_e = f.ell_element(e)

            if delta_g > 1e-12:
                ratio = ell_e / delta_g
                r_hat = max(r_hat, ratio)
            else:
                r_hat = float('inf')

    if r_hat < 1.0:
        c_g_cert = alpha_g / (1.0 - r_hat)
        c_g_bar = max(1.0, c_g_cert)
        guarantee = (1.0 - np.exp(-c_g_bar)) / c_g_bar
    else:
        c_g_cert = float('inf')
        guarantee = 0.0

    return {
        'r_hat': r_hat,
        'c_g_cert': c_g_cert,
        'guarantee': guarantee,
    }


# Factory functions matching run_all.py
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


def run_app(name, factory, cost_scales):
    print(f"\n=== {name} ===")
    results = []
    for cs in cost_scales:
        seed_results = []
        for seed in range(N_SEEDS):
            f = factory(cs, seed)
            alpha_g = f.alpha_g_exact if hasattr(f, 'alpha_g_exact') else None
            if alpha_g is None:
                print(f"  WARNING: no alpha_g_exact for {name} seed={seed}")
                continue

            gp = greedy_pruning(f, K, trace=True)
            traj = gp.get('trajectory', [])
            cert = compute_removal_marginal_cert(f, traj, alpha_g)

            # Also get the singleton s_hat for comparison
            s_hat = 0.0
            all_elements = set()
            for si in traj:
                all_elements |= si['set']
            for e in all_elements:
                g_e = f.g_evaluate(frozenset([e]))
                ell_e = f.ell_element(e)
                if g_e > 1e-12:
                    s_hat = max(s_hat, ell_e / g_e)

            # Get OPT and empirical c_g
            opt_set = find_best_solution(f, K, exact_threshold=f.n + 1)
            f_opt = f.evaluate(opt_set)
            g_opt = f.g_evaluate(opt_set)
            ell_opt = f.ell_evaluate(opt_set)
            rho = ell_opt / max(g_opt, 1e-12)

            emp_curv = empirical_greedy_curvature(f, traj, opt_set)

            singl_c = alpha_g / (1.0 - s_hat) if s_hat < 1.0 else float('inf')
            singl_c_bar = max(1.0, singl_c) if singl_c != float('inf') else float('inf')
            singl_guar = (1.0 - np.exp(-singl_c_bar)) / singl_c_bar if singl_c != float('inf') else 0.0

            seed_results.append({
                'seed': seed,
                'rho': rho,
                'gp_ratio': gp['value'] / max(f_opt, 1e-12) if f_opt > 0 else float('nan'),
                'empirical_c_g': emp_curv['c_g'],
                'curv_guarantee': guarantee_curvature(emp_curv['c_g']),
                'alpha_g': alpha_g,
                's_hat': s_hat,
                'r_hat': cert['r_hat'],
                'singl_guar': singl_guar,
                'removal_guar': cert['guarantee'],
            })

        # Aggregate
        rho_mean = np.mean([r['rho'] for r in seed_results])
        gp_ratio_mean = np.mean([r['gp_ratio'] for r in seed_results])
        cg_mean = np.mean([r['empirical_c_g'] for r in seed_results])
        s_hat_mean = np.mean([r['s_hat'] for r in seed_results])
        r_hat_mean = np.mean([r['r_hat'] for r in seed_results])
        guar_sing_mean = np.mean([r['singl_guar'] for r in seed_results])
        guar_rem_mean = np.mean([r['removal_guar'] for r in seed_results])
        curv_guar_mean = np.mean([r['curv_guarantee'] for r in seed_results])

        print(f"  cs={cs}: rho={rho_mean:.2f}, c_g={cg_mean:.2f}, gp_ratio={gp_ratio_mean:.2f}")
        print(f"    s_hat={s_hat_mean:.4f}, singl_guar={guar_sing_mean:.4f}")
        print(f"    r_hat={r_hat_mean:.4f}, removal_guar={guar_rem_mean:.4f}")
        print(f"    curv_guar(exact c_g)={curv_guar_mean:.4f}")

        results.append({
            'cost_scale': cs,
            'rho_mean': round(rho_mean, 4),
            'gp_ratio_mean': round(gp_ratio_mean, 4),
            'empirical_c_g_mean': round(cg_mean, 4),
            'curv_guarantee_mean': round(curv_guar_mean, 4),
            's_hat_mean': round(s_hat_mean, 4),
            'r_hat_mean': round(r_hat_mean, 4),
            'singl_guar_mean': round(guar_sing_mean, 4),
            'removal_guar_mean': round(guar_rem_mean, 4),
        })

    return results


if __name__ == '__main__':
    all_results = {}

    all_results['experimental_design'] = run_app(
        'Experimental Design', make_design_instance, COST_SCALES_DESIGN)
    all_results['coverage'] = run_app(
        'Coverage', make_coverage_instance, COST_SCALES_COVERAGE)
    all_results['feature_selection'] = run_app(
        'Feature Selection', make_features_instance, COST_SCALES_FEATURES)

    # Print summary table
    print("\n\n=== SUMMARY TABLE ===")
    print(f"{'Application':<15} {'rho':>6} {'GP':>6} {'c_g':>6} {'Curv':>6} {'Singl':>6} {'Remov':>6} {'HFWK':>6}")
    print("-" * 75)
    for app_name, app_key in [('Exp. Design', 'experimental_design'),
                               ('Coverage', 'coverage'),
                               ('Feature Sel.', 'feature_selection')]:
        for r in all_results[app_key]:
            hfwk = (1.0 - 1.0/np.e) - r['rho_mean'] / (1.0 - r['rho_mean']) * (1.0/np.e) if r['rho_mean'] < 1.0 else float('-inf')
            hfwk_mult = ((1 - 1/np.e) * (1 - r['rho_mean']) - r['rho_mean'] * (1/np.e)) / 1.0
            # HFWK: (1-1/e)g(O*) - ell(O*); as a fraction of f(O*) = g(O*) - ell(O*):
            # = ((1-1/e)g - ell) / (g - ell) = ((1-1/e) - rho) / (1 - rho)
            if r['rho_mean'] < 1.0:
                hfwk_ratio = ((1 - 1/np.e) - r['rho_mean']) / (1.0 - r['rho_mean'])
            else:
                hfwk_ratio = float('-inf')

            print(f"{app_name:<15} {r['rho_mean']:>6.2f} {r['gp_ratio_mean']:>6.2f} "
                  f"{r['empirical_c_g_mean']:>6.2f} {r['curv_guarantee_mean']:>6.2f} "
                  f"{r['singl_guar_mean']:>6.2f} {r['removal_guar_mean']:>6.2f} "
                  f"{hfwk_ratio:>6.2f}")

    with open('results/removal_cert_values.json', 'w') as fp:
        json.dump(all_results, fp, indent=2, default=str)
    print("\nSaved to results/removal_cert_values.json")
