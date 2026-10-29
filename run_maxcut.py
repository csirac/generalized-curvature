#!/usr/bin/env python3
"""MaxCut experiment — symmetric submodular functions with c_f = 2.

Tests whether greedy+pruning beats standard greedy on
cardinality-constrained MaxCut, a canonical symmetric non-negative
submodular objective.

Usage:
    python3 run_maxcut.py                # full experiment (20 seeds)
    python3 run_maxcut.py --seeds 5      # quick test

Results saved to experiments/results/maxcut_results.json.
"""
from release_paths import RESULTS, FIGURES, ORIGINAL

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.greedy_pruning import greedy_pruning
from algorithms.standard_greedy import standard_greedy
from algorithms.random_greedy import random_greedy
from objectives.maxcut import MaxCut, make_planted_distractor, make_erdos_renyi
from opt_finder import find_best_solution

RESULTS_DIR = str(RESULTS)
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIGS = [
    ('planted_distractor', 16, 4),
    ('erdos_renyi',        16, 4),
    ('planted_distractor', 16, 5),
    ('erdos_renyi',        16, 5),
    ('planted_distractor', 16, 8),
    ('erdos_renyi',        16, 8),
    ('planted_distractor', 20, 5),
    ('erdos_renyi',        20, 5),
    ('planted_distractor', 20, 7),
    ('erdos_renyi',        20, 7),
    ('planted_distractor', 20, 10),
    ('erdos_renyi',        20, 10),
]


def run_instance(f, k, seed, instance_type, n):
    """Run all algorithms on a single MaxCut instance."""
    opt_set = find_best_solution(f, k, exact_threshold=f.n + 1)
    f_opt = f.evaluate(opt_set)

    results = []

    for alg_name, alg_fn in [
        ('greedy', lambda: standard_greedy(f, k, trace=True)),
        ('greedy_pruning', lambda: greedy_pruning(f, k, trace=True)),
        ('random', lambda: random_greedy(f, k, seed=seed, trace=False)),
    ]:
        res = alg_fn()
        sol = res['solution']
        val = res['value']
        ratio = val / f_opt if f_opt > 0 else float('nan')

        record = {
            'instance_type': instance_type,
            'n': n,
            'k': k,
            'seed': seed,
            'algorithm': alg_name,
            'f_value': val,
            'f_opt': f_opt,
            'ratio_vs_opt': ratio,
            'solution_size': len(sol),
            'selected_set': sorted(sol),
        }

        if 'trajectory' in res:
            pruning_events = sum(
                1 for t in res['trajectory'] if t.get('pruned', False)
            )
            record['pruning_events'] = pruning_events

        results.append(record)

    return results


def main():
    global CONFIGS
    parser = argparse.ArgumentParser(description='MaxCut experiment')
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--smoke', action='store_true', help='Small synthetic test, not paper settings')
    args = parser.parse_args()

    if args.smoke:
        CONFIGS = [('planted_distractor', 8, 2), ('erdos_renyi', 8, 2)]

    n_seeds = args.seeds

    print("=" * 70)
    print("MaxCut Experiment — Symmetric Submodular (c_f = 2)")
    print("=" * 70)
    print(f"Seeds per config: {n_seeds}")
    print(f"Instance types: planted_distractor (SBM), erdos_renyi (p=0.3)")
    print(f"Algorithms: greedy, greedy+pruning, random")
    print(f"OPT: exact enumeration (n <= 22)")
    print()

    all_results = []
    t_start = time.time()

    for cfg_idx, (inst_type, n, k) in enumerate(CONFIGS):
        print(f"[{cfg_idx+1}/{len(CONFIGS)}] {inst_type} n={n} k={k} ...",
              end='', flush=True)
        t0 = time.time()

        for seed in range(n_seeds):
            graph_seed = seed * 1000 + n * 100 + k

            if inst_type == 'planted_distractor':
                adj = make_planted_distractor(n, k, seed=graph_seed)
            else:
                adj = make_erdos_renyi(n, p=0.3, seed=graph_seed)

            f = MaxCut(adj)
            records = run_instance(f, k, seed, inst_type, n)
            all_results.extend(records)

        gp_ratios = [r['ratio_vs_opt'] for r in all_results
                     if r['instance_type'] == inst_type and r['n'] == n
                     and r['k'] == k and r['algorithm'] == 'greedy_pruning']
        sg_ratios = [r['ratio_vs_opt'] for r in all_results
                     if r['instance_type'] == inst_type and r['n'] == n
                     and r['k'] == k and r['algorithm'] == 'greedy']

        wins = sum(1 for g, s in zip(gp_ratios, sg_ratios) if g > s + 1e-10)
        dt = time.time() - t0
        print(f" GP={np.mean(gp_ratios):.4f} G={np.mean(sg_ratios):.4f} "
              f"wins={wins}/{n_seeds} ({dt:.1f}s)")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")

    out_path = os.path.join(RESULTS_DIR, 'maxcut_results.json')
    with open(out_path, 'w') as fp:
        json.dump({
            'config': {
                'n_seeds': n_seeds,
                'configs': [list(c) for c in CONFIGS],
                'theory_guarantee': round((1 - np.exp(-2)) / 2, 4),
                'curvature_cf': 2.0,
                'planted_distractor_params': {
                    'p_AA': 0.8, 'p_BB': 0.2, 'p_CC': 0.3,
                    'p_AB': 0.3, 'p_AC': 0.4, 'p_BC': 0.4,
                },
                'erdos_renyi_p': 0.3,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            },
            'results': all_results,
        }, fp, indent=2)
    print(f"Results saved to {out_path}")

    print_summary(all_results)


def print_summary(results):
    """Print a summary table."""
    print("\n" + "=" * 92)
    print("SUMMARY: Mean approximation ratio (vs exact OPT)")
    print("=" * 92)
    print(f"{'Instance Type':<22} {'n':>3} {'k':>3} | "
          f"{'Greedy':>8} {'GP':>8} {'Random':>8} {'mean OPT':>8} | "
          f"{'GP>G':>6} {'#prune':>6}")
    print("-" * 92)

    keys = sorted(set((r['instance_type'], r['n'], r['k']) for r in results))

    for inst_type, n, k in keys:
        subset = [r for r in results
                  if r['instance_type'] == inst_type
                  and r['n'] == n and r['k'] == k]

        by_alg = {}
        for r in subset:
            by_alg.setdefault(r['algorithm'], []).append(r)

        greedy_ratios = [r['ratio_vs_opt'] for r in by_alg.get('greedy', [])]
        gp_ratios = [r['ratio_vs_opt'] for r in by_alg.get('greedy_pruning', [])]
        rand_ratios = [r['ratio_vs_opt'] for r in by_alg.get('random', [])]
        opt_vals = [r['f_opt'] for r in by_alg.get('greedy', [])]

        gp_wins = sum(1 for g, s in zip(gp_ratios, greedy_ratios)
                      if g > s + 1e-10)

        prune_events = [r.get('pruning_events', 0)
                        for r in by_alg.get('greedy_pruning', [])]

        def fmt(vals):
            return f"{np.mean(vals):.4f}" if vals else "   N/A "

        opt_str = f"{np.mean(opt_vals):.1f}" if opt_vals else "N/A"
        n_inst = len(gp_ratios)
        print(f"{inst_type:<22} {n:>3} {k:>3} | "
              f"{fmt(greedy_ratios):>8} {fmt(gp_ratios):>8} "
              f"{fmt(rand_ratios):>8} {opt_str:>8} | "
              f"{gp_wins:>3}/{n_inst:<2} "
              f"{np.mean(prune_events):>5.1f}")

    print("-" * 92)
    print(f"Theory guarantee (c_f=2): (1-e^{{-2}})/2 = {(1-np.exp(-2))/2:.4f}")

    total_gp_wins = 0
    total_instances = 0
    for inst_type, n, k in keys:
        by_alg = {}
        for r in results:
            if r['instance_type'] == inst_type and r['n'] == n and r['k'] == k:
                by_alg.setdefault(r['algorithm'], []).append(r)
        gp = [r['ratio_vs_opt'] for r in by_alg.get('greedy_pruning', [])]
        sg = [r['ratio_vs_opt'] for r in by_alg.get('greedy', [])]
        total_gp_wins += sum(1 for g, s in zip(gp, sg) if g > s + 1e-10)
        total_instances += len(gp)

    print(f"\nOverall GP > Greedy: {total_gp_wins}/{total_instances} instances "
          f"({100*total_gp_wins/max(total_instances, 1):.1f}%)")

    # Highlight the key result
    print("\nKey finding:")
    for inst_type, n, k in keys:
        by_alg = {}
        for r in results:
            if r['instance_type'] == inst_type and r['n'] == n and r['k'] == k:
                by_alg.setdefault(r['algorithm'], []).append(r)
        gp = [r['ratio_vs_opt'] for r in by_alg.get('greedy_pruning', [])]
        sg = [r['ratio_vs_opt'] for r in by_alg.get('greedy', [])]
        wins = sum(1 for g, s in zip(gp, sg) if g > s + 1e-10)
        if wins > 0:
            delta = np.mean(gp) - np.mean(sg)
            print(f"  {inst_type} n={n} k={k}: GP wins {wins}/{len(gp)}, "
                  f"mean ratio +{delta:.4f} over greedy")


if __name__ == '__main__':
    main()
