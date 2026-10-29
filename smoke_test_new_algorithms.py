"""Smoke test: compare greedy_pruning, lazy_greedy_pruning, threshold_greedy_pruning
on an ExperimentalDesign instance (n=40, k=10).
"""

import sys
import os

# Make sure the experiments directory is on the path
sys.path.insert(0, os.path.dirname(__file__))

from objectives.experimental_design import ExperimentalDesign
from algorithms.greedy_pruning import greedy_pruning
from algorithms.lazy_greedy_pruning import lazy_greedy_pruning
from algorithms.threshold_greedy_pruning import threshold_greedy_pruning

def run():
    n, k = 40, 10
    f = ExperimentalDesign.synthetic(n_exp=n, d=8, condition_number=10.0,
                                     cost_scale=0.0, seed=7)

    print(f"Instance: n={n}, k={k}, d=8, cond=10")
    print(f"{'Algorithm':<30} {'Value':>10} {'Queries':>10}")
    print("-" * 52)

    # Standard greedy+pruning — no query_count, add manual tracking
    r_gp = greedy_pruning(f, k)
    # Estimate queries: O(n*k) greedy + O(k^2) pruning
    est_queries_gp = n * k + k * k
    print(f"{'greedy_pruning (est.)':<30} {r_gp['value']:>10.4f} {est_queries_gp:>10}")

    # Lazy greedy+pruning
    r_lg = lazy_greedy_pruning(f, k)
    print(f"{'lazy_greedy_pruning':<30} {r_lg['value']:>10.4f} {r_lg['query_count']:>10}")

    # Threshold greedy+pruning (ε=0.1)
    r_tg = threshold_greedy_pruning(f, k, eps=0.1)
    print(f"{'threshold_greedy_pruning':<30} {r_tg['value']:>10.4f} {r_tg['query_count']:>10}")

    print()

    # Sanity checks
    gp_val = r_gp['value']
    lg_val = r_lg['value']
    tg_val = r_tg['value']

    # All values should be positive
    assert gp_val > 0, f"greedy_pruning returned non-positive value: {gp_val}"
    assert lg_val > 0, f"lazy_greedy_pruning returned non-positive value: {lg_val}"
    assert tg_val > 0, f"threshold_greedy_pruning returned non-positive value: {tg_val}"

    # Values should be within 20% of each other (same objective, different heuristics)
    best = max(gp_val, lg_val, tg_val)
    worst = min(gp_val, lg_val, tg_val)
    ratio = worst / best
    print(f"Value ratio (worst/best): {ratio:.3f}  (expected > 0.80)")
    assert ratio > 0.80, f"Values diverge too much: {gp_val:.4f} {lg_val:.4f} {tg_val:.4f}"

    # Lazy should use fewer queries than the O(nk) estimate for standard greedy
    assert r_lg['query_count'] < est_queries_gp, (
        f"lazy_greedy should use fewer queries than O(nk)={est_queries_gp}, "
        f"got {r_lg['query_count']}"
    )

    # Solutions should be valid frozensets with |A| <= k
    assert len(r_gp['solution']) <= k
    assert len(r_lg['solution']) <= k
    assert len(r_tg['solution']) <= k

    print("All checks passed.")

if __name__ == "__main__":
    run()
