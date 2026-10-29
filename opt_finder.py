"""Find the best known solution for small instances.

Uses exact enumeration for small n/k, or a combination of
multiple greedy variants + local search for larger instances.
"""

import itertools
import numpy as np
from objectives.base import ObjectiveFunction


def find_best_solution(f: ObjectiveFunction, k: int,
                       exact_threshold=22) -> frozenset:
    """Find the best known solution of size ≤ k.

    For n ≤ exact_threshold: exact enumeration over all (n choose k) sets.
    Otherwise: runs multiple heuristics and returns the best.

    Args:
        f: objective function
        k: cardinality constraint
        exact_threshold: use exact enumeration if n ≤ this

    Returns:
        frozenset — best known solution
    """
    if f.n <= exact_threshold:
        return _exact_opt(f, k)
    else:
        return _heuristic_opt(f, k)


def _exact_opt(f: ObjectiveFunction, k: int) -> frozenset:
    """Exact enumeration over all subsets of size ≤ k."""
    best_val = -float('inf')
    best_set = frozenset()

    for size in range(k + 1):
        for subset in itertools.combinations(range(f.n), size):
            S = frozenset(subset)
            val = f.evaluate(S)
            if val > best_val:
                best_val = val
                best_set = S

    return best_set


def _heuristic_opt(f: ObjectiveFunction, k: int) -> frozenset:
    """Multiple heuristics + local search for larger instances."""
    from algorithms.greedy_pruning import greedy_pruning
    from algorithms.standard_greedy import standard_greedy
    from algorithms.random_greedy import random_greedy

    candidates = []

    # Standard greedy
    sg = standard_greedy(f, k)
    candidates.append(sg['solution'])

    # Greedy with pruning
    gp = greedy_pruning(f, k)
    candidates.append(gp['solution'])

    # Scale random greedy starts and local search with instance size
    n_rg = min(20, max(5, 200 // max(f.n, 1)))
    ls_iters = min(50, max(10, 2000 // max(f.n, 1)))

    for seed in range(n_rg):
        rg = random_greedy(f, k, seed=seed)
        candidates.append(rg['solution'])

    # Local search on each candidate
    improved = []
    for S in candidates:
        S_improved = _local_search(f, S, k, max_iters=ls_iters)
        improved.append(S_improved)

    # Also try empty set (f(∅) = 0 might be best if f < 0 everywhere)
    improved.append(frozenset())

    # Return best
    best_set = max(improved, key=lambda S: f.evaluate(S))
    return best_set


def _local_search(f, S, k, max_iters=50):
    from algorithms.local_search import local_search
    return local_search(f, k, initial=S, max_iters=max_iters)['solution']
