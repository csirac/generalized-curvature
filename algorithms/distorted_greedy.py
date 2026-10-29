"""Distorted Greedy (Harshaw, Feldman, Ward, Karbasi, ICML 2019).

Requires a DecomposableObjective (access to g and ℓ separately).
Uses the distorted objective g_t(e|S) = (1-1/k)^{k-t} * Δg(e|S) - ℓ(e).
"""

import numpy as np
from objectives.base import DecomposableObjective
from .batch_oracle import neighbor_values


def distorted_greedy(f: DecomposableObjective, k: int, *, trace=False):
    """Run the distorted greedy algorithm.

    Args:
        f: decomposable objective (g - ℓ) with separate access
        k: cardinality constraint
        trace: if True, return trajectory info

    Returns:
        result dict with 'solution', 'value', 'g_value', 'ell_value'
    """
    if not isinstance(f, DecomposableObjective):
        raise TypeError("Distorted greedy requires a DecomposableObjective "
                        "(needs g and ℓ separately).")

    A = frozenset()
    trajectory = []

    for t in range(1, k + 1):
        # Distortion weight: (1 - 1/k)^{k-t}
        weight = (1.0 - 1.0 / k) ** (k - t)

        best_e, best_val = None, -float('inf')
        es = [e for e in range(f.n) if e not in A]
        g_base = f.g_evaluate(A)
        vals = neighbor_values(f, A, es, component='g', base=g_base)
        for e, candidate in zip(es, vals):
            distorted = weight * (candidate - g_base) - f.ell_element(e)
            if distorted > best_val:
                best_val, best_e = distorted, e

        # Only add if distorted marginal is positive
        if best_val <= 0 or best_e is None:
            if trace:
                trajectory.append({
                    'step': t,
                    'set': A,
                    'value': f.evaluate(A),
                    'element_added': None,
                    'distorted_marginal': best_val,
                })
            continue

        A = A | {best_e}

        if trace:
            trajectory.append({
                'step': t,
                'set': A,
                'value': f.evaluate(A),
                'element_added': best_e,
                'distorted_marginal': best_val,
            })

    result = {
        'solution': A,
        'value': f.evaluate(A),
        'g_value': f.g_evaluate(A),
        'ell_value': f.ell_evaluate(A),
    }
    if trace:
        result['trajectory'] = trajectory
    return result
