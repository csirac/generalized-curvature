"""Greedy with Pruning (Algorithm 1 from the paper).

Works on any ObjectiveFunction via value oracle only.
No decomposition needed.
"""

from objectives.base import ObjectiveFunction
from .batch_oracle import neighbor_values


def greedy_pruning(f: ObjectiveFunction, k: int, *, trace=False):
    """Run greedy with pruning.

    Args:
        f: objective function (value oracle)
        k: cardinality constraint
        trace: if True, return trajectory info for curvature estimation

    Returns:
        result dict with keys:
          'solution': frozenset — selected elements
          'value': float — f(solution)
          'trajectory': list of (step, set, value, element_added) if trace=True
    """
    cache = {}
    def value(S):
        if S not in cache:
            cache[S] = f.evaluate(S)
        return cache[S]

    A = frozenset()
    trajectory = []

    for i in range(k):
        # Find element with largest marginal
        candidates = [e for e in range(f.n) if e not in A and A | {e} not in cache]
        values = neighbor_values(f, A, candidates, base=value(A))
        for e, val in zip(candidates, values):
            cache[A | {e}] = float(val)
        best_e, best_mg = None, -float('inf')
        for e in range(f.n):
            if e not in A:
                mg = value(A | {e}) - value(A)
                if mg > best_mg:
                    best_mg = mg
                    best_e = e

        # Stop if best marginal is negative
        if best_mg < 0 or best_e is None:
            break

        A_prime = A | {best_e}

        # Pruning loop: remove elements with negative marginal to the rest
        changed = True
        while changed:
            changed = False
            for a in sorted(A_prime):
                rest = A_prime - {a}
                if value(A_prime) - value(rest) < 0:
                    A_prime = rest
                    changed = True
                    break  # restart the check

        if trace:
            trajectory.append({
                'step': i,
                'set': A_prime,
                'value': value(A_prime),
                'element_added': best_e,
                'marginal': best_mg,
                'pruned': A | {best_e} != A_prime,
            })

        A = A_prime

    result = {
        'solution': A,
        'value': value(A),
    }
    if trace:
        result['trajectory'] = trajectory
    return result
