"""Standard greedy algorithm (baseline).

Adds the element with largest marginal at each step.
No pruning, no distortion.  Works only if f is monotone (or approximately so).
"""

from objectives.base import ObjectiveFunction
from .batch_oracle import neighbor_values


def standard_greedy(f: ObjectiveFunction, k: int, *, trace=False):
    """Run the standard greedy algorithm.

    Args:
        f: objective function (value oracle)
        k: cardinality constraint
        trace: if True, return trajectory info

    Returns:
        result dict with 'solution', 'value'
    """
    A = frozenset()
    trajectory = []
    val = f.evaluate(A)

    for i in range(k):
        best_e, best_mg = None, -float('inf')
        es = [e for e in range(f.n) if e not in A]
        values = neighbor_values(f, A, es, base=val)
        for e, candidate in zip(es, values):
            mg = candidate - val
            if mg > best_mg:
                best_mg, best_e = mg, e

        if best_e is None:
            break

        A = A | {best_e}
        val += best_mg

        if trace:
            trajectory.append({
                'step': i,
                'set': A,
                'value': val,
                'element_added': best_e,
                'marginal': best_mg,
            })

    result = {
        'solution': A,
        'value': val,
    }
    if trace:
        result['trajectory'] = trajectory
    return result
