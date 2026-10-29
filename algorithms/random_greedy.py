"""Random Greedy (Buchbinder, Feldman, Naor, Schwartz, SODA 2014).

Achieves 1/e approximation for non-negative submodular maximization
under cardinality constraint.  Requires f ≥ 0.
"""

import numpy as np
from objectives.base import ObjectiveFunction
from .batch_oracle import neighbor_values


def legacy_random_greedy(f: ObjectiveFunction, k: int, *, seed=None, trace=False):
    """Run the BFNS random greedy algorithm.

    At each step, compute marginals of all remaining elements,
    keep only those with non-negative marginals, and sample
    one uniformly at random from the top-k candidates.

    Args:
        f: objective function (value oracle, should be non-negative)
        k: cardinality constraint
        seed: random seed
        trace: if True, return trajectory info

    Returns:
        result dict with 'solution', 'value'
    """
    rng = np.random.RandomState(seed)
    A = frozenset()
    trajectory = []

    for i in range(k):
        # Compute all marginals
        candidates = []
        for e in range(f.n):
            if e not in A:
                mg = f.marginal(e, A)
                if mg >= 0:
                    candidates.append((e, mg))

        if not candidates:
            break

        # Sort by marginal descending, sample uniformly from top-k
        candidates.sort(key=lambda x: -x[1])
        pool = candidates[:max(1, k - i)]

        idx = rng.randint(len(pool))
        chosen_e, chosen_mg = pool[idx]
        A = A | {chosen_e}

        if trace:
            trajectory.append({
                'step': i,
                'set': A,
                'value': f.evaluate(A),
                'element_added': chosen_e,
                'marginal': chosen_mg,
                'pool_size': len(pool),
            })

    result = {
        'solution': A,
        'value': f.evaluate(A),
    }
    if trace:
        result['trajectory'] = trajectory
    return result


def random_greedy(f: ObjectiveFunction, k: int, *, seed=None, trace=False):
    """BFNS14 Algorithm 1: fixed top-k pool, padded with zero-value dummies.

    Unlike the submitted shrinking-pool heuristic (retained above), sample
    from k entries on every iteration. Dummy choices leave the real set alone.
    Guarantees require nonnegative submodular f; otherwise this is a heuristic.
    """
    if not 0 <= k <= f.n:
        raise ValueError('Require 0 <= k <= n')
    rng = np.random.RandomState(seed)
    A = frozenset()
    val = f.evaluate(A)
    trajectory = []
    for i in range(k):
        es = [e for e in range(f.n) if e not in A]
        values = neighbor_values(f, A, es, base=val)
        candidates = [(e, float(value)-val) for e, value in zip(es, values)]
        candidates = sorted((x for x in candidates if x[1] > 0),
                            key=lambda x: (-x[1], x[0]))[:k]
        j = rng.randint(k)
        chosen, gain = candidates[j] if j < len(candidates) else (None, 0.0)
        if chosen is not None:
            A = A | {chosen}
            val += gain
        if trace:
            trajectory.append(dict(step=i, set=A, value=val,
                                   element_added=chosen, marginal=gain,
                                   pool_size=k))
    result = dict(solution=A, value=val)
    if trace:
        result['trajectory'] = trajectory
    return result
