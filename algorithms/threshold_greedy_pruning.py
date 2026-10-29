"""Threshold Greedy with Pruning (Badanidiyuru-Vondrák 2014 style, adapted for non-monotone f).

Query complexity: O((n/ε) log(max_f / ε) + k² log(1/ε))
The n/ε term dominates in practice since n >> k.
"""

import math
from objectives.base import ObjectiveFunction


def threshold_greedy_pruning(f: ObjectiveFunction, k: int, *, eps: float = 0.1, trace: bool = False):
    """Run threshold greedy with pruning.

    Threshold schedule (Badanidiyuru-Vondrák style):
      - Start τ = max single-element value (estimated by a linear scan).
      - At each threshold level, accept any element with marginal ≥ τ,
        then run the pruning loop on the accumulated solution.
      - Decrease τ by factor (1 - ε) and repeat until τ < ε * τ_0 / n
        (i.e., O(log(1/ε) + log n) threshold levels).
      - Stop early when |A| == k.

    Args:
        f:     objective function (value oracle)
        k:     cardinality constraint
        eps:   geometric decay factor (default 0.1)
        trace: if True, return trajectory info

    Returns:
        result dict with keys:
          'solution':    frozenset — selected elements
          'value':       float — f(solution)
          'query_count': int — number of oracle evaluations
          'trajectory':  list of dicts (if trace=True), one per threshold level
    """
    if not (0 < eps < 1):
        raise ValueError(f"eps must be in (0, 1), got {eps}")

    query_count = 0
    A = frozenset()
    trajectory = []

    # --- Estimate τ_0: max single-element marginal gain ---
    tau_0 = 0.0
    for e in range(f.n):
        mg = f.marginal(e, frozenset())
        query_count += 2  # evaluate({e}) and evaluate({})  (base is cached in practice)
        if mg > tau_0:
            tau_0 = mg

    if tau_0 <= 0:
        # All single-element values non-positive; return empty set.
        result = {'solution': A, 'value': f.evaluate(A), 'query_count': query_count}
        query_count += 1
        if trace:
            result['trajectory'] = trajectory
        return result

    # Threshold stops when τ < ε * τ_0 / n  (guarantees O(log n / ε) levels)
    tau_stop = eps * tau_0 / f.n
    tau = tau_0
    level = 0

    while tau >= tau_stop and len(A) < k:
        added_this_level = []

        for e in range(f.n):
            if len(A) >= k:
                break
            if e in A:
                continue
            mg = f.marginal(e, A)
            query_count += 2
            if mg >= tau:
                A = A | {e}
                added_this_level.append(e)

        # Pruning loop after each threshold level
        if added_this_level:
            changed = True
            while changed:
                changed = False
                for a in sorted(A):
                    rest = A - {a}
                    mg_a = f.marginal(a, rest)
                    query_count += 2
                    if mg_a < 0:
                        A = rest
                        changed = True
                        break  # restart

        if trace:
            val = f.evaluate(A)
            query_count += 1
            trajectory.append({
                'level':           level,
                'tau':             tau,
                'set':             A,
                'value':           val,
                'added':           added_this_level,
                'size':            len(A),
            })

        tau *= (1.0 - eps)
        level += 1

    result = {
        'solution':    A,
        'value':       f.evaluate(A),
        'query_count': query_count,
    }
    query_count += 1  # for the final evaluate above (already charged to query_count below)
    if trace:
        result['trajectory'] = trajectory
    return result
