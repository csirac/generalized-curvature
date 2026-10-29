"""Lazy (accelerated) Greedy with Pruning.

Uses a max-heap of stale marginal gains to avoid recomputing marginals for
every element at every step.  Correctness relies on submodularity: marginal
gains are non-increasing as the solution grows, so a stale upper bound is
still an upper bound — the lazy evaluation is exact.

For f = g - ℓ where ℓ is modular, marginals of f are also non-increasing
(g contributes a non-increasing component; ℓ-marginals are constant).
Hence the lazy trick applies to f directly.

Expected query complexity: O(n + k log n) for well-conditioned instances,
vs. O(nk) for standard greedy.
"""

import heapq
from objectives.base import ObjectiveFunction


def lazy_greedy_pruning(f: ObjectiveFunction, k: int, *, trace: bool = False):
    """Run lazy greedy with pruning.

    Args:
        f:     objective function (value oracle)
        k:     cardinality constraint
        trace: if True, return trajectory info

    Returns:
        result dict with keys:
          'solution':    frozenset — selected elements
          'value':       float — f(solution)
          'query_count': int — number of oracle evaluations
          'trajectory':  list of dicts if trace=True, one per greedy step
    """
    query_count = 0
    A = frozenset()
    trajectory = []

    # --- Initialise the heap with single-element marginals ---
    # Python's heapq is a min-heap; negate gains for max-heap behaviour.
    # Heap entries: (-marginal_gain, element_index)
    heap = []
    f_empty = f.evaluate(frozenset())
    query_count += 1

    for e in range(f.n):
        mg = f.evaluate(frozenset([e])) - f_empty
        query_count += 1
        heapq.heappush(heap, (-mg, e))

    for step in range(k):
        best_e = None

        # Lazy evaluation: pop until we find an element whose marginal is
        # still the best (i.e., its stale value equals its recomputed value,
        # or it stays on top after a fresh recompute).
        while heap:
            neg_stale, e = heapq.heappop(heap)

            if e in A:
                continue  # already selected; discard

            # Recompute true marginal
            mg_true = f.marginal(e, A)
            query_count += 2

            if not heap:
                # Only candidate left — take it if gain is positive.
                best_e = e
                best_mg = mg_true
                break

            neg_next, _ = heap[0]  # peek at next-best stale gain
            if mg_true >= -neg_next:
                # This element is still the best after recomputing.
                best_e = e
                best_mg = mg_true
                break
            else:
                # Push back with refreshed (true) marginal and retry.
                heapq.heappush(heap, (-mg_true, e))

        if best_e is None or best_mg < 0:
            break  # no positive-gain element remains

        A_prime = A | {best_e}

        # Pruning loop: remove elements with negative marginal to the rest
        changed = True
        while changed:
            changed = False
            for a in sorted(A_prime):
                rest = A_prime - {a}
                if f.marginal(a, rest) < 0:
                    query_count += 2
                    A_prime = rest
                    changed = True
                    break
                query_count += 2

        if trace:
            val = f.evaluate(A_prime)
            query_count += 1
            trajectory.append({
                'step':          step,
                'set':           A_prime,
                'value':         val,
                'element_added': best_e,
                'marginal':      best_mg,
                'pruned':        A | {best_e} != A_prime,
            })

        A = A_prime

    result = {
        'solution':    A,
        'value':       f.evaluate(A),
        'query_count': query_count,
    }
    query_count += 1
    if trace:
        result['trajectory'] = trajectory
    return result
