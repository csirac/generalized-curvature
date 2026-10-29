"""First-improvement 1-swap/add/delete search, factored from opt_finder.

Start at the best positive singleton (or empty). Lexicographic scans try swaps,
then feasible additions, then deletions. Accept gain > 1e-10; stop at a full
unsuccessful scan or max_iters accepted moves. A capped run is not a certified
local optimum. No approximation guarantee is claimed for signed objectives.
"""

from .batch_oracle import neighbor_values

def local_search(f, k, *, initial=None, max_iters=200, trace=False):
    if initial is None:
        S = frozenset()
        best = f.evaluate(S)
        vals = neighbor_values(f, S, range(f.n), base=best)
        for e, val in enumerate(vals):
            if val > best + 1e-10:
                S, best = frozenset({e}), float(val)
    else:
        S = frozenset(initial)
    if len(S) > k:
        raise ValueError('Initial set exceeds cardinality')
    if k == 0:
        return dict(solution=frozenset(), value=f.evaluate(frozenset()), converged=True,
                    trajectory=[])
    current = frozenset(S)
    current_val = f.evaluate(current)

    trajectory = [dict(step=0, set=current, value=current_val)]
    converged = False
    for iteration in range(max_iters):
        improved = False

        # Try swaps: remove a, add b
        for a in sorted(current):
            es = [b for b in range(f.n) if b not in current]
            vals = neighbor_values(f, current - {a}, es)
            for b, new_val in zip(es, vals):
                new_set = (current - {a}) | {b}
                if new_val > current_val + 1e-10:
                    current = new_set
                    current_val = new_val
                    improved = True
                    break
            if improved:
                break

        # Try pure additions (if |S| < k)
        if not improved and len(current) < k:
            es = [b for b in range(f.n) if b not in current]
            vals = neighbor_values(f, current, es, base=current_val)
            for b, new_val in zip(es, vals):
                new_set = current | {b}
                if new_val > current_val + 1e-10:
                    current = new_set
                    current_val = new_val
                    improved = True
                    break

        # Try pure removals
        if not improved:
            for a in sorted(current):
                new_set = current - {a}
                new_val = f.evaluate(new_set)
                if new_val > current_val + 1e-10:
                    current = new_set
                    current_val = new_val
                    improved = True
                    break

        if not improved:
            converged = True
            break
        if trace:
            trajectory.append(dict(step=iteration+1, set=current, value=current_val))

    result = dict(solution=current, value=current_val, converged=converged,
                  accepted_moves=len(trajectory)-1, max_iters=max_iters)
    if trace:
        result["trajectory"] = trajectory
    return result
