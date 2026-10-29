import numpy as np

class GraphCutLin:
    """Lin-Bilmes graph cut: f(S) = Σ_i w_i Σ_{j∈S} s_{ij} - λ Σ_{i≠j∈S} s_{ij}."""
    def __init__(self, sim_matrix, query_weights, lam=0.5):
        self.sim = np.array(sim_matrix, dtype=np.float64)
        self.qw = np.array(query_weights, dtype=np.float64)
        self.lam = lam
        self.n = self.sim.shape[0]
        self.ground_set = set(range(self.n))
        self._diag = np.diag(self.sim).copy()

    def __call__(self, S):
        S = sorted(S)
        if not S:
            return 0.0
        S_idx = np.array(S)
        coverage = float(np.dot(self.qw, np.sum(self.sim[:, S_idx], axis=1)))
        if len(S) > 1:
            intra = np.sum(self.sim[np.ix_(S_idx, S_idx)]) - np.sum(self._diag[S_idx])
        else:
            intra = 0.0
        return coverage - self.lam * float(intra)

    def marginal(self, e, S):
        return self(set(S) | {e}) - self(set(S))

    def singleton(self, e):
        return float(np.dot(self.qw, self.sim[:, e]))


def greedy_best_prefix(elements, k, f):
    """
    Greedy with best-prefix selection for non-monotone f.

    Returns:
        best_set: the prefix A_i maximizing f(A_i)
        trajectory: list of dicts per step with 'element', 'marginal',
                    'f_value', 'set' (as sorted list)
    """
    available = sorted(elements)
    S = set()
    trajectory = []
    best_val = 0.0
    best_set = set()

    for step in range(k):
        best_e, best_gain = None, -float('inf')
        for e in available:
            gain = f.marginal(e, S)
            # Exact duplicate passages are mathematically tied. Prevent a
            # round-off-only improvement from replacing the earlier index.
            # All archived selections use this first-index convention.
            equivalent = (best_e is not None and hasattr(f, 'equivalent')
                          and f.equivalent(e, best_e))
            if gain > best_gain and not equivalent:
                best_gain = gain
                best_e = e
        if best_e is None:
            break
        S.add(best_e)
        available.remove(best_e)
        val = f(S)
        trajectory.append({
            'step': step,
            'element': best_e,
            'marginal': best_gain,
            'f_value': val,
            'set': sorted(S),
        })
        if val > best_val:
            best_val = val
            best_set = set(S)

    return best_set, trajectory


def trajectory_curvature(f, trajectory):
    """
    Compute trajectory-based curvature lower bound.

    ĉ_traj = 1 - min_i [f(e_i | A_{i-1}) / f({e_i})]

    This is a lower bound on c_f restricted to the greedy trajectory.
    """
    ratios = []
    for step_info in trajectory:
        e = step_info['element']
        mg = step_info['marginal']
        singleton_val = f.singleton(e)
        if singleton_val > 1e-12:
            ratios.append(mg / singleton_val)

    if not ratios:
        return 0.0, []
    c_hat = 1.0 - min(ratios)
    return c_hat, ratios


def heuristic_best_known(elements, k, f, greedy_set, greedy_val, n_restarts=10, seed=42):
    """
    Estimate best achievable f(S) via multiple randomized greedy restarts
    plus swap-based local search.  Returns a lower bound on f(O*).

    The multilinear extension of GCLin is NOT concave (off-diagonal
    similarity matrix is not PSD), so continuous relaxation fails to
    provide a valid UB.  Instead we use the best discrete value found
    across many restarts as a proxy.
    """
    rng = np.random.RandomState(seed)
    best_val = greedy_val
    best_set = set(greedy_set)
    elems = sorted(elements)

    for trial in range(n_restarts):
        order = rng.permutation(elems)
        S = set()
        for e in order:
            if len(S) >= k:
                break
            if f.marginal(int(e), S) > 0:
                S.add(int(e))
        if len(S) < k:
            remaining = [e for e in order if e not in S]
            for e in remaining:
                if len(S) >= k:
                    break
                S.add(int(e))

        # Best-prefix selection
        prefix_best_val = 0.0
        prefix_best_set = set()
        cur = set()
        for e in sorted(S):
            cur.add(e)
            v = f(cur)
            if v > prefix_best_val:
                prefix_best_val = v
                prefix_best_set = set(cur)

        # One pass of swap-based local search from best prefix
        for e_out in list(prefix_best_set):
            swapped = False
            for e_in in elements - prefix_best_set:
                new_set = (prefix_best_set - {e_out}) | {e_in}
                new_val = f(new_set)
                if new_val > prefix_best_val + 1e-12:
                    prefix_best_val = new_val
                    prefix_best_set = new_set
                    swapped = True
                    break
            if swapped:
                break

        if prefix_best_val > best_val:
            best_val = prefix_best_val
            best_set = prefix_best_set

    return best_val, best_set
