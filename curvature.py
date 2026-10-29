"""Curvature estimation from algorithm trajectories.

Computes:
  - Empirical greedy curvature c_g along a trajectory
  - Per-step curvature ratios
  - CC curvature α_g for decomposable objectives (when g is accessible)
  - Theoretical guarantee bounds
"""

import numpy as np
from objectives.base import ObjectiveFunction, DecomposableObjective


def empirical_greedy_curvature(f: ObjectiveFunction, trajectory: list,
                                opt_set: frozenset) -> dict:
    r"""Estimate greedy curvature from a trajectory.

    The greedy curvature is:
      c_g = 1 - min_i [f(O ∪ A_i) - f(O)] / f(A_i \ O)

    where O is the optimal set and A_i is the greedy set at step i.

    Args:
        f: the objective function
        trajectory: list of dicts from greedy_pruning(trace=True)
        opt_set: an optimal (or best known) solution

    Returns:
        dict with 'c_g', 'per_step_ratios', 'per_step_curvatures'
    """
    f_O = f.evaluate(opt_set)
    ratios = []
    curvatures = []

    for step_info in trajectory:
        A_i = step_info['set']
        A_minus_O = A_i - opt_set

        f_A_minus_O = f.evaluate(A_minus_O)
        f_union = f.evaluate(opt_set | A_i)

        if abs(f_A_minus_O) < 1e-12:
            # Degenerate case: A_i ⊆ O or f(A_i \ O) ≈ 0
            ratio = None
            curv = None
        else:
            ratio = (f_union - f_O) / f_A_minus_O
            curv = 1.0 - ratio

        ratios.append(ratio)
        curvatures.append(curv)

    # c_g = max of per-step curvatures (= 1 - min of ratios)
    valid_curvatures = [c for c in curvatures if c is not None]
    c_g = max(valid_curvatures) if valid_curvatures else 0.0

    return {
        'c_g': c_g,
        'per_step_ratios': ratios,
        'per_step_curvatures': curvatures,
    }


def decomposable_curvature_bound(f: DecomposableObjective,
                                  trajectory: list,
                                  opt_set: frozenset) -> dict:
    r"""Compute the theoretical greedy curvature bound for f = g - ℓ.

    c_g ≤ α_g / (1 - r)  where r = max_i ℓ(A_i \ O) / g(A_i \ O)

    Also computes per-element cost ratios s_i = ℓ(e) / g({e}).

    Args:
        f: decomposable objective
        trajectory: greedy trajectory
        opt_set: optimal set

    Returns:
        dict with 'alpha_g_approx', 'r_max', 'c_g_bound',
                   'per_step_r', 'per_element_s'
    """
    # Per-step cost ratios r_i = ℓ(A_i \ O) / g(A_i \ O)
    per_step_r = []
    for step_info in trajectory:
        A_i = step_info['set']
        A_minus_O = A_i - opt_set
        if len(A_minus_O) == 0:
            per_step_r.append(0.0)
            continue
        g_val = f.g_evaluate(A_minus_O)
        ell_val = f.ell_evaluate(A_minus_O)
        if g_val > 1e-12:
            per_step_r.append(ell_val / g_val)
        else:
            per_step_r.append(float('inf'))

    r_max = max(per_step_r) if per_step_r else 0.0

    # Per-element cost ratios s_i = ℓ({e}) / g({e})
    per_element_s = {}
    for e in range(f.n):
        g_e = f.g_evaluate(frozenset([e]))
        ell_e = f.ell_element(e)
        if g_e > 1e-12:
            per_element_s[e] = ell_e / g_e
        else:
            per_element_s[e] = float('inf')

    # Approximate α_g (CC curvature of g) by sampling
    alpha_g = _estimate_cc_curvature(f, n_samples=200)

    # Bound
    if r_max < 1.0:
        c_g_bound = alpha_g / (1.0 - r_max)
    else:
        c_g_bound = float('inf')

    return {
        'alpha_g_approx': alpha_g,
        'r_max': r_max,
        'c_g_bound': c_g_bound,
        'per_step_r': per_step_r,
        'per_element_s': per_element_s,
    }


def _estimate_cc_curvature(f: DecomposableObjective, n_samples=200) -> float:
    """Estimate CC curvature α_g = 1 - min_{e,S} Δg(e|S) / g({e}).

    Uses random sampling of (element, set) pairs.
    """
    rng = np.random.RandomState(0)
    min_ratio = 1.0

    for _ in range(n_samples):
        e = rng.randint(f.n)
        g_e = f.g_evaluate(frozenset([e]))
        if g_e < 1e-12:
            continue

        # Random set not containing e
        mask = rng.rand(f.n) < 0.3
        mask[e] = False
        S = frozenset(np.where(mask)[0])

        g_mg = f.g_marginal(e, S)
        ratio = g_mg / g_e
        min_ratio = min(min_ratio, ratio)

    return 1.0 - min_ratio


def opt_free_curvature_bound(f: DecomposableObjective,
                             trajectory: list,
                             alpha_g: float = None) -> dict:
    """Compute an OPT-free upper bound on greedy curvature.

    Uses c_g ≤ α_g / (1 - ŝ) where ŝ = max_{e ∈ A} ℓ(e)/g({e})
    over ALL elements in the greedy solution (not just A \\ O).
    This is looser than the OPT-aware bound but requires no knowledge
    of the optimal solution.

    Args:
        f: decomposable objective (provides g, ℓ separately)
        trajectory: greedy trajectory (list of step dicts with 'set' key)
        alpha_g: CC curvature of g. If None, uses f.alpha_g_exact if
                 available, else falls back to sampling.

    Returns:
        dict with 'alpha_g', 's_hat', 'c_g_bound', 'guarantee',
                   'per_element_s'
    """
    # Get the final greedy solution
    A = trajectory[-1]['set'] if trajectory else frozenset()

    # CC curvature of g
    if alpha_g is not None:
        pass
    elif hasattr(f, 'alpha_g_exact'):
        alpha_g = f.alpha_g_exact
    else:
        alpha_g = _estimate_cc_curvature(f, n_samples=500)

    # Per-element cost ratios for elements in A
    per_element_s = {}
    for e in A:
        g_e = f.g_evaluate(frozenset([e]))
        ell_e = f.ell_element(e)
        if g_e > 1e-12:
            per_element_s[e] = ell_e / g_e
        else:
            per_element_s[e] = float('inf')

    s_hat = max(per_element_s.values()) if per_element_s else 0.0

    # Bound
    if s_hat < 1.0:
        c_g_bound = alpha_g / (1.0 - s_hat)
    else:
        c_g_bound = float('inf')

    g = guarantee_curvature(c_g_bound)

    return {
        'alpha_g': alpha_g,
        's_hat': s_hat,
        'c_g_bound': c_g_bound,
        'guarantee': g,
        'per_element_s': per_element_s,
    }


def guarantee_curvature(c: float) -> float:
    """Theorem 1 bound with c_bar=max(1,c), applied before averaging."""
    c = max(1.0, c)
    return (1.0 - np.exp(-c)) / c


def guarantee_harshaw(rho: float) -> float:
    """Compute Harshaw multiplicative guarantee: 1 - 1/(e(1-ρ))."""
    if rho >= 1.0:
        return -float('inf')
    return 1.0 - 1.0 / (np.e * (1.0 - rho))


def hardness_bound(c_g: float) -> float:
    """Compute hardness: (e-1) / (c_g + e - 1)."""
    return (np.e - 1.0) / (c_g + np.e - 1.0)
