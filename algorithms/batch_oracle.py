"""Vectorized exact set-value oracles for one-element neighbours.

Each returned neighbour counts as ONE f (or g) value query, including repeats.
The base value is supplied by the caller or queried separately. Kernels exploit
objective structure to compute exactly the same values as scalar evaluation;
counts are oracle-equivalent values, not matrix operations or cache misses.
"""
import numpy as np
from objectives.experimental_design import ExperimentalDesign, DOptimalDesign
from objectives.feature_selection import FeatureSelection
from objectives.coverage_cost import CoverageCost
from objectives.maxcut import MaxCut


def neighbor_values(f, S, elements, *, remove=False, component='f', base=None):
    elements = np.asarray(list(elements), dtype=int)
    if not len(elements):
        return np.empty(0)
    oracle = f.evaluate if component == 'f' else f.g_evaluate
    if base is None:
        base = oracle(S)
    sign = -1 if remove else 1
    if hasattr(f, 'neighbor_values_exact'):
        values = f.neighbor_values_exact(S, elements, remove, component, base)
    elif isinstance(f, ExperimentalDesign):
        X = f._X / np.sqrt(f._noise_var)
        Xs = X[sorted(S)]
        inv = np.linalg.inv(f._prior_inv + Xs.T @ Xs)
        U = X[elements]
        UP = U @ inv
        denominator = 1 + sign * np.einsum('ij,ij->i', UP, U)
        if isinstance(f, DOptimalDesign):
            values = base + np.log(denominator)
        else:
            gain = np.einsum('ij,ij->i', UP, UP) / denominator
            values = base + sign * gain
    elif isinstance(f, FeatureSelection):
        idx = sorted(S)
        if not hasattr(f, '_noise_cov'):
            f._noise_cov = f._cov / f._noise_var
        C = f._noise_cov
        M = np.eye(len(idx)) + C[np.ix_(idx, idx)]
        if remove:
            inverse = np.linalg.inv(M)
            positions = {e: j for j, e in enumerate(idx)}
            values = base + .5 * np.log([inverse[positions[e], positions[e]] for e in elements])
        else:
            cross = C[np.ix_(elements, idx)]
            schur = 1 + C[elements, elements]
            if idx:
                schur = schur - np.einsum('ij,ij->i', cross @ np.linalg.inv(M), cross)
            values = base + .5 * np.log(schur)
    elif isinstance(f, CoverageCost):
        if not hasattr(f, '_cover_matrix'):
            mat = np.zeros((f.n, f._m), dtype=bool)
            for e, items in enumerate(f._covers):
                mat[e, list(items)] = True
            f._cover_matrix = mat
        mat = f._cover_matrix
        counts = mat[sorted(S)].sum(axis=0) if S else np.zeros(f._m, dtype=int)
        weights = f._weights[:f._m] * (counts == (1 if remove else 0))
        gain = mat[elements] @ weights
        values = base + sign * gain
    elif isinstance(f, MaxCut) and component == 'f':
        inside = f._adj[elements][:, sorted(S)].sum(axis=1)
        gain = f._adj[elements].sum(axis=1) - 2 * inside
        values = base + sign * gain
    else:
        return np.array([oracle(S - {int(e)} if remove else S | {int(e)})
                         for e in elements])
    if component == 'f' and hasattr(f, '_costs'):
        values = values - sign * f._costs[elements]
    f.record_queries(component, len(elements))
    return np.asarray(values, dtype=float)


def sampled_gradient(f, x, rng, m):
    """Unbiased paired samples of partial_e F(x), n+1 values per sample.

    For e in R use f(R)-f(R-e); otherwise use f(R+e)-f(R).
    Removing e from R gives the required independent Bernoulli law.
    """
    total = np.zeros(f.n)
    for _ in range(m):
        mask = rng.random(f.n) < x
        S = frozenset(np.flatnonzero(mask))
        base = f.evaluate(S)
        added = np.flatnonzero(~mask)
        removed = np.flatnonzero(mask)
        total[added] += neighbor_values(f, S, added, base=base) - base
        total[removed] += base - neighbor_values(f, S, removed, remove=True, base=base)
    return total / m
