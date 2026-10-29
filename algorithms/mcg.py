"""Discretized measured continuous greedy (FNS, FOCS 2011), heuristic.

Uniform matroid: max w.v over 0<=v<=1, sum v<=k selects up to k positive
coordinates. w=(1-x)*estimated_gradient; x += (1-x)*v/T.
Finite T,m are experimental parameters, not the paper's asymptotic guarantee.
Reference: https://www.cs.haifa.ac.il/~moranfe/publications/FOCS2011.pdf
"""
import numpy as np
from .batch_oracle import sampled_gradient


def pipage_round(x, k, rng):
    """Randomized pipage rounding in the uniform independence polytope.

    Pair exchanges preserve total mass and coordinate expectations. Once only
    one fractional coordinate remains, Bernoulli rounding is feasible since
    total mass <= integer k. For submodular f the multilinear extension is
    convex on exchange lines, so expected f cannot decrease. No f calls.
    """
    y = np.asarray(x, dtype=float).copy()
    if np.any(y < -1e-9) or np.any(y > 1+1e-9) or y.sum() > k+1e-8:
        raise ValueError('Point outside uniform matroid polytope')
    y = np.clip(y, 0, 1)
    while True:
        frac = np.flatnonzero((y > 1e-12) & (y < 1-1e-12))
        if len(frac) < 2:
            break
        i, j = frac[:2]
        plus = min(1-y[i], y[j])
        minus = min(y[i], 1-y[j])
        step = plus if rng.random() < minus/(plus+minus) else -minus
        y[i] += step
        y[j] -= step
    if len(frac):
        i = frac[0]
        y[i] = float(rng.random() < y[i])
    S = frozenset(np.flatnonzero(y > .5))
    assert len(S) <= k
    return S


def measured_continuous_greedy(f, k, *, T=50, m=50, seed=0, trace=False):
    if T <= 0 or m <= 0 or not 0 <= k <= f.n:
        raise ValueError('Invalid MCG parameters')
    rng = np.random.default_rng(seed)
    x = np.zeros(f.n)
    trajectory = []
    for t in range(T):
        grad = sampled_gradient(f, x, rng, m)
        scores = (1-x) * grad
        chosen = np.argsort(-scores, kind='stable')[:k]
        chosen = chosen[scores[chosen] > 0]
        x[chosen] += (1-x[chosen]) / T
        assert x.sum() <= k + 1e-8
        if trace:
            trajectory.append(dict(step=t+1, fractional=x.copy()))
    S = pipage_round(x, k, rng)
    val = f.evaluate(S)
    empty = f.evaluate(frozenset())
    if val < empty:
        S, val = frozenset(), empty
    result = dict(solution=S, value=float(val), fractional=x, T=T, m=m, seed=seed,
                  guarantee='heuristic: finite sampling/discretization; signed f allowed')
    if trace:
        result['fractional_trajectory'] = trajectory
    return result
