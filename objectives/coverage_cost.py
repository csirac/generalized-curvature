"""Directed vertex cover / weighted coverage minus costs.

g(S) = |{edges covered by S}|  (or weighted coverage)
ℓ(S) = Σ_{v ∈ S} cost_v

This is the second application from Harshaw et al. (ICML 2019).
Coverage is monotone submodular with CC curvature α_g = 1.
"""

import numpy as np
from .base import DecomposableObjective


class CoverageCost(DecomposableObjective):
    """Weighted coverage minus vertex costs.

    Ground set: n vertices.
    Each vertex v covers a subset of m items (edges/targets).
    g(S) = Σ_j w_j * 1[j covered by some v ∈ S]
    ℓ(S) = Σ_{v ∈ S} cost_v
    """

    def __init__(self, covers, weights=None, costs=None):
        """
        Args:
            covers: list of sets; covers[v] = set of items covered by vertex v
            weights: (m,) item weights.  None = unit weights.
            costs: (n,) vertex costs.  None = zero costs.
        """
        self._n_vertices = len(covers)
        self._covers = [frozenset(c) for c in covers]

        # Determine universe of items
        all_items = set()
        for c in self._covers:
            all_items |= c
        self._m = max(all_items) + 1 if all_items else 0

        self._weights = (np.ones(self._m) if weights is None
                         else np.asarray(weights, dtype=float))
        self._costs = (np.zeros(self._n_vertices) if costs is None
                       else np.asarray(costs, dtype=float))

    @property
    def n(self) -> int:
        return self._n_vertices

    @property
    def alpha_g_exact(self) -> float:
        """CC curvature of coverage is exactly 1.

        Any element's coverage can be fully subsumed by other elements,
        so min_{e,S} Δg(e|S)/g({e}) = 0, giving α_g = 1.
        """
        return 1.0

    def g_evaluate(self, S: frozenset) -> float:
        covered = set()
        for v in S:
            covered |= self._covers[v]
        return sum(self._weights[j] for j in covered)

    def ell_evaluate(self, S: frozenset) -> float:
        return sum(self._costs[v] for v in S)

    @classmethod
    def random_bipartite(cls, n_vertices=60, n_items=100,
                         edge_prob=0.15, cost_scale=0.0, seed=42):
        """Random bipartite coverage instance.

        Args:
            n_vertices: number of candidate vertices
            n_items: number of items/edges to cover
            edge_prob: probability of each (vertex, item) edge
            cost_scale: costs ~ cost_scale * degree(v) / mean_degree
            seed: reproducibility
        """
        rng = np.random.RandomState(seed)
        covers = []
        for v in range(n_vertices):
            mask = rng.rand(n_items) < edge_prob
            covers.append(set(np.where(mask)[0]))

        weights = np.ones(n_items)

        if cost_scale > 0:
            degrees = np.array([len(c) for c in covers], dtype=float)
            mean_deg = max(degrees.mean(), 1.0)
            costs = cost_scale * degrees / mean_deg
        else:
            costs = np.zeros(n_vertices)

        return cls(covers, weights=weights, costs=costs)

    @classmethod
    def power_law_graph(cls, n_vertices=60, n_items=100,
                        alpha=2.5, cost_scale=0.0, seed=42):
        """Coverage instance where vertex degrees follow a power law.

        High-degree vertices are expensive (natural cost structure).
        """
        rng = np.random.RandomState(seed)
        # Power-law degrees: P(deg) ~ deg^{-alpha}
        raw_degrees = (rng.pareto(alpha - 1, n_vertices) + 1)
        degrees = np.clip(raw_degrees * n_items * 0.1, 1, n_items).astype(int)

        covers = []
        for v in range(n_vertices):
            items = rng.choice(n_items, size=min(degrees[v], n_items), replace=False)
            covers.append(set(items))

        weights = np.ones(n_items)

        if cost_scale > 0:
            deg_arr = np.array([len(c) for c in covers], dtype=float)
            costs = cost_scale * deg_arr / max(deg_arr.mean(), 1.0)
        else:
            costs = np.zeros(n_vertices)

        return cls(covers, weights=weights, costs=costs)
