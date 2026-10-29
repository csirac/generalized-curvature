r"""MaxCut objective — symmetric non-negative submodular with c_f = 2.

f(S) = |δ(S)| = number of edges with exactly one endpoint in S.
Symmetric: f(S) = f(V \ S).  Non-negative.  Submodular.
Classical curvature c_f = 2 (Proposition in the curvature paper).
"""

import numpy as np
from .base import ObjectiveFunction


class MaxCut(ObjectiveFunction):
    """Cardinality-constrained MaxCut: f(S) = |δ(S)|, |S| ≤ k.

    The adjacency matrix is stored as a dense numpy array.
    Elements are vertices 0..n-1.
    """

    def __init__(self, adj_matrix):
        """
        Args:
            adj_matrix: (n, n) symmetric binary adjacency matrix (0/1).
                        Diagonal is ignored.
        """
        A = np.asarray(adj_matrix, dtype=float)
        assert A.shape[0] == A.shape[1]
        np.fill_diagonal(A, 0)
        self._adj = (A + A.T) / 2
        self._adj = (self._adj > 0.5).astype(float)
        self._n = A.shape[0]

    @property
    def n(self) -> int:
        return self._n

    def evaluate(self, S: frozenset) -> float:
        if not S:
            return 0.0
        s_list = sorted(S)
        complement = [v for v in range(self._n) if v not in S]
        if not complement:
            return 0.0
        return self._adj[np.ix_(s_list, complement)].sum()

    def marginal(self, e: int, S: frozenset) -> float:
        if e in S:
            return 0.0
        edges_to_outside = 0.0
        edges_to_inside = 0.0
        for v in range(self._n):
            if self._adj[e, v] == 0:
                continue
            if v in S:
                edges_to_inside += 1.0
            elif v != e:
                edges_to_outside += 1.0
        return edges_to_outside - edges_to_inside


def make_sbm_graph(n, community_sizes, edge_probs, seed=42):
    """Generate a stochastic block model graph.

    Args:
        n: total number of vertices (must equal sum of community_sizes)
        community_sizes: list of community sizes
        edge_probs: dict mapping (i,j) community pairs to edge probability.
                    Must include (i,i) for intra-community edges.
                    Pairs (i,j) and (j,i) are treated as the same.
        seed: random seed for reproducibility

    Returns:
        (n, n) symmetric binary adjacency matrix
    """
    assert sum(community_sizes) == n
    rng = np.random.RandomState(seed)

    adj = np.zeros((n, n), dtype=float)
    offsets = [0]
    for s in community_sizes:
        offsets.append(offsets[-1] + s)

    for ci in range(len(community_sizes)):
        for cj in range(ci, len(community_sizes)):
            p = edge_probs.get((ci, cj), edge_probs.get((cj, ci), 0.0))
            for u in range(offsets[ci], offsets[ci + 1]):
                start = offsets[cj] if ci != cj else u + 1
                for v in range(start, offsets[cj + 1]):
                    if rng.rand() < p:
                        adj[u, v] = 1.0
                        adj[v, u] = 1.0

    return adj


def make_planted_distractor(n, k, seed=42):
    """SBM with a dense bait cluster that misleads greedy.

    Three communities:
      A (size k): dense internal edges (p=0.8) — bait cluster
      B (size k): sparse internal edges (p=0.2)
      C (size n-2k): moderate internal edges (p=0.3)

    Cross-community edges:
      A-B: 0.3,  A-C: 0.4,  B-C: 0.4

    Greedy is drawn to A-vertices early (high degree from dense internal
    edges), but the optimal cut of size k likely mixes communities.
    """
    assert n >= 2 * k, f"n={n} must be >= 2k={2*k}"
    c_sizes = [k, k, n - 2 * k]
    probs = {
        (0, 0): 0.8,
        (1, 1): 0.2,
        (2, 2): 0.3,
        (0, 1): 0.3,
        (0, 2): 0.4,
        (1, 2): 0.4,
    }
    return make_sbm_graph(n, c_sizes, probs, seed=seed)


def make_erdos_renyi(n, p=0.3, seed=42):
    """Erdos-Renyi random graph G(n, p)."""
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=float)
    for u in range(n):
        for v in range(u + 1, n):
            if rng.rand() < p:
                adj[u, v] = 1.0
                adj[v, u] = 1.0
    return adj
