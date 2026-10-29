"""Personalized movie recommendation objective.

f(S) = Σ_{i ∈ V} Σ_{j ∈ S} s_{i,j}  -  λ Σ_{i ∈ S} Σ_{j ∈ S} s_{i,j}

where s_{i,j} is similarity between items i and j, and λ controls
the diversity-relevance tradeoff.

This is NOT decomposable as g - ℓ (the penalty is quadratic, not modular).
Global curvature c_f ≤ 2λ.

From Mirzasoleiman et al. (2016), used in our paper Prop 9.
"""

import numpy as np
from .base import ObjectiveFunction


class MovieRecommendation(ObjectiveFunction):
    """Personalized movie recommendation with diversity penalty."""

    def __init__(self, similarity_matrix, lam=0.5):
        """
        Args:
            similarity_matrix: (n, n) pairwise item similarities, non-negative.
            lam: diversity tradeoff parameter λ ∈ [0, 1].
        """
        self._sim = np.asarray(similarity_matrix, dtype=float)
        self._n_items = self._sim.shape[0]
        self._lam = lam
        assert 0.0 <= lam <= 1.0

    @property
    def n(self) -> int:
        return self._n_items

    def evaluate(self, S: frozenset) -> float:
        if len(S) == 0:
            return 0.0
        idx = sorted(S)
        # Relevance: sum of similarities from all items to selected
        relevance = self._sim[:, idx].sum()
        # Diversity penalty: similarities within selected set
        penalty = self._sim[np.ix_(idx, idx)].sum()
        return relevance - self._lam * penalty

    @property
    def curvature_bound(self) -> float:
        """Theoretical curvature bound: c_f ≤ 2λ."""
        return 2.0 * self._lam

    @classmethod
    def from_movielens(cls, ratings_file, lam=0.5, n_items=100, seed=42):
        """Load from a MovieLens ratings file (u.data format).

        Computes item-item cosine similarity from user-item ratings.
        Selects the top n_items most-rated items.
        """
        import pandas as pd

        df = pd.read_csv(ratings_file, sep='\t', header=None,
                         names=['user', 'item', 'rating', 'timestamp'])

        # Select top items by number of ratings
        item_counts = df['item'].value_counts()
        top_items = item_counts.head(n_items).index.tolist()
        df = df[df['item'].isin(top_items)]

        # Build user-item matrix
        users = sorted(df['user'].unique())
        items = sorted(top_items)
        user_idx = {u: i for i, u in enumerate(users)}
        item_idx = {it: i for i, it in enumerate(items)}

        R = np.zeros((len(users), len(items)))
        for _, row in df.iterrows():
            R[user_idx[row['user']], item_idx[row['item']]] = row['rating']

        # Item-item cosine similarity
        norms = np.linalg.norm(R, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        R_normed = R / norms
        sim = R_normed.T @ R_normed
        np.fill_diagonal(sim, 0.0)  # no self-similarity
        sim = np.maximum(sim, 0.0)  # ensure non-negative

        return cls(sim, lam=lam)

    @classmethod
    def synthetic(cls, n_items=50, n_clusters=5, lam=0.5, seed=42):
        """Generate synthetic instance with clustered item similarities.

        Items within a cluster are similar; across clusters less so.
        """
        rng = np.random.RandomState(seed)
        cluster_size = n_items // n_clusters
        remainder = n_items % n_clusters

        sim = np.zeros((n_items, n_items))
        idx = 0
        for c in range(n_clusters):
            sz = cluster_size + (1 if c < remainder else 0)
            # Within-cluster: high similarity
            for i in range(idx, idx + sz):
                for j in range(idx, idx + sz):
                    if i != j:
                        sim[i, j] = 0.6 + 0.3 * rng.rand()
            idx += sz

        # Cross-cluster: low similarity
        for i in range(n_items):
            for j in range(i + 1, n_items):
                if sim[i, j] == 0:
                    sim[i, j] = 0.1 * rng.rand()
                    sim[j, i] = sim[i, j]

        np.fill_diagonal(sim, 0.0)
        return cls(sim, lam=lam)
