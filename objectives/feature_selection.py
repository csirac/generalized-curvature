"""Feature selection: mutual information minus ℓ₁ cost.

g(S) = I(X_S ; Y)  (mutual information between selected features and target)
ℓ(S) = Σ_{e ∈ S} λ_e  (ℓ₁ penalty)

Mutual information of a Gaussian is:
  I(X_S; Y) = ½ log det(I + σ⁻² Σ_{S,S})

This is monotone submodular.  CC curvature α_g depends on the
eigenvalue spread of the feature covariance.
"""

import numpy as np
from .base import DecomposableObjective


class FeatureSelection(DecomposableObjective):
    """Feature selection with MI objective and ℓ₁ cost."""

    def __init__(self, cov_matrix, noise_var=1.0, costs=None):
        """
        Args:
            cov_matrix: (p, p) feature covariance matrix Σ
            noise_var: observation noise σ²
            costs: (p,) per-feature costs.  None = zero costs.
        """
        self._cov = np.asarray(cov_matrix, dtype=float)
        self._p = self._cov.shape[0]
        self._noise_var = noise_var
        self._costs = (np.zeros(self._p) if costs is None
                       else np.asarray(costs, dtype=float))
        self._cache = {}

    @property
    def n(self) -> int:
        return self._p

    def g_evaluate(self, S: frozenset) -> float:
        """I(X_S; Y) = ½ log det(I + σ⁻² Σ_{S,S})."""
        key = S
        if key in self._cache:
            return self._cache[key]
        if len(S) == 0:
            val = 0.0
        else:
            idx = sorted(S)
            sub_cov = self._cov[np.ix_(idx, idx)]
            val = 0.5 * np.linalg.slogdet(
                np.eye(len(idx)) + sub_cov / self._noise_var
            )[1]
        self._cache[key] = val
        return val

    def ell_evaluate(self, S: frozenset) -> float:
        return sum(self._costs[e] for e in S)

    @property
    def alpha_g_exact(self) -> float:
        """Analytic CC curvature α_g = 1 - λ_min/λ_max of (I + Σ/σ²).

        For Gaussian MI, g(S) = ½ log det(I + Σ_{S,S}/σ²), and the
        curvature is determined by the eigenvalue spread of the matrix
        I + Σ/σ². This is an upper bound that becomes tight for the
        full covariance; for subsets it may be loose.
        """
        M = np.eye(self._p) + self._cov / self._noise_var
        eigvals = np.linalg.eigvalsh(M)
        return 1.0 - eigvals[0] / eigvals[-1]

    @classmethod
    def synthetic(cls, p=40, n_groups=5, within_corr=0.7,
                  cost_scale=0.0, noise_var=1.0, seed=42):
        """Generate a grouped-feature instance.

        Features come in correlated groups (simulating redundancy).
        Higher within_corr → more redundancy → higher CC curvature.

        Args:
            p: number of features
            n_groups: number of correlated groups
            within_corr: pairwise correlation within a group
            cost_scale: per-feature cost = cost_scale * (1 + noise)
            noise_var: observation noise
            seed: reproducibility
        """
        rng = np.random.RandomState(seed)
        group_size = p // n_groups
        remainder = p % n_groups

        # Build block-diagonal covariance
        cov = np.eye(p)
        idx = 0
        for g in range(n_groups):
            sz = group_size + (1 if g < remainder else 0)
            for i in range(idx, idx + sz):
                for j in range(idx, idx + sz):
                    if i != j:
                        cov[i, j] = within_corr
            idx += sz

        # Add small cross-group correlations
        cov += 0.05 * rng.randn(p, p)
        cov = (cov + cov.T) / 2
        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals[0] < 0.01:
            cov += (0.02 - eigvals[0]) * np.eye(p)

        if cost_scale > 0:
            costs = cost_scale * (1.0 + 0.3 * rng.randn(p))
            costs = np.maximum(costs, 0.01)
        else:
            costs = np.zeros(p)

        return cls(cov, noise_var=noise_var, costs=costs)
