"""Bayesian A-optimality experimental design.

g(S) = tr(Σ_prior) - tr(Σ_{S|})  (reduction in posterior variance)
ℓ(S) = Σ_{e ∈ S} cost_e

This is the lead application from Harshaw et al. (ICML 2019).
For a Gaussian model Y = Xβ + ε with prior β ~ N(0, Σ_prior),
selecting experiment subset S gives posterior covariance
  Σ_{S|} = (Σ_prior^{-1} + X_S^T Σ_noise^{-1} X_S)^{-1}
and g(S) = tr(Σ_prior) - tr(Σ_{S|}).

We use synthetic instances with controllable condition number.
"""

import numpy as np
from .base import DecomposableObjective


class ExperimentalDesign(DecomposableObjective):
    """Bayesian A-optimality: g(S) = variance reduction, ℓ(S) = costs."""

    def __init__(self, X, noise_var=1.0, costs=None, seed=None):
        """
        Args:
            X: (n_experiments, d) design matrix. Each row is one experiment.
            noise_var: observation noise variance σ².
            costs: (n_experiments,) per-experiment costs.  None = zero costs.
            seed: random seed for cost generation if costs='random'.
        """
        self._X = np.asarray(X, dtype=float)
        self._n_exp, self._d = self._X.shape
        self._noise_var = noise_var

        # Prior: Σ_prior = I_d (standard)
        self._prior_inv = np.eye(self._d)
        self._prior_trace = float(self._d)

        # Costs
        if costs is None:
            self._costs = np.zeros(self._n_exp)
        else:
            self._costs = np.asarray(costs, dtype=float)
        assert self._costs.shape == (self._n_exp,)

        # Cache
        self._cache = {}

    @property
    def n(self) -> int:
        return self._n_exp

    def _posterior_trace(self, S: frozenset) -> float:
        key = S
        if key in self._cache:
            return self._cache[key]
        if len(S) == 0:
            val = self._prior_trace
        else:
            idx = sorted(S)
            X_S = self._X[idx]
            # Σ_{S|} = (Σ_prior^{-1} + X_S^T X_S / σ²)^{-1}
            info = self._prior_inv + X_S.T @ X_S / self._noise_var
            post_cov = np.linalg.inv(info)
            val = np.trace(post_cov)
        self._cache[key] = val
        return val

    def g_evaluate(self, S: frozenset) -> float:
        """Variance reduction: tr(Σ_prior) - tr(Σ_{S|})."""
        return self._prior_trace - self._posterior_trace(S)

    def ell_evaluate(self, S: frozenset) -> float:
        return sum(self._costs[e] for e in S)

    @property
    def cc_curvature_approx(self) -> float:
        """Approximate CC curvature α_g from condition number of X^T X."""
        M = self._X.T @ self._X / self._noise_var + self._prior_inv
        eigvals = np.linalg.eigvalsh(M)
        kappa = eigvals[-1] / max(eigvals[0], 1e-12)
        return 1.0 - 1.0 / kappa

    @property
    def alpha_g_exact(self) -> float:
        """Legacy eigenvalue diagnostic; NOT an exact CC curvature.

        A-optimality is not generally submodular. Kept for archived runs only.
        """
        M = self._X.T @ self._X / self._noise_var + self._prior_inv
        eigvals = np.linalg.eigvalsh(M)
        return 1.0 - eigvals[0] / eigvals[-1]

    @classmethod
    def synthetic(cls, n_exp=50, d=10, condition_number=5.0,
                  cost_scale=0.0, noise_var=1.0, seed=42):
        """Generate a synthetic instance with controlled condition number.

        Args:
            n_exp: number of candidate experiments
            d: dimension of parameter space
            condition_number: κ(X^T X); controls α_g ≈ 1 - 1/κ
            cost_scale: costs drawn as cost_scale * |x_i|; 0 = free
            noise_var: observation noise σ²
            seed: reproducibility
        """
        rng = np.random.RandomState(seed)

        # Build X with prescribed condition number via SVD
        # U: (n_exp, d), S: (d,), V: (d, d)
        U, _ = np.linalg.qr(rng.randn(n_exp, d), mode='reduced')
        V, _ = np.linalg.qr(rng.randn(d, d))
        singular_vals = np.linspace(1.0, np.sqrt(condition_number), d)
        S_mat = np.diag(singular_vals)  # (d, d)
        X = U @ S_mat @ V  # (n_exp, d)

        # Costs proportional to experiment "effort" (norm of design row)
        if cost_scale > 0:
            row_norms = np.linalg.norm(X, axis=1)
            costs = cost_scale * row_norms / row_norms.mean()
        else:
            costs = np.zeros(n_exp)

        return cls(X, noise_var=noise_var, costs=costs, seed=seed)


class DOptimalDesign(ExperimentalDesign):
    """Normalized log information determinant with modular experiment costs.

    The inherited generator gives exactly the same X, prior, noise and cost
    vectors as A-optimality. Logdet rank-one PSD increments are monotone and
    submodular: the marginal is log(1+x_e^T M_S^{-1} x_e / sigma^2), and
    M_T >= M_S implies M_T^{-1} <= M_S^{-1} for S subset T.
    """

    def g_evaluate(self, S):
        S = frozenset(S)
        if S not in self._cache:
            xs = self._X[sorted(S)]
            sign, val = np.linalg.slogdet(self._prior_inv + xs.T @ xs / self._noise_var)
            prior_sign, prior_val = np.linalg.slogdet(self._prior_inv)
            if sign <= 0 or prior_sign <= 0:
                raise ValueError('Information matrices must be positive definite')
            self._cache[S] = float(val - prior_val)
        return self._cache[S]

    @property
    def alpha_g_exact(self):
        X = self._X / np.sqrt(self._noise_var)
        prior = np.linalg.inv(self._prior_inv)
        full = np.linalg.inv(self._prior_inv + X.T @ X)
        singleton = np.log1p(np.einsum('ij,ij->i', X @ prior, X))
        removal = -np.log1p(-np.einsum('ij,ij->i', X @ full, X))
        valid = singleton > 0
        return float(1 - np.min(removal[valid] / singleton[valid])) if np.any(valid) else 0.0
