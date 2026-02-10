"""
TULCA (Tensor ULCA) - Tensor-based discriminant analysis algorithm.
Ported from the original Dash application.
"""

import numpy as np
import tensorly as tl
from scipy import linalg
from factor_analyzer import Rotator
import pymanopt
from pymanopt.manifolds import Grassmann
from pymanopt.optimizers import TrustRegions
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings(
    "ignore", 
    message="n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism."
)


def _generate_covs(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Generate within-class and between-class covariance matrices."""
    classes = np.unique(y)
    modes = np.arange(X.ndim - 1)
    n_samples = X.shape[0]
    n_classes = len(classes)
    n_modes = len(modes)

    matrices = np.empty(n_modes, dtype=object)
    for m in modes:
        matrices[m] = np.swapaxes(
            tl.unfold(X, m + 1).reshape(
                X.shape[m + 1], n_samples, int(X.size / X.shape[m + 1] / n_samples)
            ),
            0, 1,
        )

    matrices_by_class = np.empty((n_classes, n_modes), dtype=object)
    for c in classes:
        for m in modes:
            matrices_by_class[c, m] = matrices[m][y == c]

    means = np.empty(n_modes, dtype=object)
    for m in modes:
        means[m] = matrices[m].mean(axis=0)

    means_by_class = np.empty((n_classes, n_modes), dtype=object)
    for c in classes:
        for m in modes:
            means_by_class[c, m] = matrices_by_class[c, m].mean(axis=0)

    Cws_by_class = np.empty((n_classes, n_modes), dtype=object)
    for c in classes:
        for m in modes:
            _mats = matrices_by_class[c, m] - means_by_class[c, m]
            Cws_by_class[c, m] = np.matmul(_mats, np.swapaxes(_mats, 1, 2)).sum(axis=0)

    Cbs_by_class = np.empty((n_classes, n_modes), dtype=object)
    for c in classes:
        n_class_samples = np.sum(y == c)
        for m in modes:
            _means = means_by_class[c, m] - means[m]
            Cbs_by_class[c, m] = n_class_samples * _means @ _means.T

    return Cws_by_class, Cbs_by_class


def _combine_covs(
    Cws_by_class: np.ndarray,
    Cbs_by_class: np.ndarray,
    w_tg: Optional[np.ndarray] = None,
    w_bg: Optional[np.ndarray] = None,
    w_bw: Optional[np.ndarray] = None,
    gamma0: float = 0,
    gamma1: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine covariance matrices with weights."""
    n_classes, n_modes = Cws_by_class.shape

    if w_tg is None:
        w_tg = np.zeros(n_classes)
    if w_bg is None:
        w_bg = np.ones(n_classes)
    if w_bw is None:
        w_bw = np.ones(n_classes)

    Cw_tgs = np.sum([Cws_by_class[c] * w_tg[c] for c in range(n_classes)], axis=0)
    Cw_bgs = np.sum([Cws_by_class[c] * w_bg[c] for c in range(n_classes)], axis=0)
    Cbs = np.sum([Cbs_by_class[c] * w_bw[c] for c in range(n_classes)], axis=0)

    C0s = np.empty(n_modes, dtype=object)
    C1s = np.empty(n_modes, dtype=object)
    for m in range(n_modes):
        C0s[m] = Cw_tgs[m] + Cbs[m] + gamma0 * np.eye(*Cw_tgs[m].shape)
        C1s[m] = Cw_bgs[m] + gamma1 * np.eye(*Cw_bgs[m].shape)

    return C0s, C1s


def gen_cost_tulca(manifold, C0: np.ndarray, C1: np.ndarray, alpha: float):
    """Generate cost function for TULCA optimization."""
    @pymanopt.function.autograd(manifold)
    def cost(M):
        return np.trace(M.T @ C1 @ M) / np.trace(M.T @ C0 @ M)

    @pymanopt.function.autograd(manifold)
    def cost_with_alpha(M):
        return np.trace(M.T @ (alpha * C1 - C0) @ M)

    return cost_with_alpha if alpha else cost


class TULCA:
    """Tensor ULCA for supervised dimensionality reduction of tensor data."""

    def __init__(
        self,
        n_components: Optional[np.ndarray] = None,
        w_tg: Optional[np.ndarray] = None,
        w_bg: Optional[np.ndarray] = None,
        w_bw: Optional[np.ndarray] = None,
        gamma0: float = 0,
        gamma1: float = 0,
        alphas: Optional[np.ndarray] = None,
        convergence_ratio: float = 1e-2,
        max_iterations: int = 100,
        optimization_method: str = "evd",
        manifold_generator=Grassmann,
        manifold_optimizer=TrustRegions(),
        apply_varimax: bool = False,
        apply_consist_axes: bool = True,
        verbosity: bool = False,
    ):
        self.n_components = n_components
        self.w_tg = w_tg
        self.w_bg = w_bg
        self.w_bw = w_bw
        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.alphas = alphas
        self.convergence_ratio = convergence_ratio
        self.max_iterations = max_iterations
        self.optimization_method = optimization_method
        self.manifold_generator = manifold_generator
        self.manifold_optimizer = manifold_optimizer
        self.apply_varimax = apply_varimax
        self.apply_consist_axes = apply_consist_axes
        self.verbosity = verbosity

    def _apply_evd(self, C0: np.ndarray, C1: np.ndarray, alpha: float, n_components: int) -> np.ndarray:
        """Apply eigenvalue decomposition."""
        C = C0 - alpha * C1
        schur_form, v = linalg.schur(C)
        w = linalg.eigvals(schur_form)
        top_eigen_indices = np.argsort(-w)
        return v[:, top_eigen_indices[:n_components]]

    def _optimize_with_evd(self, C0: np.ndarray, C1: np.ndarray, alpha: Optional[float], n_components: int) -> Tuple[np.ndarray, float]:
        """Optimize using eigenvalue decomposition."""
        if alpha is not None:
            M = self._apply_evd(C0, C1, alpha, n_components)
        else:
            alpha = 0
            M = self._apply_evd(C0, C1, alpha, n_components)

            for _ in range(self.max_iterations):
                prev_alpha = alpha
                alpha = np.trace(M.T @ C0 @ M) / np.trace(M.T @ C1 @ M)
                M = self._apply_evd(C0, C1, alpha, n_components)

                improved_ratio = np.abs(prev_alpha - alpha) / alpha
                if self.verbosity:
                    print(f"alpha: {alpha}, improved: {improved_ratio}")
                if improved_ratio < self.convergence_ratio:
                    break

        return M, alpha

    def _optimize_with_manopt(self, C0: np.ndarray, C1: np.ndarray, alpha: Optional[float], n_components: int) -> Tuple[np.ndarray, float]:
        """Optimize using manifold optimization."""
        mode_length = C0.shape[0]
        manifold = self.manifold_generator(mode_length, n_components)
        problem = pymanopt.Problem(manifold, gen_cost_tulca(manifold, C0, C1, alpha))
        self.manifold_optimizer._verbosity = self.verbosity
        M = self.manifold_optimizer.run(problem).point

        if alpha is None:
            alpha = 1 / problem.cost(M)

        return M, alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TULCA':
        """Fit the TULCA model."""
        modes = np.arange(X.ndim - 1)
        n_modes = len(modes)
        self.alphas_ = self.alphas
        self.Ms_ = np.empty(n_modes, dtype=object)

        if self.alphas_ is None:
            self.alphas_ = np.array([None] * n_modes)
        elif np.isscalar(self.alphas_):
            self.alphas_ = np.array([self.alphas_] * n_modes)

        if self.n_components is None:
            self.n_components = (np.array(X.shape[1:]) / 2).astype(int)
        elif np.isscalar(self.n_components):
            self.n_components = np.array([self.n_components] * n_modes)

        self.Cws_by_class_, self.Cbs_by_class_ = _generate_covs(X, y)
        self.optimize()

        return self

    def fit_with_new_weights(
        self,
        w_tg: Optional[List[float]] = None,
        w_bg: Optional[List[float]] = None,
        w_bw: Optional[List[float]] = None,
        gamma0: Optional[float] = None,
        gamma1: Optional[float] = None,
    ) -> 'TULCA':
        """Re-optimize with new weights."""
        if w_tg is not None:
            self.w_tg = w_tg
        if w_bg is not None:
            self.w_bg = w_bg
        if w_bw is not None:
            self.w_bw = w_bw
        if gamma0 is not None:
            self.gamma0 = gamma0
        if gamma1 is not None:
            self.gamma1 = gamma1

        self.optimize()
        return self

    def optimize(self) -> 'TULCA':
        """Perform optimization."""
        C0s, C1s = _combine_covs(
            self.Cws_by_class_,
            self.Cbs_by_class_,
            self.w_tg,
            self.w_bg,
            self.w_bw,
            self.gamma0,
            self.gamma1,
        )

        n_modes = len(C0s)
        for m in range(n_modes):
            C0 = C0s[m] if np.any(C0s[m]) else np.eye(*C0s[m].shape)
            C1 = C1s[m] if np.any(C1s[m]) else np.eye(*C1s[m].shape)

            if self.optimization_method == "evd":
                M, alpha = self._optimize_with_evd(C0, C1, self.alphas_[m], self.n_components[m])
            else:
                M, alpha = self._optimize_with_manopt(C0, C1, self.alphas_[m], self.n_components[m])
            
            self.Ms_[m] = M
            self.alphas_[m] = alpha

            if self.apply_varimax and self.n_components[m] > 1:
                self.Ms_[m] = Rotator(method="varimax").fit_transform(self.Ms_[m])
            if self.apply_consist_axes:
                self.Ms_[m] = self.Ms_[m] * np.sign(self.Ms_[m].sum(axis=0))
                self.Ms_[m] = self.Ms_[m][:, np.argsort(-self.Ms_[m].max(axis=0))]

        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Transform data using fitted projection matrices."""
        X_compressed = X
        for mode, M in enumerate(self.Ms_):
            X_compressed = tl.tenalg.mode_dot(X_compressed, M.T, mode + 1)
        return X_compressed

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X, y)

    def get_projection_matrices(self) -> np.ndarray:
        """Get projection matrices."""
        return np.copy(self.Ms_)

    def get_current_alphas(self) -> np.ndarray:
        """Get current alpha values."""
        return np.copy(self.alphas_)
