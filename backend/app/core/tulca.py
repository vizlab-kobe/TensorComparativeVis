"""
TULCA（Tensor ULCA）アルゴリズム実装

テンソルデータに対する教師あり次元削減アルゴリズム。
共分散行列の固有値分解またはGrassmann多様体上の最適化により、
クラスタ判別を最大化する射影行列を各モードに対して求める。

数学的背景:
  - 入力テンソル X: (T x S x V) の3次元データ
  - 目的: 各モード m に対して射影行列 M_m を求め、
    クラス間分散を最大化しつつクラス内分散を最小化する
  - 最適化目標: trace(M^T C0 M) / trace(M^T C1 M) の最大化
    C0: ターゲットクラス内分散 + クラス間分散（最大化したい）
    C1: 背景クラス内分散（最小化したい）

参考文献:
  C. Hayashi and K. Mueller, "TULCA" (原著論文)
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

# ランダムフォレストの不要な警告を抑制
warnings.filterwarnings(
    "ignore",
    message="n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism."
)


# ── 共分散行列の生成 ──────────────────────────────────────────────────────────

def _generate_covs(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """テンソルデータからクラス内・クラス間共分散行列を生成する。

    各モード m について、テンソルを mode-m 展開した行列を用いて
    クラス別の共分散行列を計算する。

    Args:
        X: 入力テンソル (T x S x V)
        y: クラスラベル (T,)

    Returns:
        (クラス内共分散, クラス間共分散) のタプル。
        各要素は (n_classes x n_modes) のオブジェクト配列。
    """
    classes = np.unique(y)
    modes = np.arange(X.ndim - 1)
    n_samples = X.shape[0]
    n_classes = len(classes)
    n_modes = len(modes)

    # 各モードの展開行列を計算
    matrices = np.empty(n_modes, dtype=object)
    for m in modes:
        matrices[m] = np.swapaxes(
            tl.unfold(X, m + 1).reshape(
                X.shape[m + 1], n_samples, int(X.size / X.shape[m + 1] / n_samples)
            ),
            0, 1,
        )

    # クラス別に分割
    matrices_by_class = np.empty((n_classes, n_modes), dtype=object)
    for c in classes:
        for m in modes:
            matrices_by_class[c, m] = matrices[m][y == c]

    # 全体平均とクラス別平均の計算
    means = np.empty(n_modes, dtype=object)
    for m in modes:
        means[m] = matrices[m].mean(axis=0)

    means_by_class = np.empty((n_classes, n_modes), dtype=object)
    for c in classes:
        for m in modes:
            means_by_class[c, m] = matrices_by_class[c, m].mean(axis=0)

    # クラス内共分散の計算: Cw = Σ(x_i - μ_c)(x_i - μ_c)^T
    Cws_by_class = np.empty((n_classes, n_modes), dtype=object)
    for c in classes:
        for m in modes:
            _mats = matrices_by_class[c, m] - means_by_class[c, m]
            Cws_by_class[c, m] = np.matmul(
                _mats, np.swapaxes(_mats, 1, 2)
            ).sum(axis=0)

    # クラス間共分散の計算: Cb = n_c * (μ_c - μ)(μ_c - μ)^T
    Cbs_by_class = np.empty((n_classes, n_modes), dtype=object)
    for c in classes:
        n_class_samples = np.sum(y == c)
        for m in modes:
            _means = means_by_class[c, m] - means[m]
            Cbs_by_class[c, m] = n_class_samples * _means @ _means.T

    return Cws_by_class, Cbs_by_class


# ── 共分散行列の重み付き結合 ──────────────────────────────────────────────────

def _combine_covs(
    Cws_by_class: np.ndarray,
    Cbs_by_class: np.ndarray,
    w_tg: Optional[np.ndarray] = None,
    w_bg: Optional[np.ndarray] = None,
    w_bw: Optional[np.ndarray] = None,
    gamma0: float = 0,
    gamma1: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """共分散行列を重み付きで結合して最適化用の行列ペアを構築する。

    TULCA の目的関数で使用される C0（最大化ターゲット）と
    C1（最小化ターゲット）を構築する:
      C0 = w_tg * Cw_tg + w_bw * Cb + gamma0 * I  （ターゲット分散 + クラス間）
      C1 = w_bg * Cw_bg + gamma1 * I               （背景分散）

    Args:
        Cws_by_class: クラス別クラス内共分散 (n_classes x n_modes)
        Cbs_by_class: クラス別クラス間共分散 (n_classes x n_modes)
        w_tg: ターゲットクラス内重み（注目クラスの分散を保持）
        w_bg: 背景クラス内重み（他クラスの分散を縮小）
        w_bw: クラス間重み（クラス間の分離を促進）
        gamma0: C0 の正則化パラメータ
        gamma1: C1 の正則化パラメータ

    Returns:
        (C0s, C1s) のタプル（各モードに対する行列配列）
    """
    n_classes, n_modes = Cws_by_class.shape

    # デフォルト重みの設定
    if w_tg is None:
        w_tg = np.zeros(n_classes)
    if w_bg is None:
        w_bg = np.ones(n_classes)
    if w_bw is None:
        w_bw = np.ones(n_classes)

    # 重み付き合計の計算
    Cw_tgs = np.sum([Cws_by_class[c] * w_tg[c] for c in range(n_classes)], axis=0)
    Cw_bgs = np.sum([Cws_by_class[c] * w_bg[c] for c in range(n_classes)], axis=0)
    Cbs = np.sum([Cbs_by_class[c] * w_bw[c] for c in range(n_classes)], axis=0)

    # 最適化用の行列ペアを構築
    C0s = np.empty(n_modes, dtype=object)
    C1s = np.empty(n_modes, dtype=object)
    for m in range(n_modes):
        C0s[m] = Cw_tgs[m] + Cbs[m] + gamma0 * np.eye(*Cw_tgs[m].shape)
        C1s[m] = Cw_bgs[m] + gamma1 * np.eye(*Cw_bgs[m].shape)

    return C0s, C1s


# ── 最適化コスト関数 ──────────────────────────────────────────────────────────

def gen_cost_tulca(manifold, C0: np.ndarray, C1: np.ndarray, alpha: float):
    """TULCA のGrassmann多様体上のコスト関数を生成する。

    alpha が指定されている場合は Rayleigh 商の代わりに
    線形コスト trace(M^T (alpha*C1 - C0) M) を使用する。

    Args:
        manifold: pymanopt の多様体オブジェクト
        C0: 最大化ターゲット行列
        C1: 最小化ターゲット行列
        alpha: トレードオフパラメータ（None で自動調整）

    Returns:
        pymanopt 互換のコスト関数
    """
    @pymanopt.function.autograd(manifold)
    def cost(M):
        return np.trace(M.T @ C1 @ M) / np.trace(M.T @ C0 @ M)

    @pymanopt.function.autograd(manifold)
    def cost_with_alpha(M):
        return np.trace(M.T @ (alpha * C1 - C0) @ M)

    return cost_with_alpha if alpha else cost


# ── TULCA モデルクラス ────────────────────────────────────────────────────────

class TULCA:
    """テンソルデータに対する教師あり次元削減モデル。

    各テンソルモードに対して射影行列を学習し、
    クラス判別を最大化する低次元表現を生成する。

    使用例:
        model = TULCA(n_components=np.array([10, 3]))
        model.fit(tensor_X, tensor_y)
        low_dim = model.transform(tensor_X)
    """

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
        """TULCAモデルを構成する。

        Args:
            n_components: 各モードの目標次元数の配列（例: [10, 3]）
            w_tg: ターゲットクラス内重み（クラスごと）
            w_bg: 背景クラス内重み（クラスごと）
            w_bw: クラス間重み（クラスごと）
            gamma0: C0 正則化パラメータ
            gamma1: C1 正則化パラメータ
            alphas: 各モードの alpha 初期値（None で自動調整）
            convergence_ratio: EVD の収束判定閾値
            max_iterations: EVD の最大反復回数
            optimization_method: "evd"（固有値分解）or "manopt"（多様体最適化）
            manifold_generator: 多様体の種類（デフォルト: Grassmann）
            manifold_optimizer: 多様体の最適化器（デフォルト: TrustRegions）
            apply_varimax: Varimax回転の適用フラグ
            apply_consist_axes: 軸の符号・順序の一貫性を保証するフラグ
            verbosity: 詳細ログの出力フラグ
        """
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

    # ── EVD ベースの最適化 ────────────────────────────────────────────────────

    def _apply_evd(
        self, C0: np.ndarray, C1: np.ndarray, alpha: float, n_components: int
    ) -> np.ndarray:
        """固有値分解による射影行列の計算。

        C = C0 - alpha * C1 の Schur 分解を行い、
        上位 n_components 個の固有ベクトルを射影行列とする。

        Args:
            C0: 最大化ターゲット行列
            C1: 最小化ターゲット行列
            alpha: トレードオフパラメータ
            n_components: 射影先の次元数

        Returns:
            射影行列 M (元次元 x n_components)
        """
        C = C0 - alpha * C1
        schur_form, v = linalg.schur(C)
        w = linalg.eigvals(schur_form)
        top_eigen_indices = np.argsort(-w)
        return v[:, top_eigen_indices[:n_components]]

    def _optimize_with_evd(
        self, C0: np.ndarray, C1: np.ndarray,
        alpha: Optional[float], n_components: int,
    ) -> Tuple[np.ndarray, float]:
        """EVD による反復最適化。

        alpha が未指定の場合、Rayleigh 商の反復更新で
        最適な alpha を自動探索する。

        Args:
            C0: 最大化ターゲット行列
            C1: 最小化ターゲット行列
            alpha: 初期 alpha（None で自動探索）
            n_components: 目標次元数

        Returns:
            (射影行列, 最適 alpha) のタプル
        """
        if alpha is not None:
            M = self._apply_evd(C0, C1, alpha, n_components)
        else:
            alpha = 0
            M = self._apply_evd(C0, C1, alpha, n_components)

            # alpha の反復更新（Rayleigh 商の最大化）
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

    # ── 多様体最適化 ──────────────────────────────────────────────────────────

    def _optimize_with_manopt(
        self, C0: np.ndarray, C1: np.ndarray,
        alpha: Optional[float], n_components: int,
    ) -> Tuple[np.ndarray, float]:
        """Grassmann 多様体上の最適化。

        pymanopt の TrustRegions ソルバーを用いて、
        Grassmann 多様体上でコスト関数を直接最適化する。

        Args:
            C0: 最大化ターゲット行列
            C1: 最小化ターゲット行列
            alpha: トレードオフパラメータ（None で Rayleigh 商を使用）
            n_components: 目標次元数

        Returns:
            (射影行列, alpha) のタプル
        """
        mode_length = C0.shape[0]
        manifold = self.manifold_generator(mode_length, n_components)
        problem = pymanopt.Problem(
            manifold, gen_cost_tulca(manifold, C0, C1, alpha)
        )
        self.manifold_optimizer._verbosity = self.verbosity
        M = self.manifold_optimizer.run(problem).point

        if alpha is None:
            alpha = 1 / problem.cost(M)

        return M, alpha

    # ── 公開API ───────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TULCA':
        """テンソルデータにモデルをフィッティングする。

        共分散行列を生成し、各モードの射影行列を最適化する。
        初回呼び出し時のみ共分散行列の計算を行い、以降は
        fit_with_new_weights() で重みのみ更新可能。

        Args:
            X: 入力テンソル (T x S x V)
            y: クラスラベル (T,)

        Returns:
            フィッティング済みの self
        """
        modes = np.arange(X.ndim - 1)
        n_modes = len(modes)
        self.alphas_ = self.alphas
        self.Ms_ = np.empty(n_modes, dtype=object)

        # alpha の初期化
        if self.alphas_ is None:
            self.alphas_ = np.array([None] * n_modes)
        elif np.isscalar(self.alphas_):
            self.alphas_ = np.array([self.alphas_] * n_modes)

        # 次元数の初期化（未指定時は元次元の半分）
        if self.n_components is None:
            self.n_components = (np.array(X.shape[1:]) / 2).astype(int)
        elif np.isscalar(self.n_components):
            self.n_components = np.array([self.n_components] * n_modes)

        # 共分散行列の計算と最適化
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
        """重みを更新して再最適化する。

        共分散行列は再計算せず、新しい重みで射影行列のみ更新する。
        フロントエンドのスライダー操作時に呼ばれる。

        Args:
            w_tg: 新しいターゲットクラス内重み
            w_bg: 新しい背景クラス内重み
            w_bw: 新しいクラス間重み
            gamma0: 新しい C0 正則化パラメータ
            gamma1: 新しい C1 正則化パラメータ

        Returns:
            再最適化済みの self
        """
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
        """現在の重みで各モードの射影行列を最適化する。

        各モードについて EVD or ManOpt で射影行列を求めた後、
        任意で Varimax 回転と軸方向の一貫性調整を適用する。

        Returns:
            最適化済みの self
        """
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
            # ゼロ行列の場合は単位行列で代替（数値安定性）
            C0 = C0s[m] if np.any(C0s[m]) else np.eye(*C0s[m].shape)
            C1 = C1s[m] if np.any(C1s[m]) else np.eye(*C1s[m].shape)

            if self.optimization_method == "evd":
                M, alpha = self._optimize_with_evd(
                    C0, C1, self.alphas_[m], self.n_components[m]
                )
            else:
                M, alpha = self._optimize_with_manopt(
                    C0, C1, self.alphas_[m], self.n_components[m]
                )

            self.Ms_[m] = M
            self.alphas_[m] = alpha

            # Varimax 回転の適用（2次元以上の場合のみ）
            if self.apply_varimax and self.n_components[m] > 1:
                self.Ms_[m] = Rotator(method="varimax").fit_transform(self.Ms_[m])

            # 軸の符号・順序の一貫性を保証
            if self.apply_consist_axes:
                self.Ms_[m] = self.Ms_[m] * np.sign(self.Ms_[m].sum(axis=0))
                self.Ms_[m] = self.Ms_[m][:, np.argsort(-self.Ms_[m].max(axis=0))]

        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """学習済み射影行列でテンソルを低次元に変換する。

        各モードに対して mode-m 積を適用し、テンソルを縮約する。

        Args:
            X: 入力テンソル (T x S x V)
            y: 未使用（scikit-learn 互換性のため保持）

        Returns:
            低次元テンソル (T x s_prime x v_prime)
        """
        X_compressed = X
        for mode, M in enumerate(self.Ms_):
            X_compressed = tl.tenalg.mode_dot(X_compressed, M.T, mode + 1)
        return X_compressed

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """フィットと変換を一連で実行する。

        Args:
            X: 入力テンソル
            y: クラスラベル

        Returns:
            低次元テンソル
        """
        return self.fit(X, y).transform(X, y)

    def get_projection_matrices(self) -> np.ndarray:
        """学習済み射影行列の配列を返す（コピー）。

        Returns:
            射影行列の配列 [M_space, M_variable]
        """
        return np.copy(self.Ms_)

    def get_current_alphas(self) -> np.ndarray:
        """現在の alpha 値の配列を返す（コピー）。

        Returns:
            各モードの alpha 値の配列
        """
        return np.copy(self.alphas_)
