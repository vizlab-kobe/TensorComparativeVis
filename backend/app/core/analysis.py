"""
分析処理モジュール（ドメイン非依存）

テンソルデータのクラスター分析に必要な全ての計算処理を提供する。
ドメイン戦略パターンにより、変数名やラベル変換はドメインオブジェクトに委譲する。

主要な処理フロー:
  1. テンソルの展開と標準化 (unfold_and_scale_tensor)
  2. PaCMAP次元削減 (apply_pacmap_reduction)
  3. ランダムフォレストによる特徴量重要度算出 (compute_feature_importance)
  4. 射影行列を用いた寄与度行列の計算 (analyze_tensor_contribution)
  5. 上位重要因子の特定 (get_top_important_factors)
  6. 統計的有意性の評価 (evaluate_statistical_significance)
"""

import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Tuple, Optional, Protocol, runtime_checkable
import pacmap
import warnings


# ── ドメインプロトコル（構造的部分型） ────────────────────────────────────────

@runtime_checkable
class DomainProtocol(Protocol):
    """分析関数が要求するドメインオブジェクトの構造的型定義。

    具象クラス（HPCDomain, AirDataDomain 等）は BaseDomain を継承するが、
    分析モジュールは構造的部分型のみに依存し、疎結合を実現する。
    """

    @property
    def variables(self) -> List[str]:
        """変数名リスト（例: ['AirIn', 'AirOut', 'CPU', 'Water']）"""
        ...

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """空間グリッドの形状 (行数, 列数)"""
        ...

    def index_to_label(self, index: int) -> str:
        """空間インデックスを人間可読なラベルに変換する"""
        ...


# ── 機械学習パラメータ設定 ────────────────────────────────────────────────────

# ランダムフォレストのデフォルトハイパーパラメータ
_RF_DEFAULTS: Dict = {
    'n_estimators': 300,
    'max_depth': 12,
    'min_samples_leaf': 10,
    'min_samples_split': 20,
    'max_features': 0.5,
    'bootstrap': True,
    'oob_score': True,
    'n_jobs': -1,
    'random_state': 42,
}

# PaCMAPのデフォルトパラメータ
_PACMAP_DEFAULTS: Dict = {
    'n_components': 2,
    'n_neighbors': None,  # データサイズに基づき自動決定
}

# 実行時に使用するパラメータ（configure() で上書き可能）
RF_PARAMS: Dict = {**_RF_DEFAULTS}
PACMAP_PARAMS: Dict = {**_PACMAP_DEFAULTS}


def configure(
    *,
    random_forest: Optional[Dict] = None,
    pacmap_params: Optional[Dict] = None,
) -> None:
    """YAML設定から機械学習パラメータを適用する。

    アプリ起動時に main.py から一度だけ呼ばれる。
    指定されたキーのみが上書きされ、残りはデフォルト値を維持する。

    Args:
        random_forest: ランダムフォレストのパラメータ辞書（部分指定可）
        pacmap_params: PaCMAPのパラメータ辞書（部分指定可）
    """
    global RF_PARAMS, PACMAP_PARAMS
    if random_forest:
        RF_PARAMS = {**_RF_DEFAULTS, **random_forest}
    if pacmap_params:
        PACMAP_PARAMS = {**_PACMAP_DEFAULTS, **pacmap_params}


# ── テンソル処理関数 ──────────────────────────────────────────────────────────

def unfold_and_scale_tensor(tensor: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """テンソルを2次元に展開し、標準化する。

    3次元テンソル (T x S x V) を (T x S*V) に展開した後、
    各特徴量を平均0・分散1に標準化する。

    Args:
        tensor: 3次元テンソル (T x S x V)

    Returns:
        (標準化済みデータ, フィット済みスケーラー) のタプル
    """
    T, S, V = tensor.shape
    unfolded_tensor = tensor.reshape(T, S * V)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(unfolded_tensor)
    return scaled_data, scaler


def apply_pacmap_reduction(scaled_data: np.ndarray) -> np.ndarray:
    """PaCMAP次元削減を適用する。

    標準化済みの高次元データを2次元に射影し、
    散布図での可視化に使用する座標を生成する。

    Args:
        scaled_data: 標準化済みデータ (T x features)

    Returns:
        2次元埋め込み座標 (T x 2)
    """
    return pacmap.PaCMAP(**PACMAP_PARAMS).fit_transform(scaled_data)


# ── 特徴量重要度分析 ──────────────────────────────────────────────────────────

def create_binary_classification_data(
    cluster1_indices: List[int],
    cluster2_indices: List[int],
    scaled_data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """2つのクラスターから2値分類データセットを構築する。

    赤クラスター (C1) をラベル1、青クラスター (C2) をラベル0として、
    ランダムフォレスト学習用のデータを作成する。

    Args:
        cluster1_indices: 赤クラスターのサンプルインデックス
        cluster2_indices: 青クラスターのサンプルインデックス
        scaled_data: 標準化済みデータ

    Returns:
        (結合データ, 2値ラベル) のタプル
    """
    combined_data = np.concatenate(
        [scaled_data[cluster1_indices], scaled_data[cluster2_indices]], axis=0
    )
    combined_labels = np.array(
        [1] * len(cluster1_indices) + [0] * len(cluster2_indices)
    )
    return combined_data, combined_labels


def compute_feature_importance(
    combined_data: np.ndarray,
    combined_labels: np.ndarray,
) -> np.ndarray:
    """ランダムフォレストで特徴量重要度を算出する。

    2値分類問題として学習し、各特徴量がクラスター判別に
    どの程度寄与するかを Gini 重要度で評価する。

    Args:
        combined_data: 結合済みデータ
        combined_labels: 2値ラベル

    Returns:
        特徴量重要度の配列 (features,)
    """
    # ラベルが1種類しかない場合はゼロベクトルを返す
    if len(np.unique(combined_labels)) == 1:
        return np.zeros(combined_data.shape[1])

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(combined_data, combined_labels)
    return rf.feature_importances_


def analyze_tensor_contribution(
    cluster1_indices: List[int],
    cluster2_indices: List[int],
    scaled_data: np.ndarray,
    Ms: np.ndarray,
    Mv: np.ndarray,
    S: int,
    V: int,
) -> np.ndarray:
    """テンソル寄与度行列を計算する。

    射影後の特徴量空間で算出した重要度を、射影行列のクロネッカー積を用いて
    元の空間・変数次元に逆射影し、S x V の寄与度行列を得る。

    Args:
        cluster1_indices: 赤クラスターのインデックス
        cluster2_indices: 青クラスターのインデックス
        scaled_data: 射影後の標準化データ
        Ms: 空間モード射影行列
        Mv: 変数モード射影行列
        S: 空間次元数
        V: 変数次元数

    Returns:
        寄与度行列 (S x V)
    """
    combined_data, combined_labels = create_binary_classification_data(
        cluster1_indices, cluster2_indices, scaled_data
    )
    feature_importance = compute_feature_importance(combined_data, combined_labels)
    # クロネッカー積による逆射影: 低次元の重要度 → 元次元の寄与度
    all_importances = (np.kron(Ms, Mv) @ feature_importance).reshape(S, V)
    return all_importances


def standardize_contributions(contribution_matrix: np.ndarray) -> np.ndarray:
    """寄与度行列を変数ごとに標準化する。

    各変数（列）について平均0・標準偏差1に正規化することで、
    スケールの異なる変数間での公平な比較を可能にする。

    Args:
        contribution_matrix: 寄与度行列 (S x V)

    Returns:
        標準化済み寄与度行列 (S x V)
    """
    _, V = contribution_matrix.shape
    standardized = np.zeros_like(contribution_matrix)

    for v in range(V):
        contrib_v = contribution_matrix[:, v]
        mean_v = contrib_v.mean()
        std_v = contrib_v.std()

        if std_v > 0:
            standardized[:, v] = (contrib_v - mean_v) / std_v
        else:
            standardized[:, v] = 0

    return standardized


def get_top_important_factors(
    contribution_matrix: np.ndarray,
    domain: DomainProtocol,
    top_k: int = 10,
) -> List[Dict]:
    """重要度上位の因子を抽出する。

    標準化済み寄与度の絶対値でソートし、上位 top_k 個の因子について
    空間ラベル・変数名・スコアを含む辞書リストを返す。

    Args:
        contribution_matrix: 寄与度行列 (S x V)
        domain: ドメイン戦略インスタンス（ラベル変換に使用）
        top_k: 返す因子数の上限

    Returns:
        上位因子の辞書リスト
    """
    standardized_contrib = standardize_contributions(contribution_matrix)
    importance_scores = np.abs(standardized_contrib)

    _, V = importance_scores.shape
    flat_scores = importance_scores.flatten()
    flat_indices = np.argsort(flat_scores)[::-1]

    top_factors = []
    for i in range(min(top_k, len(flat_indices))):
        flat_idx = flat_indices[i]
        # フラットインデックスから空間・変数インデックスを復元
        s = flat_idx // V
        v = flat_idx % V
        score = float(flat_scores[flat_idx])

        top_factors.append({
            'rank': i + 1,
            'rack': domain.index_to_label(s),
            'variable': domain.variables[v],
            'score': score,
            'rack_idx': int(s),
            'var_idx': int(v),
        })

    return top_factors


# ── 統計的有意性評価 ──────────────────────────────────────────────────────────

def calculate_cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d 効果量を計算する。

    プール標準偏差を用いた2群間の標準化平均差を算出する。
    効果量の解釈基準: 0.2=小, 0.5=中, 0.8=大

    Args:
        group1: 第1群のデータ
        group2: 第2群のデータ

    Returns:
        Cohen's d の値
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1))
        / (n1 + n2 - 2)
    )
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0


def evaluate_statistical_significance(
    cluster1_indices: List[int],
    cluster2_indices: List[int],
    rack_idx: int,
    var_idx: int,
    original_data: np.ndarray,
    domain: DomainProtocol,
) -> Dict:
    """2つのクラスター間の統計的有意性を評価する。

    Welch の t検定と Cohen's d 効果量を算出し、
    差の方向・有意性・効果量の解釈を含む結果辞書を返す。

    Args:
        cluster1_indices: 赤クラスターのインデックス
        cluster2_indices: 青クラスターのインデックス
        rack_idx: 空間インデックス
        var_idx: 変数インデックス
        original_data: 元スケールのデータ (T x S x V)
        domain: ドメイン戦略インスタンス

    Returns:
        統計検定結果の辞書
    """
    time_series = original_data[:, rack_idx, var_idx]

    cluster1_values = time_series[cluster1_indices]
    cluster2_values = time_series[cluster2_indices]

    # 最小サンプルサイズの確認
    min_sample_size = 3
    if len(cluster1_values) < min_sample_size or len(cluster2_values) < min_sample_size:
        return {
            'rack': domain.index_to_label(rack_idx),
            'variable': domain.variables[var_idx],
            'direction': "Insufficient samples",
            'mean_diff': 0.0,
            'p_value': 1.0,
            'cohen_d': 0.0,
            'significance': "Cannot determine",
            'effect_size': "Cannot determine",
        }

    # 分散ゼロの場合の特殊処理
    if np.var(cluster1_values) == 0 and np.var(cluster2_values) == 0:
        if np.mean(cluster1_values) == np.mean(cluster2_values):
            direction = "No difference"
            mean_diff = 0.0
        else:
            direction = (
                "Higher in Cluster 1"
                if np.mean(cluster1_values) > np.mean(cluster2_values)
                else "Lower in Cluster 1"
            )
            mean_diff = abs(np.mean(cluster1_values) - np.mean(cluster2_values))

        return {
            'rack': domain.index_to_label(rack_idx),
            'variable': domain.variables[var_idx],
            'direction': direction,
            'mean_diff': mean_diff,
            'p_value': 0.0 if direction != "No difference" else 1.0,
            'cohen_d': float('inf') if direction != "No difference" else 0.0,
            'significance': "Perfect separation" if direction != "No difference" else "No difference",
            'effect_size': "Infinite" if direction != "No difference" else "None",
        }

    # Welch の t検定を実施
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_stat, p_value = stats.ttest_ind(
                cluster1_values, cluster2_values, equal_var=False
            )

        if np.isnan(p_value) or np.isnan(t_stat):
            p_value = 1.0
    except Exception:
        p_value = 1.0

    # Cohen's d 効果量を算出
    try:
        cohen_d = calculate_cohen_d(cluster1_values, cluster2_values)
        if np.isnan(cohen_d) or np.isinf(cohen_d):
            cohen_d = 0.0
    except Exception:
        cohen_d = 0.0

    mean_diff = np.mean(cluster1_values) - np.mean(cluster2_values)
    direction = "Higher in Cluster 1" if mean_diff > 0 else "Lower in Cluster 1"

    # 効果量の解釈（Cohen の基準に基づく）
    abs_cohen_d = abs(cohen_d)
    if abs_cohen_d >= 0.8:
        effect_size = "Large"
    elif abs_cohen_d >= 0.5:
        effect_size = "Medium"
    elif abs_cohen_d >= 0.2:
        effect_size = "Small"
    else:
        effect_size = "Very small"

    return {
        'rack': domain.index_to_label(rack_idx),
        'variable': domain.variables[var_idx],
        'direction': direction,
        'mean_diff': float(abs(mean_diff)),
        'p_value': float(p_value),
        'cohen_d': float(abs_cohen_d),
        'significance': "Significant" if p_value < 0.05 else "Not significant",
        'effect_size': effect_size,
    }
