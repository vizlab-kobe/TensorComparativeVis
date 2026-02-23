"""
Pydanticリクエスト/レスポンスモデル定義

FastAPI エンドポイントで使用される全てのリクエストボディと
レスポンスボディの型定義を集約する。
"""

from pydantic import BaseModel
from typing import List, Dict, Any


# ── TULCA重み関連 ────────────────────────────────────────────────────────────

class ClassWeight(BaseModel):
    """単一クラスの重み設定。

    TULCA（Tensor ULCA）の最適化で使用される3種類の重み:
      - w_tg: ターゲットクラス内分散の重み
      - w_bw: クラス間分散の重み
      - w_bg: 背景クラス内分散の重み
    """
    w_tg: float = 0.0
    w_bw: float = 1.0
    w_bg: float = 1.0


class ComputeEmbeddingRequest(BaseModel):
    """埋め込み計算リクエスト。

    各クラスの重みリストを受け取り、TULCA再最適化 + PaCMAP次元削減を実行する。
    """
    class_weights: List[ClassWeight]


class ComputeEmbeddingResponse(BaseModel):
    """埋め込み計算レスポンス。

    Attributes:
        embedding: PaCMAP 2次元埋め込み座標 (T x 2)
        scaled_data: 標準化済み展開テンソル (T x S*V)
        Ms: 空間モード射影行列
        Mv: 変数モード射影行列
        labels: 各サンプルのクラスラベル
    """
    embedding: List[List[float]]
    scaled_data: List[List[float]]
    Ms: List[List[float]]
    Mv: List[List[float]]
    labels: List[int]


# ── クラスター分析関連 ────────────────────────────────────────────────────────

class ClusterAnalysisRequest(BaseModel):
    """クラスター分析リクエスト。

    2つのクラスター（赤・青）のインデックスと、
    分析に必要なデータ（スケーリング済み、射影行列）を受け取る。
    """
    cluster1_indices: List[int]
    cluster2_indices: List[int]
    scaled_data: List[List[float]]
    Ms: List[List[float]]
    Mv: List[List[float]]


class StatisticalResult(BaseModel):
    """統計的有意性の検定結果。

    Welch の t検定と Cohen's d 効果量を含む。
    """
    rack: str           # 空間ラベル（例: "A1", "Station-5"）
    variable: str       # 変数名（例: "CPU", "PM2.5"）
    direction: str      # 差の方向（"Higher in Cluster 1" 等）
    mean_diff: float    # 平均値の差の絶対値
    p_value: float      # p値
    cohen_d: float      # Cohen's d 効果量
    significance: str   # 有意性判定（"Significant" / "Not significant"）
    effect_size: str    # 効果量の解釈（"Large" / "Medium" / "Small" / "Very small"）


class FeatureImportance(BaseModel):
    """特徴量の重要度（統計的分析付き）。

    ランダムフォレストによる特徴量重要度スコアと、
    クラスター間の時系列データ・統計検定結果を含む。
    """
    rank: int                           # 重要度順位
    rack: str                           # 空間ラベル
    variable: str                       # 変数名
    score: float                        # 寄与度スコア（標準化済み）
    importance: float                   # 重要度（score と同値）
    cluster1_data: List[float]          # 赤クラスターの元データ値
    cluster2_data: List[float]          # 青クラスターの元データ値
    cluster1_time: List[str]            # 赤クラスターの時間軸ラベル
    cluster2_time: List[str]            # 青クラスターの時間軸ラベル
    mean_diff: float                    # 平均値の差
    statistical_result: StatisticalResult  # 統計検定結果


class ClusterAnalysisResponse(BaseModel):
    """クラスター分析レスポンス。

    Attributes:
        top_features: 重要度上位の特徴量リスト
        contribution_matrix: S x V 寄与度行列（ヒートマップ表示用）
    """
    top_features: List[FeatureImportance]
    contribution_matrix: List[List[float]]


# ── AI解釈関連 ────────────────────────────────────────────────────────────────

class InterpretationRequest(BaseModel):
    """AI解釈リクエスト。

    上位特徴量と各クラスターのサイズを渡してLLMに解釈を依頼する。
    """
    top_features: List[Dict[str, Any]]
    cluster1_size: int
    cluster2_size: int


class InterpretationSection(BaseModel):
    """AI解釈の1セクション。

    Attributes:
        title: セクションタイトル（例: "Key Findings"）
        text: セクション本文（自然言語テキスト）
        highlights: 強調キーワードのリスト
    """
    title: str
    text: str
    highlights: List[str] = []


class InterpretationResponse(BaseModel):
    """AI解釈レスポンス。複数セクションで構成される構造化された解釈結果。"""
    sections: List[InterpretationSection]


# ── 分析比較関連 ──────────────────────────────────────────────────────────────

class AnalysisSummary(BaseModel):
    """保存された分析の要約情報（比較用）。"""
    cluster1_size: int
    cluster2_size: int
    significant_count: int      # 統計的に有意な特徴量の数
    top_variables: List[str]    # 上位変数名
    top_racks: List[str]        # 上位空間ラベル
    top_features: List[Dict[str, Any]]  # 上位特徴量の詳細


class CompareRequest(BaseModel):
    """2つの分析結果の比較リクエスト。"""
    analysis_a: AnalysisSummary
    analysis_b: AnalysisSummary


class CompareResponse(BaseModel):
    """分析比較レスポンス。AI生成の比較セクションを返す。"""
    sections: List[InterpretationSection]


# ── 設定レスポンス ────────────────────────────────────────────────────────────

class ConfigResponse(BaseModel):
    """アプリケーション設定レスポンス。

    フロントエンド初期化時に必要な設定情報を返す。
    """
    variables: List[str]            # 変数名リスト
    n_classes: int                  # クラス数
    grid_shape: List[int]           # 空間グリッドの形状 [行数, 列数]
    colors: Dict[str, Any]          # 色設定（クラス色、クラスター色）
    visualization_type: str = "grid"  # 可視化タイプ（"grid" or "geo_map"）
    class_labels: List[str] = []    # ドメイン固有のクラスラベル（例: ["FY2014", ...]）
