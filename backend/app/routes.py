"""
APIルート定義モジュール

全てのエンドポイントハンドラをこのファイルに集約する。
各ハンドラは `request.app.state` 経由で共有依存オブジェクト
（ドメイン、データローダー、AIインタープリター等）にアクセスする。

エンドポイント一覧:
  GET  /api/config           - アプリケーション設定の取得
  GET  /api/coordinates      - 地理座標の取得（geo_map用）
  POST /api/compute-embedding - TULCA + PaCMAP 埋め込み計算
  POST /api/analyze-clusters  - クラスター間差異分析
  POST /api/interpret-clusters - AI解釈の生成
  GET  /api/health           - ヘルスチェック
"""

import os
import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from app.core import (
    unfold_and_scale_tensor,
    apply_pacmap_reduction,
    analyze_tensor_contribution,
    get_top_important_factors,
    evaluate_statistical_significance,
    apply_fdr_correction,
)
from app.core.tulca import TULCA
from models import (
    ComputeEmbeddingRequest, ComputeEmbeddingResponse,
    ClusterAnalysisRequest, ClusterAnalysisResponse,
    InterpretationRequest, InterpretationResponse,
    ConfigResponse, FeatureImportance, StatisticalResult,
    ComparisonContext, SeparationFactors, SuggestedExploration,
)

logger = logging.getLogger(__name__)

# APIルーターの作成（全エンドポイントに /api プレフィックスを適用）
router = APIRouter(prefix="/api")


# ── ヘルパー関数 ─────────────────────────────────────────────────────────────

def _get_tulca_model(state) -> TULCA:
    """TULCAモデルを取得または初期化する。

    初回呼び出し時にモデルを作成してフィッティングし、
    以降はキャッシュされたモデルを返す（遅延初期化パターン）。

    Args:
        state: FastAPIアプリケーションの共有状態

    Returns:
        フィッティング済みのTULCAモデル
    """
    if state.tulca_model is None:
        params = state.tulca_params
        state.tulca_model = TULCA(
            n_components=np.array([params['s_prime'], params['v_prime']]),
            optimization_method=params['optimization_method'],
        )
        state.tulca_model.fit(
            state.data_loader.tensor_X,
            state.data_loader.tensor_y,
        )
    return state.tulca_model


# ── エンドポイント定義 ────────────────────────────────────────────────────────

@router.get("/config", response_model=ConfigResponse)
async def get_config(request: Request):
    """アプリケーション設定を返す。

    フロントエンド初期化時に呼ばれ、変数名・クラス数・色設定等を提供する。
    """
    s = request.app.state
    return ConfigResponse(
        variables=s.domain.variables,
        n_classes=s.data_loader.n_classes,
        grid_shape=list(s.domain.grid_shape),
        colors=s.colors,
        visualization_type=s.domain.visualization_type,
        class_labels=s.domain.class_labels,
    )


@router.get("/coordinates")
async def get_coordinates(request: Request):
    """空間座標データを返す（geo_map可視化用）。

    ドメインが地理座標を持たない場合は空リストを返す。
    """
    coords = request.app.state.domain.get_coordinates()
    if not coords:
        return {"coordinates": [], "available": False}
    return {"coordinates": coords, "available": True}


@router.post("/compute-embedding", response_model=ComputeEmbeddingResponse)
async def compute_embedding(body: ComputeEmbeddingRequest, request: Request):
    """TULCA + PaCMAP 埋め込みを計算する。

    処理フロー:
      1. リクエストからクラス重みを抽出
      2. TULCAモデルを新しい重みで再最適化
      3. テンソルデータを低次元空間に射影
      4. 射影後のテンソルを展開・標準化
      5. PaCMAPで2次元埋め込みを生成
    """
    try:
        s = request.app.state
        n_classes = s.data_loader.n_classes

        # クラスごとの重みを抽出
        w_tgs = [body.class_weights[i].w_tg for i in range(n_classes)]
        w_bgs = [body.class_weights[i].w_bg for i in range(n_classes)]
        w_bws = [body.class_weights[i].w_bw for i in range(n_classes)]

        # TULCAモデルの取得と重み再最適化
        model = _get_tulca_model(s)
        model.fit_with_new_weights(w_tgs, w_bgs, w_bws)

        # テンソルの低次元射影
        low_dim_tensor = model.transform(s.data_loader.tensor_X)
        projection_matrices = model.get_projection_matrices()

        # 射影行列の取得（空間モード・変数モード）
        Ms = np.asarray(projection_matrices[0])
        Mv = np.asarray(projection_matrices[1])

        # 展開・標準化と2次元埋め込み
        scaled_data, _ = unfold_and_scale_tensor(low_dim_tensor)
        embedding = apply_pacmap_reduction(scaled_data)

        return ComputeEmbeddingResponse(
            embedding=embedding.tolist(),
            scaled_data=scaled_data.tolist(),
            Ms=Ms.tolist(),
            Mv=Mv.tolist(),
            labels=s.data_loader.tensor_y.tolist(),
        )
    except Exception as e:
        logger.exception("埋め込み計算中にエラーが発生")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-clusters", response_model=ClusterAnalysisResponse)
async def analyze_clusters(body: ClusterAnalysisRequest, request: Request):
    """2つのクラスター間の差異を分析する。

    処理フロー:
      1. ランダムフォレストで特徴量重要度を算出
      2. 射影行列を用いて元の空間・変数次元に逆射影（寄与度行列）
      3. 上位10個の重要因子を特定
      4. 各因子について統計的有意性を評価（t検定 + Cohen's d）
    """
    try:
        s = request.app.state
        scaled_data = np.array(body.scaled_data)
        Ms = np.array(body.Ms)
        Mv = np.array(body.Mv)

        # テンソル全体の形状 (時間 x 空間 x 変数)
        T, S, V = s.data_loader.shape

        # 寄与度行列の計算（特徴量重要度 × 射影行列の逆射影）
        contribution_matrix = analyze_tensor_contribution(
            body.cluster1_indices, body.cluster2_indices,
            scaled_data, Ms, Mv, S, V,
        )

        # 上位10個の重要因子を取得
        top_factors = get_top_important_factors(contribution_matrix, s.domain, top_k=10)

        # クラスターの時間軸ラベルを取得
        cluster1_array = np.array(body.cluster1_indices)
        cluster2_array = np.array(body.cluster2_indices)
        cluster1_time = [str(t) for t in s.data_loader.time_axis[cluster1_array]]
        cluster2_time = [str(t) for t in s.data_loader.time_axis[cluster2_array]]

        # 各因子の詳細データを構築
        # まず統計検定結果を全因子分まとめて収集する（FDR補正のため）
        stat_results = []
        factor_data = []
        for factor in top_factors:
            rack_idx = factor['rack_idx']
            var_idx = factor['var_idx']

            # 元データから各クラスターの値を抽出
            cluster1_data = s.data_loader.original_data[cluster1_array, rack_idx, var_idx].tolist()
            cluster2_data = s.data_loader.original_data[cluster2_array, rack_idx, var_idx].tolist()

            # 統計的有意性の評価（Welch t検定 + Mann-Whitney U + Cohen's d）
            stat_result = evaluate_statistical_significance(
                body.cluster1_indices, body.cluster2_indices,
                rack_idx, var_idx,
                s.data_loader.original_data, s.domain,
            )
            stat_results.append(stat_result)
            factor_data.append((factor, cluster1_data, cluster2_data))

        # Benjamini-Hochberg FDR 補正を全因子のp値に一括適用
        apply_fdr_correction(stat_results)

        # 補正済み結果を使って FeatureImportance を構築
        features = []
        for (factor, cluster1_data, cluster2_data), stat_result in zip(factor_data, stat_results):
            mean_diff = float(np.mean(cluster1_data) - np.mean(cluster2_data))

            features.append(FeatureImportance(
                rank=factor['rank'],
                rack=factor['rack'],
                variable=factor['variable'],
                score=factor['score'],
                importance=factor['score'],
                cluster1_data=cluster1_data,
                cluster2_data=cluster2_data,
                cluster1_time=cluster1_time,
                cluster2_time=cluster2_time,
                mean_diff=mean_diff,
                statistical_result=StatisticalResult(**stat_result),
            ))

        return ClusterAnalysisResponse(
            top_features=features,
            contribution_matrix=contribution_matrix.tolist(),
        )
    except Exception as e:
        logger.exception("クラスター分析中にエラーが発生")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interpret-clusters", response_model=InterpretationResponse)
async def interpret_clusters(body: InterpretationRequest, request: Request):
    """クラスター差異のAI解釈を生成する。

    Gemini API を使用して、特徴量データから自然言語の構造化された解釈を生成する。
    API が利用できない場合はフォールバック（データベースの要約）を返す。
    """
    try:
        s = request.app.state

        # タイムスタンプをデータローダーの time_axis から取得
        timestamps = [str(t) for t in s.data_loader.time_axis]

        result = s.ai_interpreter.interpret(
            top_features=body.top_features,
            cluster1_size=body.cluster1_size,
            cluster2_size=body.cluster2_size,
            cluster1_indices=body.cluster1_indices,
            cluster2_indices=body.cluster2_indices,
            timestamps=timestamps,
        )

        return InterpretationResponse(
            comparison_context=ComparisonContext(**result.get('comparison_context', {})),
            separation_factors=SeparationFactors(**result.get('separation_factors', {})),
            suggested_exploration=SuggestedExploration(**result.get('suggested_exploration', {})),
        )
    except Exception as e:
        logger.exception("AI解釈生成中にエラーが発生")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(request: Request):
    """ヘルスチェックエンドポイント。

    サーバーの稼働状態、データ読み込み状態、ドメイン設定を返す。
    """
    s = request.app.state
    return {
        "status": "healthy",
        "data_loaded": s.data_loader.tensor_X is not None,
        "domain": s.domain.name,
        "config": os.getenv('APP_CONFIG', 'hpc_default'),
    }
