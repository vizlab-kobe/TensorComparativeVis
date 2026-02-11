"""
API Routes - All endpoint handlers extracted from main_new.py.

Each handler accesses shared dependencies via `request.app.state`.
"""

from fastapi import APIRouter, HTTPException, Request
import numpy as np
import os
import logging

from app.core import (
    unfold_and_scale_tensor,
    apply_pacmap_reduction,
    analyze_tensor_contribution,
    get_top_important_factors,
    evaluate_statistical_significance,
)
from models import (
    ComputeEmbeddingRequest, ComputeEmbeddingResponse,
    ClusterAnalysisRequest, ClusterAnalysisResponse,
    InterpretationRequest, InterpretationResponse,
    CompareRequest, CompareResponse,
    ConfigResponse, FeatureImportance, StatisticalResult,
)
from tulca import TULCA

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# ── Helper ───────────────────────────────────────────────────────────────────

def _get_tulca_model(state) -> TULCA:
    """Get or initialize TULCA model from app.state."""
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


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/config", response_model=ConfigResponse)
async def get_config(request: Request):
    """Get application configuration."""
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
    """Get spatial coordinates for geo_map visualization."""
    coords = request.app.state.domain.get_coordinates()
    if not coords:
        return {"coordinates": [], "available": False}
    return {"coordinates": coords, "available": True}


@router.post("/compute-embedding", response_model=ComputeEmbeddingResponse)
async def compute_embedding(body: ComputeEmbeddingRequest, request: Request):
    """Compute TULCA + PaCMAP embedding with given weights."""
    try:
        s = request.app.state
        n_classes = s.data_loader.n_classes
        w_tgs = [body.class_weights[i].w_tg for i in range(n_classes)]
        w_bgs = [body.class_weights[i].w_bg for i in range(n_classes)]
        w_bws = [body.class_weights[i].w_bw for i in range(n_classes)]

        model = _get_tulca_model(s)
        model.fit_with_new_weights(w_tgs, w_bgs, w_bws)

        low_dim_tensor = model.transform(s.data_loader.tensor_X)
        projection_matrices = model.get_projection_matrices()

        Ms = np.asarray(projection_matrices[0])
        Mv = np.asarray(projection_matrices[1])

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
        logger.exception("Error computing embedding")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-clusters", response_model=ClusterAnalysisResponse)
async def analyze_clusters(body: ClusterAnalysisRequest, request: Request):
    """Analyze differences between two clusters."""
    try:
        s = request.app.state
        scaled_data = np.array(body.scaled_data)
        Ms = np.array(body.Ms)
        Mv = np.array(body.Mv)

        T, S, V = s.data_loader.shape

        contribution_matrix = analyze_tensor_contribution(
            body.cluster1_indices, body.cluster2_indices,
            scaled_data, Ms, Mv, S, V,
        )

        top_factors = get_top_important_factors(contribution_matrix, s.domain, top_k=10)

        cluster1_array = np.array(body.cluster1_indices)
        cluster2_array = np.array(body.cluster2_indices)
        cluster1_time = [str(t) for t in s.data_loader.time_axis[cluster1_array]]
        cluster2_time = [str(t) for t in s.data_loader.time_axis[cluster2_array]]

        features = []
        for factor in top_factors:
            rack_idx = factor['rack_idx']
            var_idx = factor['var_idx']

            cluster1_data = s.data_loader.original_data[cluster1_array, rack_idx, var_idx].tolist()
            cluster2_data = s.data_loader.original_data[cluster2_array, rack_idx, var_idx].tolist()

            stat_result = evaluate_statistical_significance(
                body.cluster1_indices, body.cluster2_indices,
                rack_idx, var_idx,
                s.data_loader.original_data, s.domain,
            )

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
        logger.exception("Error analyzing clusters")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interpret-clusters", response_model=InterpretationResponse)
async def interpret_clusters(body: InterpretationRequest, request: Request):
    """Generate AI interpretation of cluster differences."""
    try:
        result = request.app.state.ai_interpreter.interpret(
            body.top_features,
            body.cluster1_size,
            body.cluster2_size,
        )
        return InterpretationResponse(sections=result.get('sections', []))
    except Exception as e:
        logger.exception("Error interpreting clusters")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-analyses", response_model=CompareResponse)
async def compare_analyses(body: CompareRequest, request: Request):
    """Compare two saved analyses using AI."""
    try:
        analysis_a = {
            'cluster1_size': body.analysis_a.cluster1_size,
            'cluster2_size': body.analysis_a.cluster2_size,
            'summary': {
                'significant_count': body.analysis_a.significant_count,
                'top_variables': body.analysis_a.top_variables,
                'top_racks': body.analysis_a.top_racks,
            },
            'top_features': body.analysis_a.top_features,
        }
        analysis_b = {
            'cluster1_size': body.analysis_b.cluster1_size,
            'cluster2_size': body.analysis_b.cluster2_size,
            'summary': {
                'significant_count': body.analysis_b.significant_count,
                'top_variables': body.analysis_b.top_variables,
                'top_racks': body.analysis_b.top_racks,
            },
            'top_features': body.analysis_b.top_features,
        }
        result = request.app.state.ai_interpreter.compare_analyses(analysis_a, analysis_b)
        return CompareResponse(sections=result.get('sections', []))
    except Exception as e:
        logger.exception("Error comparing analyses")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(request: Request):
    """Health check endpoint."""
    s = request.app.state
    return {
        "status": "healthy",
        "data_loaded": s.data_loader.tensor_X is not None,
        "domain": s.domain.name,
        "config": os.getenv('APP_CONFIG', 'hpc_default'),
    }
