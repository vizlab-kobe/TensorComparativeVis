"""
FastAPI Application for HPC Dashboard Backend.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from typing import List, Dict
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables FIRST before any other imports that might use them
load_dotenv()

from models import (
    ComputeEmbeddingRequest, ComputeEmbeddingResponse,
    ClusterAnalysisRequest, ClusterAnalysisResponse,
    InterpretationRequest, InterpretationResponse,
    CompareRequest, CompareResponse,
    ConfigResponse, ClassWeight, FeatureImportance, StatisticalResult
)
from data_loader import DataLoader, VARIABLES, GRID_SHAPE, index_to_label, label_to_index
from tulca import TULCA
from analysis import (
    unfold_and_scale_tensor, apply_pacmap_reduction,
    analyze_tensor_contribution, get_top_important_factors,
    evaluate_statistical_significance
)
from interpreter import GeminiInterpreter


# Initialize FastAPI app
app = FastAPI(
    title="HPC Dashboard API",
    description="Backend API for HPC Tensor Data Visualization Dashboard",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TULCA parameters
TULCA_PARAMS = {
    's_prime': 10,
    'v_prime': 3,
    'optimization_method': 'evd'
}

# Colors configuration
COLORS = {
    'class_colors': ["#F58518", "#54A24B", "#B279A2"],
    'cluster1': '#C0392B',
    'cluster2': '#2874A6'
}

# Initialize data loader and models
data_loader = DataLoader(
    data_dir=str(Path(__file__).parent.parent.parent / "dash-app" / "data" / "processed" / "HPC")
)

# Global state for TULCA model (initialized on first request)
tulca_model = None
# GeminiInterpreter will read GEMINI_API_KEY from environment (loaded by load_dotenv above)
ai_interpreter = GeminiInterpreter()


def get_tulca_model() -> TULCA:
    """Get or initialize TULCA model."""
    global tulca_model
    if tulca_model is None:
        tulca_model = TULCA(
            n_components=np.array([TULCA_PARAMS['s_prime'], TULCA_PARAMS['v_prime']]),
            optimization_method=TULCA_PARAMS['optimization_method']
        )
        tulca_model.fit(data_loader.tensor_X, data_loader.tensor_y)
    return tulca_model


@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """Get application configuration."""
    return ConfigResponse(
        variables=VARIABLES,
        n_classes=data_loader.n_classes,
        grid_shape=list(GRID_SHAPE),
        colors=COLORS
    )


@app.post("/api/compute-embedding", response_model=ComputeEmbeddingResponse)
async def compute_embedding(request: ComputeEmbeddingRequest):
    """Compute TULCA + PaCMAP embedding with given weights."""
    try:
        # Extract weights
        n_classes = data_loader.n_classes
        w_tgs = [request.class_weights[i].w_tg for i in range(n_classes)]
        w_bgs = [request.class_weights[i].w_bg for i in range(n_classes)]
        w_bws = [request.class_weights[i].w_bw for i in range(n_classes)]

        # Get TULCA model and update weights
        model = get_tulca_model()
        model.fit_with_new_weights(w_tgs, w_bgs, w_bws)
        
        # Transform data
        low_dim_tensor = model.transform(data_loader.tensor_X)
        projection_matrices = model.get_projection_matrices()
        
        # Extract projection matrices (they are in a numpy object array)
        Ms = np.asarray(projection_matrices[0])
        Mv = np.asarray(projection_matrices[1])
        
        # Scale and apply PaCMAP
        scaled_data, _ = unfold_and_scale_tensor(low_dim_tensor)
        embedding = apply_pacmap_reduction(scaled_data)

        return ComputeEmbeddingResponse(
            embedding=embedding.tolist(),
            scaled_data=scaled_data.tolist(),
            Ms=Ms.tolist(),
            Mv=Mv.tolist(),
            labels=data_loader.tensor_y.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-clusters", response_model=ClusterAnalysisResponse)
async def analyze_clusters(request: ClusterAnalysisRequest):
    """Analyze differences between two clusters."""
    try:
        scaled_data = np.array(request.scaled_data)
        Ms = np.array(request.Ms)
        Mv = np.array(request.Mv)
        
        T, S, V = data_loader.shape
        
        # Compute contribution matrix
        contribution_matrix = analyze_tensor_contribution(
            request.cluster1_indices,
            request.cluster2_indices,
            scaled_data, Ms, Mv, S, V
        )
        
        # Get top important factors
        top_factors = get_top_important_factors(contribution_matrix, top_k=10)
        
        # Get time axis data
        cluster1_array = np.array(request.cluster1_indices)
        cluster2_array = np.array(request.cluster2_indices)
        cluster1_time = [str(t) for t in data_loader.time_axis[cluster1_array]]
        cluster2_time = [str(t) for t in data_loader.time_axis[cluster2_array]]
        
        # Enrich factors with additional data
        features = []
        for factor in top_factors:
            rack_idx = factor['rack_idx']
            var_idx = factor['var_idx']
            
            # Get cluster data - use original_data (not standardized) for time series visualization
            cluster1_data = data_loader.original_data[cluster1_array, rack_idx, var_idx].tolist()
            cluster2_data = data_loader.original_data[cluster2_array, rack_idx, var_idx].tolist()
            
            # Statistical analysis
            stat_result = evaluate_statistical_significance(
                request.cluster1_indices,
                request.cluster2_indices,
                rack_idx, var_idx,
                data_loader.original_data
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
                statistical_result=StatisticalResult(**stat_result)
            ))

        return ClusterAnalysisResponse(
            top_features=features,
            contribution_matrix=contribution_matrix.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/interpret-clusters", response_model=InterpretationResponse)
async def interpret_clusters(request: InterpretationRequest):
    """Generate AI interpretation of cluster differences."""
    try:
        result = ai_interpreter.interpret(
            request.top_features,
            request.cluster1_size,
            request.cluster2_size
        )
        # Convert dict response to InterpretationResponse
        return InterpretationResponse(sections=result.get('sections', []))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "data_loaded": data_loader.tensor_X is not None}


@app.post("/api/compare-analyses", response_model=CompareResponse)
async def compare_analyses(request: CompareRequest):
    """Compare two saved analyses using AI."""
    try:
        analysis_a = {
            'cluster1_size': request.analysis_a.cluster1_size,
            'cluster2_size': request.analysis_a.cluster2_size,
            'summary': {
                'significant_count': request.analysis_a.significant_count,
                'top_variables': request.analysis_a.top_variables,
                'top_racks': request.analysis_a.top_racks,
            },
            'top_features': request.analysis_a.top_features,
        }
        analysis_b = {
            'cluster1_size': request.analysis_b.cluster1_size,
            'cluster2_size': request.analysis_b.cluster2_size,
            'summary': {
                'significant_count': request.analysis_b.significant_count,
                'top_variables': request.analysis_b.top_variables,
                'top_racks': request.analysis_b.top_racks,
            },
            'top_features': request.analysis_b.top_features,
        }
        result = ai_interpreter.compare_analyses(analysis_a, analysis_b)
        return CompareResponse(sections=result.get('sections', []))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
