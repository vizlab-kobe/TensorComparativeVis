from pydantic import BaseModel
from typing import List, Dict, Any


class ClassWeight(BaseModel):
    """Weight configuration for a single class."""
    w_tg: float = 0.0
    w_bw: float = 1.0
    w_bg: float = 1.0


class ComputeEmbeddingRequest(BaseModel):
    """Request to compute embedding with class weights."""
    class_weights: List[ClassWeight]


class ComputeEmbeddingResponse(BaseModel):
    """Response containing computed embedding data."""
    embedding: List[List[float]]  # T x 2 PaCMAP embedding
    scaled_data: List[List[float]]  # T x (S*V) scaled data
    Ms: List[List[float]]  # Projection matrix for spatial mode
    Mv: List[List[float]]  # Projection matrix for variable mode
    labels: List[int]  # Class labels for each sample


class ClusterAnalysisRequest(BaseModel):
    """Request to analyze differences between two clusters."""
    cluster1_indices: List[int]
    cluster2_indices: List[int]
    scaled_data: List[List[float]]
    Ms: List[List[float]]
    Mv: List[List[float]]


class StatisticalResult(BaseModel):
    """Statistical analysis result for a feature."""
    rack: str
    variable: str
    direction: str
    mean_diff: float
    p_value: float
    cohen_d: float
    significance: str
    effect_size: str


class FeatureImportance(BaseModel):
    """Feature importance with statistical analysis."""
    rank: int
    rack: str
    variable: str
    score: float
    importance: float
    cluster1_data: List[float]
    cluster2_data: List[float]
    cluster1_time: List[str]
    cluster2_time: List[str]
    mean_diff: float
    statistical_result: StatisticalResult


class ClusterAnalysisResponse(BaseModel):
    """Response containing cluster analysis results."""
    top_features: List[FeatureImportance]
    contribution_matrix: List[List[float]]  # S x V contribution matrix


class InterpretationRequest(BaseModel):
    """Request for AI interpretation."""
    top_features: List[Dict[str, Any]]
    cluster1_size: int
    cluster2_size: int


class InterpretationSection(BaseModel):
    """A section of the AI interpretation."""
    title: str
    text: str
    highlights: List[str] = []


class InterpretationResponse(BaseModel):
    """Response containing structured AI interpretation."""
    sections: List[InterpretationSection]


class AnalysisSummary(BaseModel):
    """Summary of a saved analysis for comparison."""
    cluster1_size: int
    cluster2_size: int
    significant_count: int
    top_variables: List[str]
    top_racks: List[str]
    top_features: List[Dict[str, Any]]


class CompareRequest(BaseModel):
    """Request to compare two analyses."""
    analysis_a: AnalysisSummary
    analysis_b: AnalysisSummary


class CompareResponse(BaseModel):
    """Response containing comparison result."""
    sections: List[InterpretationSection]


class ConfigResponse(BaseModel):
    """Application configuration."""
    variables: List[str]
    n_classes: int
    grid_shape: List[int]
    colors: Dict[str, Any]  # Contains both strings and lists
