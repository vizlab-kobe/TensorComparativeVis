/**
 * API client for HPC Dashboard backend
 */
import axios from 'axios';
import type {
    ClassWeight,
    ComputeEmbeddingResponse,
    ClusterAnalysisResponse,
    ConfigResponse,
    FeatureImportance,
    InterpretationResponse,
} from '../types';

const API_BASE = 'http://localhost:8000/api';

const api = axios.create({
    baseURL: API_BASE,
    headers: {
        'Content-Type': 'application/json',
    },
});

export async function getConfig(): Promise<ConfigResponse> {
    const response = await api.get<ConfigResponse>('/config');
    return response.data;
}

export interface Coordinate {
    index: number;
    lat: number;
    lon: number;
    name: string;
}

export async function getCoordinates(): Promise<{ coordinates: Coordinate[]; available: boolean }> {
    const response = await api.get<{ coordinates: Coordinate[]; available: boolean }>('/coordinates');
    return response.data;
}

export async function computeEmbedding(
    classWeights: ClassWeight[]
): Promise<ComputeEmbeddingResponse> {
    const response = await api.post<ComputeEmbeddingResponse>('/compute-embedding', {
        class_weights: classWeights,
    });
    return response.data;
}

export async function analyzeClusters(
    cluster1Indices: number[],
    cluster2Indices: number[],
    scaledData: number[][],
    Ms: number[][],
    Mv: number[][]
): Promise<ClusterAnalysisResponse> {
    const response = await api.post<ClusterAnalysisResponse>('/analyze-clusters', {
        cluster1_indices: cluster1Indices,
        cluster2_indices: cluster2Indices,
        scaled_data: scaledData,
        Ms: Ms,
        Mv: Mv,
    });
    return response.data;
}

export async function interpretClusters(
    topFeatures: FeatureImportance[],
    cluster1Size: number,
    cluster2Size: number
): Promise<InterpretationResponse> {
    const response = await api.post<InterpretationResponse>('/interpret-clusters', {
        top_features: topFeatures.map((f) => ({
            rack: f.rack,
            variable: f.variable,
            importance: f.importance,
            score: f.score,
            mean_diff: f.mean_diff,
            statistical_result: f.statistical_result,
        })),
        cluster1_size: cluster1Size,
        cluster2_size: cluster2Size,
    });
    return response.data;
}

export interface AnalysisSummaryInput {
    cluster1_size: number;
    cluster2_size: number;
    significant_count: number;
    top_variables: string[];
    top_racks: string[];
    top_features: Record<string, unknown>[];
}

export async function compareAnalyses(
    analysisA: AnalysisSummaryInput,
    analysisB: AnalysisSummaryInput
): Promise<InterpretationResponse> {
    const response = await api.post<InterpretationResponse>('/compare-analyses', {
        analysis_a: analysisA,
        analysis_b: analysisB,
    });
    return response.data;
}
