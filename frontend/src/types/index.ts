/**
 * TypeScript type definitions for Tensor Comparative Visualization
 */

// API Response Types
export interface ClassWeight {
    w_tg: number;
    w_bw: number;
    w_bg: number;
}

export interface StatisticalResult {
    rack: string;
    variable: string;
    direction: string;
    mean_diff: number;
    p_value: number;
    cohen_d: number;
    significance: string;
    effect_size: string;
}

export interface FeatureImportance {
    rank: number;
    rack: string;
    variable: string;
    score: number;
    importance: number;
    cluster1_data: number[];
    cluster2_data: number[];
    cluster1_time: string[];
    cluster2_time: string[];
    mean_diff: number;
    statistical_result: StatisticalResult;
}

export interface ComputeEmbeddingResponse {
    embedding: number[][];
    scaled_data: number[][];
    Ms: number[][];
    Mv: number[][];
    labels: number[];
}

export interface ClusterAnalysisResponse {
    top_features: FeatureImportance[];
    contribution_matrix: number[][];
}

export interface ConfigResponse {
    variables: string[];
    n_classes: number;
    grid_shape: number[];
    colors: {
        class_colors: string[];
        cluster1: string;
        cluster2: string;
    };
    visualization_type: 'grid' | 'geo_map';
    class_labels: string[];
}

// Internal State Types
export interface EmbeddingPoint {
    x: number;
    y: number;
    index: number;
    label: number;
}

export interface ClusterSelection {
    cluster1: number[] | null;
    cluster2: number[] | null;
}

// Chart Data Types
export interface TimeSeriesDataPoint {
    time: Date;
    value: number;
    cluster: 'cluster1' | 'cluster2';
}

export interface HeatmapCell {
    row: number;
    col: number;
    value: number;
    rack: string;
}

// AI Interpretation Types
export interface InterpretationSection {
    title: string;
    text: string;
    highlights: string[];
}

export interface InterpretationResponse {
    sections: InterpretationSection[];
}
