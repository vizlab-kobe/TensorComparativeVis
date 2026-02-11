/**
 * Zustand store for application state management
 */
import { create } from 'zustand';
import type {
    ClassWeight,
    EmbeddingPoint,
    ClusterSelection,
    FeatureImportance,
    ConfigResponse,
    InterpretationSection,
} from '../types';

// Saved analysis for history
export interface SavedAnalysis {
    id: string;
    timestamp: Date;
    cluster1_indices: number[];
    cluster2_indices: number[];
    cluster1_size: number;
    cluster2_size: number;
    top_features: FeatureImportance[];
    interpretation: InterpretationSection[];
    summary: {
        significant_count: number;
        top_variables: string[];
        top_racks: string[];
    };
}

interface DashboardState {
    // Config
    config: ConfigResponse | null;
    setConfig: (config: ConfigResponse) => void;

    // Class weights
    classWeights: ClassWeight[];
    selectedClass: number;
    setSelectedClass: (classIndex: number) => void;
    updateWeight: (classIndex: number, weight: Partial<ClassWeight>) => void;
    initializeWeights: (nClasses: number) => void;

    // Embedding data
    embeddingData: EmbeddingPoint[];
    scaledData: number[][] | null;
    Ms: number[][] | null;
    Mv: number[][] | null;
    setEmbeddingData: (
        embedding: number[][],
        labels: number[],
        scaledData: number[][],
        Ms: number[][],
        Mv: number[][]
    ) => void;

    // Cluster selection
    clusters: ClusterSelection;
    selectCluster1: (indices: number[]) => void;
    selectCluster2: (indices: number[]) => void;
    clearCluster1: () => void;
    clearCluster2: () => void;
    resetClusters: () => void;

    // Analysis results
    topFeatures: FeatureImportance[] | null;
    contributionMatrix: number[][] | null;
    setAnalysisResults: (features: FeatureImportance[], matrix: number[][]) => void;

    // AI Interpretation (structured)
    interpretation: InterpretationSection[] | null;
    setInterpretation: (sections: InterpretationSection[]) => void;

    // Analysis History
    analysisHistory: SavedAnalysis[];
    saveCurrentAnalysis: () => void;
    clearHistory: () => void;
    selectedHistoryIds: string[];
    toggleHistorySelection: (id: string) => void;
    clearHistorySelection: () => void;

    // UI State
    currentFeatureIndex: number;
    setCurrentFeatureIndex: (index: number) => void;
    activeTab: 'ranking' | 'heatmap';
    setActiveTab: (tab: 'ranking' | 'heatmap') => void;
    selectedVariable: number;
    setSelectedVariable: (variable: number) => void;
    interpretationTab: 'summary' | 'history' | 'compare';
    setInterpretationTab: (tab: 'summary' | 'history' | 'compare') => void;

    // Loading states
    isLoading: boolean;
    setIsLoading: (loading: boolean) => void;
}

const MAX_HISTORY_SIZE = 10;

export const useDashboardStore = create<DashboardState>((set, get) => ({
    // Config
    config: null,
    setConfig: (config) => set({ config }),

    // Class weights
    classWeights: [],
    selectedClass: 0,
    setSelectedClass: (selectedClass) => set({ selectedClass }),
    updateWeight: (classIndex, weight) =>
        set((state) => ({
            classWeights: state.classWeights.map((w, i) =>
                i === classIndex ? { ...w, ...weight } : w
            ),
        })),
    initializeWeights: (nClasses) =>
        set({
            classWeights: Array.from({ length: nClasses }, () => ({
                w_tg: 0,
                w_bw: 1.0,
                w_bg: 1.0,
            })),
        }),

    // Embedding data
    embeddingData: [],
    scaledData: null,
    Ms: null,
    Mv: null,
    setEmbeddingData: (embedding, labels, scaledData, Ms, Mv) =>
        set({
            embeddingData: embedding.map((point, index) => ({
                x: point[0],
                y: point[1],
                index,
                label: labels[index],
            })),
            scaledData,
            Ms,
            Mv,
            // Clear history when embedding changes (new TULCA weights)
            analysisHistory: [],
            selectedHistoryIds: [],
        }),

    // Cluster selection
    clusters: { cluster1: null, cluster2: null },
    // Intentionally resets cluster2: selecting a new C1 starts a fresh comparison
    selectCluster1: (indices) =>
        set({ clusters: { cluster1: indices, cluster2: null } }),
    selectCluster2: (indices) =>
        set((state) => ({
            clusters: { ...state.clusters, cluster2: indices },
        })),
    clearCluster1: () =>
        set((state) => ({
            clusters: { ...state.clusters, cluster1: null },
            topFeatures: null,
            contributionMatrix: null,
            interpretation: null,
        })),
    clearCluster2: () =>
        set((state) => ({
            clusters: { ...state.clusters, cluster2: null },
            topFeatures: null,
            contributionMatrix: null,
            interpretation: null,
        })),
    resetClusters: () =>
        set({
            clusters: { cluster1: null, cluster2: null },
            topFeatures: null,
            contributionMatrix: null,
            interpretation: null,
        }),

    // Analysis results
    topFeatures: null,
    contributionMatrix: null,
    setAnalysisResults: (features, matrix) =>
        set({ topFeatures: features, contributionMatrix: matrix }),

    // AI Interpretation (structured)
    interpretation: null,
    setInterpretation: (interpretation) => set({ interpretation }),

    // Analysis History
    analysisHistory: [],
    saveCurrentAnalysis: () => {
        const state = get();
        if (!state.clusters.cluster1 || !state.clusters.cluster2 || !state.topFeatures || !state.interpretation) {
            return;
        }

        // Calculate summary statistics
        const significantFeatures = state.topFeatures.filter(
            f => f.statistical_result.p_value < 0.05
        );
        const variableCounts = state.topFeatures.reduce((acc, f) => {
            acc[f.variable] = (acc[f.variable] || 0) + 1;
            return acc;
        }, {} as Record<string, number>);
        const sortedVars = Object.entries(variableCounts)
            .sort((a, b) => b[1] - a[1])
            .map(([v]) => v);
        const racks = state.topFeatures.slice(0, 5).map(f => f.rack);

        const newAnalysis: SavedAnalysis = {
            id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            timestamp: new Date(),
            cluster1_indices: state.clusters.cluster1,
            cluster2_indices: state.clusters.cluster2,
            cluster1_size: state.clusters.cluster1.length,
            cluster2_size: state.clusters.cluster2.length,
            top_features: state.topFeatures,
            interpretation: state.interpretation,
            summary: {
                significant_count: significantFeatures.length,
                top_variables: sortedVars.slice(0, 3),
                top_racks: racks,
            },
        };

        set((state) => {
            const newHistory = [newAnalysis, ...state.analysisHistory];
            // Keep only the most recent MAX_HISTORY_SIZE entries
            return {
                analysisHistory: newHistory.slice(0, MAX_HISTORY_SIZE),
            };
        });
    },
    clearHistory: () => set({ analysisHistory: [], selectedHistoryIds: [] }),
    selectedHistoryIds: [],
    toggleHistorySelection: (id: string) => {
        set((state) => {
            const isSelected = state.selectedHistoryIds.includes(id);
            if (isSelected) {
                return {
                    selectedHistoryIds: state.selectedHistoryIds.filter(hid => hid !== id),
                };
            } else {
                // Max 2 selections for comparison
                const newSelection = [...state.selectedHistoryIds, id].slice(-2);
                return { selectedHistoryIds: newSelection };
            }
        });
    },
    clearHistorySelection: () => set({ selectedHistoryIds: [] }),

    // UI State
    currentFeatureIndex: 0,
    setCurrentFeatureIndex: (currentFeatureIndex) => set({ currentFeatureIndex }),
    activeTab: 'ranking',
    setActiveTab: (activeTab) => set({ activeTab }),
    selectedVariable: 0,
    setSelectedVariable: (selectedVariable) => set({ selectedVariable }),
    interpretationTab: 'summary',
    setInterpretationTab: (interpretationTab) => set({ interpretationTab }),

    // Loading states
    isLoading: false,
    setIsLoading: (isLoading) => set({ isLoading }),
}));
