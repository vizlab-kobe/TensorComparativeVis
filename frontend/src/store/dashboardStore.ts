/**
 * ダッシュボード状態管理ストア（Zustand）
 *
 * アプリケーション全体のグローバル状態を一元管理する。
 * 各コンポーネントは useDashboardStore() フックで必要な
 * スライスだけを取得し、状態変更時に自動再レンダリングされる。
 *
 * 状態構成:
 *   - config:         バックエンドから取得したアプリ設定
 *   - classWeights:   TULCA クラス重み（Sidebar スライダー）
 *   - embeddingData:  PaCMAP 2次元座標（ScatterPlot）
 *   - clusters:       ブラシ選択されたクラスターインデックス
 *   - topFeatures:    ランダムフォレスト特徴量ランキング
 *   - interpretation: AI解釈結果
 *   - analysisHistory: 保存済み分析の履歴
 *   - UI状態:          タブ選択、フォーカス中の特徴量等
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

// ── 保存済み分析の型定義 ─────────────────────────────────────────────────────

/** 保存済み分析結果（履歴タブ・比較機能で使用） */
export interface SavedAnalysis {
    /** 一意の識別子（タイムスタンプ + ランダム文字列） */
    id: string;
    /** 保存日時 */
    timestamp: Date;
    /** 赤クラスターのサンプルインデックス */
    cluster1_indices: number[];
    /** 青クラスターのサンプルインデックス */
    cluster2_indices: number[];
    /** 赤クラスターのサンプル数 */
    cluster1_size: number;
    /** 青クラスターのサンプル数 */
    cluster2_size: number;
    /** 上位特徴量リスト */
    top_features: FeatureImportance[];
    /** AI解釈セクション */
    interpretation: InterpretationSection[];
    /** 集計サマリー（比較プロンプト用） */
    summary: {
        significant_count: number;
        top_variables: string[];
        top_racks: string[];
    };
}

// ── ストア状態の型定義 ───────────────────────────────────────────────────────

interface DashboardState {
    // ── 設定 ──
    config: ConfigResponse | null;
    setConfig: (config: ConfigResponse) => void;

    // ── TULCA クラス重み ──
    classWeights: ClassWeight[];
    selectedClass: number;
    setSelectedClass: (classIndex: number) => void;
    updateWeight: (classIndex: number, weight: Partial<ClassWeight>) => void;
    initializeWeights: (nClasses: number) => void;

    // ── 埋め込みデータ ──
    embeddingData: EmbeddingPoint[];
    scaledData: number[][] | null;
    Ms: number[][] | null;  // 空間モード射影行列
    Mv: number[][] | null;  // 変数モード射影行列
    setEmbeddingData: (
        embedding: number[][],
        labels: number[],
        scaledData: number[][],
        Ms: number[][],
        Mv: number[][],
    ) => void;

    // ── クラスター選択 ──
    clusters: ClusterSelection;
    selectCluster1: (indices: number[]) => void;
    selectCluster2: (indices: number[]) => void;
    clearCluster1: () => void;
    clearCluster2: () => void;
    resetClusters: () => void;

    // ── 分析結果 ──
    topFeatures: FeatureImportance[] | null;
    contributionMatrix: number[][] | null;
    setAnalysisResults: (features: FeatureImportance[], matrix: number[][]) => void;

    // ── AI解釈 ──
    interpretation: InterpretationSection[] | null;
    setInterpretation: (sections: InterpretationSection[]) => void;

    // ── 分析履歴 ──
    analysisHistory: SavedAnalysis[];
    saveCurrentAnalysis: () => void;
    clearHistory: () => void;
    selectedHistoryIds: string[];
    toggleHistorySelection: (id: string) => void;
    clearHistorySelection: () => void;

    // ── UI状態 ──
    currentFeatureIndex: number;
    setCurrentFeatureIndex: (index: number) => void;
    activeTab: 'ranking' | 'heatmap';
    setActiveTab: (tab: 'ranking' | 'heatmap') => void;
    selectedVariable: number;
    setSelectedVariable: (variable: number) => void;
    interpretationTab: 'summary' | 'history' | 'compare';
    setInterpretationTab: (tab: 'summary' | 'history' | 'compare') => void;

    // ── ローディング状態 ──
    isLoading: boolean;
    setIsLoading: (loading: boolean) => void;
}

// ── 定数 ─────────────────────────────────────────────────────────────────────

/** 分析履歴の最大保存件数 */
const MAX_HISTORY_SIZE = 10;

// ── ストア実装 ───────────────────────────────────────────────────────────────

export const useDashboardStore = create<DashboardState>((set, get) => ({
    // ── 設定 ──
    config: null,
    setConfig: (config) => set({ config }),

    // ── TULCA クラス重み ──
    classWeights: [],
    selectedClass: 0,
    setSelectedClass: (selectedClass) => set({ selectedClass }),
    updateWeight: (classIndex, weight) =>
        set((state) => ({
            classWeights: state.classWeights.map((w, i) =>
                i === classIndex ? { ...w, ...weight } : w
            ),
        })),
    /** 全クラスの重みをデフォルト値で初期化する */
    initializeWeights: (nClasses) =>
        set({
            classWeights: Array.from({ length: nClasses }, () => ({
                w_tg: 0,
                w_bw: 1.0,
                w_bg: 1.0,
            })),
        }),

    // ── 埋め込みデータ ──
    embeddingData: [],
    scaledData: null,
    Ms: null,
    Mv: null,
    /**
     * 埋め込みデータを設定し、分析履歴をリセットする。
     * TULCA 重みが変更されると射影が変わるため、
     * 既存の分析履歴は無効になる。
     */
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
            // 埋め込み変更時に分析履歴をクリア
            analysisHistory: [],
            selectedHistoryIds: [],
        }),

    // ── クラスター選択 ──
    clusters: { cluster1: null, cluster2: null },
    /**
     * 赤クラスター(C1)を選択し、C2をリセットする。
     * 新しいC1選択は新しい比較の起点となるため、
     * 意図的にC2もクリアする。
     */
    selectCluster1: (indices) =>
        set({ clusters: { cluster1: indices, cluster2: null } }),
    selectCluster2: (indices) =>
        set((state) => ({
            clusters: { ...state.clusters, cluster2: indices },
        })),
    /** C1クリア時に分析結果もリセット */
    clearCluster1: () =>
        set((state) => ({
            clusters: { ...state.clusters, cluster1: null },
            topFeatures: null,
            contributionMatrix: null,
            interpretation: null,
        })),
    /** C2クリア時に分析結果もリセット */
    clearCluster2: () =>
        set((state) => ({
            clusters: { ...state.clusters, cluster2: null },
            topFeatures: null,
            contributionMatrix: null,
            interpretation: null,
        })),
    /** 全クラスター選択と分析結果をリセット */
    resetClusters: () =>
        set({
            clusters: { cluster1: null, cluster2: null },
            topFeatures: null,
            contributionMatrix: null,
            interpretation: null,
        }),

    // ── 分析結果 ──
    topFeatures: null,
    contributionMatrix: null,
    setAnalysisResults: (features, matrix) =>
        set({ topFeatures: features, contributionMatrix: matrix }),

    // ── AI解釈 ──
    interpretation: null,
    setInterpretation: (interpretation) => set({ interpretation }),

    // ── 分析履歴 ──
    analysisHistory: [],
    /**
     * 現在の分析結果を履歴に保存する。
     *
     * 有意な特徴量数、上位変数・位置の集計を行い、
     * SavedAnalysis として先頭に追加する。
     * MAX_HISTORY_SIZE を超える場合は古いものから削除。
     */
    saveCurrentAnalysis: () => {
        const state = get();
        if (
            !state.clusters.cluster1 ||
            !state.clusters.cluster2 ||
            !state.topFeatures ||
            !state.interpretation
        ) {
            return;
        }

        // サマリー統計の算出
        const significantFeatures = state.topFeatures.filter(
            (f) => f.statistical_result.p_value < 0.05
        );
        const variableCounts = state.topFeatures.reduce(
            (acc, f) => {
                acc[f.variable] = (acc[f.variable] || 0) + 1;
                return acc;
            },
            {} as Record<string, number>
        );
        const sortedVars = Object.entries(variableCounts)
            .sort((a, b) => b[1] - a[1])
            .map(([v]) => v);
        const racks = state.topFeatures.slice(0, 5).map((f) => f.rack);

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
            // 最新 MAX_HISTORY_SIZE 件のみ保持
            return {
                analysisHistory: newHistory.slice(0, MAX_HISTORY_SIZE),
            };
        });
    },
    clearHistory: () => set({ analysisHistory: [], selectedHistoryIds: [] }),

    // ── 比較用選択 ──
    selectedHistoryIds: [],
    /**
     * 履歴アイテムの選択を切り替える。
     * 比較は最大2件まで: 3件目の選択は最初の選択を押し出す。
     */
    toggleHistorySelection: (id: string) => {
        set((state) => {
            const isSelected = state.selectedHistoryIds.includes(id);
            if (isSelected) {
                return {
                    selectedHistoryIds: state.selectedHistoryIds.filter(
                        (hid) => hid !== id
                    ),
                };
            } else {
                // 最大2件の選択（FIFOで古い選択を押し出す）
                const newSelection = [...state.selectedHistoryIds, id].slice(-2);
                return { selectedHistoryIds: newSelection };
            }
        });
    },
    clearHistorySelection: () => set({ selectedHistoryIds: [] }),

    // ── UI状態 ──
    currentFeatureIndex: 0,
    setCurrentFeatureIndex: (currentFeatureIndex) => set({ currentFeatureIndex }),
    activeTab: 'ranking',
    setActiveTab: (activeTab) => set({ activeTab }),
    selectedVariable: 0,
    setSelectedVariable: (selectedVariable) => set({ selectedVariable }),
    interpretationTab: 'summary',
    setInterpretationTab: (interpretationTab) => set({ interpretationTab }),

    // ── ローディング状態 ──
    isLoading: false,
    setIsLoading: (isLoading) => set({ isLoading }),
}));
