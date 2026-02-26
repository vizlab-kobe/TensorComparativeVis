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
 *   - UI状態:          タブ選択、フォーカス中の特徴量等
 */
import { create } from 'zustand';
import type {
    ClassWeight,
    EmbeddingPoint,
    ClusterSelection,
    FeatureImportance,
    ConfigResponse,
    InterpretationResult,
} from '../types';

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
    interpretation: InterpretationResult | null;
    setInterpretation: (interpretation: InterpretationResult) => void;

    // ── UI状態 ──
    currentFeatureIndex: number;
    setCurrentFeatureIndex: (index: number) => void;
    activeTab: 'ranking' | 'heatmap';
    setActiveTab: (tab: 'ranking' | 'heatmap') => void;
    selectedVariable: number;
    setSelectedVariable: (variable: number) => void;

    // ── ローディング状態 ──
    isLoading: boolean;
    setIsLoading: (loading: boolean) => void;
}

// ── ストア実装 ───────────────────────────────────────────────────────────────

export const useDashboardStore = create<DashboardState>((set) => ({
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
     * 埋め込みデータを設定する。
     * TULCA 重みが変更されると射影が変わるため、
     * 分析結果をリセットする。
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

    // ── UI状態 ──
    currentFeatureIndex: 0,
    setCurrentFeatureIndex: (currentFeatureIndex) => set({ currentFeatureIndex }),
    activeTab: 'ranking',
    setActiveTab: (activeTab) => set({ activeTab }),
    selectedVariable: 0,
    setSelectedVariable: (selectedVariable) => set({ selectedVariable }),

    // ── ローディング状態 ──
    isLoading: false,
    setIsLoading: (isLoading) => set({ isLoading }),
}));
