/**
 * バックエンドAPIクライアントモジュール
 *
 * FastAPI バックエンドとの全ての通信を一元管理する。
 * 各関数は対応する API エンドポイントを呼び出し、
 * 型付きレスポンスを返す。
 *
 * エンドポイント対応表:
 *   getConfig()           → GET  /api/config
 *   getCoordinates()      → GET  /api/coordinates
 *   computeEmbedding()    → POST /api/compute-embedding
 *   analyzeClusters()     → POST /api/analyze-clusters
 *   interpretClusters()   → POST /api/interpret-clusters
 *   compareAnalyses()     → POST /api/compare-analyses
 */
import axios from 'axios';
import type { ConfigResponse, FeatureImportance, InterpretationSection } from '../types';

// バックエンドのベースURL（開発環境: localhost:8000）
const API_BASE = 'http://localhost:8000/api';

/** ステーション座標データの型（GeoMapVis で使用） */
export interface Coordinate {
    /** ステーションの空間インデックス */
    index: number;
    /** ステーション名 */
    name: string;
    /** 緯度 */
    lat: number;
    /** 経度 */
    lon: number;
}

/**
 * アプリケーション設定を取得する。
 *
 * フロントエンド初期化時に呼ばれ、変数名・クラス数・色設定・
 * 可視化タイプ等を取得する。
 */
export async function getConfig(): Promise<ConfigResponse> {
    const res = await axios.get(`${API_BASE}/config`);
    return res.data;
}

/**
 * 地理座標データを取得する（geo_map 可視化用）。
 *
 * @returns 座標データの配列と利用可否フラグ
 */
export async function getCoordinates() {
    const res = await axios.get(`${API_BASE}/coordinates`);
    return res.data;
}

/**
 * TULCA + PaCMAP 埋め込みを計算する。
 *
 * クラス重みを送信し、TULCA再最適化 → テンソル射影 →
 * PaCMAP 2次元埋め込みの結果を受け取る。
 *
 * @param classWeights - 各クラスのTULCA重み設定
 * @returns 埋め込み座標・スケーリングデータ・射影行列・クラスラベル
 */
export async function computeEmbedding(classWeights: Array<{ w_tg: number; w_bw: number; w_bg: number }>) {
    const res = await axios.post(`${API_BASE}/compute-embedding`, {
        class_weights: classWeights,
    });
    return res.data as {
        embedding: number[][];
        scaled_data: number[][];
        Ms: number[][];
        Mv: number[][];
        labels: number[];
    };
}

/**
 * 2つのクラスター間の差異を分析する。
 *
 * ランダムフォレストによる特徴量重要度算出、射影行列による
 * 寄与度行列計算、統計的有意性評価を実施する。
 *
 * @param cluster1 - 赤クラスター(C1)のサンプルインデックス
 * @param cluster2 - 青クラスター(C2)のサンプルインデックス
 * @param scaledData - 標準化済み展開テンソル
 * @param Ms - 空間モード射影行列
 * @param Mv - 変数モード射影行列
 * @returns 上位特徴量リストと寄与度行列
 */
export async function analyzeClusters(
    cluster1: number[],
    cluster2: number[],
    scaledData: number[][],
    Ms: number[][],
    Mv: number[][],
) {
    const res = await axios.post(`${API_BASE}/analyze-clusters`, {
        cluster1_indices: cluster1,
        cluster2_indices: cluster2,
        scaled_data: scaledData,
        Ms,
        Mv,
    });
    return res.data as {
        top_features: FeatureImportance[];
        contribution_matrix: number[][];
    };
}

/**
 * クラスター差異のAI解釈を生成する。
 *
 * 特徴量データをGemini APIに送信し、構造化された
 * 自然言語解釈を受け取る。
 *
 * @param topFeatures - 上位特徴量リスト
 * @param cluster1Size - 赤クラスターのサンプル数
 * @param cluster2Size - 青クラスターのサンプル数
 * @returns 解釈セクションの配列
 */
export async function interpretClusters(
    topFeatures: FeatureImportance[],
    cluster1Size: number,
    cluster2Size: number,
) {
    const res = await axios.post(`${API_BASE}/interpret-clusters`, {
        top_features: topFeatures,
        cluster1_size: cluster1Size,
        cluster2_size: cluster2Size,
    });
    return res.data as {
        sections: InterpretationSection[];
    };
}

/**
 * 2つの保存済み分析結果をAIで比較する。
 *
 * 各分析のクラスターサイズ、上位特徴量、統計情報を比較し、
 * 共通点と差異に関する構造化された解釈を生成する。
 *
 * @param analysisA - 1つ目の分析結果の要約
 * @param analysisB - 2つ目の分析結果の要約
 * @returns 比較セクションの配列
 */
export async function compareAnalyses(
    analysisA: {
        cluster1_size: number;
        cluster2_size: number;
        significant_count: number;
        top_variables: string[];
        top_racks: string[];
        top_features: Array<Record<string, unknown>>;
    },
    analysisB: {
        cluster1_size: number;
        cluster2_size: number;
        significant_count: number;
        top_variables: string[];
        top_racks: string[];
        top_features: Array<Record<string, unknown>>;
    },
) {
    const res = await axios.post(`${API_BASE}/compare-analyses`, {
        analysis_a: analysisA,
        analysis_b: analysisB,
    });
    return res.data as {
        sections: InterpretationSection[];
    };
}
