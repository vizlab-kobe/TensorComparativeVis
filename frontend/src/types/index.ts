/**
 * TypeScript 型定義モジュール
 *
 * バックエンドAPIのレスポンス型、フロントエンド内部状態の型、
 * チャートデータの型を一元管理する。
 * バックエンドの Pydantic モデル (models.py) と対応関係がある。
 */

// ── API レスポンス型 ─────────────────────────────────────────────────────────

/** TULCA クラス重みパラメータ（Sidebar スライダーの値） */
export interface ClassWeight {
    /** ターゲットクラス内重み */
    w_tg: number;
    /** クラス間重み */
    w_bw: number;
    /** 背景重み */
    w_bg: number;
}

/** アプリケーション設定レスポンス（GET /api/config） */
export interface ConfigResponse {
    /** 変数名リスト（例: ["AirIn", "AirOut", "CPU", "Water"]） */
    variables: string[];
    /** データセット内のクラス数 */
    n_classes: number;
    /** 空間グリッドの形状 [行数, 列数] */
    grid_shape: number[];
    /** UI表示色の設定 */
    colors: {
        class_colors: string[];
        cluster1: string;
        cluster2: string;
    };
    /** 可視化タイプ: "grid"（ヒートマップ）or "geo_map"（地図） */
    visualization_type?: string;
    /** ドメイン固有のクラスラベル（例: ["FY2014", "FY2015", "FY2016"]） */
    class_labels?: string[];
}

/** 統計的有意性の検定結果 */
export interface StatisticalResult {
    rack: string;            // 空間ラベル
    variable: string;        // 変数名
    direction: string;       // 差の方向（"Higher in Cluster 1" 等）
    mean_diff: number;       // 平均値の差の絶対値
    p_value: number;         // p値（Welch t検定）
    cohen_d: number;         // Cohen's d 効果量
    significance: string;    // 有意性判定（探索的表現）
    effect_size: string;     // 効果量の解釈
    // Mann-Whitney U 検定結果
    mwu_p_value?: number;
    mwu_statistic?: number;
    mwu_significance?: string;
    // FDR 補正結果（Benjamini-Hochberg 法）
    adjusted_p_value?: number;
    fdr_significance?: string;
    adjusted_mwu_p_value?: number;
    fdr_mwu_significance?: string;
}

/** 特徴量の重要度（統計的分析付き） */
export interface FeatureImportance {
    rank: number;                       // 重要度順位（1始まり）
    rack: string;                       // 空間ラベル
    variable: string;                   // 変数名
    score: number;                      // 寄与度スコア（標準化済み）
    importance: number;                 // 重要度（score と同値）
    cluster1_data: number[];            // 赤クラスター(C1)の元データ値
    cluster2_data: number[];            // 青クラスター(C2)の元データ値
    cluster1_time: string[];            // C1 の時間軸ラベル
    cluster2_time: string[];            // C2 の時間軸ラベル
    mean_diff: number;                  // 平均値の差
    statistical_result: StatisticalResult;  // 統計検定結果
}

// ── AI解釈レスポンス型 ────────────────────────────────────────────────────────

/** AI解釈の比較コンテキスト */
export interface ComparisonContext {
    cluster1_range: string;
    cluster2_range: string;
    cluster1_size: number;
    cluster2_size: number;
    text: string;
}

/** AI解釈の分離要因 */
export interface SeparationFactors {
    text: string;
}

/** AI解釈の探索提案 */
export interface SuggestedExploration {
    text: string;
}

/** AI解釈レスポンス（3フィールド構造） */
export interface InterpretationResult {
    comparison_context: ComparisonContext;
    separation_factors: SeparationFactors;
    suggested_exploration: SuggestedExploration;
}

// ── フロントエンド内部状態型 ──────────────────────────────────────────────────

/** クラスター選択状態（散布図のブラシ選択で更新） */
export interface ClusterSelection {
    cluster1: number[] | null;  // 赤クラスターのサンプルインデックス
    cluster2: number[] | null;  // 青クラスターのサンプルインデックス
}

// ── チャートデータ型 ──────────────────────────────────────────────────────────

/** 散布図の1データポイント */
export interface EmbeddingPoint {
    x: number;      // PaCMAP 2次元埋め込みの X 座標
    y: number;      // PaCMAP 2次元埋め込みの Y 座標
    label: number;  // クラスラベル (0, 1, 2, ...)
    index: number;  // 元のサンプルインデックス
}

/** 時系列プロットの1データポイント */
export interface TimeSeriesDataPoint {
    time: string;   // 時間軸ラベル
    value: number;  // 計測値
}

/** ヒートマップの1セル */
export interface HeatmapCell {
    row: number;      // 行インデックス
    col: number;      // 列インデックス
    value: number;    // 寄与度スコア
    label: string;    // 空間ラベル
    variable: string; // 変数名
}
