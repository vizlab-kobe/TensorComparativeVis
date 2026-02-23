/**
 * テーマ定義モジュール
 *
 * アプリケーション全体で使用するカラーパレット、フォント、
 * Chakra UIテーマを一元管理する。
 * 各コンポーネントはこのモジュールから色定数をインポートして使用する。
 */
import { extendTheme } from '@chakra-ui/react';

// ---------------------------------------------------------------------------
// カラー定数（コンポーネント間で共有）
// ---------------------------------------------------------------------------

/** クラスター選択色（散布図、特徴量ランキング、時系列、サイドバーで使用） */
export const CLUSTER_COLORS = {
    cluster1: '#d62728',  // tab10 赤 - 赤クラスター（C1）
    cluster2: '#1f77b4',  // tab10 青 - 青クラスター（C2）
} as const;

/** デフォルトのクラス色（散布図の各クラスの点に適用、設定で上書き可能） */
export const DEFAULT_CLASS_COLORS = ['#E07B54', '#5B8BD0', '#6BAF6B'];

/** UI基本色（テキスト、枠線、背景など） */
export const UI_COLORS = {
    text: '#333',           // メインテキスト色
    textSecondary: '#666',  // セカンダリテキスト色
    textMuted: '#888',      // 控えめなテキスト色
    border: '#e0e0e0',      // 枠線色
    bgSubtle: '#fafafa',    // サブ背景色
    accent: '#555',         // アクセント色
    accentHover: '#444',    // アクセントホバー色
    grid: '#f0f0f0',        // グリッド線色
    axis: '#888',           // 軸ラベル色
} as const;

/** AI解釈セクションカードのアクセント色マッピング */
export const SECTION_COLORS: Record<string, string> = {
    'key findings': '#2d9596',
    'pattern analysis': '#27ae60',
    'statistical summary': '#9467bd',
    'caveats': '#8c564b',
};

/**
 * セクションタイトルからアクセント色を取得する
 * @param title - セクションタイトル（大文字小文字不問）
 * @returns 対応するアクセント色のHEXコード
 */
export function getSectionColor(title: string): string {
    const key = title.toLowerCase();
    return SECTION_COLORS[key] || '#555';
}

// ---------------------------------------------------------------------------
// Chakra UI テーマ定義
// ---------------------------------------------------------------------------

/**
 * アカデミック研究テーマ
 * 学術発表に適した、クリーンで洗練されたデザインを提供する。
 */
const theme = extendTheme({
    /** カラーモード設定（ライトモード固定） */
    config: {
        initialColorMode: 'light',
        useSystemColorMode: false,
    },

    /** グローバルスタイル */
    styles: {
        global: {
            body: {
                bg: '#ffffff',
                color: '#333',
            },
        },
    },

    /** フォント設定（Inter を優先使用） */
    fonts: {
        heading: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        body: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    },

    /** テーマカラー */
    colors: {
        brand: {
            50: '#FDF2EF',
            100: '#F9DDD5',
            500: '#E07B54',
            600: '#D16A43',
            700: '#C25A33',
        },
        academic: {
            coral: '#E07B54',
            blue: '#5B8BD0',
            green: '#6BAF6B',
            text: '#333',
            textSecondary: '#666',
            textMuted: '#888',
            border: '#e0e0e0',
            bgSubtle: '#fafafa',
        },
    },

    /** コンポーネント固有のスタイル上書き */
    components: {
        Button: {
            defaultProps: {
                colorScheme: 'brand',
            },
        },
        Tabs: {
            variants: {
                line: {
                    tab: {
                        _selected: {
                            color: 'academic.coral',
                            borderColor: 'academic.coral',
                        },
                    },
                },
            },
        },
    },
});

export default theme;
