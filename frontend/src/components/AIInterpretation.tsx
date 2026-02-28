/**
 * AI解釈コンポーネント
 *
 * Gemini API による自然言語分析結果を3つのセクションカードで表示する。
 *
 * セクション構成:
 *   1. Comparison Context — 比較コンテキスト（時間範囲・クラスタサイズ）
 *   2. Separation Factors — 主な分離要因
 *   3. Suggested Exploration — 推奨する探索ステップ
 *
 * テキスト内の <<rack-variable>> マーカーはクリッカブルに変換され、
 * クリック時に時系列モーダルを表示する。
 */
import React, { useState, useCallback } from 'react';
import {
    Box,
    VStack,
    HStack,
    Text,
    Spinner,
} from '@chakra-ui/react';
import { useDashboardStore } from '../store/dashboardStore';
import { ScreenshotButton } from './ScreenshotButton';
import { TimeSeriesModal } from './TimeSeriesModal';
import { CLUSTER_COLORS, UI_COLORS } from '../theme';

// Map theme colors to component usage
const COLORS = {
    cluster1: CLUSTER_COLORS.cluster1,
    cluster2: CLUSTER_COLORS.cluster2,
    text: UI_COLORS.text,
    textSecondary: UI_COLORS.textSecondary,
    textMuted: UI_COLORS.textMuted,
    border: UI_COLORS.border,
};

// セクションキーごとの色定義
const SECTION_STYLE: Record<string, { color: string; label: string }> = {
    comparison_context: {
        color: UI_COLORS.textSecondary,
        label: 'COMPARISON CONTEXT',
    },
    separation_factors: {
        color: CLUSTER_COLORS.cluster1,
        label: 'SEPARATION FACTORS',
    },
    suggested_exploration: {
        color: '#1f77b4',
        label: 'SUGGESTED EXPLORATION',
    },
};

// ── マーカーパース ────────────────────────────────────────────────────────────

/**
 * <<rack-variable>> マーカー文字列を rack と variable に分解する。
 *
 * variable 名にハイフンを含む場合（PM2.5 等）やrack 名にハイフンを含む場合
 * （Station-5 等）に対応するため、既知の variable リストを使った逆方向パースを行う。
 *
 * @param markerContent - マーカー内のテキスト（例: "Jefferson-PM2.5", "Rockingham"）
 * @param variables - 既知の変数名リスト
 * @returns { rack, variable } — variable が見つからない場合は null
 */
function parseMarker(
    markerContent: string,
    variables: string[],
): { rack: string; variable: string | null } {
    // 末尾から既知の variable 名を探す（最長一致）
    for (const v of variables) {
        const suffix = `-${v}`;
        if (markerContent.endsWith(suffix)) {
            return {
                rack: markerContent.slice(0, -suffix.length),
                variable: v,
            };
        }
    }
    // ハイフンがあっても既知 variable に一致しない → rack 全体参照
    return { rack: markerContent, variable: null };
}

// ── クリッカブルテキスト描画 ──────────────────────────────────────────────────

/**
 * テキスト内の <<...>> マーカーをクリッカブル要素に変換する。
 * マーカー以外の部分はプレーンテキストとして返す。
 */
function renderClickableText(
    text: string,
    variables: string[],
    onMarkerClick: (rack: string, variable: string | null) => void,
): React.ReactNode {
    const markerRegex = /<<([^>]+)>>/g;
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;
    let key = 0;

    while ((match = markerRegex.exec(text)) !== null) {
        // マーカー前のプレーンテキスト
        if (match.index > lastIndex) {
            parts.push(text.slice(lastIndex, match.index));
        }

        // マーカー部分（クリッカブル）
        const markerContent = match[1];
        const { rack, variable } = parseMarker(markerContent, variables);

        parts.push(
            <Text
                as="span"
                key={`marker-${key++}`}
                color={COLORS.cluster1}
                textDecoration="underline"
                textDecorationColor={`${COLORS.cluster1}66`}
                cursor="pointer"
                fontWeight="600"
                _hover={{
                    bg: `${COLORS.cluster1}10`,
                    borderRadius: '2px',
                    textDecorationColor: COLORS.cluster1,
                }}
                transition="all 0.15s ease"
                onClick={() => onMarkerClick(rack, variable)}
            >
                {variable ? `${rack}-${variable}` : rack}
            </Text>,
        );

        lastIndex = match.index + match[0].length;
    }

    // 残りのプレーンテキスト
    if (lastIndex < text.length) {
        parts.push(text.slice(lastIndex));
    }

    return parts.length > 0 ? parts : text;
}

// ── セクションカード ─────────────────────────────────────────────────────────

function SectionCard({
    sectionKey,
    text,
    variables,
    onMarkerClick,
}: {
    sectionKey: string;
    text: string;
    variables: string[];
    onMarkerClick: (rack: string, variable: string | null) => void;
}) {
    const style = SECTION_STYLE[sectionKey] ?? {
        color: '#555',
        label: sectionKey.toUpperCase(),
    };

    return (
        <Box
            p={3}
            bg="white"
            borderRadius="6px"
            border="1px solid"
            borderColor={COLORS.border}
            boxShadow="0 1px 2px rgba(0,0,0,0.04)"
            borderLeft="3px solid"
            borderLeftColor={style.color}
        >
            <Text
                fontSize="11px"
                fontWeight="700"
                color={style.color}
                mb={1.5}
                textTransform="uppercase"
                letterSpacing="0.05em"
            >
                {style.label}
            </Text>
            <Text
                fontSize="12px"
                color={COLORS.text}
                lineHeight="1.7"
            >
                {renderClickableText(text, variables, onMarkerClick)}
            </Text>
        </Box>
    );
}

// ── メインコンポーネント ─────────────────────────────────────────────────────

export function AIInterpretation() {
    const { interpretation, isLoading, clusters, config } = useDashboardStore();
    const panelRef = React.useRef<HTMLDivElement>(null);
    const hasBothClusters = clusters.cluster1 && clusters.cluster2;

    // モーダル Localstate
    const [modalState, setModalState] = useState<{
        isOpen: boolean;
        rack: string;
        variable: string | null;
    }>({ isOpen: false, rack: '', variable: null });

    const handleMarkerClick = useCallback((rack: string, variable: string | null) => {
        setModalState({ isOpen: true, rack, variable });
    }, []);

    const handleModalClose = useCallback(() => {
        setModalState(prev => ({ ...prev, isOpen: false }));
    }, []);

    const variables = config?.variables ?? [];

    // コンテンツ部分のレンダリング
    const renderContent = () => {
        // ローディング中（クラスター選択済みの場合のみスピナー表示）
        if (isLoading && hasBothClusters) {
            return (
                <Box flex="1" display="flex" alignItems="center" justifyContent="center">
                    <VStack spacing={2}>
                        <Spinner size="md" color={COLORS.cluster1} thickness="2px" />
                        <Text color={COLORS.textMuted} fontSize="sm">Generating interpretation...</Text>
                    </VStack>
                </Box>
            );
        }

        // 解釈結果がない場合（空状態）
        if (!interpretation) {
            return (
                <Box flex="1" display="flex" alignItems="center" justifyContent="center" p={4}>
                    <Text color={COLORS.textMuted} fontSize="sm" textAlign="center">
                        {hasBothClusters
                            ? 'Analysis in progress...'
                            : 'Select two clusters to generate interpretation'}
                    </Text>
                </Box>
            );
        }

        // 3つのセクションカードを縦に並べる
        return (
            <Box flex="1" overflow="auto" p={3}>
                <VStack spacing={3} align="stretch">
                    <SectionCard
                        sectionKey="comparison_context"
                        text={interpretation.comparison_context.text}
                        variables={variables}
                        onMarkerClick={handleMarkerClick}
                    />
                    <SectionCard
                        sectionKey="separation_factors"
                        text={interpretation.separation_factors.text}
                        variables={variables}
                        onMarkerClick={handleMarkerClick}
                    />
                    <SectionCard
                        sectionKey="suggested_exploration"
                        text={interpretation.suggested_exploration.text}
                        variables={variables}
                        onMarkerClick={handleMarkerClick}
                    />
                </VStack>
            </Box>
        );
    };

    return (
        <Box
            ref={panelRef}
            h="100%"
            bg="white"
            borderRadius="4px"
            border="1px solid"
            borderColor={COLORS.border}
            display="flex"
            flexDirection="column"
            overflow="hidden"
        >
            {/* Header */}
            <Box px={3} py={2} borderBottom="1px solid" borderColor={COLORS.border} flexShrink={0}>
                <HStack justify="space-between" align="center">
                    <Text fontSize="sm" fontWeight="600" color={COLORS.text}>
                        AI Interpretation
                    </Text>
                    <ScreenshotButton targetRef={panelRef} filename="ai_interpretation" />
                </HStack>
            </Box>

            {/* Content */}
            {renderContent()}

            {/* Time Series Modal */}
            <TimeSeriesModal
                isOpen={modalState.isOpen}
                onClose={handleModalClose}
                rack={modalState.rack}
                variable={modalState.variable}
            />
        </Box>
    );
}
