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
 * テキスト内の <<マーカー>> 記法は表示時に除去する（将来のクリッカブル実装予約）。
 */
import React from 'react';
import {
    Box,
    VStack,
    HStack,
    Text,
    Spinner,
} from '@chakra-ui/react';
import { useDashboardStore } from '../store/dashboardStore';
import { ScreenshotButton } from './ScreenshotButton';
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

/**
 * <<...>> マーカーを除去してプレーンテキストとして表示する。
 * 将来的にはクリッカブルなリンクに変換する予定。
 */
function renderPlainText(text: string): string {
    return text.replace(/<<([^>]+)>>/g, '$1');
}

/**
 * 1つのセクションをカードとして表示するコンポーネント。
 */
function SectionCard({
    sectionKey,
    text,
}: {
    sectionKey: string;
    text: string;
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
                {renderPlainText(text)}
            </Text>
        </Box>
    );
}

// Main Component
export function AIInterpretation() {
    const { interpretation, isLoading, clusters } = useDashboardStore();
    const panelRef = React.useRef<HTMLDivElement>(null);
    const hasBothClusters = clusters.cluster1 && clusters.cluster2;

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
                    />
                    <SectionCard
                        sectionKey="separation_factors"
                        text={interpretation.separation_factors.text}
                    />
                    <SectionCard
                        sectionKey="suggested_exploration"
                        text={interpretation.suggested_exploration.text}
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
        </Box>
    );
}
