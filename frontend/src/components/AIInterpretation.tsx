/**
 * AI解釈コンポーネント（3タブ構成）
 *
 * Gemini API による自然言語分析結果を構造化して表示する。
 *
 * タブ構成:
 *   1. Summary  - 現在のクラスター分析の AI 解釈をセクションカードで表示
 *   2. History  - 過去の分析結果を一覧表示、2件選択して比較可能
 *   3. Compare  - 2件の分析結果の特徴量差分 + AI 比較分析
 *
 * サブコンポーネント:
 *   - SectionCard:  セクション1件を左ボーダー色付きカードで表示
 *   - SummaryTab:   解釈セクション一覧 + 「Save to History」ボタン
 *   - HistoryTab:   保存済み分析のチェックボックスリスト
 *   - HistoryItem:  1件の保存済み分析のサマリー行
 *   - CompareTab:   2件の分析の共通/固有特徴量表示 + AI比較
 */
import React from 'react';
import {
    Box,
    VStack,
    HStack,
    Text,
    Spinner,
    Circle,
    Tabs,
    TabList,
    TabPanels,
    Tab,
    TabPanel,
    Button,
    Badge,
    Checkbox,
    Divider,
    Grid,
    GridItem,
} from '@chakra-ui/react';
import { useDashboardStore } from '../store/dashboardStore';
import type { SavedAnalysis } from '../store/dashboardStore';
import type { InterpretationSection } from '../types';
import { ScreenshotButton } from './ScreenshotButton';
import { CLUSTER_COLORS, UI_COLORS, getSectionColor } from '../theme';

// Map theme colors to component usage
const COLORS = {
    cluster1: CLUSTER_COLORS.cluster1,
    cluster2: CLUSTER_COLORS.cluster2,
    text: UI_COLORS.text,
    textSecondary: UI_COLORS.textSecondary,
    textMuted: UI_COLORS.textMuted,
    border: UI_COLORS.border,
};

// Render text - remove brackets but display as plain text (no highlighting)
function renderPlainText(text: string) {
    // Remove brackets but keep the content
    const cleanedText = text.replace(/\[([^\]]+)\]/g, '$1');
    return cleanedText;
}

// Section Card Component
function SectionCard({ section }: { section: InterpretationSection }) {
    const sectionColor = getSectionColor(section.title);

    return (
        <Box
            p={3}
            bg="white"
            borderRadius="6px"
            border="1px solid"
            borderColor={COLORS.border}
            boxShadow="0 1px 2px rgba(0,0,0,0.04)"
            borderLeft="3px solid"
            borderLeftColor={sectionColor}
        >
            <Text
                fontSize="11px"
                fontWeight="700"
                color={sectionColor}
                mb={1.5}
                textTransform="uppercase"
                letterSpacing="0.05em"
            >
                {section.title}
            </Text>
            <Text
                fontSize="12px"
                color={COLORS.text}
                lineHeight="1.7"
            >
                {renderPlainText(section.text)}
            </Text>
        </Box>
    );
}

// Summary Tab Content
function SummaryTab() {
    const { interpretation, isLoading, clusters, saveCurrentAnalysis, topFeatures } = useDashboardStore();
    const hasBothClusters = clusters.cluster1 && clusters.cluster2;

    // Only show loading spinner if clusters are selected AND loading (i.e., generating interpretation)
    // Don't show loading during embedding computation (which resets clusters)
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

    if (!interpretation || interpretation.length === 0) {
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

    const canSave = topFeatures && topFeatures.length > 0;

    return (
        <Box flex="1" overflow="auto" p={3}>
            <VStack spacing={3} align="stretch">
                {interpretation.map((section, index) => (
                    <SectionCard key={index} section={section} />
                ))}

                {canSave && (
                    <Button
                        size="sm"
                        variant="outline"
                        colorScheme="gray"
                        onClick={saveCurrentAnalysis}
                        alignSelf="flex-end"
                        fontSize="xs"
                    >
                        Save to History
                    </Button>
                )}
            </VStack>
        </Box>
    );
}

// History Tab Content
function HistoryTab() {
    const {
        analysisHistory,
        selectedHistoryIds,
        toggleHistorySelection,
        clearHistory,
        setInterpretationTab,
    } = useDashboardStore();

    if (analysisHistory.length === 0) {
        return (
            <Box flex="1" display="flex" alignItems="center" justifyContent="center" p={4}>
                <Text color={COLORS.textMuted} fontSize="sm" textAlign="center">
                    No saved analyses yet.
                    <br />
                    Save analyses from the Summary tab to compare them.
                </Text>
            </Box>
        );
    }

    const formatTime = (date: Date) => {
        return new Date(date).toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' });
    };

    return (
        <Box flex="1" overflow="auto" p={3}>
            <VStack spacing={2} align="stretch">
                <HStack justify="space-between" mb={1}>
                    <Text fontSize="xs" color={COLORS.textMuted}>
                        {selectedHistoryIds.length === 2 ? 'Select Compare tab to compare' : 'Select 2 items to compare'}
                    </Text>
                    <Button size="xs" variant="ghost" colorScheme="red" onClick={clearHistory}>
                        Clear All
                    </Button>
                </HStack>

                {analysisHistory.map((analysis) => (
                    <HistoryItem
                        key={analysis.id}
                        analysis={analysis}
                        isSelected={selectedHistoryIds.includes(analysis.id)}
                        onToggle={() => toggleHistorySelection(analysis.id)}
                        formatTime={formatTime}
                    />
                ))}

                {selectedHistoryIds.length === 2 && (
                    <Button
                        size="sm"
                        colorScheme="blue"
                        onClick={() => setInterpretationTab('compare')}
                        mt={2}
                    >
                        Compare Selected
                    </Button>
                )}
            </VStack>
        </Box>
    );
}

function HistoryItem({
    analysis,
    isSelected,
    onToggle,
}: {
    analysis: SavedAnalysis;
    isSelected: boolean;
    onToggle: () => void;
}) {
    return (
        <Box
            p={2}
            bg={isSelected ? 'blue.50' : 'white'}
            borderRadius="4px"
            border="1px solid"
            borderColor={isSelected ? 'blue.200' : COLORS.border}
            cursor="pointer"
            onClick={onToggle}
            _hover={{ borderColor: 'blue.300' }}
        >
            <HStack spacing={2}>
                <Checkbox isChecked={isSelected} onChange={onToggle} size="sm" />
                <VStack align="start" spacing={0} flex="1">
                    <HStack spacing={2}>
                        <HStack spacing={1}>
                            <Circle size="6px" bg={COLORS.cluster1} />
                            <Text fontSize="10px" color={COLORS.textMuted}>{analysis.cluster1_size}</Text>
                            <Circle size="6px" bg={COLORS.cluster2} />
                            <Text fontSize="10px" color={COLORS.textMuted}>{analysis.cluster2_size}</Text>
                        </HStack>
                    </HStack>
                    <HStack spacing={1} flexWrap="wrap">
                        {analysis.summary.top_variables.slice(0, 2).map((v, i) => (
                            <Badge key={i} size="sm" fontSize="9px" colorScheme="gray">
                                {v}
                            </Badge>
                        ))}
                        <Text fontSize="10px" color={COLORS.textMuted}>
                            {analysis.summary.significant_count} significant
                        </Text>
                    </HStack>
                </VStack>
            </HStack>
        </Box>
    );
}

// Compare Tab Content
function CompareTab() {
    const { analysisHistory, selectedHistoryIds, clearHistorySelection } = useDashboardStore();
    const [comparisonResult, setComparisonResult] = React.useState<InterpretationSection[] | null>(null);
    const [isComparing, setIsComparing] = React.useState(false);

    const selectedAnalyses = analysisHistory.filter((a) =>
        selectedHistoryIds.includes(a.id)
    );

    if (selectedAnalyses.length !== 2) {
        return (
            <Box flex="1" display="flex" alignItems="center" justifyContent="center" p={4}>
                <Text color={COLORS.textMuted} fontSize="sm" textAlign="center">
                    Select 2 analyses from the History tab to compare.
                </Text>
            </Box>
        );
    }

    const [analysis1, analysis2] = selectedAnalyses;

    // Check if same base cluster (Cluster 1)
    const sameBaseCluster = analysis1.cluster1_size === analysis2.cluster1_size;

    // Find common and different features
    const features1 = new Set(analysis1.top_features.slice(0, 10).map(f => `${f.rack}-${f.variable}`));
    const features2 = new Set(analysis2.top_features.slice(0, 10).map(f => `${f.rack}-${f.variable}`));
    const common = [...features1].filter(f => features2.has(f));
    const onlyIn1 = [...features1].filter(f => !features2.has(f));
    const onlyIn2 = [...features2].filter(f => !features1.has(f));

    const handleAnalyzeDifference = async () => {
        setIsComparing(true);
        try {
            const { compareAnalyses } = await import('../api/client');
            const result = await compareAnalyses(
                {
                    cluster1_size: analysis1.cluster1_size,
                    cluster2_size: analysis1.cluster2_size,
                    significant_count: analysis1.summary.significant_count,
                    top_variables: analysis1.summary.top_variables,
                    top_racks: analysis1.summary.top_racks,
                    top_features: analysis1.top_features.slice(0, 10).map(f => ({
                        rack: f.rack,
                        variable: f.variable,
                        score: f.score,
                        mean_diff: f.mean_diff,
                    })),
                },
                {
                    cluster1_size: analysis2.cluster1_size,
                    cluster2_size: analysis2.cluster2_size,
                    significant_count: analysis2.summary.significant_count,
                    top_variables: analysis2.summary.top_variables,
                    top_racks: analysis2.summary.top_racks,
                    top_features: analysis2.top_features.slice(0, 10).map(f => ({
                        rack: f.rack,
                        variable: f.variable,
                        score: f.score,
                        mean_diff: f.mean_diff,
                    })),
                }
            );
            setComparisonResult(result.sections);
        } catch (error) {
            console.error('Compare analysis error:', error);
        } finally {
            setIsComparing(false);
        }
    };

    return (
        <Box flex="1" overflow="auto" p={3}>
            <VStack spacing={3} align="stretch">
                <HStack justify="space-between">
                    <Text fontSize="xs" fontWeight="600" color={COLORS.text}>Comparison</Text>
                    <Button size="xs" variant="ghost" onClick={() => { clearHistorySelection(); setComparisonResult(null); }}>
                        Clear Selection
                    </Button>
                </HStack>

                {/* Base Cluster Indicator */}
                {
                    sameBaseCluster && (
                        <Box p={2} bg="blue.50" borderRadius="4px" border="1px solid" borderColor="blue.200">
                            <Text fontSize="10px" color="blue.700">
                                ✓ Same base cluster (Red: {analysis1.cluster1_size} pts)
                            </Text>
                        </Box>
                    )
                }

                {/* Overview */}
                <Grid templateColumns="1fr 1fr" gap={2}>
                    <GridItem>
                        <Box p={2} bg="gray.50" borderRadius="4px" border="1px solid" borderColor={COLORS.border}>
                            <Text fontSize="10px" color={COLORS.textMuted}>Analysis A</Text>
                            <HStack spacing={1} mt={1}>
                                <Circle size="6px" bg={COLORS.cluster1} />
                                <Text fontSize="11px">{analysis1.cluster1_size}</Text>
                                <Circle size="6px" bg={COLORS.cluster2} />
                                <Text fontSize="11px">{analysis1.cluster2_size}</Text>
                            </HStack>
                        </Box>
                    </GridItem>
                    <GridItem>
                        <Box p={2} bg="gray.50" borderRadius="4px" border="1px solid" borderColor={COLORS.border}>
                            <Text fontSize="10px" color={COLORS.textMuted}>Analysis B</Text>
                            <HStack spacing={1} mt={1}>
                                <Circle size="6px" bg={COLORS.cluster1} />
                                <Text fontSize="11px">{analysis2.cluster1_size}</Text>
                                <Circle size="6px" bg={COLORS.cluster2} />
                                <Text fontSize="11px">{analysis2.cluster2_size}</Text>
                            </HStack>
                        </Box>
                    </GridItem>
                </Grid>

                <Divider />

                {/* Feature Comparison */}
                <Box>
                    <Text fontSize="11px" fontWeight="600" color={COLORS.text} mb={2}>Top 10 Features Comparison</Text>
                    <VStack spacing={1} align="stretch">
                        {common.length > 0 && (
                            <HStack>
                                <Badge colorScheme="green" fontSize="9px">Common</Badge>
                                <Text fontSize="10px" color={COLORS.textSecondary}>
                                    {common.join(', ')}
                                </Text>
                            </HStack>
                        )}
                        {onlyIn1.length > 0 && (
                            <HStack>
                                <Badge colorScheme="orange" fontSize="9px">Only A</Badge>
                                <Text fontSize="10px" color={COLORS.textSecondary}>
                                    {onlyIn1.join(', ')}
                                </Text>
                            </HStack>
                        )}
                        {onlyIn2.length > 0 && (
                            <HStack>
                                <Badge colorScheme="purple" fontSize="9px">Only B</Badge>
                                <Text fontSize="10px" color={COLORS.textSecondary}>
                                    {onlyIn2.join(', ')}
                                </Text>
                            </HStack>
                        )}
                    </VStack>
                </Box>

                <Divider />

                {/* AI Comparison Analysis */}
                <Box>
                    <HStack justify="space-between" mb={2}>
                        <Text fontSize="11px" fontWeight="600" color={COLORS.text}>AI Comparison Analysis</Text>
                        <Button
                            size="xs"
                            colorScheme="blue"
                            onClick={handleAnalyzeDifference}
                            isLoading={isComparing}
                            loadingText="Analyzing..."
                        >
                            Analyze Difference
                        </Button>
                    </HStack>

                    {comparisonResult ? (
                        <VStack spacing={2} align="stretch">
                            {comparisonResult.map((section, i) => (
                                <Box
                                    key={i}
                                    p={3}
                                    bg="gray.50"
                                    borderRadius="4px"
                                    borderLeft="3px solid"
                                    borderLeftColor={getSectionColor(section.title)}
                                >
                                    <Text fontSize="11px" fontWeight="600" color={getSectionColor(section.title)} mb={1}>
                                        {section.title}
                                    </Text>
                                    <Text fontSize="11px" color={COLORS.textSecondary} lineHeight="1.6">
                                        {renderPlainText(section.text)}
                                    </Text>
                                </Box>
                            ))}
                        </VStack>
                    ) : (
                        <Box p={3} bg="gray.50" borderRadius="4px" textAlign="center">
                            <Text fontSize="10px" color={COLORS.textMuted}>
                                Click "Analyze Difference" to get AI-powered comparison insights
                            </Text>
                        </Box>
                    )}
                </Box>
            </VStack >
        </Box >
    );
}

// Main Component
export function AIInterpretation() {
    const { interpretationTab, setInterpretationTab } = useDashboardStore();
    const panelRef = React.useRef<HTMLDivElement>(null);

    const tabIndex = { summary: 0, history: 1, compare: 2 }[interpretationTab];

    const handleTabChange = (index: number) => {
        const tabs: ('summary' | 'history' | 'compare')[] = ['summary', 'history', 'compare'];
        setInterpretationTab(tabs[index]);
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
            {/* Header with cluster info */}
            <Box px={3} py={2} borderBottom="1px solid" borderColor={COLORS.border} flexShrink={0}>
                <HStack justify="space-between" align="center">
                    <Text fontSize="sm" fontWeight="600" color={COLORS.text}>
                        AI Analysis Summary
                    </Text>
                    <ScreenshotButton targetRef={panelRef} filename="ai_analysis" />
                </HStack>
            </Box>

            {/* Tabs */}
            <Tabs
                index={tabIndex}
                onChange={handleTabChange}
                size="sm"
                variant="line"
                flex="1"
                display="flex"
                flexDirection="column"
                overflow="hidden"
            >
                <TabList px={2} borderBottom="1px solid" borderColor={COLORS.border}>
                    <Tab fontSize="xs" _selected={{ color: COLORS.cluster1, borderColor: COLORS.cluster1 }}>
                        Summary
                    </Tab>
                    <Tab fontSize="xs" _selected={{ color: COLORS.cluster1, borderColor: COLORS.cluster1 }}>
                        History
                    </Tab>
                    <Tab fontSize="xs" _selected={{ color: COLORS.cluster1, borderColor: COLORS.cluster1 }}>
                        Compare
                    </Tab>
                </TabList>

                <TabPanels flex="1" overflow="hidden" display="flex" flexDirection="column">
                    <TabPanel p={0} flex="1" display="flex" flexDirection="column" overflow="hidden">
                        <SummaryTab />
                    </TabPanel>
                    <TabPanel p={0} flex="1" display="flex" flexDirection="column" overflow="hidden">
                        <HistoryTab />
                    </TabPanel>
                    <TabPanel p={0} flex="1" display="flex" flexDirection="column" overflow="hidden">
                        <CompareTab />
                    </TabPanel>
                </TabPanels>
            </Tabs>
        </Box>
    );
}
