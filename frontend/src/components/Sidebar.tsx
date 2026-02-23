/**
 * サイドバーコンポーネント
 *
 * TULCA パラメータ制御とクラスター選択状態の表示を行う。
 *
 * セクション構成:
 *   1. クラスター選択状態 - C1/C2 の選択サンプル数の表示と解除
 *   2. TULCA パラメータ   - ターゲットクラス選択、w_tg/w_bw/w_bg スライダー
 *   3. 実行ボタン         - TULCA 再計算 → PaCMAP 埋め込みのトリガー
 */
import {
    Box,
    VStack,
    HStack,
    Text,
    Select,
    Slider,
    SliderTrack,
    SliderFilledTrack,
    SliderThumb,
    Button,
    Divider,
    Circle,
} from '@chakra-ui/react';
import { useDashboardStore } from '../store/dashboardStore';
import { CLUSTER_COLORS, UI_COLORS } from '../theme';

// Map theme colors to component usage
const COLORS = {
    cluster1: CLUSTER_COLORS.cluster1,
    cluster2: CLUSTER_COLORS.cluster2,
    accent: UI_COLORS.accent,
    accentHover: UI_COLORS.accentHover,
    text: UI_COLORS.text,
    textSecondary: UI_COLORS.textSecondary,
    textMuted: UI_COLORS.textMuted,
    border: UI_COLORS.border,
    bgSubtle: UI_COLORS.bgSubtle,
};

interface SidebarProps {
    onExecute: () => void;
    isLoading: boolean;
}

export function Sidebar({ onExecute, isLoading }: SidebarProps) {
    const {
        config,
        classWeights,
        selectedClass,
        setSelectedClass,
        updateWeight,
        clusters,
        clearCluster1,
        clearCluster2,
        resetClusters,
    } = useDashboardStore();

    const currentWeights = classWeights[selectedClass] || { w_tg: 0, w_bw: 1, w_bg: 1 };

    return (
        <Box
            w="280px"
            minW="280px"
            h="100vh"
            bg="white"
            borderRight="1px solid"
            borderColor={COLORS.border}
            display="flex"
            flexDirection="column"
        >
            {/* Content - vertically aligned with main grid */}
            <Box flex="1" overflowY="auto" px={4} py={5}>
                <VStack spacing={5} align="stretch">
                    {/* Cluster Selection Status */}
                    <Box>
                        <Text
                            fontSize="10px"
                            fontWeight="600"
                            color={COLORS.textMuted}
                            mb={3}
                            textTransform="uppercase"
                            letterSpacing="0.05em"
                        >
                            Cluster Selection
                        </Text>
                        <VStack spacing={2} align="stretch">
                            <HStack
                                p={3}
                                bg="white"
                                borderRadius="4px"
                                border="1px solid"
                                borderColor={clusters.cluster1 ? COLORS.cluster1 : COLORS.border}
                                justify="space-between"
                            >
                                <HStack spacing={2}>
                                    <Circle size="10px" bg={COLORS.cluster1} />
                                    <Text fontSize="sm" color={COLORS.text}>Cluster 1</Text>
                                </HStack>
                                <HStack spacing={2}>
                                    <Text
                                        fontSize="xs"
                                        color={clusters.cluster1 ? COLORS.cluster1 : COLORS.textMuted}
                                        fontWeight="500"
                                    >
                                        {clusters.cluster1 ? `${clusters.cluster1.length} pts` : '—'}
                                    </Text>
                                    {clusters.cluster1 && (
                                        <Box
                                            as="button"
                                            fontSize="xs"
                                            color={COLORS.textMuted}
                                            _hover={{ color: COLORS.cluster1 }}
                                            onClick={clearCluster1}
                                            cursor="pointer"
                                            title="Clear Cluster 1"
                                        >
                                            ✕
                                        </Box>
                                    )}
                                </HStack>
                            </HStack>
                            <HStack
                                p={3}
                                bg="white"
                                borderRadius="4px"
                                border="1px solid"
                                borderColor={clusters.cluster2 ? COLORS.cluster2 : COLORS.border}
                                justify="space-between"
                            >
                                <HStack spacing={2}>
                                    <Circle size="10px" bg={COLORS.cluster2} />
                                    <Text fontSize="sm" color={COLORS.text}>Cluster 2</Text>
                                </HStack>
                                <HStack spacing={2}>
                                    <Text
                                        fontSize="xs"
                                        color={clusters.cluster2 ? COLORS.cluster2 : COLORS.textMuted}
                                        fontWeight="500"
                                    >
                                        {clusters.cluster2 ? `${clusters.cluster2.length} pts` : '—'}
                                    </Text>
                                    {clusters.cluster2 && (
                                        <Box
                                            as="button"
                                            fontSize="xs"
                                            color={COLORS.textMuted}
                                            _hover={{ color: COLORS.cluster2 }}
                                            onClick={clearCluster2}
                                            cursor="pointer"
                                            title="Clear Cluster 2"
                                        >
                                            ✕
                                        </Box>
                                    )}
                                </HStack>
                            </HStack>
                        </VStack>
                        <Button
                            size="xs"
                            variant="outline"
                            mt={2}
                            w="full"
                            onClick={resetClusters}
                            borderColor={COLORS.border}
                            color={COLORS.textSecondary}
                            _hover={{ bg: COLORS.bgSubtle }}
                        >
                            Reset All
                        </Button>
                    </Box>

                    <Divider borderColor={COLORS.border} />

                    {/* TULCA Parameters */}
                    <Box>
                        <Text
                            fontSize="10px"
                            fontWeight="600"
                            color={COLORS.textMuted}
                            mb={3}
                            textTransform="uppercase"
                            letterSpacing="0.05em"
                        >
                            TULCA Parameters
                        </Text>

                        <Box mb={4}>
                            <Text fontSize="xs" color={COLORS.textSecondary} mb={1}>
                                Target Class
                            </Text>
                            <Select
                                size="sm"
                                bg="white"
                                borderColor={COLORS.border}
                                fontSize="sm"
                                value={selectedClass}
                                onChange={(e) => setSelectedClass(Number(e.target.value))}
                            >
                                {Array.from({ length: config?.n_classes || 3 }, (_, i) => (
                                    <option key={i} value={i}>Class {i + 1}</option>
                                ))}
                            </Select>
                        </Box>

                        <VStack spacing={4}>
                            <SliderControl
                                label="Target Weight"
                                value={currentWeights.w_tg}
                                onChange={(v) => updateWeight(selectedClass, { w_tg: v })}
                            />
                            <SliderControl
                                label="Between-class"
                                value={currentWeights.w_bw}
                                onChange={(v) => updateWeight(selectedClass, { w_bw: v })}
                            />
                            <SliderControl
                                label="Background"
                                value={currentWeights.w_bg}
                                onChange={(v) => updateWeight(selectedClass, { w_bg: v })}
                            />
                        </VStack>

                        {/* Execute Button - moved up */}
                        <Button
                            w="full"
                            size="md"
                            mt={4}
                            bg={COLORS.accent}
                            color="white"
                            _hover={{ bg: COLORS.accentHover }}
                            _active={{ bg: '#333' }}
                            onClick={onExecute}
                            isLoading={isLoading}
                            loadingText="Computing..."
                            fontWeight="500"
                        >
                            Execute
                        </Button>
                    </Box>
                </VStack>
            </Box>
        </Box>
    );
}

interface SliderControlProps {
    label: string;
    value: number;
    onChange: (value: number) => void;
}

function SliderControl({ label, value, onChange }: SliderControlProps) {
    return (
        <Box w="full">
            <HStack justify="space-between" mb={2}>
                <Text fontSize="xs" color={COLORS.textSecondary}>
                    {label}
                </Text>
                <Text fontSize="xs" color={COLORS.textMuted} fontFamily="mono">
                    {value.toFixed(1)}
                </Text>
            </HStack>
            <Slider
                min={0}
                max={1}
                step={0.1}
                value={value}
                onChange={onChange}
            >
                <SliderTrack bg="#e0e0e0" h="4px" borderRadius="2px">
                    <SliderFilledTrack bg={COLORS.accent} />
                </SliderTrack>
                <SliderThumb boxSize={3} bg={COLORS.accent} border="2px solid white" boxShadow="sm" />
            </Slider>
        </Box>
    );
}

