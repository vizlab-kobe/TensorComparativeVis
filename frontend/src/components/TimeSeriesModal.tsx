/**
 * 時系列モーダルコンポーネント
 *
 * AI解釈テキスト内のクリッカブルマーカーから呼び出され、
 * 指定されたrack-variableの時系列グラフをモーダル内に表示する。
 *
 * - variable 指定時: 1つのグラフ（C1赤 / C2青の2系列）
 * - variable=null 時: rack 配下の全variable分のグラフを縦に並べて表示
 */
import { useEffect, useRef } from 'react';
import {
    Modal,
    ModalOverlay,
    ModalContent,
    ModalHeader,
    ModalBody,
    ModalCloseButton,
    Box,
    Text,
    VStack,
    HStack,
    Circle,
} from '@chakra-ui/react';
import * as d3 from 'd3';
import { useDashboardStore } from '../store/dashboardStore';
import { CLUSTER_COLORS, UI_COLORS } from '../theme';
import type { FeatureImportance } from '../types';

// ── Props ────────────────────────────────────────────────────────────────────

interface TimeSeriesModalProps {
    isOpen: boolean;
    onClose: () => void;
    rack: string;
    variable: string | null;
}

// ── 単一グラフ描画コンポーネント ──────────────────────────────────────────────

function TimeSeriesChart({ feature }: { feature: FeatureImportance }) {
    const svgRef = useRef<SVGSVGElement>(null);
    const width = 560;
    const height = 220;
    const margin = { top: 16, right: 16, bottom: 40, left: 50 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    useEffect(() => {
        if (!svgRef.current) return;

        const svg = d3.select(svgRef.current);
        svg.selectAll('*').remove();

        // データ準備
        const c1Data = feature.cluster1_time.map((t, i) => ({
            time: new Date(t),
            value: feature.cluster1_data[i],
        }));
        const c2Data = feature.cluster2_time.map((t, i) => ({
            time: new Date(t),
            value: feature.cluster2_data[i],
        }));

        const allData = [...c1Data, ...c2Data];
        const xExtent = d3.extent(allData, d => d.time) as [Date, Date];
        const yExtent = d3.extent(allData, d => d.value) as [number, number];
        const yPadding = (yExtent[1] - yExtent[0]) * 0.15 || 0.1;

        const xScale = d3.scaleTime().domain(xExtent).range([0, innerWidth]);
        const yScale = d3.scaleLinear()
            .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
            .range([innerHeight, 0]);

        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        // クリップパス
        const clipId = `modal-clip-${Math.random().toString(36).substr(2, 9)}`;
        svg.append('defs').append('clipPath')
            .attr('id', clipId)
            .append('rect')
            .attr('width', innerWidth)
            .attr('height', innerHeight);
        const chartArea = g.append('g').attr('clip-path', `url(#${clipId})`);

        // グリッド線
        chartArea.selectAll('line.grid')
            .data(yScale.ticks(5))
            .enter().append('line')
            .attr('x1', 0).attr('x2', innerWidth)
            .attr('y1', d => yScale(d)).attr('y2', d => yScale(d))
            .attr('stroke', UI_COLORS.grid);

        // 折れ線描画
        const line = d3.line<{ time: Date; value: number }>()
            .x(d => xScale(d.time))
            .y(d => yScale(d.value))
            .curve(d3.curveMonotoneX);

        const sorted1 = [...c1Data].sort((a, b) => a.time.getTime() - b.time.getTime());
        const sorted2 = [...c2Data].sort((a, b) => a.time.getTime() - b.time.getTime());

        chartArea.append('path')
            .datum(sorted1)
            .attr('fill', 'none')
            .attr('stroke', CLUSTER_COLORS.cluster1)
            .attr('stroke-width', 2)
            .attr('d', line);

        chartArea.append('path')
            .datum(sorted2)
            .attr('fill', 'none')
            .attr('stroke', CLUSTER_COLORS.cluster2)
            .attr('stroke-width', 2)
            .attr('d', line);

        // データポイント
        const drawPoints = (
            data: { time: Date; value: number }[],
            color: string,
            className: string,
        ) => {
            chartArea.selectAll(`.${className}`)
                .data(data)
                .enter().append('circle')
                .attr('class', className)
                .attr('cx', d => xScale(d.time))
                .attr('cy', d => yScale(d.value))
                .attr('r', 2)
                .attr('fill', color)
                .attr('stroke', 'white')
                .attr('stroke-width', 0.5)
                .attr('opacity', 0.8);
        };

        drawPoints(c1Data, CLUSTER_COLORS.cluster1, 'p1');
        drawPoints(c2Data, CLUSTER_COLORS.cluster2, 'p2');

        // 軸
        const xAxis = d3.axisBottom(xScale).ticks(5).tickFormat(d3.timeFormat('%Y-%m') as any);
        const yAxis = d3.axisLeft(yScale).ticks(5);

        g.append('g')
            .attr('transform', `translate(0,${innerHeight})`)
            .call(xAxis)
            .call(g => g.select('.domain').attr('stroke', '#ddd'))
            .call(g => g.selectAll('.tick line').attr('stroke', '#ddd'))
            .call(g => g.selectAll('.tick text')
                .attr('fill', UI_COLORS.textMuted)
                .attr('font-size', '10px')
                .attr('transform', 'rotate(-30)')
                .attr('text-anchor', 'end'));

        g.append('g')
            .call(yAxis)
            .call(g => g.select('.domain').attr('stroke', '#ddd'))
            .call(g => g.selectAll('.tick line').attr('stroke', '#ddd'))
            .call(g => g.selectAll('.tick text')
                .attr('fill', UI_COLORS.textMuted)
                .attr('font-size', '10px'));

    }, [feature, innerWidth, innerHeight, margin.left, margin.top]);

    // 統計情報
    const stat = feature.statistical_result;
    const effectiveP = stat.adjusted_p_value ?? stat.p_value;
    const isSignificant = effectiveP < 0.05;
    const pLabel = stat.adjusted_p_value != null ? 'p(FDR)' : 'p';

    return (
        <Box>
            <svg
                ref={svgRef}
                width="100%"
                viewBox={`0 0 ${width} ${height}`}
                preserveAspectRatio="xMidYMid meet"
                style={{ display: 'block', maxWidth: '100%' }}
            />
            <HStack spacing={3} mt={1} px={2} pb={1} fontSize="xs">
                <Box
                    px={2}
                    py={0.5}
                    borderRadius="3px"
                    bg={isSignificant ? '#e8f5e9' : '#f5f5f5'}
                    border="1px solid"
                    borderColor={isSignificant ? '#c8e6c9' : '#e0e0e0'}
                >
                    <Text
                        color={isSignificant ? '#2e7d32' : '#888'}
                        fontWeight="600"
                        fontSize="10px"
                    >
                        {isSignificant ? '✓ Suggests difference' : 'No clear difference'}
                    </Text>
                </Box>
                <Text color="#888">
                    {pLabel}={effectiveP.toFixed(3)}, d={stat.cohen_d.toFixed(2)}
                </Text>
            </HStack>
        </Box>
    );
}

// ── メインモーダルコンポーネント ──────────────────────────────────────────────

export function TimeSeriesModal({ isOpen, onClose, rack, variable }: TimeSeriesModalProps) {
    const { topFeatures, config } = useDashboardStore();

    // 対象featureを検索
    const findFeature = (r: string, v: string): FeatureImportance | undefined =>
        topFeatures?.find(f => f.rack === r && f.variable === v);

    // 表示対象を決定
    let featuresToShow: { variable: string; feature: FeatureImportance | undefined }[] = [];

    if (variable) {
        // 単一variable指定
        featuresToShow = [{ variable, feature: findFeature(rack, variable) }];
    } else {
        // rack全体 → configのvariablesリストから全variableを表示
        const variables = config?.variables ?? [];
        featuresToShow = variables.map(v => ({
            variable: v,
            feature: findFeature(rack, v),
        }));
    }

    const headerTitle = variable ? `${rack} - ${variable}` : `${rack} (all variables)`;

    return (
        <Modal isOpen={isOpen} onClose={onClose} size="xl" scrollBehavior="inside">
            <ModalOverlay bg="blackAlpha.400" />
            <ModalContent maxW="660px">
                <ModalHeader
                    fontSize="md"
                    fontWeight="600"
                    color={UI_COLORS.text}
                    pb={2}
                    borderBottom="1px solid"
                    borderColor={UI_COLORS.border}
                >
                    <HStack spacing={4}>
                        <Text>{headerTitle}</Text>
                        <HStack spacing={3} ml="auto" mr={8}>
                            <HStack spacing={1}>
                                <Circle size="8px" bg={CLUSTER_COLORS.cluster1} />
                                <Text fontSize="10px" color={UI_COLORS.textMuted}>C1</Text>
                            </HStack>
                            <HStack spacing={1}>
                                <Circle size="8px" bg={CLUSTER_COLORS.cluster2} />
                                <Text fontSize="10px" color={UI_COLORS.textMuted}>C2</Text>
                            </HStack>
                        </HStack>
                    </HStack>
                </ModalHeader>
                <ModalCloseButton />
                <ModalBody px={4} py={3}>
                    <VStack spacing={4} align="stretch">
                        {featuresToShow.map(({ variable: v, feature }) => (
                            <Box key={v}>
                                {/* variable名ヘッダー（複数variable時のみ） */}
                                {!variable && (
                                    <Text
                                        fontSize="11px"
                                        fontWeight="700"
                                        color={UI_COLORS.textSecondary}
                                        mb={1}
                                        textTransform="uppercase"
                                        letterSpacing="0.04em"
                                    >
                                        {v}
                                    </Text>
                                )}
                                {feature ? (
                                    <TimeSeriesChart feature={feature} />
                                ) : (
                                    <Box
                                        p={4}
                                        bg="#fafafa"
                                        borderRadius="4px"
                                        border="1px solid"
                                        borderColor={UI_COLORS.border}
                                        textAlign="center"
                                    >
                                        <Text fontSize="xs" color={UI_COLORS.textMuted}>
                                            No data available for {rack} - {v}
                                        </Text>
                                    </Box>
                                )}
                            </Box>
                        ))}
                    </VStack>
                </ModalBody>
            </ModalContent>
        </Modal>
    );
}
