/**
 * 特徴量ランキングコンポーネント（ダイバージングバーチャート）
 *
 * クラスター間の平均値差 (mean_diff) を左右に分岐する
 * 横棒グラフで表示する。正方向（右）は C1 の方が高い特徴量、
 * 負方向（左）は C2 の方が高い特徴量を示す。
 *
 * 視覚エンコーディング:
 *   - バーの長さ: 平均値の差の絶対値
 *   - バーの色:   C1 側 = cluster1カラー、C2 側 = cluster2カラー
 *   - バーの透明度: p < 0.05 なら高透明度、それ以外は低透明度
 *   - アスタリスク(*): 統計的に有意な特徴量
 */
import { useEffect, useRef, useState } from 'react';
import { Box, Text, Center, HStack, Circle } from '@chakra-ui/react';
import * as d3 from 'd3';
import { useDashboardStore } from '../store/dashboardStore';
import { ScreenshotButton } from './ScreenshotButton';
import { CLUSTER_COLORS, UI_COLORS } from '../theme';

// Map theme colors to component usage
const COLORS = {
    positive: CLUSTER_COLORS.cluster1,
    negative: CLUSTER_COLORS.cluster2,
    text: UI_COLORS.text,
    textMuted: UI_COLORS.textMuted,
    bg: UI_COLORS.bgSubtle,
};

export function FeatureRanking() {
    const containerRef = useRef<HTMLDivElement>(null);
    const panelRef = useRef<HTMLDivElement>(null);
    const svgRef = useRef<SVGSVGElement>(null);
    const [dimensions, setDimensions] = useState({ width: 400, height: 300 });
    const { topFeatures } = useDashboardStore();

    // Responsive sizing
    useEffect(() => {
        if (!containerRef.current) return;

        const resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                setDimensions({ width: Math.max(width, 200), height: Math.max(height, 200) });
            }
        });

        resizeObserver.observe(containerRef.current);
        return () => resizeObserver.disconnect();
    }, []);

    // Draw diverging bar chart
    useEffect(() => {
        if (!svgRef.current || !topFeatures || topFeatures.length === 0) return;

        const svg = d3.select(svgRef.current);
        svg.selectAll('*').remove();

        const margin = { top: 10, right: 50, bottom: 10, left: 100 };
        const width = dimensions.width - margin.left - margin.right;
        const height = dimensions.height - margin.top - margin.bottom;

        if (width <= 0 || height <= 0) return;

        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        // Calculate bar height to fit all items in available space
        const totalItems = topFeatures.length;
        const barGap = 4;
        const barHeight = Math.max(12, Math.min(24, (height - (totalItems - 1) * barGap) / totalItems));

        // Prepare data - use mean_diff for direction
        const data = topFeatures.map((f, i) => ({
            label: `#${f.rank}  ${f.rack}-${f.variable}`,
            value: f.mean_diff, // positive = higher in C1, negative = higher in C2
            score: f.score,
            rank: f.rank,
            pValue: f.statistical_result.adjusted_p_value ?? f.statistical_result.p_value,
            index: i,
        }));

        // Scale for bars - symmetric around zero
        const maxAbs = d3.max(data, d => Math.abs(d.value)) || 1;
        const xScale = d3.scaleLinear()
            .domain([-maxAbs, maxAbs])
            .range([0, width]);

        const centerX = xScale(0);

        // Draw center line
        g.append('line')
            .attr('x1', centerX)
            .attr('x2', centerX)
            .attr('y1', 0)
            .attr('y2', data.length * (barHeight + barGap))
            .attr('stroke', '#ddd')
            .attr('stroke-width', 1);

        // Draw bars
        const bars = g.selectAll('.bar-group')
            .data(data)
            .enter()
            .append('g')
            .attr('class', 'bar-group')
            .attr('transform', (_, i) => `translate(0, ${i * (barHeight + barGap)})`);

        // Bar rectangles
        bars.append('rect')
            .attr('x', d => d.value >= 0 ? centerX : xScale(d.value))
            .attr('y', 0)
            .attr('width', d => Math.abs(xScale(d.value) - centerX))
            .attr('height', barHeight)
            .attr('fill', d => d.value >= 0 ? COLORS.positive : COLORS.negative)
            .attr('opacity', d => d.pValue < 0.05 ? 0.9 : 0.5)
            .attr('rx', 2);

        // Feature labels (left side)
        bars.append('text')
            .attr('x', -8)
            .attr('y', barHeight / 2)
            .attr('dy', '0.35em')
            .attr('text-anchor', 'end')
            .attr('fill', COLORS.text)
            .attr('font-size', '11px')
            .attr('font-family', 'Inter, -apple-system, sans-serif')
            .attr('font-weight', (_, i) => i < 3 ? '600' : '400')
            .text(d => d.label);

        // Value labels (on bars) - position inside if bar is too long
        const chartRightEdge = width - 50; // Leave margin for labels
        bars.append('text')
            .attr('x', d => {
                const barEnd = xScale(d.value);
                const isPositive = d.value >= 0;
                // Check if label would go outside chart
                if (isPositive && barEnd > chartRightEdge) {
                    return barEnd - 6; // Inside bar (right-aligned)
                } else if (!isPositive && barEnd < 50) {
                    return barEnd + 6; // Inside bar (left-aligned)
                }
                // Normal: outside bar
                return isPositive ? barEnd + 4 : barEnd - 4;
            })
            .attr('y', barHeight / 2)
            .attr('dy', '0.35em')
            .attr('text-anchor', d => {
                const barEnd = xScale(d.value);
                const isPositive = d.value >= 0;
                if (isPositive && barEnd > chartRightEdge) return 'end';
                if (!isPositive && barEnd < 50) return 'start';
                return isPositive ? 'start' : 'end';
            })
            .attr('fill', d => {
                const barEnd = xScale(d.value);
                const isPositive = d.value >= 0;
                // White text when inside bar
                if ((isPositive && barEnd > chartRightEdge) || (!isPositive && barEnd < 50)) {
                    return '#fff';
                }
                return COLORS.textMuted;
            })
            .attr('font-size', '9px')
            .attr('font-family', 'Inter, -apple-system, sans-serif')
            .attr('font-weight', '500')
            .text(d => d.value.toFixed(2));

        // Significance indicator
        bars.filter(d => d.pValue < 0.05)
            .append('text')
            .attr('x', d => d.value >= 0 ? centerX - 4 : centerX + 4)
            .attr('y', barHeight / 2)
            .attr('dy', '0.35em')
            .attr('text-anchor', d => d.value >= 0 ? 'end' : 'start')
            .attr('fill', d => d.value >= 0 ? COLORS.positive : COLORS.negative)
            .attr('font-size', '9px')
            .attr('font-family', 'Inter, -apple-system, sans-serif')
            .attr('font-weight', '600')
            .text('*');

    }, [topFeatures, dimensions]);

    if (!topFeatures || topFeatures.length === 0) {
        return (
            <Box ref={containerRef} h="100%" display="flex" flexDirection="column">
                <Box px={4} py={3} borderBottom="1px solid" borderColor="#e0e0e0">
                    <HStack spacing={4}>
                        <HStack spacing={1}>
                            <Circle size="8px" bg={COLORS.positive} />
                            <Text fontSize="10px" color="#888">Higher in C1</Text>
                        </HStack>
                        <HStack spacing={1}>
                            <Circle size="8px" bg={COLORS.negative} />
                            <Text fontSize="10px" color="#888">Higher in C2</Text>
                        </HStack>
                    </HStack>
                </Box>
                <Center flex="1">
                    <Text color="#888" fontSize="sm">
                        Select two clusters to view summary
                    </Text>
                </Center>
            </Box>
        );
    }

    return (
        <Box ref={(el) => { panelRef.current = el; containerRef.current = el; }} h="100%" display="flex" flexDirection="column" overflow="hidden">
            {/* Header with legend only - no title */}
            <Box px={4} py={3} borderBottom="1px solid" borderColor="#e0e0e0" flexShrink={0}>
                <HStack spacing={4}>
                    <HStack spacing={1}>
                        <Circle size="8px" bg={COLORS.positive} />
                        <Text fontSize="10px" color="#888">Higher in C1</Text>
                    </HStack>
                    <HStack spacing={1}>
                        <Circle size="8px" bg={COLORS.negative} />
                        <Text fontSize="10px" color="#888">Higher in C2</Text>
                    </HStack>
                    <ScreenshotButton targetRef={panelRef} filename="feature_ranking" />
                </HStack>
            </Box>

            {/* Chart - no overflow, fit to container */}
            <Box flex="1" p={2} overflow="hidden">
                <svg
                    ref={svgRef}
                    width="100%"
                    height="100%"
                    viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
                    preserveAspectRatio="xMidYMid meet"
                    style={{ display: 'block' }}
                />
            </Box>
        </Box>
    );
}
