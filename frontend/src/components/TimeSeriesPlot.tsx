/**
 * 時系列プロットコンポーネント（D3.js）
 *
 * 選択された特徴量の時系列データを C1/C2 の2系列で描画する。
 * 特徴量ナビゲーション（前/次ボタン）で上位特徴量を順番に閲覧可能。
 *
 * 機能:
 *   - マウスホイールでズーム、ドラッグでパン
 *   - 特徴量切り替え時に自動ズームリセット
 *   - ツールチップによるデータポイント詳細表示
 *   - フッターに統計情報（p値, Cohen's d, 有意性判定）
 *   - Clip-path によるチャート領域外の描画抑制
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import { Box, HStack, IconButton, Text, Circle, VStack, Tooltip } from '@chakra-ui/react';
import { ChevronLeftIcon, ChevronRightIcon, RepeatIcon } from '@chakra-ui/icons';
import * as d3 from 'd3';
import { useDashboardStore } from '../store/dashboardStore';
import { ScreenshotButton } from './ScreenshotButton';
import { CLUSTER_COLORS, UI_COLORS } from '../theme';

// Map theme colors to component usage
const COLORS = {
    cluster1: CLUSTER_COLORS.cluster1,
    cluster2: CLUSTER_COLORS.cluster2,
    grid: UI_COLORS.grid,
    axis: UI_COLORS.axis,
    text: UI_COLORS.text,
    textMuted: UI_COLORS.textMuted,
};

export function TimeSeriesPlot() {
    const containerRef = useRef<HTMLDivElement>(null);
    const panelRef = useRef<HTMLDivElement>(null);
    const svgRef = useRef<SVGSVGElement>(null);
    const [dimensions, setDimensions] = useState({ width: 400, height: 250 });
    const { topFeatures, currentFeatureIndex, setCurrentFeatureIndex } = useDashboardStore();

    // Zoom state
    const [isZoomed, setIsZoomed] = useState(false);
    const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
    const currentTransformRef = useRef<d3.ZoomTransform>(d3.zoomIdentity);

    const margin = { top: 20, right: 20, bottom: 50, left: 55 };
    const innerWidth = dimensions.width - margin.left - margin.right;
    const innerHeight = dimensions.height - margin.top - margin.bottom;

    const feature = topFeatures?.[currentFeatureIndex];

    // Reset zoom when feature changes
    useEffect(() => {
        if (svgRef.current && zoomRef.current) {
            d3.select(svgRef.current).call(zoomRef.current.transform, d3.zoomIdentity);
            currentTransformRef.current = d3.zoomIdentity;
            setIsZoomed(false);
        }
    }, [currentFeatureIndex]);

    // Reset zoom handler
    const handleResetZoom = useCallback(() => {
        if (svgRef.current && zoomRef.current) {
            d3.select(svgRef.current)
                .transition()
                .duration(300)
                .call(zoomRef.current.transform, d3.zoomIdentity);
            currentTransformRef.current = d3.zoomIdentity;
            setIsZoomed(false);
        }
    }, []);

    // Responsive sizing
    useEffect(() => {
        if (!containerRef.current) return;

        const resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                setDimensions({ width: Math.max(width, 200), height: Math.max(height - 100, 150) });
            }
        });

        resizeObserver.observe(containerRef.current);
        return () => resizeObserver.disconnect();
    }, []);

    const handlePrev = () => {
        if (currentFeatureIndex > 0) setCurrentFeatureIndex(currentFeatureIndex - 1);
    };

    const handleNext = () => {
        if (topFeatures && currentFeatureIndex < topFeatures.length - 1) {
            setCurrentFeatureIndex(currentFeatureIndex + 1);
        }
    };

    useEffect(() => {
        if (!svgRef.current || !feature || innerWidth <= 0 || innerHeight <= 0) return;

        const svg = d3.select(svgRef.current);
        svg.selectAll('*').remove();

        // Create or select tooltip
        let tooltip = d3.select('body').select('.ts-tooltip');
        if (tooltip.empty()) {
            tooltip = d3.select('body').append('div')
                .attr('class', 'ts-tooltip')
                .style('position', 'absolute')
                .style('background', 'rgba(0,0,0,0.8)')
                .style('color', 'white')
                .style('padding', '8px 12px')
                .style('border-radius', '4px')
                .style('font-size', '11px')
                .style('pointer-events', 'none')
                .style('opacity', 0)
                .style('z-index', 9999);
        }

        const cluster1Data = feature.cluster1_time.map((t, i) => ({
            time: new Date(t),
            value: feature.cluster1_data[i],
        }));

        const cluster2Data = feature.cluster2_time.map((t, i) => ({
            time: new Date(t),
            value: feature.cluster2_data[i],
        }));

        const allData = [...cluster1Data, ...cluster2Data];
        const xExtent = d3.extent(allData, (d) => d.time) as [Date, Date];
        const yExtent = d3.extent(allData, (d) => d.value) as [number, number];
        const yPadding = (yExtent[1] - yExtent[0]) * 0.15;

        const xScale = d3.scaleTime().domain(xExtent).range([0, innerWidth]);
        const yScale = d3.scaleLinear().domain([yExtent[0] - yPadding, yExtent[1] + yPadding]).range([innerHeight, 0]);

        // Store original scales for zoom reset
        const xScaleOriginal = xScale.copy();
        const yScaleOriginal = yScale.copy();

        // Create clip path for chart area
        const clipId = 'chart-clip-' + Math.random().toString(36).substr(2, 9);
        svg.append('defs').append('clipPath')
            .attr('id', clipId)
            .append('rect')
            .attr('x', 0)
            .attr('y', 0)
            .attr('width', innerWidth)
            .attr('height', innerHeight);

        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        // Chart content group with clip path
        const chartArea = g.append('g').attr('clip-path', `url(#${clipId})`);

        // Grid lines group
        const gridGroup = chartArea.append('g').attr('class', 'grid');

        // Lines group
        const linesGroup = chartArea.append('g').attr('class', 'lines');

        // Points group
        const pointsGroup = chartArea.append('g').attr('class', 'points');

        // Axes groups
        const xAxisGroup = g.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0,${innerHeight})`);
        const yAxisGroup = g.append('g').attr('class', 'y-axis');

        const sorted1 = [...cluster1Data].sort((a, b) => a.time.getTime() - b.time.getTime());
        const sorted2 = [...cluster2Data].sort((a, b) => a.time.getTime() - b.time.getTime());

        // Function to update chart based on current scales
        function updateChart(xS: d3.ScaleTime<number, number>, yS: d3.ScaleLinear<number, number>) {
            // Update grid lines
            gridGroup.selectAll('line.grid').remove();
            gridGroup.selectAll('line.grid')
                .data(yS.ticks(5))
                .enter().append('line')
                .attr('class', 'grid')
                .attr('x1', 0).attr('x2', innerWidth)
                .attr('y1', (d) => yS(d)).attr('y2', (d) => yS(d))
                .attr('stroke', COLORS.grid);

            // Update line generator
            const line = d3.line<{ time: Date; value: number }>()
                .x((d) => xS(d.time))
                .y((d) => yS(d.value))
                .curve(d3.curveMonotoneX);

            // Update lines
            linesGroup.selectAll('path').remove();
            linesGroup.append('path')
                .datum(sorted1)
                .attr('fill', 'none')
                .attr('stroke', COLORS.cluster1)
                .attr('stroke-width', 2)
                .attr('d', line);

            linesGroup.append('path')
                .datum(sorted2)
                .attr('fill', 'none')
                .attr('stroke', COLORS.cluster2)
                .attr('stroke-width', 2)
                .attr('d', line);

            // Update points
            pointsGroup.selectAll('circle').remove();

            pointsGroup.selectAll('.p1')
                .data(cluster1Data)
                .enter().append('circle')
                .attr('class', 'p1')
                .attr('cx', (d) => xS(d.time))
                .attr('cy', (d) => yS(d.value))
                .attr('r', 2)
                .attr('fill', COLORS.cluster1)
                .attr('stroke', 'white')
                .attr('stroke-width', 0.5)
                .attr('opacity', 0.8)
                .style('cursor', 'pointer')
                .on('mouseover', function (event, d) {
                    d3.select(this).attr('r', 4).attr('stroke-width', 1.5);
                    tooltip
                        .style('opacity', 1)
                        .html(`<strong>Cluster 1</strong><br/>Time: ${d3.timeFormat('%Y-%m-%d %H:%M')(d.time)}<br/>Value: ${d.value.toFixed(3)}`)
                        .style('left', (event.pageX + 10) + 'px')
                        .style('top', (event.pageY - 28) + 'px');
                })
                .on('mouseout', function () {
                    d3.select(this).attr('r', 2).attr('stroke-width', 0.5);
                    tooltip.style('opacity', 0);
                });

            pointsGroup.selectAll('.p2')
                .data(cluster2Data)
                .enter().append('circle')
                .attr('class', 'p2')
                .attr('cx', (d) => xS(d.time))
                .attr('cy', (d) => yS(d.value))
                .attr('r', 2)
                .attr('fill', COLORS.cluster2)
                .attr('stroke', 'white')
                .attr('stroke-width', 0.5)
                .attr('opacity', 0.8)
                .style('cursor', 'pointer')
                .on('mouseover', function (event, d) {
                    d3.select(this).attr('r', 4).attr('stroke-width', 1.5);
                    tooltip
                        .style('opacity', 1)
                        .html(`<strong>Cluster 2</strong><br/>Time: ${d3.timeFormat('%Y-%m-%d %H:%M')(d.time)}<br/>Value: ${d.value.toFixed(3)}`)
                        .style('left', (event.pageX + 10) + 'px')
                        .style('top', (event.pageY - 28) + 'px');
                })
                .on('mouseout', function () {
                    d3.select(this).attr('r', 2).attr('stroke-width', 0.5);
                    tooltip.style('opacity', 0);
                });

            // Update axes
            const xAxis = d3.axisBottom(xS).ticks(5).tickFormat(d3.timeFormat('%Y-%m') as any);
            const yAxis = d3.axisLeft(yS).ticks(5);

            xAxisGroup.call(xAxis)
                .call(g => g.select('.domain').attr('stroke', '#ddd'))
                .call(g => g.selectAll('.tick line').attr('stroke', '#ddd'))
                .call(g => g.selectAll('.tick text')
                    .attr('fill', COLORS.textMuted)
                    .attr('font-size', '10px')
                    .attr('transform', 'rotate(-30)')
                    .attr('text-anchor', 'end'));

            yAxisGroup.call(yAxis)
                .call(g => g.select('.domain').attr('stroke', '#ddd'))
                .call(g => g.selectAll('.tick line').attr('stroke', '#ddd'))
                .call(g => g.selectAll('.tick text').attr('fill', COLORS.textMuted).attr('font-size', '10px'));
        }

        // Initial render
        updateChart(xScale, yScale);

        // Setup zoom behavior
        const zoom = d3.zoom<SVGSVGElement, unknown>()
            .scaleExtent([1, 20])
            .translateExtent([[0, 0], [dimensions.width, dimensions.height]])
            .extent([[0, 0], [dimensions.width, dimensions.height]])
            .on('zoom', (event) => {
                const transform = event.transform;
                currentTransformRef.current = transform;

                // Rescale axes based on zoom transform
                const newXScale = transform.rescaleX(xScaleOriginal);
                const newYScale = transform.rescaleY(yScaleOriginal);

                updateChart(newXScale, newYScale);

                // Update zoom state
                setIsZoomed(transform.k !== 1 || transform.x !== 0 || transform.y !== 0);
            });

        zoomRef.current = zoom;
        svg.call(zoom);

        // Apply current transform if exists
        if (currentTransformRef.current && currentTransformRef.current.k !== 1) {
            svg.call(zoom.transform, currentTransformRef.current);
        }

        // Add zoom hint overlay (appears on hover)
        const hintGroup = g.append('g').attr('class', 'zoom-hint').style('opacity', 0);
        hintGroup.append('rect')
            .attr('x', innerWidth - 120)
            .attr('y', 5)
            .attr('width', 115)
            .attr('height', 20)
            .attr('rx', 3)
            .attr('fill', 'rgba(0,0,0,0.6)');
        hintGroup.append('text')
            .attr('x', innerWidth - 62)
            .attr('y', 18)
            .attr('text-anchor', 'middle')
            .attr('fill', 'white')
            .attr('font-size', '10px')
            .text('Scroll to zoom, drag to pan');

        svg.on('mouseenter', () => hintGroup.transition().duration(200).style('opacity', 1));
        svg.on('mouseleave', () => hintGroup.transition().duration(200).style('opacity', 0));

    }, [feature, innerWidth, innerHeight, margin, dimensions.width, dimensions.height]);

    if (!topFeatures || topFeatures.length === 0) {
        return (
            <Box
                ref={containerRef}
                h="100%"
                bg="white"
                borderRadius="4px"
                border="1px solid"
                borderColor="#e0e0e0"
                display="flex"
                flexDirection="column"
            >
                <Box px={4} py={3} borderBottom="1px solid" borderColor="#e0e0e0">
                    <Text fontSize="sm" fontWeight="600" color="#333">
                        Time Series Comparison
                    </Text>
                </Box>
                <Box flex="1" display="flex" alignItems="center" justifyContent="center">
                    <Text color="#888" fontSize="sm">Select clusters to view time series</Text>
                </Box>
            </Box>
        );
    }

    const stat = feature?.statistical_result;

    return (
        <Box
            ref={(el) => { panelRef.current = el; containerRef.current = el; }}
            h="100%"
            bg="white"
            borderRadius="4px"
            border="1px solid"
            borderColor="#e0e0e0"
            display="flex"
            flexDirection="column"
            overflow="hidden"
        >
            {/* Header */}
            <Box px={4} py={3} borderBottom="1px solid" borderColor="#e0e0e0" flexShrink={0}>
                <HStack justify="space-between" align="center">
                    <VStack align="start" spacing={0}>
                        <Text fontSize="sm" fontWeight="600" color="#333">
                            Time Series Comparison
                        </Text>
                        {feature && (
                            <Text fontSize="xs" color="#888">
                                #{feature.rank} {feature.rack}-{feature.variable}
                            </Text>
                        )}
                    </VStack>
                    {/* Legend */}
                    <HStack spacing={4}>
                        <HStack spacing={1}>
                            <Circle size="8px" bg={COLORS.cluster1} />
                            <Text fontSize="10px" color="#888">Cluster 1</Text>
                        </HStack>
                        <HStack spacing={1}>
                            <Circle size="8px" bg={COLORS.cluster2} />
                            <Text fontSize="10px" color="#888">Cluster 2</Text>
                        </HStack>
                        {isZoomed && (
                            <Tooltip label="Reset zoom" placement="top" hasArrow>
                                <IconButton
                                    aria-label="Reset zoom"
                                    icon={<RepeatIcon />}
                                    size="xs"
                                    variant="ghost"
                                    colorScheme="blue"
                                    onClick={handleResetZoom}
                                />
                            </Tooltip>
                        )}
                        <ScreenshotButton targetRef={panelRef} filename="time_series" />
                    </HStack>
                </HStack>
            </Box>

            {/* Chart */}
            <Box flex="1" p={2} overflow="hidden">
                <svg
                    ref={svgRef}
                    width="100%"
                    height="100%"
                    viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
                    preserveAspectRatio="xMidYMid meet"
                    style={{ display: 'block', maxWidth: '100%', maxHeight: '100%' }}
                />
            </Box>

            {/* Navigation & Stats */}
            <Box borderTop="1px solid" borderColor="#e0e0e0" px={4} py={2} flexShrink={0}>
                <HStack justify="space-between" align="center">
                    {/* Stats with significance badge */}
                    {stat && (
                        <HStack spacing={3} fontSize="xs">
                            <Box
                                px={2}
                                py={0.5}
                                borderRadius="3px"
                                bg={stat.p_value < 0.05 ? '#e8f5e9' : '#f5f5f5'}
                                border="1px solid"
                                borderColor={stat.p_value < 0.05 ? '#c8e6c9' : '#e0e0e0'}
                            >
                                <Text
                                    color={stat.p_value < 0.05 ? '#2e7d32' : '#888'}
                                    fontWeight="600"
                                    fontSize="10px"
                                >
                                    {stat.p_value < 0.05 ? '✓ Significant' : 'Not Significant'}
                                </Text>
                            </Box>
                            <Text color="#888">
                                p={stat.p_value.toFixed(3)}, d={stat.cohen_d.toFixed(2)}
                            </Text>
                        </HStack>
                    )}

                    {/* Navigation */}
                    <HStack spacing={2}>
                        <IconButton
                            aria-label="Previous"
                            icon={<ChevronLeftIcon />}
                            size="xs"
                            variant="ghost"
                            isDisabled={currentFeatureIndex === 0}
                            onClick={handlePrev}
                        />
                        <Text fontSize="xs" color="#888" minW="50px" textAlign="center">
                            {currentFeatureIndex + 1} / {topFeatures.length}
                        </Text>
                        <IconButton
                            aria-label="Next"
                            icon={<ChevronRightIcon />}
                            size="xs"
                            variant="ghost"
                            isDisabled={currentFeatureIndex === topFeatures.length - 1}
                            onClick={handleNext}
                        />
                    </HStack>
                </HStack>
            </Box>
        </Box>
    );
}
