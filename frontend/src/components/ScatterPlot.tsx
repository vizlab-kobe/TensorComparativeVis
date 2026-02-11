/**
 * D3.js Scatter Plot - Academic research theme
 * Fixed layout, no axes/grid, stable scale
 */
import { useEffect, useRef, useCallback, useState } from 'react';
import { Box, Text, Spinner, HStack, Circle } from '@chakra-ui/react';
import * as d3 from 'd3';
import { useDashboardStore } from '../store/dashboardStore';
import type { EmbeddingPoint } from '../types';
import { ScreenshotButton } from './ScreenshotButton';
import { CLUSTER_COLORS, DEFAULT_CLASS_COLORS, UI_COLORS } from '../theme';

// Map theme colors to component usage
const COLORS = {
    class: DEFAULT_CLASS_COLORS,
    cluster1: CLUSTER_COLORS.cluster1,
    cluster2: CLUSTER_COLORS.cluster2,
    text: UI_COLORS.text,
    textMuted: UI_COLORS.textMuted,
};

export function ScatterPlot() {
    const containerRef = useRef<HTMLDivElement>(null);
    const panelRef = useRef<HTMLDivElement>(null);
    const svgRef = useRef<SVGSVGElement>(null);
    const [dimensions, setDimensions] = useState({ width: 400, height: 300 });

    const {
        embeddingData,
        config,
        clusters,
        selectCluster1,
        selectCluster2,
    } = useDashboardStore();

    const margin = { top: 30, right: 30, bottom: 30, left: 30 };
    const innerWidth = dimensions.width - margin.left - margin.right;
    const innerHeight = dimensions.height - margin.top - margin.bottom;

    const classColors = config?.colors.class_colors || COLORS.class;
    const cluster1Color = COLORS.cluster1;
    const cluster2Color = COLORS.cluster2;

    // Responsive sizing - but keep consistent layout
    useEffect(() => {
        if (!containerRef.current) return;

        const resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                // Subtract header height (approximately 50px) for chart area
                setDimensions({ width: Math.max(width - 16, 200), height: Math.max(height - 66, 200) });
            }
        });

        resizeObserver.observe(containerRef.current);
        return () => resizeObserver.disconnect();
    }, []);

    const handleBrushEnd = useCallback(
        (event: d3.D3BrushEvent<EmbeddingPoint>) => {
            if (!event.selection || embeddingData.length === 0) return;

            const [[x0, y0], [x1, y1]] = event.selection as [[number, number], [number, number]];

            // Calculate scale from fresh data extent
            const xExtent = d3.extent(embeddingData, (d: EmbeddingPoint) => d.x) as [number, number];
            const yExtent = d3.extent(embeddingData, (d: EmbeddingPoint) => d.y) as [number, number];
            const xRange = xExtent[1] - xExtent[0];
            const yRange = yExtent[1] - yExtent[0];
            const xPadding = xRange * 0.15;
            const yPadding = yRange * 0.15;

            const xScale = d3.scaleLinear()
                .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
                .range([0, innerWidth]);
            const yScale = d3.scaleLinear()
                .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
                .range([innerHeight, 0]);

            const cluster1Set = new Set(clusters.cluster1 || []);
            const cluster2Set = new Set(clusters.cluster2 || []);

            const selectedIndices = embeddingData
                .filter((d) => {
                    if (cluster1Set.has(d.index) || cluster2Set.has(d.index)) return false;
                    const px = xScale(d.x);
                    const py = yScale(d.y);
                    return px >= x0 && px <= x1 && py >= y0 && py <= y1;
                })
                .map((d) => d.index);

            if (selectedIndices.length === 0) return;

            if (!clusters.cluster1) {
                selectCluster1(selectedIndices);
            } else if (!clusters.cluster2) {
                selectCluster2(selectedIndices);
            } else {
                selectCluster1(selectedIndices);
            }

            d3.select(svgRef.current).select('.brush').call(d3.brush().clear as any);
        },
        [embeddingData, clusters, innerWidth, innerHeight, selectCluster1, selectCluster2]
    );

    useEffect(() => {
        if (!svgRef.current || embeddingData.length === 0 || innerWidth <= 0 || innerHeight <= 0) return;

        const svg = d3.select(svgRef.current);
        svg.selectAll('*').remove();

        // Calculate extent from ALL data points
        const xExtent = d3.extent(embeddingData, (d) => d.x) as [number, number];
        const yExtent = d3.extent(embeddingData, (d) => d.y) as [number, number];

        // Add generous padding to ensure all points fit within bounds
        const xRange = xExtent[1] - xExtent[0];
        const yRange = yExtent[1] - yExtent[0];
        const xPadding = xRange * 0.15;
        const yPadding = yRange * 0.15;

        // Create scale directly from data - no caching
        const xScale = d3.scaleLinear()
            .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
            .range([0, innerWidth]);

        const yScale = d3.scaleLinear()
            .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
            .range([innerHeight, 0]);

        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        // No grid or axes - just points

        // Points
        const cluster1Set = new Set(clusters.cluster1 || []);
        const cluster2Set = new Set(clusters.cluster2 || []);

        // Helper: Draw ellipse around cluster points (with semi-transparent fill)
        const drawClusterEllipse = (points: EmbeddingPoint[], color: string) => {
            if (points.length < 2) return;

            // Calculate bounding ellipse
            const xCoords = points.map(d => xScale(d.x));
            const yCoords = points.map(d => yScale(d.y));

            const cx = (Math.min(...xCoords) + Math.max(...xCoords)) / 2;
            const cy = (Math.min(...yCoords) + Math.max(...yCoords)) / 2;

            // Use half the range plus padding for radii
            const rx = (Math.max(...xCoords) - Math.min(...xCoords)) / 2 + 25;
            const ry = (Math.max(...yCoords) - Math.min(...yCoords)) / 2 + 25;

            g.append('ellipse')
                .attr('cx', cx)
                .attr('cy', cy)
                .attr('rx', rx)
                .attr('ry', ry)
                .attr('fill', color)
                .attr('fill-opacity', 0.12)
                .attr('stroke', color)
                .attr('stroke-width', 2)
                .attr('opacity', 0.9);
        };

        // Draw ellipses FIRST (as background layer)
        const cluster1Points = embeddingData.filter((d) => cluster1Set.has(d.index));
        const cluster2Points = embeddingData.filter((d) => cluster2Set.has(d.index));
        drawClusterEllipse(cluster1Points, cluster1Color);
        drawClusterEllipse(cluster2Points, cluster2Color);

        // THEN draw ALL points on top with their original label colors
        g.selectAll('.point')
            .data(embeddingData)
            .enter().append('circle')
            .attr('class', 'point')
            .attr('cx', (d) => xScale(d.x))
            .attr('cy', (d) => yScale(d.y))
            .attr('r', 3)
            .attr('fill', (d) => classColors[d.label % classColors.length])
            .attr('stroke', 'white')
            .attr('stroke-width', 0.5)
            .attr('opacity', 0.9);

        // Brush
        const brush = d3.brush<EmbeddingPoint>()
            .extent([[0, 0], [innerWidth, innerHeight]])
            .on('end', handleBrushEnd);
        g.append('g').attr('class', 'brush').call(brush);

    }, [embeddingData, clusters, innerWidth, innerHeight, margin, classColors, cluster1Color, cluster2Color, handleBrushEnd]);

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
                    <Text fontSize="sm" fontWeight="600" color="#333">
                        Scatter Plot
                    </Text>
                    {/* Legend */}
                    <HStack spacing={4}>
                        {classColors.map((color, i) => (
                            <HStack key={i} spacing={1}>
                                <Circle size="8px" bg={color} />
                                <Text fontSize="10px" color="#888">{config?.class_labels?.[i] || `Class ${i + 1}`}</Text>
                            </HStack>
                        ))}
                        <ScreenshotButton targetRef={panelRef} filename="scatter_plot" />
                    </HStack>
                </HStack>
            </Box>

            {/* Chart */}
            <Box flex="1" p={2} overflow="hidden" position="relative">
                {embeddingData.length === 0 ? (
                    <Box position="absolute" top="50%" left="50%" transform="translate(-50%, -50%)" textAlign="center">
                        <Spinner size="md" color="#555" mb={3} thickness="2px" />
                        <Text color="#888" fontSize="sm">Click "Execute" to compute embedding</Text>
                    </Box>
                ) : (
                    <svg
                        ref={svgRef}
                        width="100%"
                        height="100%"
                        viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
                        preserveAspectRatio="xMidYMid meet"
                        style={{ display: 'block', maxWidth: '100%', maxHeight: '100%' }}
                    />
                )}
            </Box>
        </Box>
    );
}
