/**
 * グリッドヒートマップコンポーネント（D3.js）
 *
 * 寄与度行列の空間分布をヒートマップとして可視化する。
 * HPCドメインでは 36行×24列 のラック配置に対応する。
 *
 * 機能:
 *   - 変数セレクターで表示変数を切り替え
 *   - YlOrRd カラースケールで寄与度の大きさを表現
 *   - ホバー時にラックラベルと寄与度値をツールチップ表示
 *   - カラーレジェンド（グラデーションバー）
 *   - ResizeObserver によるレスポンシブリサイズ
 */
import { useEffect, useRef, useState } from 'react';
import { Box, Text, Center, Select, HStack } from '@chakra-ui/react';
import * as d3 from 'd3';
import { useDashboardStore } from '../store/dashboardStore';
import { ScreenshotButton } from './ScreenshotButton';

const COLS = 24;
const ROWS = 36;

// Academic color palette - using blue sequential scale
const COLORS = {
    text: '#333',
    textMuted: '#888',
};

export function Heatmap() {
    const containerRef = useRef<HTMLDivElement>(null);
    const panelRef = useRef<HTMLDivElement>(null);
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const svgRef = useRef<SVGSVGElement>(null);
    const [dimensions, setDimensions] = useState({ width: 400, height: 300 });
    const { contributionMatrix, selectedVariable, setSelectedVariable, config } = useDashboardStore();

    const variables = config?.variables || ['AirIn', 'AirOut', 'CPU', 'Water'];

    // Responsive sizing - observe the chart container, not the whole component
    useEffect(() => {
        if (!chartContainerRef.current) return;

        const resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                // Use full available space with minimal padding
                setDimensions({ width: Math.max(width - 8, 200), height: Math.max(height - 8, 200) });
            }
        });

        resizeObserver.observe(chartContainerRef.current);
        return () => resizeObserver.disconnect();
    }, []);

    useEffect(() => {
        if (!svgRef.current || !contributionMatrix) return;

        const svg = d3.select(svgRef.current);
        svg.selectAll('*').remove();

        // Margins increased for x-axis labels and legend visibility
        const margin = { top: 10, right: 60, bottom: 40, left: 25 };
        const width = dimensions.width - margin.left - margin.right;
        const height = dimensions.height - margin.top - margin.bottom;

        if (width <= 0 || height <= 0) return;

        const data = contributionMatrix.map((row) => row[selectedVariable]);
        const values: { row: number; col: number; value: number }[] = [];

        for (let i = 0; i < Math.min(data.length, ROWS * COLS); i++) {
            values.push({
                row: Math.floor(i / COLS),
                col: i % COLS,
                value: Math.abs(data[i]),
            });
        }

        const maxVal = d3.max(values, (d) => d.value) || 1;

        // Custom color scale - white to warm orange-red for better visibility of positive values
        const colorScale = d3.scaleSequential()
            .domain([0, maxVal])
            .interpolator(d3.interpolateYlOrRd);

        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        const cellWidth = width / COLS;
        const cellHeight = height / ROWS;

        // Create tooltip
        const tooltip = d3.select('body').append('div')
            .attr('class', 'heatmap-tooltip')
            .style('position', 'absolute')
            .style('visibility', 'hidden')
            .style('background', 'rgba(0,0,0,0.8)')
            .style('color', 'white')
            .style('padding', '6px 10px')
            .style('border-radius', '4px')
            .style('font-size', '11px')
            .style('pointer-events', 'none')
            .style('z-index', '1000');

        // Helper to convert index to rack label (A01, B02, etc.)
        const indexToRackLabel = (col: number, row: number) => {
            const letter = String.fromCharCode(65 + col); // A-X
            const number = String(row + 1).padStart(2, '0'); // 01-36
            return `${letter}${number}`;
        };

        // Cells with hover
        g.selectAll('.cell')
            .data(values)
            .enter().append('rect')
            .attr('class', 'cell')
            .attr('x', (d) => d.col * cellWidth)
            .attr('y', (d) => d.row * cellHeight)
            .attr('width', Math.max(cellWidth - 0.5, 1))
            .attr('height', Math.max(cellHeight - 0.5, 1))
            .attr('fill', (d) => colorScale(d.value))
            .attr('rx', 0.5)
            .style('cursor', 'pointer')
            .on('mouseover', function (event, d) {
                const rackLabel = indexToRackLabel(d.col, d.row);
                tooltip
                    .style('visibility', 'visible')
                    .html(`<strong>${rackLabel}</strong><br/>Contribution: ${d.value.toFixed(4)}`);
                d3.select(this).attr('stroke', '#333').attr('stroke-width', 1);
            })
            .on('mousemove', function (event) {
                tooltip
                    .style('top', (event.pageY - 10) + 'px')
                    .style('left', (event.pageX + 10) + 'px');
            })
            .on('mouseout', function () {
                tooltip.style('visibility', 'hidden');
                d3.select(this).attr('stroke', 'none');
            });

        // X axis labels (every 6 columns: A, G, M, S)
        const xLabelCols = [0, 6, 12, 18];
        g.selectAll('.x-label')
            .data(xLabelCols)
            .enter().append('text')
            .attr('class', 'x-label')
            .attr('x', (d) => d * cellWidth + cellWidth / 2)
            .attr('y', height + 15)
            .attr('dy', '0.35em')
            .attr('text-anchor', 'middle')
            .attr('fill', COLORS.textMuted)
            .attr('font-size', '9px')
            .attr('font-family', 'Inter, -apple-system, sans-serif')
            .text((d) => String.fromCharCode(65 + d));

        // Y axis labels (every 6th row: 1, 7, 13, 19, 25, 31)
        const yLabelRows = [0, 6, 12, 18, 24, 30];
        g.selectAll('.y-label')
            .data(yLabelRows)
            .enter().append('text')
            .attr('class', 'y-label')
            .attr('x', -4)
            .attr('y', (d) => d * cellHeight + cellHeight / 2)
            .attr('dy', '0.35em')
            .attr('text-anchor', 'end')
            .attr('fill', COLORS.textMuted)
            .attr('font-size', '9px')
            .attr('font-family', 'Inter, -apple-system, sans-serif')
            .text((d) => d + 1);

        // Cleanup tooltip on unmount
        return () => {
            d3.selectAll('.heatmap-tooltip').remove();
        };

        // Color legend
        const legendHeight = height * 0.5;
        const legendG = svg.append('g')
            .attr('transform', `translate(${dimensions.width - margin.right + 8}, ${margin.top + (height - legendHeight) / 2})`);

        const defs = svg.append('defs');
        const gradient = defs.append('linearGradient')
            .attr('id', 'hm-grad')
            .attr('x1', '0%').attr('y1', '100%')
            .attr('x2', '0%').attr('y2', '0%');

        gradient.selectAll('stop')
            .data(d3.range(0, 1.1, 0.1))
            .enter().append('stop')
            .attr('offset', (d) => `${d * 100}%`)
            .attr('stop-color', (d) => colorScale(d * maxVal));

        legendG.append('rect')
            .attr('width', 8)
            .attr('height', legendHeight)
            .attr('fill', 'url(#hm-grad)')
            .attr('rx', 1);

        const legendScale = d3.scaleLinear().domain([0, maxVal]).range([legendHeight, 0]);
        legendG.append('g')
            .attr('transform', 'translate(8, 0)')
            .call(d3.axisRight(legendScale).ticks(3).tickFormat(d3.format('.2f')))
            .call(g => g.select('.domain').remove())
            .call(g => g.selectAll('.tick line').attr('stroke', '#ddd'))
            .call(g => g.selectAll('.tick text').attr('fill', COLORS.textMuted).attr('font-size', '8px'));

    }, [contributionMatrix, selectedVariable, dimensions]);

    if (!contributionMatrix) {
        return (
            <Box ref={containerRef} h="100%" display="flex" flexDirection="column">
                <Box px={4} py={3} borderBottom="1px solid" borderColor="#e0e0e0">
                    <HStack spacing={2}>
                        <Text fontSize="xs" color="#888">Variable:</Text>
                        <Select
                            size="xs"
                            w="80px"
                            borderColor="#ddd"
                            fontSize="xs"
                            value={selectedVariable}
                            onChange={(e) => setSelectedVariable(Number(e.target.value))}
                            isDisabled
                        >
                            {variables.map((v, i) => (
                                <option key={i} value={i}>{v}</option>
                            ))}
                        </Select>
                    </HStack>
                </Box>
                <Center flex="1">
                    <Text color="#888" fontSize="sm">Select two clusters</Text>
                </Center>
            </Box>
        );
    }

    return (
        <Box ref={(el) => { panelRef.current = el; containerRef.current = el; }} h="100%" display="flex" flexDirection="column" overflow="hidden">
            {/* Compact header with variable selector */}
            <Box px={3} py={1} borderBottom="1px solid" borderColor="#e0e0e0" flexShrink={0}>
                <HStack spacing={2}>
                    <Text fontSize="xs" color="#888">Variable:</Text>
                    <Select
                        size="xs"
                        w="80px"
                        borderColor="#ddd"
                        fontSize="xs"
                        value={selectedVariable}
                        onChange={(e) => setSelectedVariable(Number(e.target.value))}
                    >
                        {variables.map((v, i) => (
                            <option key={i} value={i}>{v}</option>
                        ))}
                    </Select>
                    <ScreenshotButton targetRef={panelRef} filename="heatmap" />
                </HStack>
            </Box>

            {/* Chart - takes all remaining space */}
            <Box ref={chartContainerRef} flex="1" minH={0} overflow="visible" p={1}>
                <svg
                    ref={svgRef}
                    width="100%"
                    height="100%"
                    viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
                    preserveAspectRatio="xMidYMid meet"
                    style={{ display: 'block', maxWidth: '100%', maxHeight: '100%' }}
                />
            </Box>
        </Box>
    );
}
