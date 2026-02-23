/**
 * 地理マップ可視化コンポーネント（D3.js + TopoJSON）
 *
 * 大気データドメイン用の空間可視化。米国地図上にモニタリング
 * ステーションをマーカーとして描画し、寄与度の大きさを
 * 円のサイズと色で表現する。
 *
 * 機能:
 *   - D3 Albers USA 投影による米国地図描画
 *   - TopoJSON (us-atlas) から州境界を取得
 *   - バックエンドからステーション座標を遅延読み込み
 *   - YlOrRd カラースケール + 面積比例スケールのマーカー
 *   - ホバー時にステーション名と寄与度をツールチップ表示
 *   - カラーレジェンド（グラデーションバー）
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import { Box, Text, Center, Select, HStack } from '@chakra-ui/react';
import * as d3 from 'd3';
import { feature } from 'topojson-client';
import type { Topology, GeometryCollection } from 'topojson-specification';
import { useDashboardStore } from '../store/dashboardStore';
import { getCoordinates, type Coordinate } from '../api/client';
import { ScreenshotButton } from './ScreenshotButton';
import { UI_COLORS } from '../theme';

const COLORS = {
    text: UI_COLORS.text,
    textMuted: UI_COLORS.textMuted,
    land: UI_COLORS.grid,     // reuse grid color for map land
    border: '#ccc',
    stateBorder: '#ddd',
};

// US Atlas TopoJSON URL (free CDN)
const US_ATLAS_URL = 'https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json';

export function GeoMapVis() {
    const containerRef = useRef<HTMLDivElement>(null);
    const panelRef = useRef<HTMLDivElement>(null);
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const svgRef = useRef<SVGSVGElement>(null);
    const [dimensions, setDimensions] = useState({ width: 500, height: 350 });
    const [coordinates, setCoordinates] = useState<Coordinate[]>([]);
    const [usGeo, setUsGeo] = useState<GeoJSON.FeatureCollection | null>(null);
    const { contributionMatrix, selectedVariable, setSelectedVariable, config } = useDashboardStore();

    const variables = config?.variables || [];

    // Load coordinates and US map data on mount
    useEffect(() => {
        async function loadData() {
            try {
                const coordResult = await getCoordinates();
                if (coordResult.available) {
                    setCoordinates(coordResult.coordinates);
                }
            } catch (error) {
                console.error('Failed to load coordinates:', error);
            }

            try {
                const response = await fetch(US_ATLAS_URL);
                const topology = (await response.json()) as Topology<{
                    states: GeometryCollection;
                }>;
                const states = feature(
                    topology,
                    topology.objects.states
                ) as unknown as GeoJSON.FeatureCollection;
                setUsGeo(states);
            } catch (error) {
                console.error('Failed to load US map:', error);
            }
        }
        loadData();
    }, []);

    // Responsive sizing
    useEffect(() => {
        if (!chartContainerRef.current) return;

        const resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                setDimensions({
                    width: Math.max(width - 8, 200),
                    height: Math.max(height - 8, 150),
                });
            }
        });

        resizeObserver.observe(chartContainerRef.current);
        return () => resizeObserver.disconnect();
    }, []);

    // Draw the map
    const draw = useCallback(() => {
        if (!svgRef.current || !contributionMatrix || !usGeo || coordinates.length === 0) return;

        const svg = d3.select(svgRef.current);
        svg.selectAll('*').remove();

        const margin = { top: 5, right: 60, bottom: 5, left: 5 };
        const width = dimensions.width - margin.left - margin.right;
        const height = dimensions.height - margin.top - margin.bottom;

        if (width <= 0 || height <= 0) return;

        // Extract contribution data for the selected variable
        const data = contributionMatrix.map((row) => Math.abs(row[selectedVariable]));
        const maxVal = d3.max(data) || 1;

        // Color scale
        const colorScale = d3.scaleSequential()
            .domain([0, maxVal])
            .interpolator(d3.interpolateYlOrRd);

        // Size scale for circles (min 3, max scales with view)
        const maxRadius = Math.min(width, height) * 0.035;
        const radiusScale = d3.scaleSqrt()
            .domain([0, maxVal])
            .range([2.5, Math.max(maxRadius, 6)]);

        // Projection
        const projection = d3.geoAlbersUsa()
            .fitSize([width, height], usGeo);

        const path = d3.geoPath().projection(projection);

        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        // Draw state boundaries
        g.selectAll('.state')
            .data(usGeo.features)
            .enter()
            .append('path')
            .attr('class', 'state')
            .attr('d', path)
            .attr('fill', COLORS.land)
            .attr('stroke', COLORS.stateBorder)
            .attr('stroke-width', 0.5);

        // Create tooltip
        const tooltip = d3.select('body').append('div')
            .attr('class', 'geomap-tooltip')
            .style('position', 'absolute')
            .style('visibility', 'hidden')
            .style('background', 'rgba(0,0,0,0.85)')
            .style('color', 'white')
            .style('padding', '8px 12px')
            .style('border-radius', '6px')
            .style('font-size', '11px')
            .style('pointer-events', 'none')
            .style('z-index', '1000')
            .style('box-shadow', '0 2px 8px rgba(0,0,0,0.3)');

        // Draw station markers
        const stationData = coordinates
            .map((coord) => {
                const projected = projection([coord.lon, coord.lat]);
                if (!projected) return null;
                return {
                    ...coord,
                    x: projected[0],
                    y: projected[1],
                    value: data[coord.index] ?? 0,
                };
            })
            .filter((d): d is NonNullable<typeof d> => d !== null);

        // Sort by value so smaller circles render on top of larger ones
        stationData.sort((a, b) => b.value - a.value);

        g.selectAll('.station')
            .data(stationData)
            .enter()
            .append('circle')
            .attr('class', 'station')
            .attr('cx', (d) => d.x)
            .attr('cy', (d) => d.y)
            .attr('r', (d) => radiusScale(d.value))
            .attr('fill', (d) => colorScale(d.value))
            .attr('stroke', '#fff')
            .attr('stroke-width', 0.8)
            .attr('opacity', 0.85)
            .style('cursor', 'pointer')
            .on('mouseover', function (_event, d) {
                tooltip
                    .style('visibility', 'visible')
                    .html(`<strong>${d.name}</strong><br/>Contribution: ${d.value.toFixed(4)}`);
                d3.select(this)
                    .attr('stroke', '#333')
                    .attr('stroke-width', 2)
                    .attr('opacity', 1);
            })
            .on('mousemove', function (_event) {
                tooltip
                    .style('top', _event.pageY - 10 + 'px')
                    .style('left', _event.pageX + 10 + 'px');
            })
            .on('mouseout', function () {
                tooltip.style('visibility', 'hidden');
                d3.select(this)
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 0.8)
                    .attr('opacity', 0.85);
            });

        // Color legend
        const legendHeight = height * 0.4;
        const legendG = svg.append('g')
            .attr('transform', `translate(${dimensions.width - margin.right + 8}, ${margin.top + (height - legendHeight) / 2})`);

        const defs = svg.append('defs');
        const gradient = defs.append('linearGradient')
            .attr('id', 'geo-grad')
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
            .attr('fill', 'url(#geo-grad)')
            .attr('rx', 1);

        const legendScale = d3.scaleLinear().domain([0, maxVal]).range([legendHeight, 0]);
        legendG.append('g')
            .attr('transform', 'translate(8, 0)')
            .call(d3.axisRight(legendScale).ticks(3).tickFormat(d3.format('.2f')))
            .call((g) => g.select('.domain').remove())
            .call((g) => g.selectAll('.tick line').attr('stroke', '#ddd'))
            .call((g) => g.selectAll('.tick text').attr('fill', COLORS.textMuted).attr('font-size', '8px'));

        // Cleanup
        return () => {
            d3.selectAll('.geomap-tooltip').remove();
        };
    }, [contributionMatrix, selectedVariable, dimensions, usGeo, coordinates]);

    useEffect(() => {
        const cleanup = draw();
        return cleanup;
    }, [draw]);

    if (!contributionMatrix) {
        return (
            <Box ref={containerRef} h="100%" display="flex" flexDirection="column">
                <Box px={4} py={3} borderBottom="1px solid" borderColor="#e0e0e0">
                    <HStack spacing={2}>
                        <Text fontSize="xs" color="#888">Variable:</Text>
                        <Select
                            size="xs"
                            w="90px"
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
        <Box
            ref={(el) => {
                panelRef.current = el;
                containerRef.current = el;
            }}
            h="100%"
            display="flex"
            flexDirection="column"
            overflow="hidden"
        >
            {/* Header with variable selector */}
            <Box px={3} py={1} borderBottom="1px solid" borderColor="#e0e0e0" flexShrink={0}>
                <HStack spacing={2}>
                    <Text fontSize="xs" color="#888">Variable:</Text>
                    <Select
                        size="xs"
                        w="90px"
                        borderColor="#ddd"
                        fontSize="xs"
                        value={selectedVariable}
                        onChange={(e) => setSelectedVariable(Number(e.target.value))}
                    >
                        {variables.map((v, i) => (
                            <option key={i} value={i}>{v}</option>
                        ))}
                    </Select>
                    <ScreenshotButton targetRef={panelRef} filename="geo_map" />
                </HStack>
            </Box>

            {/* Map */}
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
