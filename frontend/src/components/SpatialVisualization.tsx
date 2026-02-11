/**
 * SpatialVisualization - Router component that renders
 * the appropriate spatial visualization based on domain config.
 *
 * - "grid"    → GridHeatmap (HPC rack layout)
 * - "geo_map" → GeoMapVis  (US map with station markers)
 */
import { useDashboardStore } from '../store/dashboardStore';
import { Heatmap } from './Heatmap';
import { GeoMapVis } from './GeoMapVis';

export function SpatialVisualization() {
    const { config } = useDashboardStore();
    const vizType = config?.visualization_type ?? 'grid';

    if (vizType === 'geo_map') {
        return <GeoMapVis />;
    }

    // Default: grid heatmap
    return <Heatmap />;
}
