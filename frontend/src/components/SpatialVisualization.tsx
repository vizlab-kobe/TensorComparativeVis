/**
 * SpatialVisualization - 空間可視化ルーターコンポーネント
 *
 * ドメイン設定に基づいて適切な空間可視化コンポーネントを切り替える:
 *   - "grid"    → GridHeatmap（HPCラック配置のヒートマップ）
 *   - "geo_map" → GeoMapVis（米国地図上のステーションマーカー）
 */
import { useDashboardStore } from '../store/dashboardStore';
import { Heatmap } from './Heatmap';
import { GeoMapVis } from './GeoMapVis';

export function SpatialVisualization() {
    const { config } = useDashboardStore();
    // ドメイン設定から可視化タイプを取得（デフォルト: grid）
    const vizType = config?.visualization_type ?? 'grid';

    if (vizType === 'geo_map') {
        return <GeoMapVis />;
    }

    // デフォルト: グリッドヒートマップ
    return <Heatmap />;
}
