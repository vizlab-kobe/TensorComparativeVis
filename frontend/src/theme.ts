/**
 * Shared theme constants for the application.
 * Consolidates color definitions previously duplicated across 6+ components.
 */

/** Cluster colors used in scatter plot, feature ranking, time series, and sidebar */
export const CLUSTER_COLORS = {
    cluster1: '#d62728',  // tab10 red
    cluster2: '#1f77b4',  // tab10 blue
} as const;

/** Default class colors for scatter plot points (can be overridden by config) */
export const DEFAULT_CLASS_COLORS = ['#E07B54', '#5B8BD0', '#6BAF6B'];

/** UI chrome colors */
export const UI_COLORS = {
    text: '#333',
    textSecondary: '#666',
    textMuted: '#888',
    border: '#e0e0e0',
    bgSubtle: '#fafafa',
    accent: '#555',
    accentHover: '#444',
    grid: '#f0f0f0',
    axis: '#888',
} as const;

/** Section header accent colors for AI interpretation cards */
export const SECTION_COLORS: Record<string, string> = {
    'key findings': '#2d9596',
    'pattern analysis': '#27ae60',
    'statistical summary': '#9467bd',
    'caveats': '#8c564b',
};

/** Map section title (lowercased) to its accent color */
export function getSectionColor(title: string): string {
    const key = title.toLowerCase();
    return SECTION_COLORS[key] || '#555';
}
