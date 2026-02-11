"""
AirData Domain - Configuration for US Air Quality monitoring station data.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from pathlib import Path
from .base_domain import BaseDomain


class AirDataDomain(BaseDomain):
    """Domain configuration for US Air Quality tensor data."""
    
    _VARIABLES = ['CO2', 'NO', 'Ozone', 'PM10', 'PM2.5']
    
    def __init__(self, data_dir: str = None):
        """Initialize and optionally load coordinate data for labels."""
        self._coordinates = None
        self._data_dir = data_dir
    
    @property
    def name(self) -> str:
        return "AirData"
    
    @property
    def data_dir(self) -> str:
        return "data/processed/AirData"
    
    @property
    def variables(self) -> List[str]:
        return self._VARIABLES
    
    @property
    def grid_shape(self) -> Tuple[int, int]:
        # 55 locations x 5 variables, displayed as a flat list (no 2D grid)
        return (55, 5)
    
    @property
    def file_mapping(self) -> Dict[str, str]:
        """Map logical file names to actual file names on disk."""
        return {
            'tensor_X': 'tensor_data_X_air.npy',
            'tensor_y': 'tensor_data_y_air.npy',
            'time_axis': 'time_axis.npy',
            'time_original': 'tensor_data_X_air.npy',  # Use X as original (no separate original file)
            'coordinates': 'coordinates.npy',
        }
    
    def _load_coordinates(self):
        """Lazily load coordinate data."""
        if self._coordinates is None and self._data_dir:
            coord_path = Path(self._data_dir) / self.file_mapping['coordinates']
            if coord_path.exists():
                self._coordinates = np.load(str(coord_path), allow_pickle=True)
        return self._coordinates
    
    def index_to_label(self, spatial_index: int) -> str:
        """Convert spatial index to station name."""
        coords = self._load_coordinates()
        if coords is not None and spatial_index < len(coords):
            return str(coords[spatial_index][2])  # name field
        return f"Station-{spatial_index + 1}"
    
    def label_to_index(self, label: str) -> int:
        """Convert station name to spatial index."""
        coords = self._load_coordinates()
        if coords is not None:
            for i, row in enumerate(coords):
                if str(row[2]) == label:
                    return i
        # Fallback: parse "Station-N"
        if label.startswith("Station-"):
            return int(label.split("-")[1]) - 1
        return 0
    
    @property
    def domain_knowledge(self) -> str:
        return """## US Air Quality Data Analysis Domain Knowledge

### Variables
- CO2: Carbon dioxide concentration
- NO: Nitrogen monoxide concentration
- Ozone: Ground-level ozone concentration
- PM10: Particulate matter (diameter ≤ 10μm)
- PM2.5: Fine particulate matter (diameter ≤ 2.5μm)

### Spatial Structure
- Data from 55 monitoring stations across the United States
- Stations are geographically distributed, with varying local conditions
- Spatial patterns may reflect regional pollution sources, meteorological patterns, or geographical features

### Temporal Patterns
- Weekly aggregated data covering the year 2018
- Time points are labeled by quarter (Q1-Q4)
- Seasonal variations are expected due to weather patterns and human activity cycles
- Ozone tends to peak in summer; PM levels may rise in winter or during wildfire seasons
"""
    
    def _format_features(self, features: List[Dict]) -> str:
        """Format features for prompt."""
        lines = []
        for f in features:
            stat = f.get('statistical_result', {})
            direction = "higher in C1" if stat.get('mean_diff', 0) > 0 else "lower in C1"
            sig = "*" if stat.get('p_value', 1) < 0.05 else ("+" if stat.get('p_value', 1) < 0.1 else "")
            effect = stat.get('effect_size', 'N/A')
            
            lines.append(
                f"- {f.get('rack', 'N/A')}/{f.get('variable', 'N/A')}: "
                f"score={f.get('score', 0):.3f}, {direction}, "
                f"effect={effect}{sig}"
            )
        return "\n".join(lines)
    
    def build_interpretation_prompt(
        self,
        top_features: List[Dict],
        cluster1_size: int,
        cluster2_size: int,
        preprocessed: Dict[str, Any]
    ) -> str:
        """Build LLM prompt for air quality cluster interpretation."""
        features_text = self._format_features(top_features[:30])
        
        return f"""
You are an AI assistant that generates structured summaries of air quality data cluster analysis results.
Your role is to organize and describe the data clearly - NOT to make causal claims or predictions.

{self.domain_knowledge}

## Analysis Context
- Red Cluster (Cluster 1): {cluster1_size} time points (weekly observations)
- Blue Cluster (Cluster 2): {cluster2_size} time points (weekly observations)

## Preprocessed Statistics
- Significant features: {preprocessed['significant_count']}/{preprocessed['total_count']}
- Dominant pollutant type: {preprocessed['dominant_variable']}
- Variable distribution: {preprocessed['variable_distribution']}
- Average effect size: {preprocessed['avg_effect_size']}
- Spatial pattern: {preprocessed['rack_concentration']}
- Co-occurring pollutants at same station: {len(preprocessed['co_occurrences'])} cases

## Top Features (up to 30, by contribution score)
{features_text}

## Output Instructions
Generate a JSON response with the following structure. Each section should contain natural language text (2-3 sentences) in ENGLISH for academic publication.
Do NOT use any special formatting like brackets or markdown. Write in plain text.

{{
  "sections": [
    {{
      "title": "Key Findings",
      "text": "Summarize the most important differences. Which pollutants show the largest differences? Are differences concentrated in specific regions or distributed across stations?",
      "highlights": []
    }},
    {{
      "title": "Statistical Summary",
      "text": "Describe the statistical evidence. How many features are statistically significant? What are the effect sizes?",
      "highlights": []
    }},
    {{
      "title": "Caveats",
      "text": "Note limitations. Mention cluster size imbalance if present. Note if many features lack statistical significance.",
      "highlights": []
    }}
  ]
}}

## Requirements
- Output ONLY valid JSON, no other text
- Text should be in ENGLISH (for academic publication)
- Each section should be 2-3 sentences
- Focus on DESCRIBING patterns, not explaining causation
- Do NOT use brackets, asterisks, arrows, or any special formatting - plain text only
"""
    
    def build_comparison_prompt(
        self,
        analysis_a: Dict[str, Any],
        analysis_b: Dict[str, Any]
    ) -> str:
        """Build LLM prompt for comparing two air quality analyses."""
        same_cluster1 = analysis_a.get('cluster1_size') == analysis_b.get('cluster1_size')
        same_cluster2 = analysis_a.get('cluster2_size') == analysis_b.get('cluster2_size')
        
        if same_cluster1 and same_cluster2:
            context_msg = "Both analyses use IDENTICAL cluster selections."
        elif same_cluster1:
            context_msg = "Both analyses share the SAME Red Cluster (C1). The comparison involves different Blue Cluster (C2) selections."
        elif same_cluster2:
            context_msg = "Both analyses share the SAME Blue Cluster (C2). The comparison involves different Red Cluster (C1) selections."
        else:
            context_msg = "The analyses use completely different cluster selections."
        
        features_a = analysis_a.get('top_features', [])[:10]
        features_b = analysis_b.get('top_features', [])[:10]
        
        feature_set_a = set(f"{f.get('rack')}-{f.get('variable')}" for f in features_a)
        feature_set_b = set(f"{f.get('rack')}-{f.get('variable')}" for f in features_b)
        
        common = feature_set_a & feature_set_b
        only_a = feature_set_a - feature_set_b
        only_b = feature_set_b - feature_set_a
        
        return f"""
You are comparing two cluster analysis results from a US air quality data visualization system.

## Comparison Context
{context_msg}

## Analysis A
- Red Cluster size: {analysis_a.get('cluster1_size')} weekly observations
- Blue Cluster size: {analysis_a.get('cluster2_size')} weekly observations
- Significant features: {analysis_a.get('significant_count', 'N/A')}
- Top pollutants: {', '.join(analysis_a.get('top_variables', [])[:5])}
- Top stations: {', '.join(analysis_a.get('top_racks', [])[:5])}

## Analysis B
- Red Cluster size: {analysis_b.get('cluster1_size')} weekly observations
- Blue Cluster size: {analysis_b.get('cluster2_size')} weekly observations
- Significant features: {analysis_b.get('significant_count', 'N/A')}
- Top pollutants: {', '.join(analysis_b.get('top_variables', [])[:5])}
- Top stations: {', '.join(analysis_b.get('top_racks', [])[:5])}

## Feature Overlap
- Common: {len(common)} - {list(common)[:5]}
- Only in A: {len(only_a)} - {list(only_a)[:5]}
- Only in B: {len(only_b)} - {list(only_b)[:5]}

Generate a JSON comparison:
{{
  "sections": [
    {{"title": "Comparison Overview", "text": "...", "highlights": []}},
    {{"title": "Feature Differences", "text": "...", "highlights": []}},
    {{"title": "Implications", "text": "...", "highlights": []}}
  ]
}}

## Requirements
- Output ONLY valid JSON, no other text
- Text in ENGLISH, 2-3 sentences each
- Do NOT use brackets, asterisks, or markdown formatting
"""
