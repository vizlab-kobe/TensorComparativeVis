"""
HPC Domain - Configuration for High-Performance Computing supercomputer data.
"""

from typing import List, Tuple, Dict, Any
from .base_domain import BaseDomain


class HPCDomain(BaseDomain):
    """Domain configuration for HPC supercomputer tensor data."""
    
    # HPC-specific constants
    _VARIABLES = ['AirIn', 'AirOut', 'CPU', 'Water']
    _RACK_NUMBERS = [i for i in range(1, 46) if i % 10 != 3 and i % 10 != 8]
    _GRID_SHAPE = (36, 24)
    _HEATMAP_COLS = 24
    
    @property
    def name(self) -> str:
        return "HPC"
    
    @property
    def data_dir(self) -> str:
        return "data/processed/HPC"
    
    @property
    def file_mapping(self) -> Dict[str, str]:
        """Map logical file names to actual file names on disk."""
        return {
            'tensor_X': 'HPC_tensor_X.npy',
            'tensor_y': 'HPC_tensor_y.npy',
            'time_axis': 'HPC_time_axis.npy',
            'time_original': 'HPC_time_original.npy',
        }
    
    @property
    def variables(self) -> List[str]:
        return self._VARIABLES
    
    @property
    def grid_shape(self) -> Tuple[int, int]:
        return self._GRID_SHAPE
    
    def index_to_label(self, vector_index: int) -> str:
        """Convert heatmap index to supercomputer rack label (e.g., A1, B5)."""
        col_index = vector_index % self._HEATMAP_COLS
        row_index = vector_index // self._HEATMAP_COLS
        letter = chr(ord('A') + col_index)
        return f"{letter}{self._RACK_NUMBERS[row_index]}"
    
    def label_to_index(self, label: str) -> int:
        """Convert rack label to heatmap index."""
        letter = label[0]
        number = int(label[1:])
        col_index = ord(letter) - ord('A')
        row_index = self._RACK_NUMBERS.index(number)
        return row_index * self._HEATMAP_COLS + col_index
    
    # ── Domain vocabulary hooks ──
    
    @property
    def class_labels(self) -> List[str]:
        return ["FY2014", "FY2015", "FY2016"]

    @property
    def _system_label(self) -> str:
        return "HPC tensor data"

    @property
    def domain_knowledge(self) -> str:
        return """## HPC Tensor Data Analysis Domain Knowledge

### Variables
- The tensor data contains multivariate time-series measurements from a supercomputer
- Each variable represents a different sensor type: AirIn, AirOut, CPU temperature, Water temperature
- Variables may have different scales and units

### Spatial Structure
- Data points are organized in a 2D rack layout representing physical server positions
- Adjacent racks may exhibit correlated thermal patterns
- Spatial clustering may indicate localized cooling issues or workload concentration

### Temporal Patterns
- Time points are grouped by predefined labels (e.g., job types, workload periods)
- Cluster comparisons reveal temporal dynamics in system behavior
- Statistical significance indicates reliable differences in thermal/performance characteristics
"""
    
    def build_interpretation_prompt(
        self,
        top_features: List[Dict],
        cluster1_size: int,
        cluster2_size: int,
        preprocessed: Dict[str, Any]
    ) -> str:
        """Build LLM prompt for HPC cluster interpretation."""
        features_text = self._format_features(top_features[:30])
        
        return f"""
You are an AI assistant that generates structured summaries of tensor data cluster analysis results.
Your role is to organize and describe the data clearly - NOT to make causal claims or predictions.

{self.domain_knowledge}

## Analysis Context
- Red Cluster (Cluster 1): {cluster1_size} time points
- Blue Cluster (Cluster 2): {cluster2_size} time points

## Preprocessed Statistics
- Significant features: {preprocessed['significant_count']}/{preprocessed['total_count']}
- Dominant variable type: {preprocessed['dominant_variable']}
- Variable distribution: {preprocessed['variable_distribution']}
- Average effect size: {preprocessed['avg_effect_size']}
- Spatial pattern: {preprocessed['rack_concentration']}
- Co-occurring variables in same location: {len(preprocessed['co_occurrences'])} cases

## Top Features (up to 30, by contribution score)
{features_text}

## Output Instructions
Generate a JSON response with the following structure. Each section should contain natural language text (2-3 sentences) in ENGLISH for academic publication.
Do NOT use any special formatting like brackets or markdown. Write in plain text.

{{
  "sections": [
    {{
      "title": "Key Findings",
      "text": "Summarize the most important differences. Which variables show the largest differences? Are differences concentrated in specific locations or distributed?",
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

