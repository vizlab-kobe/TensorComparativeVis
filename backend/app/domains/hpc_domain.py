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
    
    def build_comparison_prompt(
        self,
        analysis_a: Dict[str, Any],
        analysis_b: Dict[str, Any]
    ) -> str:
        """Build LLM prompt for comparing two HPC analyses."""
        # Check cluster matching
        same_cluster1 = analysis_a.get('cluster1_size') == analysis_b.get('cluster1_size')
        same_cluster2 = analysis_a.get('cluster2_size') == analysis_b.get('cluster2_size')
        
        if same_cluster1 and same_cluster2:
            context_msg = "Both analyses use IDENTICAL cluster selections. Any differences in results are due to analysis parameters."
        elif same_cluster1:
            context_msg = "Both analyses share the SAME Red Cluster (C1) as the base. The comparison involves different Blue Cluster (C2) selections."
        elif same_cluster2:
            context_msg = "Both analyses share the SAME Blue Cluster (C2) as the base. The comparison involves different Red Cluster (C1) selections."
        else:
            context_msg = "The analyses use completely different cluster selections."
        
        # Extract features
        features_a = analysis_a.get('top_features', [])[:10]
        features_b = analysis_b.get('top_features', [])[:10]
        
        feature_set_a = set(f"{f.get('rack')}-{f.get('variable')}" for f in features_a)
        feature_set_b = set(f"{f.get('rack')}-{f.get('variable')}" for f in features_b)
        
        common = feature_set_a & feature_set_b
        only_a = feature_set_a - feature_set_b
        only_b = feature_set_b - feature_set_a
        
        return f"""
You are comparing two cluster analysis results from an HPC tensor data visualization system.

## Comparison Context
{context_msg}

## Analysis A
- Red Cluster size: {analysis_a.get('cluster1_size')} time points
- Blue Cluster size: {analysis_a.get('cluster2_size')} time points
- Significant features: {analysis_a.get('significant_count', 'N/A')}
- Top variables: {', '.join(analysis_a.get('top_variables', [])[:5])}
- Top locations: {', '.join(analysis_a.get('top_racks', [])[:5])}

## Analysis B
- Red Cluster size: {analysis_b.get('cluster1_size')} time points
- Blue Cluster size: {analysis_b.get('cluster2_size')} time points
- Significant features: {analysis_b.get('significant_count', 'N/A')}
- Top variables: {', '.join(analysis_b.get('top_variables', [])[:5])}
- Top locations: {', '.join(analysis_b.get('top_racks', [])[:5])}

## Feature Overlap
- Common features (in both): {len(common)} - {list(common)[:5]}
- Only in A: {len(only_a)} - {list(only_a)[:5]}
- Only in B: {len(only_b)} - {list(only_b)[:5]}

## Cluster Matching
- Red Cluster (C1): {"SAME" if same_cluster1 else "DIFFERENT"} (A: {analysis_a.get('cluster1_size')}, B: {analysis_b.get('cluster1_size')})
- Blue Cluster (C2): {"SAME" if same_cluster2 else "DIFFERENT"} (A: {analysis_a.get('cluster2_size')}, B: {analysis_b.get('cluster2_size')})

Generate a JSON comparison with this structure:
{{
  "sections": [
    {{
      "title": "Comparison Overview",
      "text": "Summarize the key differences and similarities between the two analyses.",
      "highlights": []
    }},
    {{
      "title": "Feature Differences",
      "text": "Describe which features appear in one analysis but not the other, and what this might indicate.",
      "highlights": []
    }},
    {{
      "title": "Implications",
      "text": "What do these differences suggest about the data patterns?",
      "highlights": []
    }}
  ]
}}

## Requirements
- Output ONLY valid JSON, no other text
- Text should be in ENGLISH (for academic publication)
- Each section should be 2-3 sentences
- {"Mention that both analyses share the same base cluster" if same_cluster1 or same_cluster2 else "Note that different clusters are compared"}
- Do NOT use brackets, asterisks, arrows, or any special formatting - plain text only
"""
