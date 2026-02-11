"""
Base Domain Strategy - Abstract interface for domain-specific configurations.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any


class BaseDomain(ABC):
    """Abstract base class for domain-specific configurations."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Domain name for identification."""
        pass
    
    @property
    @abstractmethod
    def data_dir(self) -> str:
        """Relative path to data directory from project root."""
        pass
    
    @property
    @abstractmethod
    def variables(self) -> List[str]:
        """List of variable names in the dataset."""
        pass
    
    @property
    @abstractmethod
    def grid_shape(self) -> Tuple[int, int]:
        """Grid shape for spatial visualization (rows, cols)."""
        pass
    
    @abstractmethod
    def index_to_label(self, index: int) -> str:
        """Convert spatial index to human-readable label."""
        pass
    
    @abstractmethod
    def label_to_index(self, label: str) -> int:
        """Convert human-readable label to spatial index."""
        pass
    
    @property
    @abstractmethod
    def domain_knowledge(self) -> str:
        """Domain knowledge text for LLM context."""
        pass
    
    @abstractmethod
    def build_interpretation_prompt(
        self,
        top_features: List[Dict],
        cluster1_size: int,
        cluster2_size: int,
        preprocessed: Dict[str, Any]
    ) -> str:
        """Build LLM prompt for cluster interpretation."""
        pass
    
    # ── Domain vocabulary hooks (override in subclasses for domain-specific wording) ──

    @property
    def class_labels(self) -> List[str]:
        """Human-readable labels for each class in the dataset.
        Override in subclasses for domain-specific labels.
        Default: Class 1, Class 2, ...
        """
        return []  # empty = let frontend fall back to generic

    @property
    def _system_label(self) -> str:
        """Short label describing the system for LLM prompts (e.g. 'HPC tensor data')."""
        return f"{self.name} data"

    @property
    def _time_unit(self) -> str:
        """Human-readable time unit (e.g. 'time points', 'weekly observations')."""
        return "time points"

    @property
    def _variable_noun(self) -> str:
        """Noun for variables (e.g. 'variables', 'pollutants')."""
        return "variables"

    @property
    def _location_noun(self) -> str:
        """Noun for spatial locations (e.g. 'locations', 'stations')."""
        return "locations"

    # ── Shared concrete methods ──────────────────────────────────────────────

    def _format_features(self, features: List[Dict]) -> str:
        """Format feature list into readable lines for LLM prompts."""
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

    def build_comparison_prompt(
        self,
        analysis_a: Dict[str, Any],
        analysis_b: Dict[str, Any]
    ) -> str:
        """Build LLM prompt for comparing two analyses.
        
        Uses domain vocabulary hooks (_system_label, _time_unit, etc.) so
        subclasses get domain-appropriate wording without overriding.
        """
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
        
        features_a = analysis_a.get('top_features', [])[:10]
        features_b = analysis_b.get('top_features', [])[:10]
        
        feature_set_a = set(f"{f.get('rack')}-{f.get('variable')}" for f in features_a)
        feature_set_b = set(f"{f.get('rack')}-{f.get('variable')}" for f in features_b)
        
        common = feature_set_a & feature_set_b
        only_a = feature_set_a - feature_set_b
        only_b = feature_set_b - feature_set_a

        tu = self._time_unit
        vn = self._variable_noun
        ln = self._location_noun
        
        return f"""
You are comparing two cluster analysis results from a {self._system_label} visualization system.

## Comparison Context
{context_msg}

## Analysis A
- Red Cluster size: {analysis_a.get('cluster1_size')} {tu}
- Blue Cluster size: {analysis_a.get('cluster2_size')} {tu}
- Significant features: {analysis_a.get('significant_count', 'N/A')}
- Top {vn}: {', '.join(analysis_a.get('top_variables', [])[:5])}
- Top {ln}: {', '.join(analysis_a.get('top_racks', [])[:5])}

## Analysis B
- Red Cluster size: {analysis_b.get('cluster1_size')} {tu}
- Blue Cluster size: {analysis_b.get('cluster2_size')} {tu}
- Significant features: {analysis_b.get('significant_count', 'N/A')}
- Top {vn}: {', '.join(analysis_b.get('top_variables', [])[:5])}
- Top {ln}: {', '.join(analysis_b.get('top_racks', [])[:5])}

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

    @property
    def ui_metadata(self) -> Dict[str, Any]:
        """UI configuration metadata (optional, can be overridden)."""
        return {
            "title": f"{self.name} Analysis Dashboard",
            "description": "Tensor-based comparative analysis"
        }
    
    @property
    def visualization_type(self) -> str:
        """Spatial visualization type: 'grid' for heatmap, 'geo_map' for geographic map."""
        return "grid"
    
    def get_coordinates(self) -> List[Dict[str, Any]]:
        """Return spatial coordinates for geo_map visualization.
        Override in subclasses that use geo_map visualization.
        
        Returns:
            List of dicts: [{"index": 0, "lat": ..., "lon": ..., "name": "..."}]
        """
        return []

