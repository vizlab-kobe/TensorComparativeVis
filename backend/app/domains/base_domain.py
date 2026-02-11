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
    
    @abstractmethod
    def build_comparison_prompt(
        self,
        analysis_a: Dict[str, Any],
        analysis_b: Dict[str, Any]
    ) -> str:
        """Build LLM prompt for comparing two analyses."""
        pass
    
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
