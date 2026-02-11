"""Core analysis modules for TensorComparativeVis."""

from .data_loader import DataLoader
from .interpreter import GeminiInterpreter
from .analysis import (
    configure,
    DomainProtocol,
    unfold_and_scale_tensor,
    apply_pacmap_reduction,
    analyze_tensor_contribution,
    get_top_important_factors,
    evaluate_statistical_significance
)

__all__ = [
    'DataLoader',
    'GeminiInterpreter',
    'configure',
    'DomainProtocol',
    'unfold_and_scale_tensor',
    'apply_pacmap_reduction',
    'analyze_tensor_contribution',
    'get_top_important_factors',
    'evaluate_statistical_significance'
]
