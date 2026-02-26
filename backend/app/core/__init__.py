"""
コア分析モジュールパッケージ

データ読み込み、分析処理、AI解釈の各モジュールを集約する。
"""

from .data_loader import DataLoader
from .interpreter import GeminiInterpreter
from .analysis import (
    configure,
    unfold_and_scale_tensor,
    apply_pacmap_reduction,
    analyze_tensor_contribution,
    get_top_important_factors,
    evaluate_statistical_significance,
    apply_fdr_correction,
)

__all__ = [
    'DataLoader',
    'GeminiInterpreter',
    'configure',
    'unfold_and_scale_tensor',
    'apply_pacmap_reduction',
    'analyze_tensor_contribution',
    'get_top_important_factors',
    'evaluate_statistical_significance',
    'apply_fdr_correction',
]
