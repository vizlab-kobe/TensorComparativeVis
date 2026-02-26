"""
アプリケーションパッケージ初期化モジュール

外部から利用される主要クラス・関数を一括エクスポートする。
"""

from .domains import BaseDomain, HPCDomain
from .core import (
    DataLoader,
    GeminiInterpreter,
    unfold_and_scale_tensor,
    apply_pacmap_reduction,
    analyze_tensor_contribution,
    get_top_important_factors,
    evaluate_statistical_significance,
    apply_fdr_correction,
)

__all__ = [
    'BaseDomain',
    'HPCDomain',
    'DataLoader',
    'GeminiInterpreter',
    'unfold_and_scale_tensor',
    'apply_pacmap_reduction',
    'analyze_tensor_contribution',
    'get_top_important_factors',
    'evaluate_statistical_significance',
    'apply_fdr_correction',
]
