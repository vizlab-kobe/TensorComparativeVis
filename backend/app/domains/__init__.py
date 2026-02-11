"""
Domain-specific configurations for TensorComparativeVis.
Each domain defines data schema, prompts, and metadata.
"""

from .base_domain import BaseDomain
from .hpc_domain import HPCDomain
from .air_data_domain import AirDataDomain

__all__ = ['BaseDomain', 'HPCDomain', 'AirDataDomain']
