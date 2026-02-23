"""
ドメイン戦略パッケージ初期化

各ドメイン固有の設定クラスを一括エクスポートする。
新しいドメインを追加する場合は、BaseDomain を継承したクラスを作成し、
ここでエクスポートを追加する。
"""

from .base_domain import BaseDomain
from .hpc_domain import HPCDomain
from .air_data_domain import AirDataDomain

__all__ = ['BaseDomain', 'HPCDomain', 'AirDataDomain']
