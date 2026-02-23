"""
設定ファイルローダーモジュール

YAMLベースの設定ファイルを読み込み、適切なドメイン戦略インスタンスを生成する。
設定ファイルのパスは以下の優先順位で決定される:
  1. 関数引数で直接指定
  2. 環境変数 APP_CONFIG（ファイル名のみ、拡張子省略可）
  3. デフォルト: "hpc_default.yaml"
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """YAML設定ファイルを読み込んで辞書として返す。

    Args:
        config_path: 設定ファイルへのパス。
                     None の場合、環境変数 APP_CONFIG またはデフォルト値を使用。

    Returns:
        設定辞書（ドメイン名、TULCAパラメータ、色設定等を含む）
    """
    if config_path is None:
        # 環境変数からファイル名を取得（デフォルト: hpc_default）
        config_name = os.getenv('APP_CONFIG', 'hpc_default')
        if not config_name.endswith('.yaml'):
            config_name = f"{config_name}.yaml"
        # プロジェクトルートの configs/ ディレクトリから読み込む
        config_path = Path(__file__).parent.parent.parent / "configs" / config_name
    else:
        config_path = Path(config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_domain_instance(config: Dict[str, Any]):
    """設定辞書からドメイン戦略インスタンスを生成する。

    ドメイン名（config['domain']）に基づいて、
    対応するドメインクラスをインスタンス化する。

    Args:
        config: load_config() で取得した設定辞書

    Returns:
        ドメイン戦略インスタンス（HPCDomain / AirDataDomain）

    Raises:
        ValueError: 未知のドメイン名が指定された場合
    """
    domain_name = config.get('domain', 'hpc').lower()

    # プロジェクトルートパスを解決（データディレクトリの構築に使用）
    project_root = Path(__file__).parent.parent.parent

    if domain_name == 'hpc':
        from app.domains import HPCDomain
        return HPCDomain()
    elif domain_name in ('air_data', 'airdata'):
        from app.domains import AirDataDomain
        # AirDataDomain は座標ファイル読み込みのためにデータディレクトリが必要
        data_dir = str(project_root / "data" / "processed" / "AirData")
        return AirDataDomain(data_dir=data_dir)
    else:
        raise ValueError(f"未知のドメイン: {domain_name}")
