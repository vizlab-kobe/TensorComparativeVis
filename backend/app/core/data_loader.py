"""
データローダーモジュール

テンソルデータの .npy ファイルを読み込み、遅延初期化で提供する。
ドメイン戦略のファイルマッピングに従い、適切なファイル名を解決する。

読み込むデータファイル:
  - tensor_X: テンソルデータ本体 (T x S x V)
  - tensor_y: クラスラベル (T,)
  - time_axis: 時間軸ラベル (T,)
  - time_original: 元スケールの時系列データ (T x S x V)
"""

import numpy as np
from typing import Tuple
from pathlib import Path


class DataLoader:
    """テンソルデータファイルの汎用ローダー。

    ドメイン戦略に基づき、データディレクトリから必要なファイルを
    遅延的に読み込む。プロパティアクセス時に自動ロードが行われる。
    """

    def __init__(self, data_dir: str, domain):
        """データディレクトリとドメイン戦略を指定して初期化する。

        Args:
            data_dir: データディレクトリへのパス（相対または絶対）
            domain: ドメイン戦略インスタンス（ファイルマッピングを提供）
        """
        self.data_dir = Path(data_dir)
        self.domain = domain
        # 遅延初期化用の内部キャッシュ
        self._original_data = None
        self._time_axis = None
        self._tensor_X = None
        self._tensor_y = None

    def load_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """全データファイルを一括読み込みする。

        ドメインの file_mapping 属性があればそれに従い、
        なければ命名規則（{domain_name}_*.npy）でファイルを特定する。

        Returns:
            (original_data, time_axis, tensor_X, tensor_y) のタプル
        """
        # ドメイン固有のファイルマッピングを優先使用
        fm = getattr(self.domain, 'file_mapping', None)

        if fm:
            self._tensor_X = np.load(self.data_dir / fm['tensor_X'])
            self._tensor_y = np.load(self.data_dir / fm['tensor_y'], allow_pickle=True)
            self._time_axis = np.load(self.data_dir / fm['time_axis'], allow_pickle=True)
            self._original_data = np.load(self.data_dir / fm['time_original'])
        else:
            # フォールバック: プレフィックス命名規則
            prefix = f"{self.domain.name}_"
            self._tensor_X = np.load(self.data_dir / f"{prefix}tensor_X.npy")
            self._tensor_y = np.load(self.data_dir / f"{prefix}tensor_y.npy")
            self._time_axis = np.load(self.data_dir / f"{prefix}time_axis.npy")
            self._original_data = np.load(self.data_dir / f"{prefix}time_original.npy")

        return self._original_data, self._time_axis, self._tensor_X, self._tensor_y

    # ── 遅延ロードプロパティ ─────────────────────────────────────────────────

    @property
    def original_data(self) -> np.ndarray:
        """元スケールの時系列データ (T x S x V)。未ロード時は自動ロードする。"""
        if self._original_data is None:
            self.load_all()
        return self._original_data

    @property
    def time_axis(self) -> np.ndarray:
        """時間軸ラベル配列 (T,)。未ロード時は自動ロードする。"""
        if self._time_axis is None:
            self.load_all()
        return self._time_axis

    @property
    def tensor_X(self) -> np.ndarray:
        """テンソルデータ本体 (T x S x V)。未ロード時は自動ロードする。"""
        if self._tensor_X is None:
            self.load_all()
        return self._tensor_X

    @property
    def tensor_y(self) -> np.ndarray:
        """クラスラベル配列 (T,)。未ロード時は自動ロードする。"""
        if self._tensor_y is None:
            self.load_all()
        return self._tensor_y

    # ── 便利プロパティ ───────────────────────────────────────────────────────

    @property
    def shape(self) -> Tuple[int, int, int]:
        """テンソルの形状 (T, S, V) を返す。

        T: 時間ステップ数, S: 空間次元数, V: 変数次元数
        """
        return self.tensor_X.shape

    @property
    def n_classes(self) -> int:
        """データセット内のユニークなクラス数を返す。"""
        return len(np.unique(self.tensor_y))
