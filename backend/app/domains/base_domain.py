"""
ドメイン戦略 基底クラス

全ドメイン固有設定の抽象インターフェースを定義する。
新しいデータドメインを追加する場合は、このクラスを継承して
全ての抽象メソッド・プロパティを実装する。

設計方針:
  - 抽象部分: データ固有の設定（変数名、ラベル変換、プロンプト構築）
  - 具象部分: 共通処理（特徴量フォーマット、語彙フック）
  - 語彙フック: プロンプト内のドメイン固有表現をサブクラスで上書き可能
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any


class BaseDomain(ABC):
    """ドメイン戦略の抽象基底クラス。

    各ドメイン（HPC、大気データ等）はこのクラスを継承し、
    固有のデータ構造・可視化設定・LLMプロンプトを提供する。
    """

    # ── 抽象プロパティ（全サブクラスで実装必須） ──────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """ドメイン識別名（例: "HPC", "AirData"）"""
        pass

    @property
    @abstractmethod
    def data_dir(self) -> str:
        """プロジェクトルートからの相対データディレクトリパス"""
        pass

    @property
    @abstractmethod
    def variables(self) -> List[str]:
        """データセットの変数名リスト（例: ['AirIn', 'AirOut', 'CPU', 'Water']）"""
        pass

    @property
    @abstractmethod
    def grid_shape(self) -> Tuple[int, int]:
        """空間可視化のグリッド形状 (行数, 列数)"""
        pass

    # ── 抽象メソッド（全サブクラスで実装必須） ────────────────────────────────

    @abstractmethod
    def index_to_label(self, index: int) -> str:
        """空間インデックスを人間可読ラベルに変換する。

        Args:
            index: 空間次元のフラットインデックス

        Returns:
            人間可読ラベル（例: "A1", "Station-5"）
        """
        pass

    @abstractmethod
    def label_to_index(self, label: str) -> int:
        """人間可読ラベルを空間インデックスに逆変換する。

        Args:
            label: 人間可読ラベル

        Returns:
            空間次元のフラットインデックス
        """
        pass

    @property
    @abstractmethod
    def domain_knowledge(self) -> str:
        """LLMプロンプトに埋め込むドメイン知識テキスト"""
        pass

    @abstractmethod
    def build_interpretation_prompt(
        self,
        features_with_confidence: List[Dict],
        cluster1_range: Dict[str, Any],
        cluster2_range: Dict[str, Any],
        co_occurrences: List[tuple],
        rack_concentration: str,
        dominant_variable: str,
    ) -> str:
        """クラスター解釈用のLLMプロンプトを構築する。

        Args:
            features_with_confidence: confidence ラベル付き上位特徴量リスト
            cluster1_range: C1 の時間範囲 {"start", "end", "size"}
            cluster2_range: C2 の時間範囲 {"start", "end", "size"}
            co_occurrences: 同一位置での変数共起リスト
            rack_concentration: 空間集中度の文字列
            dominant_variable: 最頻出変数名

        Returns:
            LLMに送信するプロンプト文字列
        """
        pass

    # ── 語彙フック（サブクラスで上書き可能） ──────────────────────────────────

    @property
    def class_labels(self) -> List[str]:
        """データセット内の各クラスの人間可読ラベル。

        空リストの場合、フロントエンドは汎用ラベルに
        フォールバックする（例: "Class 1", "Class 2"）。
        """
        return []

    @property
    def _system_label(self) -> str:
        """LLMプロンプト用のシステム短縮ラベル（例: 'HPC tensor data'）"""
        return f"{self.name} data"

    @property
    def _time_unit(self) -> str:
        """時間単位の人間可読表現（例: 'time points', 'weekly observations'）"""
        return "time points"

    @property
    def _variable_noun(self) -> str:
        """変数の総称名詞（例: 'variables', 'pollutants'）"""
        return "variables"

    @property
    def _location_noun(self) -> str:
        """空間位置の総称名詞（例: 'locations', 'stations'）"""
        return "locations"

    # ── 共通具象メソッド ──────────────────────────────────────────────────────

    def _format_features(self, features: List[Dict]) -> str:
        """特徴量リストをLLMプロンプト用の可読テキストに変換する。

        各特徴量について、空間ラベル・変数名・スコア・方向・
        効果量・確信度を1行にまとめる。

        Args:
            features: 特徴量辞書のリスト（confidence キー付き）

        Returns:
            フォーマット済みテキスト（改行区切り）
        """
        lines = []
        for f in features:
            stat = f.get('statistical_result', {})
            direction = "higher in C1" if stat.get('mean_diff', 0) > 0 else "lower in C1"
            effect = stat.get('effect_size', 'N/A')
            confidence = f.get('confidence', 'unclear')

            lines.append(
                f"- <<{f.get('rack', 'N/A')}-{f.get('variable', 'N/A')}>>: "
                f"score={f.get('score', 0):.3f}, {direction}, "
                f"effect={effect}, confidence={confidence}"
            )
        return "\n".join(lines)

    # ── 可視化設定（サブクラスで上書き可能） ──────────────────────────────────

    @property
    def visualization_type(self) -> str:
        """空間可視化のタイプ。

        'grid': ヒートマップ表示（デフォルト、HPC等）
        'geo_map': 地図表示（大気データ等）
        """
        return "grid"

    def get_coordinates(self) -> List[Dict[str, Any]]:
        """地理座標データを返す（geo_map可視化用）。

        geo_map を使用するサブクラスでオーバーライドする。

        Returns:
            座標辞書のリスト [{"index": 0, "lat": ..., "lon": ..., "name": "..."}]
        """
        return []
