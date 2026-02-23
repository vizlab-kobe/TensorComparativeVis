"""
ドメイン戦略 基底クラス

全ドメイン固有設定の抽象インターフェースを定義する。
新しいデータドメインを追加する場合は、このクラスを継承して
全ての抽象メソッド・プロパティを実装する。

設計方針:
  - 抽象部分: データ固有の設定（変数名、ラベル変換、プロンプト構築）
  - 具象部分: 共通処理（特徴量フォーマット、比較プロンプト、語彙フック）
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
        top_features: List[Dict],
        cluster1_size: int,
        cluster2_size: int,
        preprocessed: Dict[str, Any],
    ) -> str:
        """クラスター解釈用のLLMプロンプトを構築する。

        Args:
            top_features: 上位特徴量のリスト
            cluster1_size: 赤クラスターのサンプル数
            cluster2_size: 青クラスターのサンプル数
            preprocessed: 前処理済み統計情報

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
        効果量・有意性を1行にまとめる。

        Args:
            features: 特徴量辞書のリスト

        Returns:
            フォーマット済みテキスト（改行区切り）
        """
        lines = []
        for f in features:
            stat = f.get('statistical_result', {})
            direction = "higher in C1" if stat.get('mean_diff', 0) > 0 else "lower in C1"
            # 有意性マーカー: * = p<0.05, + = p<0.1
            sig = (
                "*" if stat.get('p_value', 1) < 0.05
                else ("+" if stat.get('p_value', 1) < 0.1 else "")
            )
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
        analysis_b: Dict[str, Any],
    ) -> str:
        """2つの分析結果を比較するLLMプロンプトを構築する。

        ドメイン語彙フック（_system_label, _time_unit 等）を使用し、
        サブクラスのオーバーライドなしにドメイン適切な表現を生成する。

        Args:
            analysis_a: 1つ目の分析結果
            analysis_b: 2つ目の分析結果

        Returns:
            比較用LLMプロンプト文字列
        """
        # クラスター選択の一致度を判定
        same_cluster1 = (
            analysis_a.get('cluster1_size') == analysis_b.get('cluster1_size')
        )
        same_cluster2 = (
            analysis_a.get('cluster2_size') == analysis_b.get('cluster2_size')
        )

        if same_cluster1 and same_cluster2:
            context_msg = (
                "Both analyses use IDENTICAL cluster selections. "
                "Any differences in results are due to analysis parameters."
            )
        elif same_cluster1:
            context_msg = (
                "Both analyses share the SAME Red Cluster (C1) as the base. "
                "The comparison involves different Blue Cluster (C2) selections."
            )
        elif same_cluster2:
            context_msg = (
                "Both analyses share the SAME Blue Cluster (C2) as the base. "
                "The comparison involves different Red Cluster (C1) selections."
            )
        else:
            context_msg = "The analyses use completely different cluster selections."

        # 上位特徴量の集合演算で共通・固有の特徴を算出
        features_a = analysis_a.get('top_features', [])[:10]
        features_b = analysis_b.get('top_features', [])[:10]

        feature_set_a = set(
            f"{f.get('rack')}-{f.get('variable')}" for f in features_a
        )
        feature_set_b = set(
            f"{f.get('rack')}-{f.get('variable')}" for f in features_b
        )

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
