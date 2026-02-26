"""
大気データドメイン戦略

米国大気質モニタリングステーションデータ固有の設定を提供する。
55ヶ所の観測ステーションにおける5種類の大気汚染物質の
週次データを地理マップ上で可視化する。

データ構造:
  - テンソル形状: (T x 55 x 5) = 時間 x ステーション数 x 変数数
  - ステーションラベル: 座標ファイルから取得した地名
  - クラス: Q1〜Q4 の四半期別
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from pathlib import Path
from .base_domain import BaseDomain


class AirDataDomain(BaseDomain):
    """米国大気質テンソルデータのドメイン設定。

    55ヶ所のモニタリングステーション × 5種類の汚染物質による
    週次時系列データの分析に特化した設定を提供する。
    地理座標を用いたマップ可視化をサポートする。
    """

    # 計測変数名（大気汚染物質）
    _VARIABLES = ['CO2', 'NO', 'Ozone', 'PM10', 'PM2.5']

    def __init__(self, data_dir: str = None):
        """ドメインを初期化し、座標データの遅延読み込みを準備する。

        Args:
            data_dir: データディレクトリパス（座標ファイルの読み込みに使用）
        """
        self._coordinates = None
        self._data_dir = data_dir

    # ── 抽象プロパティの実装 ──────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """ドメイン識別名"""
        return "AirData"

    @property
    def data_dir(self) -> str:
        """データディレクトリの相対パス"""
        return "data/processed/AirData"

    @property
    def variables(self) -> List[str]:
        """変数名リスト（大気汚染物質）"""
        return self._VARIABLES

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """空間グリッドの形状。

        55ステーション × 5変数。地理マップ表示のため
        2Dグリッドとしては使用しない。
        """
        return (55, 5)

    @property
    def visualization_type(self) -> str:
        """大気データは地理マップ可視化を使用する"""
        return "geo_map"

    @property
    def file_mapping(self) -> Dict[str, str]:
        """論理ファイル名 → 物理ファイル名の対応表"""
        return {
            'tensor_X': 'tensor_data_X_air.npy',
            'tensor_y': 'tensor_data_y_air.npy',
            'time_axis': 'time_axis.npy',
            # 元スケールデータがないため X を代用
            'time_original': 'tensor_data_X_air.npy',
            'coordinates': 'coordinates.npy',
        }

    # ── 座標データの遅延読み込み ──────────────────────────────────────────────

    def _load_coordinates(self):
        """座標データを遅延読み込みする。

        初回呼び出し時のみファイルを読み込み、
        以降はキャッシュされた結果を返す。

        Returns:
            座標配列 (N x 3: [lat, lon, name]) または None
        """
        if self._coordinates is None and self._data_dir:
            coord_path = Path(self._data_dir) / self.file_mapping['coordinates']
            if coord_path.exists():
                self._coordinates = np.load(str(coord_path), allow_pickle=True)
        return self._coordinates

    def get_coordinates(self) -> List[Dict[str, Any]]:
        """全ステーションの座標データを辞書リストで返す。

        地理マップ可視化コンポーネント（GeoMapVis）で使用される。

        Returns:
            座標辞書のリスト [{"index": 0, "lat": ..., "lon": ..., "name": "..."}]
        """
        coords = self._load_coordinates()
        if coords is None:
            return []
        result = []
        for i, row in enumerate(coords):
            result.append({
                "index": i,
                "lat": float(row[0]),
                "lon": float(row[1]),
                "name": str(row[2]),
            })
        return result

    # ── ラベル変換メソッド ────────────────────────────────────────────────────

    def index_to_label(self, spatial_index: int) -> str:
        """空間インデックスをステーション名に変換する。

        座標データが読み込まれている場合はステーション名を返し、
        そうでない場合は "Station-N" 形式のフォールバックラベルを返す。

        Args:
            spatial_index: ステーションのインデックス (0〜54)

        Returns:
            ステーション名（例: "Los Angeles", "Station-5"）
        """
        coords = self._load_coordinates()
        if coords is not None and spatial_index < len(coords):
            return str(coords[spatial_index][2])
        return f"Station-{spatial_index + 1}"

    def label_to_index(self, label: str) -> int:
        """ステーション名を空間インデックスに逆変換する。

        座標データから名前を検索し、見つからない場合は
        "Station-N" 形式のフォールバックパースを試みる。

        Args:
            label: ステーション名

        Returns:
            ステーションのインデックス
        """
        coords = self._load_coordinates()
        if coords is not None:
            for i, row in enumerate(coords):
                if str(row[2]) == label:
                    return i
        # フォールバック: "Station-N" 形式のパース
        if label.startswith("Station-"):
            return int(label.split("-")[1]) - 1
        return 0

    # ── 語彙フック ────────────────────────────────────────────────────────────

    @property
    def class_labels(self) -> List[str]:
        """四半期別クラスラベル"""
        return ["Q1", "Q2", "Q3", "Q4"]

    @property
    def _system_label(self) -> str:
        return "US air quality data"

    @property
    def _time_unit(self) -> str:
        return "weekly observations"

    @property
    def _variable_noun(self) -> str:
        return "pollutants"

    @property
    def _location_noun(self) -> str:
        return "stations"

    # ── ドメイン知識とLLMプロンプト ────────────────────────────────────────────

    @property
    def domain_knowledge(self) -> str:
        """大気データ分析のドメイン知識テキスト（LLMコンテキスト用）"""
        return """## US Air Quality Data Analysis Domain Knowledge

### Variables
- CO2: Carbon dioxide concentration
- NO: Nitrogen monoxide concentration
- Ozone: Ground-level ozone concentration
- PM10: Particulate matter (diameter ≤ 10μm)
- PM2.5: Fine particulate matter (diameter ≤ 2.5μm)

### Spatial Structure
- Data from 55 monitoring stations across the United States
- Stations are geographically distributed, with varying local conditions
- Spatial patterns may reflect regional pollution sources, meteorological patterns, or geographical features

### Temporal Patterns
- Weekly aggregated data covering the year 2018
- Time points are labeled by quarter (Q1-Q4)
- Seasonal variations are expected due to weather patterns and human activity cycles
- Ozone tends to peak in summer; PM levels may rise in winter or during wildfire seasons
"""

    def build_interpretation_prompt(
        self,
        features_with_confidence: List[Dict],
        cluster1_range: Dict[str, Any],
        cluster2_range: Dict[str, Any],
        co_occurrences: List[tuple],
        rack_concentration: str,
        dominant_variable: str,
    ) -> str:
        """大気質クラスター解釈用のLLMプロンプトを構築する。

        大気汚染データに特化した文脈情報を含むプロンプトを生成し、
        構造化された3フィールドJSON解釈を要求する。

        Args:
            features_with_confidence: confidence ラベル付き上位特徴量リスト
            cluster1_range: C1 の時間範囲 {"start", "end", "size"}
            cluster2_range: C2 の時間範囲 {"start", "end", "size"}
            co_occurrences: 同一位置での変数共起リスト
            rack_concentration: 空間集中度の文字列
            dominant_variable: 最頻出変数名

        Returns:
            LLMプロンプト文字列
        """
        features_text = self._format_features(features_with_confidence[:30])

        co_occ_text = "\n".join(
            f"  - <<{station}>>: {', '.join(vars_list)}"
            for station, vars_list in co_occurrences[:10]
        ) or "  None detected"

        return f"""
You are an AI assistant analyzing US air quality data cluster comparison results.
Your role is to organize and describe observed patterns — NOT to make causal claims.

{self.domain_knowledge}

## Cluster Time Ranges
- Cluster 1 (Red): {cluster1_range['start']} 〜 {cluster1_range['end']} ({cluster1_range['size']} time points)
- Cluster 2 (Blue): {cluster2_range['start']} 〜 {cluster2_range['end']} ({cluster2_range['size']} time points)

## Pattern Summary
- Dominant pollutant: {dominant_variable}
- Spatial pattern: {rack_concentration}
- Co-occurring pollutants at same station:
{co_occ_text}

## Top Features (by contribution score, with confidence level)
{features_text}

## Output Instructions

Generate a JSON response with EXACTLY this structure:

{{
  "comparison_context": {{
    "text": "Describe what time periods / groups are being compared, and their sizes."
  }},
  "separation_factors": {{
    "text": "Describe the main factors that differentiate the two clusters. Which stations and pollutants show differences?"
  }},
  "suggested_exploration": {{
    "text": "Suggest what the user should look at next to understand the patterns better."
  }}
}}

## Formatting Rules
1. Output ONLY valid JSON, no other text.
2. Write all text in JAPANESE (日本語で記述してください).
3. When mentioning station or station-variable combinations, use marker notation: <<StationName-Ozone>> or <<StationName>>.
   Only use names that appear in the feature list above.
4. Convert confidence labels to natural language:
   - "strong" → 「明確な差が見られます」
   - "moderate" → 「差が示唆されます」
   - "weak" → 「わずかな傾向があります」
   - "unclear" → 「差は明確ではありません」
5. Do NOT include raw p-values or Cohen's d numbers in the output text.
6. Each field's text should be 2-4 sentences.
"""
