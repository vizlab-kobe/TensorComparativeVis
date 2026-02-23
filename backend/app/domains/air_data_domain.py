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
        top_features: List[Dict],
        cluster1_size: int,
        cluster2_size: int,
        preprocessed: Dict[str, Any],
    ) -> str:
        """大気質クラスター解釈用のLLMプロンプトを構築する。

        大気汚染データに特化した文脈情報を含むプロンプトを生成し、
        構造化されたJSON解釈を要求する。

        Args:
            top_features: 上位特徴量リスト
            cluster1_size: 赤クラスターのサンプル数
            cluster2_size: 青クラスターのサンプル数
            preprocessed: 前処理済み統計情報

        Returns:
            LLMプロンプト文字列
        """
        features_text = self._format_features(top_features[:30])

        return f"""
You are an AI assistant that generates structured summaries of air quality data cluster analysis results.
Your role is to organize and describe the data clearly - NOT to make causal claims or predictions.

{self.domain_knowledge}

## Analysis Context
- Red Cluster (Cluster 1): {cluster1_size} time points (weekly observations)
- Blue Cluster (Cluster 2): {cluster2_size} time points (weekly observations)

## Preprocessed Statistics
- Significant features: {preprocessed['significant_count']}/{preprocessed['total_count']}
- Dominant pollutant type: {preprocessed['dominant_variable']}
- Variable distribution: {preprocessed['variable_distribution']}
- Average effect size: {preprocessed['avg_effect_size']}
- Spatial pattern: {preprocessed['rack_concentration']}
- Co-occurring pollutants at same station: {len(preprocessed['co_occurrences'])} cases

## Top Features (up to 30, by contribution score)
{features_text}

## Output Instructions
Generate a JSON response with the following structure. Each section should contain natural language text (2-3 sentences) in ENGLISH for academic publication.
Do NOT use any special formatting like brackets or markdown. Write in plain text.

{{
  "sections": [
    {{
      "title": "Key Findings",
      "text": "Summarize the most important differences. Which pollutants show the largest differences? Are differences concentrated in specific regions or distributed across stations?",
      "highlights": []
    }},
    {{
      "title": "Statistical Summary",
      "text": "Describe the statistical evidence. How many features are statistically significant? What are the effect sizes?",
      "highlights": []
    }},
    {{
      "title": "Caveats",
      "text": "Note limitations. Mention cluster size imbalance if present. Note if many features lack statistical significance.",
      "highlights": []
    }}
  ]
}}

## Requirements
- Output ONLY valid JSON, no other text
- Text should be in ENGLISH (for academic publication)
- Each section should be 2-3 sentences
- Focus on DESCRIBING patterns, not explaining causation
- Do NOT use brackets, asterisks, arrows, or any special formatting - plain text only
"""
