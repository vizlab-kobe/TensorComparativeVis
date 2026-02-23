"""
HPCドメイン戦略

スーパーコンピュータ（HPC）データ固有の設定を提供する。
センサーデータ（吸気温度、排気温度、CPU温度、冷却水温度）を
ラック配置構造上で可視化する。

データ構造:
  - テンソル形状: (T x 864 x 4) = 時間 x ラック数(36行×24列) x 変数数
  - ラックラベル: A1〜X45 の英字+数字形式
  - クラス: FY2014, FY2015, FY2016 の年度別
"""

from typing import List, Tuple, Dict, Any
from .base_domain import BaseDomain


class HPCDomain(BaseDomain):
    """HPCスーパーコンピュータ テンソルデータのドメイン設定。

    864台のサーバーラック × 4種類の温度センサーによる
    多変量時系列データの分析に特化した設定を提供する。
    """

    # ── HPC固有の定数 ─────────────────────────────────────────────────────────

    # 計測変数名（センサー種別）
    _VARIABLES = ['AirIn', 'AirOut', 'CPU', 'Water']

    # ラック番号リスト（x3, x8 は欠番: 通路の位置）
    _RACK_NUMBERS = [i for i in range(1, 46) if i % 10 != 3 and i % 10 != 8]

    # ヒートマップのグリッド形状 (行数=36ラック行, 列数=24列)
    _GRID_SHAPE = (36, 24)

    # ヒートマップの列数（index_to_label / label_to_index で使用）
    _HEATMAP_COLS = 24

    # ── 抽象プロパティの実装 ──────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """ドメイン識別名"""
        return "HPC"

    @property
    def data_dir(self) -> str:
        """データディレクトリの相対パス"""
        return "data/processed/HPC"

    @property
    def file_mapping(self) -> Dict[str, str]:
        """論理ファイル名 → 物理ファイル名の対応表"""
        return {
            'tensor_X': 'HPC_tensor_X.npy',
            'tensor_y': 'HPC_tensor_y.npy',
            'time_axis': 'HPC_time_axis.npy',
            'time_original': 'HPC_time_original.npy',
        }

    @property
    def variables(self) -> List[str]:
        """変数名リスト"""
        return self._VARIABLES

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """空間グリッドの形状 (行, 列)"""
        return self._GRID_SHAPE

    # ── ラベル変換メソッド ────────────────────────────────────────────────────

    def index_to_label(self, vector_index: int) -> str:
        """ヒートマップインデックスをラックラベルに変換する。

        例: index 0 → "A1", index 24 → "A2", index 1 → "B1"

        Args:
            vector_index: 空間次元のフラットインデックス

        Returns:
            ラックラベル（英字+数字: "A1"〜"X45"）
        """
        col_index = vector_index % self._HEATMAP_COLS
        row_index = vector_index // self._HEATMAP_COLS
        letter = chr(ord('A') + col_index)
        return f"{letter}{self._RACK_NUMBERS[row_index]}"

    def label_to_index(self, label: str) -> int:
        """ラックラベルをヒートマップインデックスに逆変換する。

        Args:
            label: ラックラベル（例: "A1", "B5"）

        Returns:
            空間次元のフラットインデックス
        """
        letter = label[0]
        number = int(label[1:])
        col_index = ord(letter) - ord('A')
        row_index = self._RACK_NUMBERS.index(number)
        return row_index * self._HEATMAP_COLS + col_index

    # ── 語彙フック ────────────────────────────────────────────────────────────

    @property
    def class_labels(self) -> List[str]:
        """年度別クラスラベル"""
        return ["FY2014", "FY2015", "FY2016"]

    @property
    def _system_label(self) -> str:
        return "HPC tensor data"

    # ── ドメイン知識とLLMプロンプト ────────────────────────────────────────────

    @property
    def domain_knowledge(self) -> str:
        """HPCデータ分析のドメイン知識テキスト（LLMコンテキスト用）"""
        return """## HPC Tensor Data Analysis Domain Knowledge

### Variables
- The tensor data contains multivariate time-series measurements from a supercomputer
- Each variable represents a different sensor type: AirIn, AirOut, CPU temperature, Water temperature
- Variables may have different scales and units

### Spatial Structure
- Data points are organized in a 2D rack layout representing physical server positions
- Adjacent racks may exhibit correlated thermal patterns
- Spatial clustering may indicate localized cooling issues or workload concentration

### Temporal Patterns
- Time points are grouped by predefined labels (e.g., job types, workload periods)
- Cluster comparisons reveal temporal dynamics in system behavior
- Statistical significance indicates reliable differences in thermal/performance characteristics
"""

    def build_interpretation_prompt(
        self,
        top_features: List[Dict],
        cluster1_size: int,
        cluster2_size: int,
        preprocessed: Dict[str, Any],
    ) -> str:
        """HPCクラスター解釈用のLLMプロンプトを構築する。

        スーパーコンピュータの温度データに特化した文脈情報を含む
        プロンプトを生成し、構造化されたJSON解釈を要求する。

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
You are an AI assistant that generates structured summaries of tensor data cluster analysis results.
Your role is to organize and describe the data clearly - NOT to make causal claims or predictions.

{self.domain_knowledge}

## Analysis Context
- Red Cluster (Cluster 1): {cluster1_size} time points
- Blue Cluster (Cluster 2): {cluster2_size} time points

## Preprocessed Statistics
- Significant features: {preprocessed['significant_count']}/{preprocessed['total_count']}
- Dominant variable type: {preprocessed['dominant_variable']}
- Variable distribution: {preprocessed['variable_distribution']}
- Average effect size: {preprocessed['avg_effect_size']}
- Spatial pattern: {preprocessed['rack_concentration']}
- Co-occurring variables in same location: {len(preprocessed['co_occurrences'])} cases

## Top Features (up to 30, by contribution score)
{features_text}

## Output Instructions
Generate a JSON response with the following structure. Each section should contain natural language text (2-3 sentences) in ENGLISH for academic publication.
Do NOT use any special formatting like brackets or markdown. Write in plain text.

{{
  "sections": [
    {{
      "title": "Key Findings",
      "text": "Summarize the most important differences. Which variables show the largest differences? Are differences concentrated in specific locations or distributed?",
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
