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
- AirIn: 吸気温度（サーバーラック前面から取り入れる空気の温度）
- AirOut: 排気温度（サーバーラック背面から排出される空気の温度）
- CPU: CPU温度（サーバー内部のプロセッサ温度）
- Water: 冷却水温度（水冷直接冷却システムの冷却水温度）

### Spatial Structure
- データは36行×24列のラック配置構造上で構成される
- 隣接ラックは相関した熱パターンを示すことがある
- 空間的な集中は局所的な冷却問題やワークロード集中を示唆する可能性がある

### Temporal Patterns
- 時系列データは年度単位（FY2014, FY2015, FY2016）でグループ化されている
- クラスター比較によりシステムの熱的振る舞いの時間変化が明らかになる
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
        """HPCクラスター解釈用のLLMプロンプトを構築する。

        スーパーコンピュータの温度データに特化した文脈情報を含む
        プロンプトを生成し、構造化された3フィールドJSON解釈を要求する。

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

        # 共起パターンのテキスト化
        co_occ_text = "\n".join(
            f"  - <<{rack}>>: {', '.join(vars_list)}"
            for rack, vars_list in co_occurrences[:10]
        ) or "  None detected"

        return f"""
You are an AI assistant analyzing HPC (supercomputer) tensor data cluster comparison results.
Your role is to organize and describe observed patterns — NOT to make causal claims.

{self.domain_knowledge}

## Cluster Time Ranges
- Cluster 1 (Red): {cluster1_range['start']} 〜 {cluster1_range['end']} ({cluster1_range['size']} time points)
- Cluster 2 (Blue): {cluster2_range['start']} 〜 {cluster2_range['end']} ({cluster2_range['size']} time points)

## Pattern Summary
- Dominant variable: {dominant_variable}
- Spatial pattern: {rack_concentration}
- Co-occurring variables at same rack:
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
    "text": "Describe the main factors that differentiate the two clusters. Which racks and variables show differences?"
  }},
  "suggested_exploration": {{
    "text": "Suggest what the user should look at next to understand the patterns better."
  }}
}}

## Formatting Rules
1. Output ONLY valid JSON, no other text.
2. Write all text in JAPANESE (日本語で記述してください).
3. When mentioning rack names or rack-variable combinations, use marker notation: <<C12-Water>> or <<M17>>.
   Only use names that appear in the feature list above.
4. Convert confidence labels to natural language:
   - "strong" → 「明確な差が見られます」
   - "moderate" → 「差が示唆されます」
   - "weak" → 「わずかな傾向があります」
   - "unclear" → 「差は明確ではありません」
5. Do NOT include raw p-values or Cohen's d numbers in the output text.
6. Each field's text should be 2-4 sentences.
"""
