"""
AI解釈エンジンモジュール

Gemini APIを使用してクラスター分析結果を自然言語で解釈する。
ドメイン戦略パターンにより、プロンプト生成はドメインオブジェクトに委譲し、
このモジュールはAPI呼び出しとレスポンス解析に専念する。

主な機能:
  - interpret(): 特徴量データからAI解釈を生成
  - フォールバック: API不可時はデータベースの簡易要約を返す
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any
from collections import Counter
from dotenv import load_dotenv

# 環境変数を読み込む（.envファイルからAPIキー等を取得）
load_dotenv()

logger = logging.getLogger(__name__)

# Gemini APIクライアントのインポート（オプショナル依存）
try:
    from google import genai
except ImportError:
    genai = None


class GeminiInterpreter:
    """Gemini APIを用いたクラスター差異のAI解釈エンジン。

    ドメイン戦略インスタンスからプロンプトを取得し、
    Gemini APIに送信して構造化されたJSON解釈を受け取る。
    """

    def __init__(self, domain, api_key: Optional[str] = None):
        """AIインタープリターを初期化する。

        Args:
            domain: ドメイン戦略インスタンス（プロンプト生成を提供）
            api_key: Gemini APIキー（省略時は環境変数 GEMINI_API_KEY を使用）
        """
        self.domain = domain

        # APIキーの解決（引数 → 環境変数の優先順位）
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if genai and api_key:
            self.client = genai.Client(api_key=api_key)
            logger.info("Gemini APIクライアントを初期化しました")
        else:
            logger.warning(
                "Gemini APIクライアント未初期化。"
                "GEMINI_API_KEY 環境変数を設定してください。"
            )
            self.client = None

    # ── メイン解釈処理 ────────────────────────────────────────────────────────

    def interpret(
        self,
        top_features: List[Dict],
        cluster1_size: int,
        cluster2_size: int,
        cluster1_indices: List[int],
        cluster2_indices: List[int],
        timestamps: List[str],
    ) -> Dict[str, Any]:
        """クラスター差異の構造化された解釈を生成する。

        処理フロー:
          1. 時間範囲を算出
          2. 特徴量データを前処理してパターンを抽出
          3. ドメイン戦略からプロンプトを構築
          4. Gemini APIに送信
          5. JSONレスポンスをパース

        Args:
            top_features: 上位特徴量のリスト（辞書形式）
            cluster1_size: 赤クラスターのサンプル数
            cluster2_size: 青クラスターのサンプル数
            cluster1_indices: クラスタ1に含まれる時点のインデックスリスト
            cluster2_indices: クラスタ2に含まれる時点のインデックスリスト
            timestamps: 全時点の日付文字列リスト

        Returns:
            comparison_context / separation_factors / suggested_exploration
            の3フィールドを含む構造化された解釈辞書
        """
        if not self.client or not top_features:
            return self._fallback_interpretation(top_features)

        # 時間範囲の算出
        time_range1 = self._compute_time_range(cluster1_indices, timestamps)
        time_range2 = self._compute_time_range(cluster2_indices, timestamps)

        # 特徴量データを前処理してコンテキスト情報を抽出
        preprocessed = self._preprocess_features(top_features)

        # ドメイン固有のプロンプトを構築
        prompt = self.domain.build_interpretation_prompt(
            features_with_confidence=preprocessed['features_with_confidence'],
            cluster1_range=time_range1,
            cluster2_range=time_range2,
            co_occurrences=preprocessed['co_occurrences'],
            rack_concentration=preprocessed['rack_concentration'],
            dominant_variable=preprocessed['dominant_variable'],
        )

        try:
            response = self.client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=prompt,
            )
            return self._parse_json_response(response.text, top_features)
        except Exception as e:
            logger.error(f"API呼び出しエラー: {e}")
            return self._fallback_interpretation(top_features)

    # ── 時間・確信度ヘルパー ───────────────────────────────────────────────────

    @staticmethod
    def _compute_time_range(
        indices: List[int], timestamps: List[str],
    ) -> Dict[str, Any]:
        """インデックスリストとタイムスタンプリストから時間範囲を算出する。

        Args:
            indices: 対象クラスタの時点インデックス
            timestamps: 全時点の日付文字列リスト

        Returns:
            {"start": str, "end": str, "size": int}
        """
        if not indices or not timestamps:
            return {"start": "Unknown", "end": "Unknown", "size": 0}

        ts = [timestamps[i] for i in sorted(indices) if i < len(timestamps)]
        if not ts:
            return {"start": "Unknown", "end": "Unknown", "size": 0}

        return {"start": ts[0], "end": ts[-1], "size": len(indices)}

    @staticmethod
    def _compute_confidence_label(p_value: float, cohen_d: float) -> str:
        """p値とCohen's dから確信度ラベルを生成する。

        画面には出さず、プロンプトのコンテキストとしてのみ使用する。

        Returns:
            "strong" | "moderate" | "weak" | "unclear"
        """
        if p_value >= 0.05:
            return "unclear"
        if cohen_d >= 0.8:
            return "strong"
        if cohen_d >= 0.5:
            return "moderate"
        return "weak"

    # ── 前処理ヘルパー ────────────────────────────────────────────────────────

    def _preprocess_features(self, features: List[Dict]) -> Dict[str, Any]:
        """特徴量データを前処理してパターンを抽出する。

        変数の出現頻度、空間分布の集中度、統計情報、
        同一場所での変数共起パターン等を算出する。

        Args:
            features: 上位特徴量のリスト

        Returns:
            前処理結果の辞書
        """
        # 変数の出現頻度分布
        variables = [f.get('variable', '') for f in features]
        var_counts = Counter(variables)

        # 空間位置の分布
        racks = [f.get('rack', '') for f in features]

        # 統計的有意性の集計（FDR補正後のp値を優先）
        significant_count = sum(
            1 for f in features
            if f.get('statistical_result', {}).get(
                'adjusted_p_value',
                f.get('statistical_result', {}).get('p_value', 1.0),
            ) < 0.05
        )

        # 各因子の確信度ラベルを付与（画面には出さず、プロンプト用）
        features_with_confidence = [
            {
                **f,
                'confidence': self._compute_confidence_label(
                    f.get('statistical_result', {}).get('p_value', 1.0),
                    abs(f.get('statistical_result', {}).get('cohen_d', 0.0)),
                ),
            }
            for f in features
        ]

        # 同一場所での変数共起パターンの検出
        rack_vars: Dict[str, List[str]] = {}
        for f in features:
            rack = f.get('rack', '')
            var = f.get('variable', '')
            rack_vars.setdefault(rack, []).append(var)

        co_occurrences = [
            (rack, vars_list)
            for rack, vars_list in rack_vars.items()
            if len(vars_list) > 1
        ]

        return {
            'total_count': len(features),
            'significant_count': significant_count,
            'dominant_variable': (
                var_counts.most_common(1)[0][0] if var_counts else 'N/A'
            ),
            'rack_concentration': self._analyze_spatial_pattern(racks),
            'co_occurrences': co_occurrences,
            'features_with_confidence': features_with_confidence,
        }

    def _analyze_spatial_pattern(self, racks: List[str]) -> str:
        """空間位置の集中度を分析する。

        ユニーク位置数 / 全位置数 の比率で判定:
          < 0.3: 高集中, < 0.6: 中程度集中, >= 0.6: 広範囲分布

        Args:
            racks: 空間ラベルのリスト

        Returns:
            集中度を表す文字列
        """
        if not racks:
            return "no data"

        unique_count = len(set(racks))
        total = len(racks)
        concentration_ratio = unique_count / total if total > 0 else 0

        if concentration_ratio < 0.3:
            return "highly concentrated"
        elif concentration_ratio < 0.6:
            return "moderately concentrated"
        else:
            return "widely distributed"

    # ── レスポンス解析 ────────────────────────────────────────────────────────

    def _parse_json_response(
        self, response_text: str, features: List[Dict],
    ) -> Dict[str, Any]:
        """LLMレスポンスからJSONを抽出・パースする。

        マークダウンコードブロック（```json ... ```）を除去し、
        構造の妥当性を検証する。パース失敗時はフォールバックを返す。

        Args:
            response_text: LLMの生テキストレスポンス
            features: フォールバック用の特徴量データ

        Returns:
            新スキーマの3フィールドを含む解釈辞書
        """
        try:
            text = response_text.strip()
            # マークダウンコードブロックの除去
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]

            result = json.loads(text.strip())

            # 構造の検証: 必須3フィールドが存在するか
            required_keys = {
                'comparison_context',
                'separation_factors',
                'suggested_exploration',
            }
            if required_keys.issubset(result.keys()):
                return result

        except json.JSONDecodeError as e:
            logger.warning(f"JSONパースエラー: {e}")

        return self._fallback_interpretation(features)

    # ── フォールバック処理 ────────────────────────────────────────────────────

    def _fallback_interpretation(self, features: List[Dict]) -> Dict[str, Any]:
        """API不可時のフォールバック解釈を生成する。

        特徴量データから基本的な統計サマリーを構築し、
        最低限の構造化された解釈を返す。

        Args:
            features: 上位特徴量のリスト

        Returns:
            フォールバック解釈の辞書
        """
        if not features:
            return {
                "comparison_context": {
                    "cluster1_range": "Unknown",
                    "cluster2_range": "Unknown",
                    "cluster1_size": 0,
                    "cluster2_size": 0,
                    "text": "No features available for analysis.",
                },
                "separation_factors": {
                    "text": "No data available.",
                },
                "suggested_exploration": {
                    "text": "Full interpretation requires API access.",
                },
            }

        # 基本的な統計サマリーの生成
        top_vars = list(set(f.get('variable', '') for f in features[:5]))
        top_racks = list(set(f.get('rack', '') for f in features[:5]))
        top_features_str = [
            f"{f.get('rack', '?')}-{f.get('variable', '?')}"
            for f in features[:3]
        ]

        return {
            "comparison_context": {
                "cluster1_range": "Unknown",
                "cluster2_range": "Unknown",
                "cluster1_size": 0,
                "cluster2_size": 0,
                "text": "Cluster comparison context unavailable.",
            },
            "separation_factors": {
                "text": f"Top differentiating features: {', '.join(top_features_str)}.",
            },
            "suggested_exploration": {
                "text": "Full interpretation requires API access.",
            },
        }
