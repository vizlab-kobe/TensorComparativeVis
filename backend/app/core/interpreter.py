"""
AI解釈エンジンモジュール

Gemini APIを使用してクラスター分析結果を自然言語で解釈する。
ドメイン戦略パターンにより、プロンプト生成はドメインオブジェクトに委譲し、
このモジュールはAPI呼び出しとレスポンス解析に専念する。

主な機能:
  - interpret(): 特徴量データからAI解釈を生成
  - compare_analyses(): 2つの分析結果を比較するAI解釈を生成
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
    ) -> Dict[str, Any]:
        """クラスター差異の構造化された解釈を生成する。

        処理フロー:
          1. 特徴量データを前処理してパターンを抽出
          2. ドメイン戦略からプロンプトを構築
          3. Gemini APIに送信
          4. JSONレスポンスをパース

        Args:
            top_features: 上位特徴量のリスト（辞書形式）
            cluster1_size: 赤クラスターのサンプル数
            cluster2_size: 青クラスターのサンプル数

        Returns:
            sections キーを含む構造化された解釈辞書
        """
        if not self.client or not top_features:
            return self._fallback_interpretation(top_features)

        # 特徴量データを前処理してコンテキスト情報を抽出
        preprocessed = self._preprocess_features(top_features)

        # ドメイン固有のプロンプトを構築
        prompt = self.domain.build_interpretation_prompt(
            top_features=top_features,
            cluster1_size=cluster1_size,
            cluster2_size=cluster2_size,
            preprocessed=preprocessed,
        )

        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            return self._parse_json_response(response.text, top_features)
        except Exception as e:
            logger.error(f"API呼び出しエラー: {e}")
            return self._fallback_interpretation(top_features)

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

        # 統計的有意性の集計
        significant_count = sum(
            1 for f in features
            if f.get('statistical_result', {}).get('p_value', 1) < 0.05
        )

        # 効果量の平均
        effect_sizes = [
            abs(f.get('statistical_result', {}).get('cohen_d', 0))
            for f in features
        ]
        avg_effect = sum(effect_sizes) / len(effect_sizes) if effect_sizes else 0

        # 変数分布の文字列表現（プロンプト用）
        total = len(features)
        var_dist = ", ".join([
            f"{v}: {c} ({100 * c / total:.0f}%)"
            for v, c in var_counts.most_common(4)
        ])

        # 同一場所での変数共起パターンの検出
        rack_vars: Dict[str, List[str]] = {}
        for f in features:
            rack = f.get('rack', '')
            var = f.get('variable', '')
            if rack not in rack_vars:
                rack_vars[rack] = []
            rack_vars[rack].append(var)

        co_occurrences = [
            (rack, vars_list)
            for rack, vars_list in rack_vars.items()
            if len(vars_list) > 1
        ]

        return {
            'total_count': total,
            'significant_count': significant_count,
            'dominant_variable': var_counts.most_common(1)[0][0] if var_counts else 'N/A',
            'variable_distribution': var_dist,
            'avg_effect_size': f"{avg_effect:.2f}",
            'rack_concentration': self._analyze_spatial_pattern(racks),
            'co_occurrences': co_occurrences,
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
        self, response_text: str, features: List[Dict]
    ) -> Dict[str, Any]:
        """LLMレスポンスからJSONを抽出・パースする。

        マークダウンコードブロック（```json ... ```）を除去し、
        構造の妥当性を検証する。パース失敗時はフォールバックを返す。

        Args:
            response_text: LLMの生テキストレスポンス
            features: フォールバック用の特徴量データ

        Returns:
            sections キーを含む解釈辞書
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

            # 構造の検証: sections 配列が存在するか
            if 'sections' in result and isinstance(result['sections'], list):
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
                "sections": [{
                    "title": "No Data",
                    "text": "No features available for analysis.",
                    "highlights": [],
                }]
            }

        # 基本的な統計サマリーの生成
        top_vars = list(set(f.get('variable', '') for f in features[:5]))
        top_racks = list(set(f.get('rack', '') for f in features[:5]))

        return {
            "sections": [
                {
                    "title": "Key Findings",
                    "text": (
                        f"Top differentiating variables: {', '.join(top_vars[:3])}. "
                        f"Most affected locations: {', '.join(top_racks[:3])}."
                    ),
                    "highlights": top_vars[:3],
                },
                {
                    "title": "Statistical Summary",
                    "text": (
                        f"Analysis identified {len(features)} important features. "
                        "Statistical significance varies across features."
                    ),
                    "highlights": [],
                },
                {
                    "title": "Caveats",
                    "text": "This is an automated summary. Full LLM interpretation requires API access.",
                    "highlights": [],
                },
            ]
        }

    # ── 分析比較 ──────────────────────────────────────────────────────────────

    def compare_analyses(
        self,
        analysis_a: Dict[str, Any],
        analysis_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        """2つの保存済み分析結果をAIで比較する。

        各分析のクラスターサイズ、上位特徴量、統計情報を
        ドメイン固有のプロンプトで比較し、構造化された結果を返す。

        Args:
            analysis_a: 1つ目の分析結果
            analysis_b: 2つ目の分析結果

        Returns:
            比較結果の構造化辞書
        """
        if not self.client:
            return self._fallback_comparison(analysis_a, analysis_b)

        # ドメイン固有の比較プロンプトを構築
        prompt = self.domain.build_comparison_prompt(analysis_a, analysis_b)

        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            return self._parse_json_response(response.text, [])
        except Exception as e:
            logger.error(f"分析比較APIエラー: {e}")
            return self._fallback_comparison(analysis_a, analysis_b)

    def _fallback_comparison(
        self,
        analysis_a: Dict[str, Any],
        analysis_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        """API不可時の比較フォールバックを返す。

        Args:
            analysis_a: 1つ目の分析結果
            analysis_b: 2つ目の分析結果

        Returns:
            基本的な比較結果の辞書
        """
        return {
            "sections": [
                {
                    "title": "Comparison Overview",
                    "text": (
                        f"Analysis A has {analysis_a.get('cluster1_size', 0)} vs "
                        f"{analysis_a.get('cluster2_size', 0)} points. "
                        f"Analysis B has {analysis_b.get('cluster1_size', 0)} vs "
                        f"{analysis_b.get('cluster2_size', 0)} points."
                    ),
                    "highlights": [],
                },
                {
                    "title": "Feature Differences",
                    "text": "Detailed comparison requires LLM API access.",
                    "highlights": [],
                },
            ]
        }
