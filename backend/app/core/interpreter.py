"""
Refactored AI Interpreter - Now domain-agnostic using strategy pattern.
"""

import os
import json
from typing import List, Dict, Optional, Any
from collections import Counter
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from google import genai
except ImportError:
    genai = None


class GeminiInterpreter:
    """AI interpreter for cluster analysis using Gemini API with domain strategy."""

    def __init__(self, domain, api_key: Optional[str] = None):
        """Initialize with domain strategy.
        
        Args:
            domain: Domain strategy instance that provides prompts and domain knowledge
            api_key: Optional Gemini API key (reads from env var if not provided)
        """
        self.domain = domain
        
        # Read API key from parameter or environment variable
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if genai and api_key:
            self.client = genai.Client(api_key=api_key)
            print("Gemini API client initialized.") 
        else:
            print("Gemini API client not initialized. Set GEMINI_API_KEY environment variable.")
            self.client = None

    def interpret(
        self,
        top_features: List[Dict],
        cluster1_size: int,
        cluster2_size: int
    ) -> Dict[str, Any]:
        """Generate structured interpretation of cluster differences."""
        if not self.client or not top_features:
            return self._fallback_interpretation(top_features)

        # Preprocess data for better context
        preprocessed = self._preprocess_features(top_features)

        # Use domain-specific prompt builder
        prompt = self.domain.build_interpretation_prompt(
            top_features=top_features,
            cluster1_size=cluster1_size,
            cluster2_size=cluster2_size,
            preprocessed=preprocessed
        )

        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return self._parse_json_response(response.text, top_features)
        except Exception as e:
            print(f"API error: {e}")
            return self._fallback_interpretation(top_features)

    def _preprocess_features(self, features: List[Dict]) -> Dict[str, Any]:
        """Preprocess features to extract patterns."""
        # Variable distribution
        variables = [f.get('variable', '') for f in features]
        var_counts = Counter(variables)
        
        # Location distribution
        racks = [f.get('rack', '') for f in features]
        rack_counts = Counter(racks)
        
        # Statistical summary
        significant_count = sum(
            1 for f in features 
            if f.get('statistical_result', {}).get('p_value', 1) < 0.05
        )
        
        # Effect sizes
        effect_sizes = [
            abs(f.get('statistical_result', {}).get('cohen_d', 0)) 
            for f in features
        ]
        avg_effect = sum(effect_sizes) / len(effect_sizes) if effect_sizes else 0
        
        # Create distribution string
        total = len(features)
        var_dist = ", ".join([
            f"{v}: {c} ({100*c/total:.0f}%)" 
            for v, c in var_counts.most_common(4)
        ])
        
        # Check for co-occurring variables at same location
        rack_vars = {}
        for f in features:
            rack = f.get('rack', '')
            var = f.get('variable', '')
            if rack not in rack_vars:
                rack_vars[rack] = []
            rack_vars[rack].append(var)
        
        co_occurrences = [
            (rack, vars) for rack, vars in rack_vars.items() 
            if len(vars) > 1
        ]
        
        return {
            'total_count': total,
            'significant_count': significant_count,
            'dominant_variable': var_counts.most_common(1)[0][0] if var_counts else 'N/A',
            'variable_distribution': var_dist,
            'avg_effect_size': f"{avg_effect:.2f}",
            'rack_concentration': self._analyze_spatial_pattern(racks),
            'co_occurrences': co_occurrences
        }

    def _analyze_spatial_pattern(self, racks: List[str]) -> str:
        """Analyze if locations are spatially concentrated or distributed."""
        if not racks:
            return "no data"
        
        unique_racks = set(racks)
        total = len(racks)
        unique_count = len(unique_racks)
        
        concentration_ratio = unique_count / total if total > 0 else 0
        
        if concentration_ratio < 0.3:
            return "highly concentrated"
        elif concentration_ratio < 0.6:
            return "moderately concentrated"
        else:
            return "widely distributed"

    def _parse_json_response(self, response_text: str, features: List[Dict]) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        try:
            # Try to extract JSON from response
            text = response_text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            result = json.loads(text.strip())
            
            # Validate structure
            if 'sections' in result and isinstance(result['sections'], list):
                return result
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
        
        return self._fallback_interpretation(features)

    def _fallback_interpretation(self, features: List[Dict]) -> Dict[str, Any]:
        """Fallback interpretation when API is unavailable."""
        if not features:
            return {
                "sections": [
                    {
                        "title": "No Data",
                        "text": "No features available for analysis.",
                        "highlights": []
                    }
                ]
            }
        
        # Generate basic interpretation from data
        top_vars = list(set(f.get('variable', '') for f in features[:5]))
        top_racks = list(set(f.get('rack', '') for f in features[:5]))
        
        return {
            "sections": [
                {
                    "title": "Key Findings",
                    "text": f"Top differentiating variables: {', '.join(top_vars[:3])}. "
                           f"Most affected locations: {', '.join(top_racks[:3])}.",
                    "highlights": top_vars[:3]
                },
                {
                    "title": "Statistical Summary",
                    "text": f"Analysis identified {len(features)} important features. "
                           "Statistical significance varies across features.",
                    "highlights": []
                },
                {
                    "title": "Caveats",
                    "text": "This is an automated summary. Full LLM interpretation requires API access.",
                    "highlights": []
                }
            ]
        }

    def compare_analyses(
        self,
        analysis_a: Dict[str, Any],
        analysis_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two saved analyses using Gemini."""
        if not self.client:
            return self._fallback_comparison(analysis_a, analysis_b)

        # Use domain-specific comparison prompt
        prompt = self.domain.build_comparison_prompt(analysis_a, analysis_b)

        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return self._parse_json_response(response.text, [])
        except Exception as e:
            print(f"Compare API error: {e}")
            return self._fallback_comparison(analysis_a, analysis_b)

    def _fallback_comparison(
        self, 
        analysis_a: Dict[str, Any], 
        analysis_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback comparison when API is unavailable."""
        return {
            "sections": [
                {
                    "title": "Comparison Overview",
                    "text": f"Analysis A has {analysis_a.get('cluster1_size', 0)} vs {analysis_a.get('cluster2_size', 0)} points. "
                           f"Analysis B has {analysis_b.get('cluster1_size', 0)} vs {analysis_b.get('cluster2_size', 0)} points.",
                    "highlights": []
                },
                {
                    "title": "Feature Differences",
                    "text": "Detailed comparison requires LLM API access.",
                    "highlights": []
                }
            ]
        }
