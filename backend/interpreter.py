"""
AI Interpreter using Google Gemini API.
Enhanced version with structured JSON output and domain knowledge.
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
    """AI interpreter for cluster analysis using Gemini API."""

    # Domain knowledge for tensor data analysis
    DOMAIN_KNOWLEDGE = """
## Tensor Data Analysis Domain Knowledge

### Variables
- The tensor data contains multivariate time-series measurements
- Each variable represents a different measurement type or sensor
- Variables may have different scales and units

### Spatial Structure
- Data points are organized in a spatial grid structure
- Adjacent locations may exhibit correlated patterns
- Spatial clustering may indicate localized phenomena

### Temporal Patterns
- Time points are grouped by predefined labels (e.g., periods, conditions)
- Cluster comparisons reveal temporal dynamics
- Statistical significance indicates reliable differences
"""

    def __init__(self, api_key: Optional[str] = None):
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

        prompt = self._build_prompt(
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

    def _build_prompt(
        self,
        top_features: List[Dict],
        cluster1_size: int,
        cluster2_size: int,
        preprocessed: Dict[str, Any]
    ) -> str:
        """Build the prompt for LLM."""
        features_text = self._format_features(top_features[:30])

        return f"""
You are an AI assistant that generates structured summaries of tensor data cluster analysis results.
Your role is to organize and describe the data clearly - NOT to make causal claims or predictions.

{self.DOMAIN_KNOWLEDGE}

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

    def _format_features(self, features: List[Dict]) -> str:
        """Format features for prompt."""
        lines = []
        for f in features:
            stat = f.get('statistical_result', {})
            direction = "higher in C1" if stat.get('mean_diff', 0) > 0 else "lower in C1"
            sig = "*" if stat.get('p_value', 1) < 0.05 else ("+" if stat.get('p_value', 1) < 0.1 else "")
            effect = stat.get('effect_size', 'N/A')
            
            lines.append(
                f"- {f.get('rack', 'N/A')}/{f.get('variable', 'N/A')}: "
                f"score={f.get('score', 0):.3f}, {direction}, "
                f"effect={effect}{sig}"
            )
        return "\n".join(lines)

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

        # Check which clusters match between the two analyses
        same_cluster1 = analysis_a.get('cluster1_size') == analysis_b.get('cluster1_size')
        same_cluster2 = analysis_a.get('cluster2_size') == analysis_b.get('cluster2_size')
        
        # Determine comparison context
        if same_cluster1 and same_cluster2:
            context_msg = "Both analyses use IDENTICAL cluster selections. Any differences in results are due to analysis parameters."
        elif same_cluster1:
            context_msg = "Both analyses share the SAME Red Cluster (C1) as the base. The comparison involves different Blue Cluster (C2) selections."
        elif same_cluster2:
            context_msg = "Both analyses share the SAME Blue Cluster (C2) as the base. The comparison involves different Red Cluster (C1) selections."
        else:
            context_msg = "The analyses use completely different cluster selections."
        
        # Extract top features
        features_a = analysis_a.get('top_features', [])[:10]
        features_b = analysis_b.get('top_features', [])[:10]
        
        feature_set_a = set(f"{f.get('rack')}-{f.get('variable')}" for f in features_a)
        feature_set_b = set(f"{f.get('rack')}-{f.get('variable')}" for f in features_b)
        
        common = feature_set_a & feature_set_b
        only_a = feature_set_a - feature_set_b
        only_b = feature_set_b - feature_set_a

        prompt = f"""
You are comparing two cluster analysis results from a tensor data visualization system.

## Comparison Context
{context_msg}

## Analysis A
- Red Cluster size: {analysis_a.get('cluster1_size')} time points
- Blue Cluster size: {analysis_a.get('cluster2_size')} time points
- Significant features: {analysis_a.get('significant_count', 'N/A')}
- Top variables: {', '.join(analysis_a.get('top_variables', [])[:5])}
- Top locations: {', '.join(analysis_a.get('top_racks', [])[:5])}

## Analysis B
- Red Cluster size: {analysis_b.get('cluster1_size')} time points
- Blue Cluster size: {analysis_b.get('cluster2_size')} time points
- Significant features: {analysis_b.get('significant_count', 'N/A')}
- Top variables: {', '.join(analysis_b.get('top_variables', [])[:5])}
- Top locations: {', '.join(analysis_b.get('top_racks', [])[:5])}

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
