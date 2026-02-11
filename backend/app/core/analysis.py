"""
Refactored Analysis module - Now domain-agnostic.
Accepts domain strategy for variable names and label conversion.
"""

import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Tuple, Optional, Protocol, runtime_checkable
import pacmap
import warnings


# ── Domain Protocol (structural typing for domain parameter) ──────────────

@runtime_checkable
class DomainProtocol(Protocol):
    """Structural type for domain objects used by analysis functions."""

    @property
    def variables(self) -> List[str]: ...

    @property
    def grid_shape(self) -> Tuple[int, int]: ...

    def index_to_label(self, index: int) -> str: ...


# ── Configurable ML parameters ───────────────────────────────────────────

_RF_DEFAULTS: Dict = {
    'n_estimators': 300,
    'max_depth': 12,
    'min_samples_leaf': 10,
    'min_samples_split': 20,
    'max_features': 0.5,
    'bootstrap': True,
    'oob_score': True,
    'n_jobs': -1,
    'random_state': 42,
}

_PACMAP_DEFAULTS: Dict = {
    'n_components': 2,
    'n_neighbors': None,
}

# Mutable runtime copies (updated by configure())
RF_PARAMS: Dict = {**_RF_DEFAULTS}
PACMAP_PARAMS: Dict = {**_PACMAP_DEFAULTS}


def configure(
    *,
    random_forest: Optional[Dict] = None,
    pacmap_params: Optional[Dict] = None,
) -> None:
    """Apply external configuration (from YAML) to ML parameters.
    
    Called once at app startup from main_new.py.
    Only keys present in the override dict are merged; others keep defaults.
    """
    global RF_PARAMS, PACMAP_PARAMS
    if random_forest:
        RF_PARAMS = {**_RF_DEFAULTS, **random_forest}
    if pacmap_params:
        PACMAP_PARAMS = {**_PACMAP_DEFAULTS, **pacmap_params}


def unfold_and_scale_tensor(tensor: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """Unfold tensor and apply standard scaling."""
    T, S, V = tensor.shape
    unfolded_tensor = tensor.reshape(T, S * V)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(unfolded_tensor)
    return scaled_data, scaler


def apply_pacmap_reduction(scaled_data: np.ndarray) -> np.ndarray:
    """Apply PaCMAP dimensionality reduction."""
    return pacmap.PaCMAP(**PACMAP_PARAMS).fit_transform(scaled_data)


def create_binary_classification_data(
    cluster1_indices: List[int],
    cluster2_indices: List[int],
    scaled_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Create binary classification data from two selected clusters."""
    combined_data = np.concatenate(
        [scaled_data[cluster1_indices], scaled_data[cluster2_indices]], axis=0
    )
    combined_labels = np.array(
        [1] * len(cluster1_indices) + [0] * len(cluster2_indices)
    )
    return combined_data, combined_labels


def compute_feature_importance(
    combined_data: np.ndarray, 
    combined_labels: np.ndarray
) -> np.ndarray:
    """Compute feature importance using Random Forest."""
    if len(np.unique(combined_labels)) == 1:
        return np.zeros(combined_data.shape[1])

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(combined_data, combined_labels)
    return rf.feature_importances_


def analyze_tensor_contribution(
    cluster1_indices: List[int],
    cluster2_indices: List[int],
    scaled_data: np.ndarray,
    Ms: np.ndarray,
    Mv: np.ndarray,
    S: int,
    V: int
) -> np.ndarray:
    """Analyze tensor contribution using feature importance and projection matrices."""
    combined_data, combined_labels = create_binary_classification_data(
        cluster1_indices, cluster2_indices, scaled_data
    )
    feature_importance = compute_feature_importance(combined_data, combined_labels)
    all_importances = (np.kron(Ms, Mv) @ feature_importance).reshape(S, V)
    return all_importances


def standardize_contributions(contribution_matrix: np.ndarray) -> np.ndarray:
    """Standardize contribution matrix per variable."""
    _, V = contribution_matrix.shape
    standardized = np.zeros_like(contribution_matrix)

    for v in range(V):
        contrib_v = contribution_matrix[:, v]
        mean_v = contrib_v.mean()
        std_v = contrib_v.std()

        if std_v > 0:
            standardized[:, v] = (contrib_v - mean_v) / std_v
        else:
            standardized[:, v] = 0

    return standardized


def get_top_important_factors(
    contribution_matrix: np.ndarray,
    domain: DomainProtocol,
    top_k: int = 10
) -> List[Dict]:
    """Get top important factors sorted by importance."""
    standardized_contrib = standardize_contributions(contribution_matrix)
    importance_scores = np.abs(standardized_contrib)

    _, V = importance_scores.shape
    flat_scores = importance_scores.flatten()
    flat_indices = np.argsort(flat_scores)[::-1]

    top_factors = []
    for i in range(min(top_k, len(flat_indices))):
        flat_idx = flat_indices[i]
        s = flat_idx // V
        v = flat_idx % V
        score = float(flat_scores[flat_idx])

        top_factors.append({
            'rank': i + 1,
            'rack': domain.index_to_label(s),
            'variable': domain.variables[v],
            'score': score,
            'rack_idx': int(s),
            'var_idx': int(v)
        })

    return top_factors


def calculate_cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2)
    )
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0


def evaluate_statistical_significance(
    cluster1_indices: List[int],
    cluster2_indices: List[int],
    rack_idx: int,
    var_idx: int,
    original_data: np.ndarray,
    domain: DomainProtocol
) -> Dict:
    """Evaluate statistical significance between two clusters."""
    time_series = original_data[:, rack_idx, var_idx]
    
    cluster1_values = time_series[cluster1_indices]
    cluster2_values = time_series[cluster2_indices]

    min_sample_size = 3
    if len(cluster1_values) < min_sample_size or len(cluster2_values) < min_sample_size:
        return {
            'rack': domain.index_to_label(rack_idx),
            'variable': domain.variables[var_idx],
            'direction': "Insufficient samples",
            'mean_diff': 0.0,
            'p_value': 1.0,
            'cohen_d': 0.0,
            'significance': "Cannot determine",
            'effect_size': "Cannot determine"
        }

    # Handle zero variance cases
    if np.var(cluster1_values) == 0 and np.var(cluster2_values) == 0:
        if np.mean(cluster1_values) == np.mean(cluster2_values):
            direction = "No difference"
            mean_diff = 0.0
        else:
            direction = "Higher in Cluster 1" if np.mean(cluster1_values) > np.mean(cluster2_values) else "Lower in Cluster 1"
            mean_diff = abs(np.mean(cluster1_values) - np.mean(cluster2_values))

        return {
            'rack': domain.index_to_label(rack_idx),
            'variable': domain.variables[var_idx],
            'direction': direction,
            'mean_diff': mean_diff,
            'p_value': 0.0 if direction != "No difference" else 1.0,
            'cohen_d': float('inf') if direction != "No difference" else 0.0,
            'significance': "Perfect separation" if direction != "No difference" else "No difference",
            'effect_size': "Infinite" if direction != "No difference" else "None"
        }

    # Perform t-test
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_stat, p_value = stats.ttest_ind(cluster1_values, cluster2_values, equal_var=False)

        if np.isnan(p_value) or np.isnan(t_stat):
            p_value = 1.0
    except Exception:
        p_value = 1.0

    # Calculate Cohen's d
    try:
        cohen_d = calculate_cohen_d(cluster1_values, cluster2_values)
        if np.isnan(cohen_d) or np.isinf(cohen_d):
            cohen_d = 0.0
    except Exception:
        cohen_d = 0.0

    mean_diff = np.mean(cluster1_values) - np.mean(cluster2_values)
    direction = "Higher in Cluster 1" if mean_diff > 0 else "Lower in Cluster 1"

    # Effect size interpretation
    abs_cohen_d = abs(cohen_d)
    if abs_cohen_d >= 0.8:
        effect_size = "Large"
    elif abs_cohen_d >= 0.5:
        effect_size = "Medium"
    elif abs_cohen_d >= 0.2:
        effect_size = "Small"
    else:
        effect_size = "Very small"

    return {
        'rack': domain.index_to_label(rack_idx),
        'variable': domain.variables[var_idx],
        'direction': direction,
        'mean_diff': float(abs(mean_diff)),
        'p_value': float(p_value),
        'cohen_d': float(abs_cohen_d),
        'significance': "Significant" if p_value < 0.05 else "Not significant",
        'effect_size': effect_size
    }
