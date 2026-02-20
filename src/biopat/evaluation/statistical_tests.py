"""Statistical significance tests for IR evaluation.

Provides paired t-tests, bootstrap confidence intervals, and
Bonferroni correction for comparing retrieval systems.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def paired_t_test(
    scores_a: List[float],
    scores_b: List[float],
) -> Tuple[float, float]:
    """Per-query paired t-test between two systems.

    Args:
        scores_a: Per-query metric scores for system A.
        scores_b: Per-query metric scores for system B.

    Returns:
        (t_statistic, p_value) tuple.
    """
    from scipy import stats

    a = np.array(scores_a)
    b = np.array(scores_b)
    t_stat, p_val = stats.ttest_rel(a, b)
    return float(t_stat), float(p_val)


def bootstrap_confidence_interval(
    scores: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for a metric.

    Args:
        scores: Per-query metric scores.
        confidence: Confidence level (default 0.95).
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed.

    Returns:
        (mean, lower_bound, upper_bound) tuple.
    """
    rng = np.random.RandomState(seed)
    data = np.array(scores)
    n = len(data)

    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        means[i] = sample.mean()

    alpha = 1 - confidence
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    mean = float(data.mean())

    return mean, lower, upper


def bootstrap_paired_test(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap test for difference between two systems.

    Returns:
        (mean_diff, lower_ci, upper_ci) — if CI doesn't contain 0,
        the difference is significant at the corresponding level.
    """
    rng = np.random.RandomState(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    diffs = a - b
    n = len(diffs)

    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = sample.mean()

    lower = float(np.percentile(boot_means, 2.5))
    upper = float(np.percentile(boot_means, 97.5))
    mean_diff = float(diffs.mean())

    return mean_diff, lower, upper


def bonferroni_correction(
    p_values: List[float],
) -> List[float]:
    """Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values to correct.

    Returns:
        Corrected p-values (capped at 1.0).
    """
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def significance_matrix(
    per_query_scores: Dict[str, List[float]],
    metric_name: str = "",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute pairwise significance between all systems.

    Args:
        per_query_scores: {system_name: [per-query scores]}.
        metric_name: Name of the metric being compared.

    Returns:
        Nested dict: {sys_a: {sys_b: {"t_stat": ..., "p_value": ..., "significant": ...}}}.
    """
    systems = list(per_query_scores.keys())
    matrix: Dict[str, Dict[str, Dict[str, float]]] = {}

    all_pvals = []
    pairs = []

    for i, sys_a in enumerate(systems):
        matrix[sys_a] = {}
        for j, sys_b in enumerate(systems):
            if i == j:
                matrix[sys_a][sys_b] = {"t_stat": 0.0, "p_value": 1.0, "significant": False}
                continue
            if j < i:
                # Reuse symmetric result
                matrix[sys_a][sys_b] = {
                    "t_stat": -matrix[sys_b][sys_a]["t_stat"],
                    "p_value": matrix[sys_b][sys_a]["p_value"],
                    "significant": matrix[sys_b][sys_a]["significant"],
                }
                continue

            t_stat, p_val = paired_t_test(per_query_scores[sys_a], per_query_scores[sys_b])
            all_pvals.append(p_val)
            pairs.append((sys_a, sys_b))
            matrix[sys_a][sys_b] = {"t_stat": t_stat, "p_value": p_val, "significant": False}

    # Bonferroni correction
    corrected = bonferroni_correction(all_pvals)
    for (sys_a, sys_b), p_corr in zip(pairs, corrected):
        sig = p_corr < 0.05
        matrix[sys_a][sys_b]["p_value_corrected"] = p_corr
        matrix[sys_a][sys_b]["significant"] = sig
        # Fill symmetric
        if sys_b in matrix and sys_a in matrix[sys_b]:
            matrix[sys_b][sys_a]["p_value_corrected"] = p_corr
            matrix[sys_b][sys_a]["significant"] = sig

    return matrix
