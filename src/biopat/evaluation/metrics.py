"""Metrics computation module.

Computes standard IR evaluation metrics for the benchmark.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Optional
import math

logger = logging.getLogger(__name__)


class MetricsComputer:
    """Computes IR evaluation metrics."""

    @staticmethod
    def precision_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        """Compute Precision@k.

        Args:
            retrieved: List of retrieved document IDs (ranked).
            relevant: Set of relevant document IDs.
            k: Cutoff rank.

        Returns:
            Precision@k value.
        """
        if k <= 0:
            return 0.0

        retrieved_k = retrieved[:k]
        hits = sum(1 for doc_id in retrieved_k if doc_id in relevant)
        return hits / k

    @staticmethod
    def recall_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        """Compute Recall@k.

        Args:
            retrieved: List of retrieved document IDs (ranked).
            relevant: Set of relevant document IDs.
            k: Cutoff rank.

        Returns:
            Recall@k value.
        """
        if not relevant:
            return 0.0

        retrieved_k = set(retrieved[:k])
        hits = len(retrieved_k & relevant)
        return hits / len(relevant)

    @staticmethod
    def average_precision(
        retrieved: List[str],
        relevant: Set[str],
    ) -> float:
        """Compute Average Precision.

        Args:
            retrieved: List of retrieved document IDs (ranked).
            relevant: Set of relevant document IDs.

        Returns:
            Average Precision value.
        """
        if not relevant:
            return 0.0

        hits = 0
        sum_precisions = 0.0

        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                hits += 1
                precision_at_i = hits / i
                sum_precisions += precision_at_i

        return sum_precisions / len(relevant)

    @staticmethod
    def dcg_at_k(
        retrieved: List[str],
        relevance_scores: Dict[str, int],
        k: int,
    ) -> float:
        """Compute Discounted Cumulative Gain at k.

        Args:
            retrieved: List of retrieved document IDs (ranked).
            relevance_scores: Dictionary mapping doc_id to relevance score.
            k: Cutoff rank.

        Returns:
            DCG@k value.
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], 1):
            rel = relevance_scores.get(doc_id, 0)
            dcg += (2 ** rel - 1) / math.log2(i + 1)
        return dcg

    @staticmethod
    def ndcg_at_k(
        retrieved: List[str],
        relevance_scores: Dict[str, int],
        k: int,
    ) -> float:
        """Compute Normalized DCG at k.

        Args:
            retrieved: List of retrieved document IDs (ranked).
            relevance_scores: Dictionary mapping doc_id to relevance score.
            k: Cutoff rank.

        Returns:
            NDCG@k value.
        """
        dcg = MetricsComputer.dcg_at_k(retrieved, relevance_scores, k)

        # Compute ideal DCG
        sorted_rels = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(sorted_rels, 1):
            idcg += (2 ** rel - 1) / math.log2(i + 1)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def mrr(
        retrieved: List[str],
        relevant: Set[str],
    ) -> float:
        """Compute Mean Reciprocal Rank for a single query.

        Args:
            retrieved: List of retrieved document IDs (ranked).
            relevant: Set of relevant document IDs.

        Returns:
            Reciprocal rank (1/position of first relevant doc).
        """
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                return 1.0 / i
        return 0.0

    def compute_all_metrics(
        self,
        results: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        k_values: List[int] = [10, 50, 100],
    ) -> Dict[str, float]:
        """Compute all metrics for a result set.

        Args:
            results: Retrieved results {query_id: {doc_id: score}}.
            qrels: Ground truth {query_id: {doc_id: relevance}}.
            k_values: Values of k for metrics.

        Returns:
            Dictionary of aggregated metrics.
        """
        all_precisions = {k: [] for k in k_values}
        all_recalls = {k: [] for k in k_values}
        all_ndcgs = {k: [] for k in k_values}
        all_aps = []
        all_mrrs = []

        for query_id, retrieved_scores in results.items():
            if query_id not in qrels:
                continue

            query_qrels = qrels[query_id]
            relevant = set(query_qrels.keys())

            # Sort retrieved by score
            retrieved = sorted(
                retrieved_scores.keys(),
                key=lambda x: retrieved_scores[x],
                reverse=True
            )

            # Compute metrics at each k
            for k in k_values:
                all_precisions[k].append(self.precision_at_k(retrieved, relevant, k))
                all_recalls[k].append(self.recall_at_k(retrieved, relevant, k))
                all_ndcgs[k].append(self.ndcg_at_k(retrieved, query_qrels, k))

            # MAP and MRR
            all_aps.append(self.average_precision(retrieved, relevant))
            all_mrrs.append(self.mrr(retrieved, relevant))

        # Aggregate
        metrics = {}

        for k in k_values:
            if all_precisions[k]:
                metrics[f"P@{k}"] = sum(all_precisions[k]) / len(all_precisions[k])
            if all_recalls[k]:
                metrics[f"Recall@{k}"] = sum(all_recalls[k]) / len(all_recalls[k])
            if all_ndcgs[k]:
                metrics[f"NDCG@{k}"] = sum(all_ndcgs[k]) / len(all_ndcgs[k])

        if all_aps:
            metrics["MAP"] = sum(all_aps) / len(all_aps)
        if all_mrrs:
            metrics["MRR"] = sum(all_mrrs) / len(all_mrrs)

        return metrics

    def compute_per_domain_metrics(
        self,
        results: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        query_domains: Dict[str, str],
        k_values: List[int] = [10, 100],
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics broken down by domain.

        Args:
            results: Retrieved results.
            qrels: Ground truth.
            query_domains: Mapping of query_id to domain.
            k_values: Values of k for metrics.

        Returns:
            Dictionary of domain to metrics.
        """
        # Group queries by domain
        domain_results = {}
        domain_qrels = {}

        for query_id in results:
            if query_id not in qrels:
                continue

            domain = query_domains.get(query_id, "unknown")

            if domain not in domain_results:
                domain_results[domain] = {}
                domain_qrels[domain] = {}

            domain_results[domain][query_id] = results[query_id]
            domain_qrels[domain][query_id] = qrels[query_id]

        # Compute metrics per domain
        domain_metrics = {}
        for domain in domain_results:
            domain_metrics[domain] = self.compute_all_metrics(
                domain_results[domain],
                domain_qrels[domain],
                k_values
            )
            domain_metrics[domain]["num_queries"] = len(domain_results[domain])

        return domain_metrics

    def format_metrics_table(
        self,
        metrics: Dict[str, float],
        name: str = "Model",
    ) -> str:
        """Format metrics as a readable table.

        Args:
            metrics: Dictionary of metrics.
            name: Model/run name.

        Returns:
            Formatted string.
        """
        lines = [f"\n{'=' * 50}", f"Results for: {name}", "=" * 50]

        # Sort metrics for consistent display
        sorted_metrics = sorted(metrics.items())

        for metric_name, value in sorted_metrics:
            lines.append(f"{metric_name:15s}: {value:.4f}")

        lines.append("=" * 50)
        return "\n".join(lines)
