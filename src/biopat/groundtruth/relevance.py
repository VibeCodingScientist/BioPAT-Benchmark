"""Relevance assignment module.

Computes graded relevance scores for query-document pairs
based on citation source, confidence, and rejection type.
"""

import logging
from pathlib import Path
from typing import Optional
from enum import IntEnum
import polars as pl

from biopat import compat

logger = logging.getLogger(__name__)


class RelevanceLevel(IntEnum):
    """Relevance tier definitions following BioPAT specification."""
    NOT_RELEVANT = 0
    RELEVANT = 1
    HIGHLY_RELEVANT = 2
    NOVELTY_DESTROYING = 3


RELEVANCE_LABELS = {
    RelevanceLevel.NOT_RELEVANT: "not_relevant",
    RelevanceLevel.RELEVANT: "relevant",
    RelevanceLevel.HIGHLY_RELEVANT: "highly_relevant",
    RelevanceLevel.NOVELTY_DESTROYING: "novelty_destroying",
}


class RelevanceAssigner:
    """Assigns relevance scores to query-document pairs."""

    def __init__(self, processed_dir: Path):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @property
    def qrels_path(self) -> Path:
        return self.processed_dir / "qrels.parquet"

    def assign_binary_relevance(
        self,
        citation_links: pl.DataFrame,
        confidence_threshold: int = 8,
    ) -> pl.DataFrame:
        """Assign binary relevance (Phase 1).

        Phase 1 uses binary relevance:
        - 1 if high-confidence citation exists
        - 0 otherwise

        Args:
            citation_links: Validated citation links with temporal constraints applied.
            confidence_threshold: Minimum confidence for positive relevance.

        Returns:
            DataFrame with query_id, doc_id, relevance columns.
        """
        # Filter by confidence
        qrels = citation_links.filter(
            pl.col("confidence") >= confidence_threshold
        )

        # Create qrels format
        qrels = qrels.select([
            pl.col("query_id"),
            pl.col("paper_id").alias("doc_id"),
            pl.lit(1).alias("relevance"),
            pl.lit("citation").alias("evidence_source"),
            pl.col("confidence"),
            pl.col("source"),
        ])

        # Deduplicate (keep highest confidence if multiple citations)
        qrels = compat.unique(
            qrels.sort("confidence", descending=True),
            subset=["query_id", "doc_id"],
            keep="first"
        )

        logger.info(f"Created {len(qrels)} binary relevance judgments")
        return qrels

    def assign_graded_relevance(
        self,
        citation_links: pl.DataFrame,
        office_action_data: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """Assign graded relevance (Phase 2+).

        Relevance tiers:
        - 3 (novelty_destroying): Examiner cited under 102 (anticipation)
        - 2 (highly_relevant): Examiner cited under 103 OR high-confidence examiner citation
        - 1 (relevant): Medium-confidence examiner OR high-confidence applicant citation
        - 0 (not_relevant): Low confidence or no link

        Args:
            citation_links: Validated citation links.
            office_action_data: Optional Office Action rejection data (Phase 2+).

        Returns:
            DataFrame with graded relevance scores.
        """
        # Start with basic relevance from citation source and confidence
        qrels = citation_links.with_columns([
            pl.lit(0).alias("relevance"),
            pl.lit("unknown").alias("label"),
        ])

        # Apply relevance rules
        qrels = qrels.with_columns(
            pl.when(
                # High-confidence examiner citations
                (pl.col("source") == "examiner") &
                (pl.col("confidence") >= 9)
            )
            .then(pl.lit(RelevanceLevel.HIGHLY_RELEVANT))
            .when(
                # Medium-confidence examiner citations
                (pl.col("source") == "examiner") &
                (pl.col("confidence") >= 7)
            )
            .then(pl.lit(RelevanceLevel.RELEVANT))
            .when(
                # High-confidence applicant citations
                (pl.col("source") == "applicant") &
                (pl.col("confidence") >= 8)
            )
            .then(pl.lit(RelevanceLevel.RELEVANT))
            .otherwise(pl.lit(RelevanceLevel.NOT_RELEVANT))
            .alias("relevance")
        )

        # If Office Action data is provided, upgrade relevance for 102/103 rejections
        if office_action_data is not None and len(office_action_data) > 0:
            # Join with OA data
            qrels = qrels.join(
                office_action_data.select(["query_id", "doc_id", "rejection_type"]),
                on=["query_id", "doc_id"],
                how="left"
            )

            # Upgrade relevance based on rejection type
            qrels = qrels.with_columns(
                pl.when(pl.col("rejection_type") == "102")
                .then(pl.lit(RelevanceLevel.NOVELTY_DESTROYING))
                .when(pl.col("rejection_type") == "103")
                .then(pl.lit(RelevanceLevel.HIGHLY_RELEVANT))
                .otherwise(pl.col("relevance"))
                .alias("relevance")
            )

        # Add human-readable labels
        qrels = qrels.with_columns(
            compat.map_elements(
                pl.col("relevance"),
                lambda r: RELEVANCE_LABELS.get(r, "unknown"),
                return_dtype=pl.Utf8
            ).alias("label")
        )

        # Format output
        qrels = qrels.select([
            pl.col("query_id"),
            pl.col("paper_id").alias("doc_id"),
            pl.col("relevance"),
            pl.col("label"),
            pl.col("source").alias("evidence_source"),
            pl.col("confidence"),
        ])

        # Deduplicate (keep highest relevance if multiple entries)
        qrels = compat.unique(
            qrels.sort("relevance", descending=True),
            subset=["query_id", "doc_id"],
            keep="first"
        )

        # Filter out not_relevant (keep only positive judgments for qrels)
        qrels = qrels.filter(pl.col("relevance") > 0)

        logger.info(f"Created {len(qrels)} graded relevance judgments")
        self._log_relevance_distribution(qrels)

        return qrels

    def _log_relevance_distribution(self, qrels: pl.DataFrame):
        """Log the distribution of relevance levels."""
        distribution = (
            compat.group_by(qrels, ["relevance", "label"])
            .agg(pl.count().alias("count"))
            .sort("relevance", descending=True)
        )

        logger.info("Relevance distribution:")
        for row in compat.iter_rows(distribution, named=True):
            logger.info(f"  {row['label']} ({row['relevance']}): {row['count']}")

    def get_qrels_stats(self, qrels: pl.DataFrame) -> dict:
        """Get statistics about relevance judgments.

        Args:
            qrels: Qrels DataFrame.

        Returns:
            Dictionary of statistics.
        """
        stats = {
            "total_judgments": len(qrels),
            "unique_queries": qrels["query_id"].n_unique(),
            "unique_docs": qrels["doc_id"].n_unique(),
        }

        # Relevance distribution
        stats["relevance_distribution"] = (
            compat.group_by(qrels, "relevance")
            .agg(pl.count().alias("count"))
            .sort("relevance")
            .to_dicts()
        )

        # Judgments per query
        per_query = compat.group_by(qrels, "query_id").agg(pl.count().alias("count"))
        stats["mean_judgments_per_query"] = per_query["count"].mean()
        stats["median_judgments_per_query"] = per_query["count"].median()

        # Source distribution
        if "evidence_source" in qrels.columns:
            stats["source_distribution"] = (
                compat.group_by(qrels, "evidence_source")
                .agg(pl.count().alias("count"))
                .to_dicts()
            )

        return stats

    def create_qrels(
        self,
        citation_links: pl.DataFrame,
        graded: bool = False,
        office_action_data: Optional[pl.DataFrame] = None,
        confidence_threshold: int = 8,
        save: bool = True,
    ) -> pl.DataFrame:
        """Create relevance judgments (qrels) for the benchmark.

        Args:
            citation_links: Validated citation links.
            graded: If True, use graded relevance (Phase 2+). If False, binary (Phase 1).
            office_action_data: Optional Office Action data for Phase 2+.
            confidence_threshold: Minimum confidence for binary relevance.
            save: Whether to save qrels to disk.

        Returns:
            Qrels DataFrame.
        """
        if graded:
            qrels = self.assign_graded_relevance(citation_links, office_action_data)
        else:
            qrels = self.assign_binary_relevance(citation_links, confidence_threshold)

        # Log statistics
        stats = self.get_qrels_stats(qrels)
        logger.info(f"Qrels statistics: {stats}")

        if save:
            qrels.write_parquet(self.qrels_path)
            logger.info(f"Saved qrels to {self.qrels_path}")

        return qrels

    def load(self) -> pl.DataFrame:
        """Load qrels from disk.

        Returns:
            Qrels DataFrame.
        """
        if not self.qrels_path.exists():
            raise FileNotFoundError(f"Qrels not found at {self.qrels_path}")
        return pl.read_parquet(self.qrels_path)
