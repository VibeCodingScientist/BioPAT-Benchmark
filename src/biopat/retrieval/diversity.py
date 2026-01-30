"""Diversity Reranking for BioPAT.

Implements result diversification to avoid redundant results:
- MMR (Maximal Marginal Relevance)
- xQuAD (Explicit Query Aspect Diversification)
- PM-2 (Proportional Model)

Reference:
- Carbonell & Goldstein, "The Use of MMR, Diversity-Based Reranking" (1998)
- Santos et al., "Exploiting Query Reformulations for Web Search Result Diversification" (2010)
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DiverseResult:
    """A search result with diversity information."""

    doc_id: str
    relevance_score: float
    diversity_score: float
    final_score: float

    # Diversity metadata
    is_redundant: bool = False
    similar_to: List[str] = None  # IDs of similar already-selected docs

    def __post_init__(self):
        if self.similar_to is None:
            self.similar_to = []


class MMRDiversifier:
    """Maximal Marginal Relevance diversification.

    MMR balances relevance and novelty:
    MMR = λ * Sim(d, Q) - (1-λ) * max_{d_j ∈ S} Sim(d, d_j)

    where λ controls the relevance-diversity trade-off.
    """

    def __init__(
        self,
        lambda_param: float = 0.7,  # Higher = more relevance, lower = more diversity
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ):
        """
        Initialize MMR diversifier.

        Args:
            lambda_param: Trade-off parameter (0-1)
            similarity_fn: Function to compute document similarity
        """
        self.lambda_param = lambda_param
        self.similarity_fn = similarity_fn

        # Embedding cache for similarity computation
        self._embedding_cache: Dict[str, np.ndarray] = {}

    def set_embeddings(self, doc_embeddings: Dict[str, np.ndarray]):
        """Set document embeddings for similarity computation."""
        self._embedding_cache = doc_embeddings

    def _compute_similarity(
        self,
        doc1_id: str,
        doc2_id: str,
    ) -> float:
        """Compute similarity between two documents."""
        if self.similarity_fn:
            return self.similarity_fn(doc1_id, doc2_id)

        # Use embeddings if available
        if doc1_id in self._embedding_cache and doc2_id in self._embedding_cache:
            emb1 = self._embedding_cache[doc1_id]
            emb2 = self._embedding_cache[doc2_id]
            return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10))

        return 0.0

    def rerank(
        self,
        candidates: List[Tuple[str, float]],  # (doc_id, relevance_score)
        k: int = 10,
    ) -> List[DiverseResult]:
        """
        Rerank using MMR.

        Args:
            candidates: List of (doc_id, relevance_score) tuples
            k: Number of results to return

        Returns:
            List of DiverseResult with diversified ranking
        """
        if not candidates:
            return []

        # Normalize relevance scores to [0, 1]
        scores = [s for _, s in candidates]
        max_score = max(scores) if scores else 1.0
        min_score = min(scores) if scores else 0.0
        score_range = max_score - min_score if max_score != min_score else 1.0

        normalized = {
            doc_id: (score - min_score) / score_range
            for doc_id, score in candidates
        }

        selected = []
        remaining = set(doc_id for doc_id, _ in candidates)

        for _ in range(min(k, len(candidates))):
            best_doc = None
            best_mmr = float('-inf')

            for doc_id in remaining:
                # Relevance component
                relevance = normalized[doc_id]

                # Diversity component (max similarity to already selected)
                if selected:
                    max_sim = max(
                        self._compute_similarity(doc_id, sel.doc_id)
                        for sel in selected
                    )
                else:
                    max_sim = 0.0

                # MMR score
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_doc = doc_id

            if best_doc is None:
                break

            # Find similar docs
            similar_to = []
            if selected:
                for sel in selected:
                    sim = self._compute_similarity(best_doc, sel.doc_id)
                    if sim > 0.8:  # High similarity threshold
                        similar_to.append(sel.doc_id)

            result = DiverseResult(
                doc_id=best_doc,
                relevance_score=normalized[best_doc],
                diversity_score=1.0 - (max_sim if selected else 0.0),
                final_score=best_mmr,
                is_redundant=len(similar_to) > 0,
                similar_to=similar_to,
            )

            selected.append(result)
            remaining.remove(best_doc)

        return selected


class XQuADDiversifier:
    """xQuAD: Explicit Query Aspect Diversification.

    Diversifies results across query aspects/intents.
    """

    def __init__(
        self,
        lambda_param: float = 0.5,
    ):
        """
        Initialize xQuAD diversifier.

        Args:
            lambda_param: Trade-off parameter
        """
        self.lambda_param = lambda_param

    def rerank(
        self,
        candidates: List[Tuple[str, float]],
        doc_aspects: Dict[str, Set[str]],  # doc_id -> set of aspects covered
        k: int = 10,
    ) -> List[DiverseResult]:
        """
        Rerank using xQuAD.

        Args:
            candidates: List of (doc_id, relevance_score) tuples
            doc_aspects: Mapping of doc_id to aspects it covers
            k: Number of results to return

        Returns:
            Diversified results
        """
        if not candidates:
            return []

        # Get all aspects
        all_aspects = set()
        for aspects in doc_aspects.values():
            all_aspects.update(aspects)

        # Normalize scores
        scores = [s for _, s in candidates]
        max_score = max(scores) if scores else 1.0

        normalized = {
            doc_id: score / max_score if max_score > 0 else 0
            for doc_id, score in candidates
        }

        # Track aspect coverage
        aspect_coverage: Dict[str, float] = {a: 0.0 for a in all_aspects}

        selected = []
        remaining = set(doc_id for doc_id, _ in candidates)

        for _ in range(min(k, len(candidates))):
            best_doc = None
            best_score = float('-inf')

            for doc_id in remaining:
                relevance = normalized.get(doc_id, 0)
                aspects = doc_aspects.get(doc_id, set())

                # Compute diversity gain (coverage of uncovered aspects)
                diversity_gain = 0.0
                for aspect in aspects:
                    diversity_gain += (1 - aspect_coverage.get(aspect, 0))

                if aspects:
                    diversity_gain /= len(aspects)

                # xQuAD score
                score = (
                    self.lambda_param * relevance +
                    (1 - self.lambda_param) * diversity_gain
                )

                if score > best_score:
                    best_score = score
                    best_doc = doc_id

            if best_doc is None:
                break

            # Update aspect coverage
            for aspect in doc_aspects.get(best_doc, set()):
                aspect_coverage[aspect] = min(1.0, aspect_coverage[aspect] + 0.3)

            result = DiverseResult(
                doc_id=best_doc,
                relevance_score=normalized[best_doc],
                diversity_score=best_score - self.lambda_param * normalized[best_doc],
                final_score=best_score,
            )

            selected.append(result)
            remaining.remove(best_doc)

        return selected


class ClusterDiversifier:
    """Cluster-based diversification.

    Groups documents by similarity and selects representatives from each cluster.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        docs_per_cluster: int = 2,
    ):
        """
        Initialize cluster diversifier.

        Args:
            n_clusters: Number of clusters
            docs_per_cluster: Documents to select per cluster
        """
        self.n_clusters = n_clusters
        self.docs_per_cluster = docs_per_cluster

    def rerank(
        self,
        candidates: List[Tuple[str, float]],
        embeddings: Dict[str, np.ndarray],
        k: int = 10,
    ) -> List[DiverseResult]:
        """
        Rerank using clustering.

        Args:
            candidates: List of (doc_id, relevance_score) tuples
            embeddings: Document embeddings
            k: Number of results

        Returns:
            Diversified results
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            logger.warning("sklearn not available, falling back to relevance ranking")
            return [
                DiverseResult(
                    doc_id=doc_id,
                    relevance_score=score,
                    diversity_score=1.0,
                    final_score=score,
                )
                for doc_id, score in candidates[:k]
            ]

        if not candidates or not embeddings:
            return []

        # Get embeddings for candidates
        doc_ids = [doc_id for doc_id, _ in candidates if doc_id in embeddings]
        if not doc_ids:
            return []

        X = np.array([embeddings[doc_id] for doc_id in doc_ids])
        scores = {doc_id: score for doc_id, score in candidates}

        # Cluster
        n_clusters = min(self.n_clusters, len(doc_ids))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Group by cluster
        clusters: Dict[int, List[str]] = {i: [] for i in range(n_clusters)}
        for doc_id, label in zip(doc_ids, labels):
            clusters[label].append(doc_id)

        # Select from each cluster
        selected = []
        for cluster_id in range(n_clusters):
            cluster_docs = clusters[cluster_id]
            # Sort by relevance within cluster
            cluster_docs.sort(key=lambda x: scores.get(x, 0), reverse=True)

            for doc_id in cluster_docs[:self.docs_per_cluster]:
                result = DiverseResult(
                    doc_id=doc_id,
                    relevance_score=scores[doc_id],
                    diversity_score=1.0,  # Inherent diversity from clustering
                    final_score=scores[doc_id],
                )
                selected.append(result)

                if len(selected) >= k:
                    break

            if len(selected) >= k:
                break

        # Sort final results by relevance
        selected.sort(key=lambda x: x.relevance_score, reverse=True)

        return selected[:k]


class PatentDiversifier:
    """Patent-specific diversification.

    Diversifies across patent families, jurisdictions, and assignees
    to provide broader coverage of the prior art landscape.
    """

    def __init__(
        self,
        max_per_family: int = 2,
        max_per_assignee: int = 3,
        jurisdiction_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize patent diversifier.

        Args:
            max_per_family: Max patents per family
            max_per_assignee: Max patents per assignee
            jurisdiction_weights: Preference weights for jurisdictions
        """
        self.max_per_family = max_per_family
        self.max_per_assignee = max_per_assignee
        self.jurisdiction_weights = jurisdiction_weights or {
            "US": 1.0,
            "EP": 0.9,
            "WO": 0.8,
            "CN": 0.7,
            "JP": 0.7,
        }

    def rerank(
        self,
        candidates: List[Tuple[str, float]],
        patent_metadata: Dict[str, Dict[str, Any]],
        k: int = 10,
    ) -> List[DiverseResult]:
        """
        Rerank patents with diversity constraints.

        Args:
            candidates: List of (patent_id, relevance_score)
            patent_metadata: Metadata including family_id, assignee, jurisdiction
            k: Number of results

        Returns:
            Diversified patent results
        """
        if not candidates:
            return []

        # Track counts
        family_counts: Dict[str, int] = {}
        assignee_counts: Dict[str, int] = {}

        selected = []

        for patent_id, score in candidates:
            if len(selected) >= k:
                break

            metadata = patent_metadata.get(patent_id, {})
            family_id = metadata.get("family_id", patent_id)
            assignee = metadata.get("assignee", "unknown")
            jurisdiction = metadata.get("jurisdiction", "")

            # Check constraints
            if family_counts.get(family_id, 0) >= self.max_per_family:
                continue

            if assignee_counts.get(assignee, 0) >= self.max_per_assignee:
                continue

            # Apply jurisdiction weight
            jurisdiction_weight = self.jurisdiction_weights.get(jurisdiction, 0.5)
            adjusted_score = score * jurisdiction_weight

            result = DiverseResult(
                doc_id=patent_id,
                relevance_score=score,
                diversity_score=jurisdiction_weight,
                final_score=adjusted_score,
            )

            selected.append(result)
            family_counts[family_id] = family_counts.get(family_id, 0) + 1
            assignee_counts[assignee] = assignee_counts.get(assignee, 0) + 1

        # Sort by adjusted score
        selected.sort(key=lambda x: x.final_score, reverse=True)

        return selected


def create_mmr_diversifier(
    lambda_param: float = 0.7,
) -> MMRDiversifier:
    """Factory function for MMR diversifier."""
    return MMRDiversifier(lambda_param=lambda_param)


def create_xquad_diversifier(
    lambda_param: float = 0.5,
) -> XQuADDiversifier:
    """Factory function for xQuAD diversifier."""
    return XQuADDiversifier(lambda_param=lambda_param)


def create_patent_diversifier(
    max_per_family: int = 2,
    max_per_assignee: int = 3,
) -> PatentDiversifier:
    """Factory function for patent diversifier."""
    return PatentDiversifier(
        max_per_family=max_per_family,
        max_per_assignee=max_per_assignee,
    )
