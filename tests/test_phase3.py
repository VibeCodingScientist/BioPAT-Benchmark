"""Tests for Phase 3 modules."""

import pytest
from pathlib import Path
from collections import defaultdict

from biopat.evaluation.dense import (
    DenseRetriever,
    DenseRetrieverConfig,
    MODEL_REGISTRY,
)
from biopat.evaluation.hybrid import (
    ResultFusion,
    FusionConfig,
)
from biopat.evaluation.reranker import (
    CrossEncoderReranker,
    RerankerConfig,
    CROSS_ENCODER_REGISTRY,
)
from biopat.evaluation.ablation import (
    QueryRepresentationAblation,
    DocumentRepresentationAblation,
    IPCAblation,
    DomainAblation,
    TemporalAblation,
)
from biopat.evaluation.error_analysis import (
    ErrorAnalyzer,
    ErrorAnalysisConfig,
    FailureCategory,
    VocabularyAnalyzer,
)
from biopat.evaluation.metrics import MetricsComputer


class TestDenseRetrieverConfig:
    """Tests for dense retriever configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DenseRetrieverConfig()

        assert config.model_name == "contriever"
        assert config.batch_size == 32
        assert config.max_length == 512
        assert config.normalize_embeddings is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = DenseRetrieverConfig(
            model_name="specter2",
            batch_size=64,
            use_gpu=False,
        )

        assert config.model_name == "specter2"
        assert config.batch_size == 64
        assert config.use_gpu is False


class TestModelRegistry:
    """Tests for model registry."""

    def test_model_registry_has_common_models(self):
        """Test that common models are in registry."""
        assert "contriever" in MODEL_REGISTRY
        assert "specter2" in MODEL_REGISTRY
        assert "gtr-t5-base" in MODEL_REGISTRY
        assert "all-mpnet-base" in MODEL_REGISTRY

    def test_model_registry_values_are_valid(self):
        """Test that registry values look like HuggingFace IDs."""
        for name, model_id in MODEL_REGISTRY.items():
            assert "/" in model_id, f"{name} should have valid HF model ID"


class TestResultFusion:
    """Tests for result fusion methods."""

    @pytest.fixture
    def sample_results(self):
        """Create sample retrieval results."""
        results1 = {
            "q1": {"d1": 10.0, "d2": 8.0, "d3": 5.0},
            "q2": {"d1": 7.0, "d4": 6.0},
        }
        results2 = {
            "q1": {"d2": 0.9, "d3": 0.8, "d4": 0.7},
            "q2": {"d3": 0.8, "d4": 0.6},
        }
        return [results1, results2]

    def test_rrf_fusion(self, sample_results):
        """Test reciprocal rank fusion."""
        fusion = ResultFusion(FusionConfig(method="rrf"))
        fused = fusion.reciprocal_rank_fusion(sample_results, k=60)

        # Check all queries present
        assert "q1" in fused
        assert "q2" in fused

        # Check documents are combined
        assert "d1" in fused["q1"]
        assert "d2" in fused["q1"]
        assert "d3" in fused["q1"]
        assert "d4" in fused["q1"]

    def test_rrf_scores_are_positive(self, sample_results):
        """Test RRF scores are positive."""
        fusion = ResultFusion()
        fused = fusion.reciprocal_rank_fusion(sample_results)

        for qid, docs in fused.items():
            for doc_id, score in docs.items():
                assert score > 0, f"Score should be positive for {qid}/{doc_id}"

    def test_linear_fusion(self, sample_results):
        """Test linear score fusion."""
        fusion = ResultFusion(FusionConfig(method="linear"))
        fused = fusion.linear_fusion(sample_results)

        # Check all queries present
        assert "q1" in fused
        assert "q2" in fused

        # Documents from both systems should be present
        assert "d1" in fused["q1"]
        assert "d4" in fused["q1"]

    def test_weighted_fusion(self, sample_results):
        """Test weighted score fusion."""
        fusion = ResultFusion()
        weights = [0.7, 0.3]
        fused = fusion.weighted_fusion(sample_results, weights)

        assert "q1" in fused
        # Document present in both systems should have combined score
        assert "d2" in fused["q1"]
        assert "d3" in fused["q1"]


class TestCrossEncoderRegistry:
    """Tests for cross-encoder registry."""

    def test_registry_has_common_models(self):
        """Test that common cross-encoder models are available."""
        assert "ms-marco-minilm" in CROSS_ENCODER_REGISTRY
        assert "ms-marco-minilm-12" in CROSS_ENCODER_REGISTRY


class TestRerankerConfig:
    """Tests for reranker configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = RerankerConfig()

        assert config.model_name == "ms-marco-minilm"
        assert config.batch_size == 32
        assert config.top_k == 100


class TestIPCAblation:
    """Tests for IPC ablation."""

    def test_get_ipc3(self):
        """Test IPC3 extraction."""
        ablation = IPCAblation(Path("."))

        ipc3 = ablation.get_ipc3(["A61K39/395", "C07D213/00"])
        assert "A61K" in ipc3
        assert "C07D" in ipc3

    def test_get_ipc3_empty(self):
        """Test IPC3 extraction with empty input."""
        ablation = IPCAblation(Path("."))

        ipc3 = ablation.get_ipc3([])
        assert len(ipc3) == 0

    def test_split_by_ipc(self):
        """Test splitting qrels by IPC."""
        ablation = IPCAblation(Path("."))
        ablation.query_ipcs = {
            "q1": ["A61K39/395"],
            "q2": ["C07D213/00"],
            "q3": ["A61K10/00"],
        }

        qrels = {
            "q1": {"d1": 1},
            "q2": {"d2": 2},
            "q3": {"d3": 1},
        }

        splits = ablation.split_by_ipc(qrels)

        assert "A61K" in splits
        assert "C07D" in splits
        assert "q1" in splits["A61K"]
        assert "q3" in splits["A61K"]
        assert "q2" in splits["C07D"]


class TestDomainAblation:
    """Tests for domain ablation."""

    def test_split_by_domain_type(self):
        """Test splitting by domain type."""
        ablation = DomainAblation(Path("."))

        qrels = {
            "q1": {"d1": 1},
            "q2": {"d2": 2},
            "q3": {"d3": 1},
        }

        domain_classification = {
            "q1": "IN",
            "q2": "OUT",
            "q3": "IN",
        }

        splits = ablation.split_by_domain_type(qrels, domain_classification)

        assert "IN" in splits
        assert "OUT" in splits
        assert "q1" in splits["IN"]
        assert "q3" in splits["IN"]
        assert "q2" in splits["OUT"]


class TestTemporalAblation:
    """Tests for temporal ablation."""

    def test_split_by_temporal(self):
        """Test splitting by temporal period."""
        ablation = TemporalAblation(Path("."), recent_cutoff="2015-01-01")
        ablation.query_dates = {
            "q1": "2016-05-15",
            "q2": "2010-03-20",
            "q3": "2018-11-01",
        }

        qrels = {
            "q1": {"d1": 1},
            "q2": {"d2": 2},
            "q3": {"d3": 1},
        }

        splits = ablation.split_by_temporal(qrels)

        assert "recent" in splits
        assert "older" in splits
        assert "q1" in splits["recent"]
        assert "q3" in splits["recent"]
        assert "q2" in splits["older"]


class TestErrorAnalyzer:
    """Tests for error analyzer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample retrieval data."""
        results = {
            "q1": {"d1": 10.0, "d2": 8.0, "d3": 5.0},
            "q2": {"d4": 7.0, "d5": 6.0},
        }
        qrels = {
            "q1": {"d1": 3, "d10": 2},  # d10 is not retrieved
            "q2": {"d4": 1, "d20": 3},  # d20 is not retrieved
        }
        return results, qrels

    def test_identify_failures(self, sample_data):
        """Test failure identification."""
        results, qrels = sample_data
        analyzer = ErrorAnalyzer(ErrorAnalysisConfig(rank_threshold=10))

        failures = analyzer.identify_failures(results, qrels)

        # d10 and d20 are not retrieved, so they should be failures
        failure_docs = [f[1] for f in failures]
        assert "d10" in failure_docs
        assert "d20" in failure_docs

    def test_compute_failure_statistics(self, sample_data):
        """Test failure statistics computation."""
        results, qrels = sample_data
        analyzer = ErrorAnalyzer()

        failures = analyzer.identify_failures(results, qrels)
        stats = analyzer.compute_failure_statistics(failures, qrels)

        assert "total_failures" in stats
        assert "failure_rate" in stats
        assert "not_retrieved" in stats
        assert stats["total_failures"] == 2


class TestVocabularyAnalyzer:
    """Tests for vocabulary analyzer."""

    def test_tokenize(self):
        """Test tokenization."""
        analyzer = VocabularyAnalyzer()

        tokens = analyzer.tokenize("The quick brown fox jumps over the lazy dog")

        # Should exclude stopwords and very short words (len <= 2)
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens  # 3 chars is fine
        assert "the" not in tokens  # Stopword

    def test_compute_overlap(self):
        """Test vocabulary overlap computation."""
        analyzer = VocabularyAnalyzer()

        query = "novel kinase inhibitors for cancer treatment"
        doc = "kinase inhibitors have shown promise in cancer therapy"

        overlap = analyzer.compute_overlap(query, doc)

        assert "jaccard" in overlap
        assert "coverage" in overlap
        assert overlap["overlap_count"] > 0
        assert 0 <= overlap["jaccard"] <= 1
        assert 0 <= overlap["coverage"] <= 1


class TestFailureCategory:
    """Tests for failure categories."""

    def test_failure_categories_exist(self):
        """Test that all expected categories exist."""
        assert FailureCategory.VOCABULARY_MISMATCH
        assert FailureCategory.ABSTRACTION_LEVEL
        assert FailureCategory.CROSS_DOMAIN
        assert FailureCategory.SEMANTIC_GAP
        assert FailureCategory.FALSE_NEGATIVE
        assert FailureCategory.UNKNOWN


class TestMetricsComputerIntegration:
    """Integration tests for metrics with Phase 3 modules."""

    def test_compute_metrics_with_graded_relevance(self):
        """Test metrics computation with graded relevance."""
        metrics = MetricsComputer()

        results = {
            "q1": {"d1": 10.0, "d2": 8.0, "d3": 5.0, "d4": 3.0},
        }
        qrels = {
            "q1": {"d1": 3, "d2": 2, "d5": 1},  # Graded relevance
        }

        computed = metrics.compute_all_metrics(results, qrels, k_values=[3])

        assert "NDCG@3" in computed
        assert "Recall@3" in computed
        # NDCG should be positive when relevant docs are retrieved
        assert computed["NDCG@3"] > 0


class TestDenseRetrieverBasic:
    """Basic tests for DenseRetriever (no model loading)."""

    def test_config_model_resolution(self):
        """Test that model names are resolved from registry."""
        retriever = DenseRetriever(model_name="contriever")

        assert retriever.model_id == "facebook/contriever"

    def test_config_custom_model(self):
        """Test custom model ID passthrough."""
        retriever = DenseRetriever(model_name="custom/model")

        assert retriever.model_id == "custom/model"

    def test_cache_path_generation(self):
        """Test cache path generation."""
        config = DenseRetrieverConfig(cache_dir="/tmp/test_cache")
        retriever = DenseRetriever(config=config)

        cache_path = retriever._get_cache_path("test_key")

        assert cache_path is not None
        assert "test_cache" in str(cache_path)


class TestFusionConfig:
    """Tests for fusion configuration."""

    def test_default_config(self):
        """Test default fusion config."""
        config = FusionConfig()

        assert config.method == "rrf"
        assert config.rrf_k == 60
        assert config.normalize_scores is True

    def test_custom_config(self):
        """Test custom fusion config."""
        config = FusionConfig(
            method="weighted",
            rrf_k=100,
            normalize_scores=False,
        )

        assert config.method == "weighted"
        assert config.rrf_k == 100
        assert config.normalize_scores is False
