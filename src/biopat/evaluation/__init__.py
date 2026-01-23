"""Evaluation modules for BioPAT baselines."""

from .bm25 import BM25Evaluator
from .metrics import MetricsComputer
from .dense import DenseRetriever, DenseRetrieverConfig, DenseEvaluator, MODEL_REGISTRY
from .hybrid import ResultFusion, FusionConfig, HybridRetriever, BM25DenseHybrid
from .reranker import (
    CrossEncoderReranker,
    RerankerConfig,
    BM25CrossEncoderPipeline,
    DenseCrossEncoderPipeline,
    CROSS_ENCODER_REGISTRY,
)
from .ablation import (
    AblationConfig,
    AblationResult,
    QueryRepresentationAblation,
    DocumentRepresentationAblation,
    DomainAblation,
    TemporalAblation,
    IPCAblation,
    AblationRunner,
)
from .error_analysis import (
    ErrorAnalyzer,
    ErrorAnalysisConfig,
    FailureCategory,
    FailureCase,
    VocabularyAnalyzer,
    DomainAnalyzer,
    ErrorReportGenerator,
    run_error_analysis,
)

__all__ = [
    # BM25
    "BM25Evaluator",
    # Metrics
    "MetricsComputer",
    # Dense retrieval
    "DenseRetriever",
    "DenseRetrieverConfig",
    "DenseEvaluator",
    "MODEL_REGISTRY",
    # Hybrid
    "ResultFusion",
    "FusionConfig",
    "HybridRetriever",
    "BM25DenseHybrid",
    # Reranking
    "CrossEncoderReranker",
    "RerankerConfig",
    "BM25CrossEncoderPipeline",
    "DenseCrossEncoderPipeline",
    "CROSS_ENCODER_REGISTRY",
    # Ablation
    "AblationConfig",
    "AblationResult",
    "QueryRepresentationAblation",
    "DocumentRepresentationAblation",
    "DomainAblation",
    "TemporalAblation",
    "IPCAblation",
    "AblationRunner",
    # Error analysis
    "ErrorAnalyzer",
    "ErrorAnalysisConfig",
    "FailureCategory",
    "FailureCase",
    "VocabularyAnalyzer",
    "DomainAnalyzer",
    "ErrorReportGenerator",
    "run_error_analysis",
]
