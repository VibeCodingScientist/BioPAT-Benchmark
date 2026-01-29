"""SOTA Retrieval Module for BioPAT.

This module implements state-of-the-art retrieval methods:
- Dense retrieval with domain-specific embeddings
- Hybrid search (BM25 + dense fusion)
- Cross-encoder reranking
- LLM-based query expansion (HyDE)
"""

from biopat.retrieval.dense import DenseRetriever
from biopat.retrieval.hybrid import HybridRetriever
from biopat.retrieval.reranker import CrossEncoderReranker
from biopat.retrieval.hyde import HyDEQueryExpander

__all__ = [
    "DenseRetriever",
    "HybridRetriever",
    "CrossEncoderReranker",
    "HyDEQueryExpander",
]
