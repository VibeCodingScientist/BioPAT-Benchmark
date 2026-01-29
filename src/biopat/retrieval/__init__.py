"""SOTA Retrieval Module for BioPAT.

This module implements state-of-the-art retrieval methods:
- Dense retrieval with domain-specific embeddings
- Hybrid search (BM25 + dense fusion)
- Cross-encoder reranking
- LLM-based query expansion (HyDE)
- Molecular structure similarity (Morgan fingerprints, ChemBERTa)
- Biological sequence similarity (ESM-2, ProtBERT, BLAST)
"""

from biopat.retrieval.dense import DenseRetriever, create_domain_retriever
from biopat.retrieval.hybrid import HybridRetriever, SparseRetriever, create_hybrid_retriever
from biopat.retrieval.reranker import CrossEncoderReranker, LLMReranker, create_reranker
from biopat.retrieval.hyde import HyDEQueryExpander, QueryExpansionPipeline, create_hyde_expander
from biopat.retrieval.molecular import MolecularRetriever, MorganFingerprintEncoder, create_molecular_retriever
from biopat.retrieval.sequence import SequenceRetriever, BLASTSearcher, create_sequence_retriever

__all__ = [
    # Dense retrieval
    "DenseRetriever",
    "create_domain_retriever",

    # Hybrid retrieval
    "HybridRetriever",
    "SparseRetriever",
    "create_hybrid_retriever",

    # Reranking
    "CrossEncoderReranker",
    "LLMReranker",
    "create_reranker",

    # Query expansion
    "HyDEQueryExpander",
    "QueryExpansionPipeline",
    "create_hyde_expander",

    # Molecular search
    "MolecularRetriever",
    "MorganFingerprintEncoder",
    "create_molecular_retriever",

    # Sequence search
    "SequenceRetriever",
    "BLASTSearcher",
    "create_sequence_retriever",
]
