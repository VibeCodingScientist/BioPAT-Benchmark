"""SOTA Retrieval Module for BioPAT.

This module implements state-of-the-art retrieval methods:
- Dense retrieval with domain-specific embeddings
- Hybrid search (BM25 + dense fusion)
- Cross-encoder reranking
- LLM-based query expansion (HyDE)
- Molecular structure similarity (Morgan fingerprints, ChemBERTa)
- Biological sequence similarity (ESM-2, ProtBERT, BLAST)
- SPLADE learned sparse retrieval
- ColBERT late interaction
- Learning-to-Rank (LambdaMART, RankNet, ListNet)
- Biomedical NER for claim/document decomposition
- Thesaurus-based query expansion (MeSH, ChEBI, DrugBank)
- Substructure and scaffold search
- Diversity reranking (MMR, xQuAD)
"""

from biopat.retrieval.dense import DenseRetriever, create_domain_retriever
from biopat.retrieval.hybrid import HybridRetriever, SparseRetriever, create_hybrid_retriever
from biopat.retrieval.reranker import CrossEncoderReranker, LLMReranker, create_reranker
from biopat.retrieval.hyde import HyDEQueryExpander, QueryExpansionPipeline, create_hyde_expander
from biopat.retrieval.molecular import MolecularRetriever, MorganFingerprintEncoder, create_molecular_retriever
from biopat.retrieval.sequence import SequenceRetriever, BLASTSearcher, create_sequence_retriever
from biopat.retrieval.splade import SPLADERetriever, SPLADEEncoder, HybridSPLADERetriever, create_splade_retriever
from biopat.retrieval.colbert import ColBERTRetriever, ColBERTEncoder, create_colbert_retriever

# Learning-to-Rank
from biopat.retrieval.learning_to_rank import (
    RankingFeatures,
    LambdaMARTRanker,
    XGBoostRanker,
    RankNetModel,
    ListNetModel,
    EnsembleLTR,
    LTRFeatureExtractor,
    create_ltr_ranker,
)

# Biomedical NER
from biopat.retrieval.biomedical_ner import (
    BioEntity,
    ExtractedClaim,
    BiomedicalNER,
    EntityLinker,
    QueryEntityExpander,
    create_biomedical_ner,
)

# Thesaurus
from biopat.retrieval.thesaurus import (
    ThesaurusTerm,
    ExpansionResult,
    MeSHThesaurus,
    ChEBIThesaurus,
    DrugBankThesaurus,
    UnifiedThesaurus,
    create_unified_thesaurus,
)

# Substructure search
from biopat.retrieval.substructure import (
    SubstructureMatch,
    ScaffoldMatch,
    MCSResult,
    SubstructureSearcher,
    ScaffoldSearcher,
    MCSSearcher,
    MolecularDescriptorCalculator,
    create_substructure_searcher,
    create_scaffold_searcher,
    create_mcs_searcher,
)

# Diversity reranking
from biopat.retrieval.diversity import (
    DiverseResult,
    MMRDiversifier,
    XQuADDiversifier,
    ClusterDiversifier,
    PatentDiversifier,
    create_mmr_diversifier,
    create_xquad_diversifier,
    create_patent_diversifier,
)

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

    # SPLADE learned sparse
    "SPLADERetriever",
    "SPLADEEncoder",
    "HybridSPLADERetriever",
    "create_splade_retriever",

    # ColBERT late interaction
    "ColBERTRetriever",
    "ColBERTEncoder",
    "create_colbert_retriever",

    # Learning-to-Rank
    "RankingFeatures",
    "LambdaMARTRanker",
    "XGBoostRanker",
    "RankNetModel",
    "ListNetModel",
    "EnsembleLTR",
    "LTRFeatureExtractor",
    "create_ltr_ranker",

    # Biomedical NER
    "BioEntity",
    "ExtractedClaim",
    "BiomedicalNER",
    "EntityLinker",
    "QueryEntityExpander",
    "create_biomedical_ner",

    # Thesaurus
    "ThesaurusTerm",
    "ExpansionResult",
    "MeSHThesaurus",
    "ChEBIThesaurus",
    "DrugBankThesaurus",
    "UnifiedThesaurus",
    "create_unified_thesaurus",

    # Substructure search
    "SubstructureMatch",
    "ScaffoldMatch",
    "MCSResult",
    "SubstructureSearcher",
    "ScaffoldSearcher",
    "MCSSearcher",
    "MolecularDescriptorCalculator",
    "create_substructure_searcher",
    "create_scaffold_searcher",
    "create_mcs_searcher",

    # Diversity reranking
    "DiverseResult",
    "MMRDiversifier",
    "XQuADDiversifier",
    "ClusterDiversifier",
    "PatentDiversifier",
    "create_mmr_diversifier",
    "create_xquad_diversifier",
    "create_patent_diversifier",
]
