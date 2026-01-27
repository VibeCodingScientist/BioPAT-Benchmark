"""Data processing modules for BioPAT."""

from .patents import PatentProcessor
from .papers import PaperProcessor
from .linking import CitationLinker
from .npl_parser import NPLParser, NPLLinker
from .claim_mapper import ClaimMapper, ClaimCitationMapper
from .prior_patents import PriorPatentSelector
from .patent_ids import (
    Jurisdiction,
    ParsedPatentId,
    PatentIdNormalizer,
    normalize_patent_id,
    normalize_patent_ids,
    extract_jurisdiction,
    are_same_patent,
    deduplicate_patent_ids,
    group_by_jurisdiction,
    validate_patent_id,
)
from .international_patents import (
    InternationalPatent,
    InternationalCorpusConfig,
    InternationalCorpusBuilder,
    create_international_corpus_entry,
    merge_corpus_dataframes,
    get_corpus_statistics,
)

__all__ = [
    "PatentProcessor",
    "PaperProcessor",
    "CitationLinker",
    "NPLParser",
    "NPLLinker",
    "ClaimMapper",
    "ClaimCitationMapper",
    "PriorPatentSelector",
    # Patent ID normalization
    "Jurisdiction",
    "ParsedPatentId",
    "PatentIdNormalizer",
    "normalize_patent_id",
    "normalize_patent_ids",
    "extract_jurisdiction",
    "are_same_patent",
    "deduplicate_patent_ids",
    "group_by_jurisdiction",
    "validate_patent_id",
    # International corpus (Phase 6)
    "InternationalPatent",
    "InternationalCorpusConfig",
    "InternationalCorpusBuilder",
    "create_international_corpus_entry",
    "merge_corpus_dataframes",
    "get_corpus_statistics",
]
