"""Data Harmonization Layer for BioPAT.

Phase 4.0 (Advanced): Provides unified ID generation, SQL schema,
and cross-reference linking for heterogeneous entity types.
"""

from .entity_resolver import (
    BIOPAT_PREFIX,
    BioPATId,
    CrossReferenceResolver,
    EntityResolver,
    EntityType,
    PublicationSource,
    ResolvedEntity,
    SequenceType,
    create_biopat_id,
    is_valid_biopat_id,
)
from .schema import (
    Base,
    Chemical,
    CrossReference,
    DatabaseManager,
    DocumentTypeEnum,
    JurisdictionEnum,
    LinkSourceEnum,
    Patent,
    PatentChemicalLink,
    PatentPublicationCitation,
    PatentSequenceLink,
    Publication,
    PublicationChemicalLink,
    PublicationSequenceLink,
    Sequence,
    SequenceTypeEnum,
    get_database_url_from_config,
)
from .linker import (
    BidirectionalIndex,
    CrossReferenceManager,
    EntityLinker,
    LinkCandidate,
    LinkStats,
)

__all__ = [
    # Entity Resolution
    "BIOPAT_PREFIX",
    "BioPATId",
    "EntityResolver",
    "EntityType",
    "PublicationSource",
    "SequenceType",
    "ResolvedEntity",
    "CrossReferenceResolver",
    "create_biopat_id",
    "is_valid_biopat_id",
    # Schema
    "Base",
    "Patent",
    "Publication",
    "Chemical",
    "Sequence",
    "PatentChemicalLink",
    "PatentSequenceLink",
    "PublicationChemicalLink",
    "PublicationSequenceLink",
    "PatentPublicationCitation",
    "CrossReference",
    "DatabaseManager",
    "JurisdictionEnum",
    "DocumentTypeEnum",
    "SequenceTypeEnum",
    "LinkSourceEnum",
    "get_database_url_from_config",
    # Linker
    "EntityLinker",
    "BidirectionalIndex",
    "CrossReferenceManager",
    "LinkCandidate",
    "LinkStats",
]
