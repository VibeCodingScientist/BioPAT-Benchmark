"""Cross-Reference Linking System.

Phase 4.0 (Advanced): Provides functionality for linking entities across
different data sources and creating bidirectional indices for fast lookups.

This module handles:
- Paper-to-Patent citation linking
- Chemical-to-Patent/Paper linking via structure matching
- Sequence-to-Patent/Paper linking via accession numbers
- Bulk import and resolution of cross-references
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import polars as pl
from sqlalchemy.orm import Session

from biopat.harmonization.entity_resolver import (
    BioPATId,
    EntityResolver,
    EntityType,
    PublicationSource,
)
from biopat.harmonization.schema import (
    Chemical,
    CrossReference,
    DatabaseManager,
    LinkSourceEnum,
    Patent,
    PatentChemicalLink,
    PatentPublicationCitation,
    PatentSequenceLink,
    Publication,
    PublicationChemicalLink,
    PublicationSequenceLink,
    Sequence,
)

logger = logging.getLogger(__name__)


@dataclass
class LinkCandidate:
    """Candidate link between two entities."""

    source_id: str
    target_id: str
    source_type: EntityType
    target_type: EntityType
    confidence: float = 1.0
    link_source: LinkSourceEnum = LinkSourceEnum.INFERRED
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LinkStats:
    """Statistics about linking operations."""

    total_candidates: int = 0
    successful_links: int = 0
    failed_links: int = 0
    duplicate_links: int = 0
    by_source: Dict[str, int] = field(default_factory=dict)
    by_type: Dict[str, int] = field(default_factory=dict)


class EntityLinker:
    """Links entities across different data sources.

    Provides methods for:
    - Extracting identifiers from text (DOIs, PMIDs, chemical names)
    - Matching extracted identifiers to known entities
    - Creating and storing links in the database
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        entity_resolver: Optional[EntityResolver] = None,
    ):
        """Initialize entity linker.

        Args:
            db_manager: Database manager for persistence.
            entity_resolver: Entity resolver for ID normalization.
        """
        self.db_manager = db_manager
        self.entity_resolver = entity_resolver or EntityResolver()

        # Regex patterns for identifier extraction
        self._doi_pattern = re.compile(
            r'\b(10\.\d{4,}/[^\s\]\)\"\']+)',
            re.IGNORECASE
        )
        self._pmid_pattern = re.compile(
            r'\bPMID[:\s]*(\d{7,8})\b|\bPubMed[:\s]*(\d{7,8})\b',
            re.IGNORECASE
        )
        self._pmc_pattern = re.compile(
            r'\bPMC(\d{7,8})\b',
            re.IGNORECASE
        )
        self._accession_pattern = re.compile(
            r'\b([A-Z]{1,3}_?\d{5,10}(?:\.\d+)?)\b'  # GenBank/RefSeq format
        )
        self._seq_id_pattern = re.compile(
            r'SEQ\s*ID\s*NO[:\s]*(\d+)',
            re.IGNORECASE
        )

    def extract_publication_references(
        self,
        text: str,
    ) -> List[Tuple[str, PublicationSource]]:
        """Extract publication references from text.

        Args:
            text: Text to search for references.

        Returns:
            List of (identifier, source) tuples.
        """
        references = []

        # Extract DOIs
        for match in self._doi_pattern.finditer(text):
            doi = match.group(1).rstrip(".,;")
            references.append((doi, PublicationSource.DOI))

        # Extract PMIDs
        for match in self._pmid_pattern.finditer(text):
            pmid = match.group(1) or match.group(2)
            if pmid:
                references.append((pmid, PublicationSource.PMID))

        # Extract PMC IDs
        for match in self._pmc_pattern.finditer(text):
            pmc = f"PMC{match.group(1)}"
            references.append((pmc, PublicationSource.PMC))

        return references

    def extract_sequence_references(
        self,
        text: str,
    ) -> List[Dict[str, str]]:
        """Extract sequence references from text.

        Args:
            text: Text to search for sequence references.

        Returns:
            List of dicts with accession and/or SEQ ID NO.
        """
        references = []

        # Extract accession numbers
        for match in self._accession_pattern.finditer(text):
            accession = match.group(1)
            # Filter out common false positives
            if not self._is_likely_accession(accession):
                continue
            references.append({"accession": accession})

        # Extract SEQ ID NOs
        for match in self._seq_id_pattern.finditer(text):
            seq_id = match.group(1)
            references.append({"seq_id_no": seq_id})

        return references

    def _is_likely_accession(self, accession: str) -> bool:
        """Check if string is likely a sequence accession."""
        # Common prefixes for sequence databases
        prefixes = [
            "NM_", "NP_", "XM_", "XP_",  # RefSeq
            "AAA", "AAB", "AAC", "AAD",  # GenBank protein
            "U", "AF", "AY", "DQ", "EU",  # GenBank nucleotide
        ]
        return any(accession.upper().startswith(p) for p in prefixes)

    def link_patent_to_publications(
        self,
        patent: Patent,
        session: Session,
        text_fields: Optional[List[str]] = None,
    ) -> List[PatentPublicationCitation]:
        """Extract and link publication citations from a patent.

        Args:
            patent: Patent record to process.
            session: Database session.
            text_fields: Fields to search for references.

        Returns:
            List of created citation links.
        """
        if text_fields is None:
            text_fields = ["abstract", "claims_text"]

        # Gather text to search
        text_to_search = []
        for field_name in text_fields:
            value = getattr(patent, field_name, None)
            if value:
                text_to_search.append(value)

        full_text = " ".join(text_to_search)
        references = self.extract_publication_references(full_text)

        links = []
        for identifier, source in references:
            # Resolve to BioPAT ID
            biopat_id = self.entity_resolver.resolve_publication(identifier, source)

            # Find or create publication record
            publication = self._find_or_create_publication(
                session, biopat_id, identifier, source
            )

            if publication:
                # Create link if not exists
                existing = session.query(PatentPublicationCitation).filter_by(
                    patent_id=patent.id,
                    publication_id=publication.id
                ).first()

                if not existing:
                    link = PatentPublicationCitation(
                        patent_id=patent.id,
                        publication_id=publication.id,
                        link_source=LinkSourceEnum.EXTRACTED,
                    )
                    session.add(link)
                    links.append(link)

        return links

    def _find_or_create_publication(
        self,
        session: Session,
        biopat_id: BioPATId,
        identifier: str,
        source: PublicationSource,
    ) -> Optional[Publication]:
        """Find existing publication or create placeholder.

        Args:
            session: Database session.
            biopat_id: Resolved BioPAT ID.
            identifier: Original identifier.
            source: Identifier source.

        Returns:
            Publication record or None.
        """
        # Try to find by BioPAT ID
        pub = session.query(Publication).filter_by(
            biopat_id=biopat_id.canonical
        ).first()

        if pub:
            return pub

        # Try to find by specific identifier
        if source == PublicationSource.PMID:
            pub = session.query(Publication).filter_by(pmid=identifier).first()
        elif source == PublicationSource.DOI:
            pub = session.query(Publication).filter_by(
                doi=identifier.lower()
            ).first()
        elif source == PublicationSource.PMC:
            pub = session.query(Publication).filter_by(pmc_id=identifier).first()

        if pub:
            return pub

        # Create placeholder
        pub = Publication(
            biopat_id=biopat_id.canonical,
            source="extracted",
        )

        if source == PublicationSource.PMID:
            pub.pmid = identifier
        elif source == PublicationSource.DOI:
            pub.doi = identifier.lower()
        elif source == PublicationSource.PMC:
            pub.pmc_id = identifier

        session.add(pub)
        return pub

    def link_patent_to_sequences(
        self,
        patent: Patent,
        session: Session,
    ) -> List[PatentSequenceLink]:
        """Extract and link sequence references from a patent.

        Args:
            patent: Patent record to process.
            session: Database session.

        Returns:
            List of created sequence links.
        """
        # Search claims for SEQ ID NOs
        text = patent.claims_text or ""
        references = self.extract_sequence_references(text)

        links = []
        for ref in references:
            seq_id_no = ref.get("seq_id_no")
            accession = ref.get("accession")

            # Find sequence by SEQ ID NO or accession
            sequence = None
            if seq_id_no:
                sequence = session.query(Sequence).filter_by(
                    patent_seq_id=seq_id_no
                ).first()
            if not sequence and accession:
                sequence = session.query(Sequence).filter_by(
                    ncbi_accession=accession
                ).first()

            if sequence:
                existing = session.query(PatentSequenceLink).filter_by(
                    patent_id=patent.id,
                    sequence_id=sequence.id
                ).first()

                if not existing:
                    link = PatentSequenceLink(
                        patent_id=patent.id,
                        sequence_id=sequence.id,
                        link_source=LinkSourceEnum.EXTRACTED,
                        seq_id_no=seq_id_no,
                    )
                    session.add(link)
                    links.append(link)

        return links


class BidirectionalIndex:
    """In-memory bidirectional index for fast entity lookups.

    Maintains mappings in both directions for efficient cross-referencing:
    - Forward: source_id -> set of target_ids
    - Reverse: target_id -> set of source_ids
    """

    def __init__(self):
        """Initialize empty bidirectional index."""
        self._forward: Dict[str, Set[str]] = defaultdict(set)
        self._reverse: Dict[str, Set[str]] = defaultdict(set)
        self._metadata: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def add_link(
        self,
        source_id: str,
        target_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a link to the index.

        Args:
            source_id: Source entity ID.
            target_id: Target entity ID.
            metadata: Optional link metadata.
        """
        self._forward[source_id].add(target_id)
        self._reverse[target_id].add(source_id)

        if metadata:
            self._metadata[(source_id, target_id)] = metadata

    def get_targets(self, source_id: str) -> Set[str]:
        """Get all targets linked from a source.

        Args:
            source_id: Source entity ID.

        Returns:
            Set of target IDs.
        """
        return self._forward.get(source_id, set())

    def get_sources(self, target_id: str) -> Set[str]:
        """Get all sources linked to a target.

        Args:
            target_id: Target entity ID.

        Returns:
            Set of source IDs.
        """
        return self._reverse.get(target_id, set())

    def get_metadata(
        self,
        source_id: str,
        target_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific link.

        Args:
            source_id: Source entity ID.
            target_id: Target entity ID.

        Returns:
            Link metadata or None.
        """
        return self._metadata.get((source_id, target_id))

    def has_link(self, source_id: str, target_id: str) -> bool:
        """Check if a link exists.

        Args:
            source_id: Source entity ID.
            target_id: Target entity ID.

        Returns:
            True if link exists.
        """
        return target_id in self._forward.get(source_id, set())

    def remove_link(self, source_id: str, target_id: str):
        """Remove a link from the index.

        Args:
            source_id: Source entity ID.
            target_id: Target entity ID.
        """
        if source_id in self._forward:
            self._forward[source_id].discard(target_id)
        if target_id in self._reverse:
            self._reverse[target_id].discard(source_id)
        self._metadata.pop((source_id, target_id), None)

    def size(self) -> int:
        """Get total number of links."""
        return sum(len(targets) for targets in self._forward.values())

    def to_dataframe(self) -> pl.DataFrame:
        """Convert index to Polars DataFrame.

        Returns:
            DataFrame with source_id, target_id columns.
        """
        records = []
        for source_id, targets in self._forward.items():
            for target_id in targets:
                record = {
                    "source_id": source_id,
                    "target_id": target_id,
                }
                meta = self._metadata.get((source_id, target_id), {})
                record.update(meta)
                records.append(record)

        return pl.DataFrame(records) if records else pl.DataFrame()

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        source_col: str = "source_id",
        target_col: str = "target_id",
        metadata_cols: Optional[List[str]] = None,
    ) -> "BidirectionalIndex":
        """Create index from DataFrame.

        Args:
            df: DataFrame with links.
            source_col: Source ID column name.
            target_col: Target ID column name.
            metadata_cols: Optional metadata column names.

        Returns:
            BidirectionalIndex instance.
        """
        index = cls()

        for row in df.iter_rows(named=True):
            source_id = row[source_col]
            target_id = row[target_col]

            metadata = None
            if metadata_cols:
                metadata = {col: row.get(col) for col in metadata_cols}

            index.add_link(source_id, target_id, metadata)

        return index


class CrossReferenceManager:
    """Manages cross-references across the entire BioPAT system.

    Provides high-level operations for:
    - Bulk importing cross-references
    - Building and querying indices
    - Synchronizing with database
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        entity_resolver: Optional[EntityResolver] = None,
    ):
        """Initialize cross-reference manager.

        Args:
            db_manager: Database manager for persistence.
            entity_resolver: Entity resolver for ID normalization.
        """
        self.db_manager = db_manager
        self.entity_resolver = entity_resolver or EntityResolver()
        self.linker = EntityLinker(db_manager, entity_resolver)

        # In-memory indices by relationship type
        self._indices: Dict[str, BidirectionalIndex] = {}

    def get_index(self, relationship_type: str) -> BidirectionalIndex:
        """Get or create index for a relationship type.

        Args:
            relationship_type: Type of relationship (e.g., "cites", "contains").

        Returns:
            BidirectionalIndex for the relationship.
        """
        if relationship_type not in self._indices:
            self._indices[relationship_type] = BidirectionalIndex()
        return self._indices[relationship_type]

    def add_cross_reference(
        self,
        source_biopat_id: str,
        target_biopat_id: str,
        relationship_type: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        persist: bool = True,
    ):
        """Add a cross-reference between entities.

        Args:
            source_biopat_id: Source BioPAT ID.
            target_biopat_id: Target BioPAT ID.
            relationship_type: Type of relationship.
            confidence: Confidence score (0-1).
            metadata: Optional additional metadata.
            persist: Whether to persist to database.
        """
        # Add to in-memory index
        index = self.get_index(relationship_type)
        link_metadata = {"confidence": confidence}
        if metadata:
            link_metadata.update(metadata)
        index.add_link(source_biopat_id, target_biopat_id, link_metadata)

        # Persist to database
        if persist and self.db_manager:
            session = self.db_manager.get_session()
            try:
                # Parse entity types from IDs
                source_parsed = self.entity_resolver.parse_biopat_id(source_biopat_id)
                target_parsed = self.entity_resolver.parse_biopat_id(target_biopat_id)

                crossref = CrossReference(
                    source_biopat_id=source_biopat_id,
                    source_type=source_parsed.entity_type.value if source_parsed else "UNK",
                    target_biopat_id=target_biopat_id,
                    target_type=target_parsed.entity_type.value if target_parsed else "UNK",
                    relationship_type=relationship_type,
                    confidence=confidence,
                    extra_data=metadata,
                )
                session.merge(crossref)
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to persist cross-reference: {e}")
            finally:
                session.close()

    def get_related_entities(
        self,
        biopat_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "forward",
    ) -> List[str]:
        """Get entities related to a given entity.

        Args:
            biopat_id: BioPAT ID to query.
            relationship_type: Optional filter by relationship type.
            direction: "forward", "reverse", or "both".

        Returns:
            List of related BioPAT IDs.
        """
        results = set()

        indices_to_search = (
            [self._indices[relationship_type]]
            if relationship_type and relationship_type in self._indices
            else self._indices.values()
        )

        for index in indices_to_search:
            if direction in ("forward", "both"):
                results.update(index.get_targets(biopat_id))
            if direction in ("reverse", "both"):
                results.update(index.get_sources(biopat_id))

        return list(results)

    def load_from_database(self, relationship_type: Optional[str] = None):
        """Load cross-references from database into memory.

        Args:
            relationship_type: Optional filter by relationship type.
        """
        if not self.db_manager:
            logger.warning("No database manager configured")
            return

        session = self.db_manager.get_session()
        try:
            query = session.query(CrossReference)
            if relationship_type:
                query = query.filter_by(relationship_type=relationship_type)

            for crossref in query.all():
                index = self.get_index(crossref.relationship_type)
                index.add_link(
                    crossref.source_biopat_id,
                    crossref.target_biopat_id,
                    {"confidence": crossref.confidence, **(crossref.extra_data or {})},
                )

            logger.info(f"Loaded {query.count()} cross-references from database")

        finally:
            session.close()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about cross-references.

        Returns:
            Dictionary with index statistics.
        """
        stats = {
            "total_links": 0,
            "by_relationship": {},
        }

        for rel_type, index in self._indices.items():
            count = index.size()
            stats["by_relationship"][rel_type] = count
            stats["total_links"] += count

        return stats
