"""Entity Resolution and Unified ID System.

Phase 4.0 (Advanced): Provides unified identifier generation and cross-source
resolution for harmonizing heterogeneous data types across the BioPAT benchmark.

Unified ID Format:
- Patents: BP:PAT:{jurisdiction}:{number} (e.g., BP:PAT:US:10123456)
- Publications: BP:PUB:{source}:{id} (e.g., BP:PUB:PMID:12345678)
- Chemicals: BP:CHM:{InChIKey} (e.g., BP:CHM:BSYNRYMUTXBXSQ-UHFFFAOYSA-N)
- Sequences: BP:SEQ:{type}:{hash} (e.g., BP:SEQ:AA:a1b2c3d4e5f6)

This module handles:
- Canonical ID generation from various source formats
- Cross-source resolution (SMILES -> InChIKey, PMID -> DOI)
- Entity deduplication across jurisdictions and databases
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# BioPAT ID prefix
BIOPAT_PREFIX = "BP"


class EntityType(str, Enum):
    """Types of entities in the BioPAT system."""
    PATENT = "PAT"
    PUBLICATION = "PUB"
    CHEMICAL = "CHM"
    SEQUENCE = "SEQ"


class SequenceType(str, Enum):
    """Types of biological sequences."""
    AMINO_ACID = "AA"  # Protein sequences
    NUCLEOTIDE = "NT"  # DNA/RNA sequences
    UNKNOWN = "UNK"


class PublicationSource(str, Enum):
    """Sources for publication identifiers."""
    PMID = "PMID"  # PubMed ID
    DOI = "DOI"    # Digital Object Identifier
    PMC = "PMC"    # PubMed Central ID
    ARXIV = "ARXIV"
    OPENALEX = "OA"


@dataclass
class BioPATId:
    """Unified BioPAT identifier."""

    entity_type: EntityType
    identifier: str
    subtype: Optional[str] = None  # jurisdiction for patents, source for pubs
    original: str = ""  # Original identifier before normalization

    @property
    def canonical(self) -> str:
        """Return canonical BioPAT ID string."""
        if self.subtype:
            return f"{BIOPAT_PREFIX}:{self.entity_type.value}:{self.subtype}:{self.identifier}"
        return f"{BIOPAT_PREFIX}:{self.entity_type.value}:{self.identifier}"

    def __str__(self) -> str:
        return self.canonical

    def __hash__(self) -> int:
        return hash(self.canonical)

    def __eq__(self, other) -> bool:
        if isinstance(other, BioPATId):
            return self.canonical == other.canonical
        return False


@dataclass
class ResolvedEntity:
    """Entity with resolved cross-references."""

    biopat_id: BioPATId
    entity_type: EntityType
    primary_id: str
    alternate_ids: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def get_all_ids(self) -> List[str]:
        """Get all known identifiers for this entity."""
        ids = [self.primary_id]
        ids.extend(self.alternate_ids.values())
        return ids


class EntityResolver:
    """Resolves and harmonizes entity identifiers across sources.

    Handles:
    - Patent ID normalization across US/EP/WO jurisdictions
    - Publication ID resolution (PMID <-> DOI <-> PMC)
    - Chemical structure canonicalization (SMILES -> InChIKey)
    - Sequence hashing for deduplication
    """

    def __init__(
        self,
        use_rdkit: bool = True,
        cache_size: int = 10000,
    ):
        """Initialize entity resolver.

        Args:
            use_rdkit: Whether to use RDKit for chemical canonicalization.
            cache_size: Maximum cache size for resolved entities.
        """
        self.use_rdkit = use_rdkit
        self._cache: Dict[str, BioPATId] = {}
        self._cache_size = cache_size

        # Try to import RDKit for chemical handling
        self._rdkit_available = False
        if use_rdkit:
            try:
                from rdkit import Chem
                from rdkit.Chem.inchi import MolFromInchi, MolToInchi, MolToInchiKey
                self._rdkit_available = True
            except ImportError:
                logger.warning("RDKit not available. Chemical canonicalization will be limited.")

    def resolve_patent(
        self,
        patent_id: str,
        jurisdiction: Optional[str] = None,
    ) -> BioPATId:
        """Resolve a patent identifier to BioPAT ID.

        Args:
            patent_id: Raw patent identifier (e.g., "US10123456", "EP 1234567 A1").
            jurisdiction: Optional jurisdiction override.

        Returns:
            BioPATId for the patent.
        """
        cache_key = f"pat:{patent_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Import patent ID normalizer
        from biopat.processing.patent_ids import normalize_patent_id, Jurisdiction

        parsed = normalize_patent_id(patent_id)

        # Determine jurisdiction
        if jurisdiction:
            jur = jurisdiction.upper()
        elif parsed.jurisdiction != Jurisdiction.UNKNOWN:
            jur = parsed.jurisdiction.value
        else:
            jur = "UNK"

        biopat_id = BioPATId(
            entity_type=EntityType.PATENT,
            identifier=parsed.doc_number,
            subtype=jur,
            original=patent_id,
        )

        self._cache_put(cache_key, biopat_id)
        return biopat_id

    def resolve_publication(
        self,
        identifier: str,
        source: Optional[PublicationSource] = None,
    ) -> BioPATId:
        """Resolve a publication identifier to BioPAT ID.

        Args:
            identifier: Publication identifier (PMID, DOI, etc.).
            source: Optional source type hint.

        Returns:
            BioPATId for the publication.
        """
        cache_key = f"pub:{identifier}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Auto-detect source if not provided
        if source is None:
            source = self._detect_publication_source(identifier)

        # Normalize identifier based on source
        normalized_id = self._normalize_publication_id(identifier, source)

        biopat_id = BioPATId(
            entity_type=EntityType.PUBLICATION,
            identifier=normalized_id,
            subtype=source.value,
            original=identifier,
        )

        self._cache_put(cache_key, biopat_id)
        return biopat_id

    def resolve_chemical(
        self,
        structure: str,
        structure_type: str = "smiles",
    ) -> BioPATId:
        """Resolve a chemical structure to BioPAT ID via InChIKey.

        Args:
            structure: Chemical structure representation.
            structure_type: Type of input ("smiles", "inchi", "inchikey").

        Returns:
            BioPATId for the chemical.
        """
        cache_key = f"chm:{structure}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Compute InChIKey
        inchikey = self._compute_inchikey(structure, structure_type)

        if not inchikey:
            # Use hash of input as fallback
            inchikey = f"UNKNOWN-{hashlib.sha256(structure.encode()).hexdigest()[:14]}"
            logger.warning(f"Could not compute InChIKey for: {structure[:50]}...")

        biopat_id = BioPATId(
            entity_type=EntityType.CHEMICAL,
            identifier=inchikey,
            original=structure,
        )

        self._cache_put(cache_key, biopat_id)
        return biopat_id

    def resolve_sequence(
        self,
        sequence: str,
        seq_type: Optional[SequenceType] = None,
    ) -> BioPATId:
        """Resolve a biological sequence to BioPAT ID via hash.

        Args:
            sequence: Biological sequence string (amino acids or nucleotides).
            seq_type: Type of sequence (AA or NT). Auto-detected if not provided.

        Returns:
            BioPATId for the sequence.
        """
        # Normalize sequence
        normalized = self._normalize_sequence(sequence)
        cache_key = f"seq:{normalized[:100]}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Detect sequence type if not provided
        if seq_type is None:
            seq_type = self._detect_sequence_type(normalized)

        # Compute hash of normalized sequence
        seq_hash = hashlib.sha256(normalized.encode()).hexdigest()[:12]

        biopat_id = BioPATId(
            entity_type=EntityType.SEQUENCE,
            identifier=seq_hash,
            subtype=seq_type.value,
            original=sequence[:100] + ("..." if len(sequence) > 100 else ""),
        )

        self._cache_put(cache_key, biopat_id)
        return biopat_id

    def parse_biopat_id(self, biopat_id_str: str) -> Optional[BioPATId]:
        """Parse a BioPAT ID string back to components.

        Args:
            biopat_id_str: BioPAT ID string (e.g., "BP:PAT:US:10123456").

        Returns:
            Parsed BioPATId or None if invalid.
        """
        parts = biopat_id_str.split(":")

        if len(parts) < 3 or parts[0] != BIOPAT_PREFIX:
            return None

        try:
            entity_type = EntityType(parts[1])
        except ValueError:
            return None

        if len(parts) == 3:
            # No subtype: BP:CHM:InChIKey
            return BioPATId(
                entity_type=entity_type,
                identifier=parts[2],
            )
        elif len(parts) == 4:
            # With subtype: BP:PAT:US:123456
            return BioPATId(
                entity_type=entity_type,
                identifier=parts[3],
                subtype=parts[2],
            )
        else:
            # Handle DOIs with colons: BP:PUB:DOI:10.1234/abc
            return BioPATId(
                entity_type=entity_type,
                identifier=":".join(parts[3:]),
                subtype=parts[2],
            )

    def _detect_publication_source(self, identifier: str) -> PublicationSource:
        """Auto-detect publication identifier source."""
        identifier = identifier.strip()

        # Check for DOI pattern
        if identifier.startswith("10.") or identifier.startswith("doi:"):
            return PublicationSource.DOI

        # Check for PMID (numeric)
        if identifier.isdigit():
            return PublicationSource.PMID

        # Check for PMC
        if identifier.upper().startswith("PMC"):
            return PublicationSource.PMC

        # Check for arXiv
        if "arxiv" in identifier.lower():
            return PublicationSource.ARXIV

        # Check for OpenAlex
        if identifier.startswith("W") and identifier[1:].isdigit():
            return PublicationSource.OPENALEX

        # Default to DOI if contains slash
        if "/" in identifier:
            return PublicationSource.DOI

        return PublicationSource.PMID

    def _normalize_publication_id(
        self,
        identifier: str,
        source: PublicationSource,
    ) -> str:
        """Normalize publication identifier based on source."""
        identifier = identifier.strip()

        if source == PublicationSource.DOI:
            # Remove doi: prefix and normalize
            if identifier.lower().startswith("doi:"):
                identifier = identifier[4:]
            if identifier.startswith("https://doi.org/"):
                identifier = identifier[16:]
            if identifier.startswith("http://dx.doi.org/"):
                identifier = identifier[18:]
            return identifier.lower()

        elif source == PublicationSource.PMID:
            # Extract numeric part
            return re.sub(r"\D", "", identifier)

        elif source == PublicationSource.PMC:
            # Normalize to PMC + number
            num = re.sub(r"\D", "", identifier)
            return f"PMC{num}"

        return identifier

    def _compute_inchikey(
        self,
        structure: str,
        structure_type: str,
    ) -> Optional[str]:
        """Compute InChIKey from chemical structure."""
        if not self._rdkit_available:
            # If InChIKey provided directly, validate format
            if structure_type == "inchikey":
                if re.match(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$", structure):
                    return structure
            return None

        try:
            from rdkit import Chem
            from rdkit.Chem.inchi import MolToInchiKey

            mol = None

            if structure_type == "smiles":
                mol = Chem.MolFromSmiles(structure)
            elif structure_type == "inchi":
                mol = Chem.MolFromInchi(structure)
            elif structure_type == "inchikey":
                # Already an InChIKey
                if re.match(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$", structure):
                    return structure
                return None

            if mol is None:
                return None

            return MolToInchiKey(mol)

        except Exception as e:
            logger.warning(f"Failed to compute InChIKey: {e}")
            return None

    def _normalize_sequence(self, sequence: str) -> str:
        """Normalize biological sequence string."""
        # Remove whitespace, convert to uppercase
        normalized = re.sub(r"\s+", "", sequence.upper())
        # Remove common non-sequence characters
        normalized = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY*-]", "", normalized)
        return normalized

    def _detect_sequence_type(self, sequence: str) -> SequenceType:
        """Detect whether sequence is amino acid or nucleotide."""
        # Count character types
        aa_only = set("DEFHIKLMNPQRSVWY")  # AA chars not in NT
        nt_only = set("U")  # RNA-specific

        seq_set = set(sequence.upper())

        if seq_set & aa_only:
            return SequenceType.AMINO_ACID
        if "U" in seq_set and "T" not in seq_set:
            return SequenceType.NUCLEOTIDE
        if seq_set <= set("ACGT"):
            return SequenceType.NUCLEOTIDE
        if seq_set <= set("ACGU"):
            return SequenceType.NUCLEOTIDE

        # Default to amino acid for ambiguous cases
        return SequenceType.AMINO_ACID

    def _cache_put(self, key: str, value: BioPATId):
        """Add item to cache with LRU eviction."""
        if len(self._cache) >= self._cache_size:
            # Simple eviction: remove oldest item
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value

    def clear_cache(self):
        """Clear the resolution cache."""
        self._cache.clear()


class CrossReferenceResolver:
    """Resolves cross-references between different identifier systems.

    Handles mappings like:
    - PMID <-> DOI (via PubMed/CrossRef)
    - SMILES <-> InChIKey (via RDKit)
    - Patent family relationships
    """

    def __init__(
        self,
        entity_resolver: Optional[EntityResolver] = None,
    ):
        """Initialize cross-reference resolver.

        Args:
            entity_resolver: EntityResolver instance to use.
        """
        self.entity_resolver = entity_resolver or EntityResolver()

        # Caches for cross-references
        self._pmid_to_doi: Dict[str, str] = {}
        self._doi_to_pmid: Dict[str, str] = {}
        self._smiles_to_inchikey: Dict[str, str] = {}

    def add_pmid_doi_mapping(self, pmid: str, doi: str):
        """Add a PMID <-> DOI mapping.

        Args:
            pmid: PubMed ID.
            doi: Digital Object Identifier.
        """
        pmid = pmid.strip()
        doi = doi.strip().lower()
        self._pmid_to_doi[pmid] = doi
        self._doi_to_pmid[doi] = pmid

    def get_doi_from_pmid(self, pmid: str) -> Optional[str]:
        """Look up DOI from PMID."""
        return self._pmid_to_doi.get(pmid.strip())

    def get_pmid_from_doi(self, doi: str) -> Optional[str]:
        """Look up PMID from DOI."""
        return self._doi_to_pmid.get(doi.strip().lower())

    def resolve_to_canonical(
        self,
        identifier: str,
        entity_type: Optional[EntityType] = None,
    ) -> BioPATId:
        """Resolve any identifier to its canonical BioPAT ID.

        Args:
            identifier: Raw identifier of any type.
            entity_type: Optional type hint.

        Returns:
            Canonical BioPATId.
        """
        if entity_type == EntityType.PATENT:
            return self.entity_resolver.resolve_patent(identifier)
        elif entity_type == EntityType.PUBLICATION:
            return self.entity_resolver.resolve_publication(identifier)
        elif entity_type == EntityType.CHEMICAL:
            return self.entity_resolver.resolve_chemical(identifier)
        elif entity_type == EntityType.SEQUENCE:
            return self.entity_resolver.resolve_sequence(identifier)
        else:
            # Auto-detect type
            return self._auto_resolve(identifier)

    def _auto_resolve(self, identifier: str) -> BioPATId:
        """Auto-detect entity type and resolve."""
        identifier = identifier.strip()

        # Check if already a BioPAT ID
        if identifier.startswith(f"{BIOPAT_PREFIX}:"):
            parsed = self.entity_resolver.parse_biopat_id(identifier)
            if parsed:
                return parsed

        # Check for patent patterns
        patent_pattern = r"^(US|EP|WO|JP|CN|KR|GB|DE|FR|CA|AU)\s*[\d/]+"
        if re.match(patent_pattern, identifier.upper()):
            return self.entity_resolver.resolve_patent(identifier)

        # Check for PMID (pure numeric)
        if identifier.isdigit() and 1 <= len(identifier) <= 10:
            return self.entity_resolver.resolve_publication(
                identifier, PublicationSource.PMID
            )

        # Check for DOI
        if identifier.startswith("10.") or "doi.org" in identifier:
            return self.entity_resolver.resolve_publication(
                identifier, PublicationSource.DOI
            )

        # Check for InChIKey pattern
        if re.match(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$", identifier):
            return self.entity_resolver.resolve_chemical(identifier, "inchikey")

        # Check for SMILES-like pattern
        if re.match(r"^[A-Za-z0-9@+\-\[\]()=#$.\\/:]+$", identifier) and len(identifier) > 5:
            # Could be SMILES, try to resolve as chemical
            return self.entity_resolver.resolve_chemical(identifier, "smiles")

        # Check for sequence (long string of letters)
        if re.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", identifier.upper()) and len(identifier) > 20:
            return self.entity_resolver.resolve_sequence(identifier)

        # Default: treat as publication
        return self.entity_resolver.resolve_publication(identifier)

    def bulk_resolve(
        self,
        identifiers: List[str],
        entity_type: Optional[EntityType] = None,
    ) -> List[BioPATId]:
        """Resolve multiple identifiers.

        Args:
            identifiers: List of raw identifiers.
            entity_type: Optional type hint for all identifiers.

        Returns:
            List of resolved BioPATIds.
        """
        return [
            self.resolve_to_canonical(id_, entity_type)
            for id_ in identifiers
        ]


def create_biopat_id(
    entity_type: EntityType,
    identifier: str,
    subtype: Optional[str] = None,
) -> str:
    """Create a BioPAT ID string.

    Args:
        entity_type: Type of entity.
        identifier: Primary identifier.
        subtype: Optional subtype (jurisdiction, source, etc.).

    Returns:
        BioPAT ID string.
    """
    if subtype:
        return f"{BIOPAT_PREFIX}:{entity_type.value}:{subtype}:{identifier}"
    return f"{BIOPAT_PREFIX}:{entity_type.value}:{identifier}"


def is_valid_biopat_id(id_string: str) -> bool:
    """Check if a string is a valid BioPAT ID.

    Args:
        id_string: String to validate.

    Returns:
        True if valid BioPAT ID format.
    """
    if not id_string.startswith(f"{BIOPAT_PREFIX}:"):
        return False

    parts = id_string.split(":")
    if len(parts) < 3:
        return False

    try:
        EntityType(parts[1])
        return True
    except ValueError:
        return False
