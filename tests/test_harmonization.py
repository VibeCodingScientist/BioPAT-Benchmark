"""Tests for the Data Harmonization Layer (Phase 4.0)."""

import pytest
import tempfile
from pathlib import Path

from biopat.harmonization import (
    BIOPAT_PREFIX,
    BioPATId,
    EntityResolver,
    EntityType,
    PublicationSource,
    SequenceType,
    create_biopat_id,
    is_valid_biopat_id,
    BidirectionalIndex,
)


class TestBioPATId:
    """Tests for BioPATId dataclass."""

    def test_canonical_with_subtype(self):
        """Test canonical ID with subtype."""
        bid = BioPATId(
            entity_type=EntityType.PATENT,
            identifier="10123456",
            subtype="US",
        )
        assert bid.canonical == "BP:PAT:US:10123456"

    def test_canonical_without_subtype(self):
        """Test canonical ID without subtype."""
        bid = BioPATId(
            entity_type=EntityType.CHEMICAL,
            identifier="BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
        )
        assert bid.canonical == "BP:CHM:BSYNRYMUTXBXSQ-UHFFFAOYSA-N"

    def test_equality(self):
        """Test BioPATId equality based on canonical form."""
        bid1 = BioPATId(EntityType.PATENT, "123", "US")
        bid2 = BioPATId(EntityType.PATENT, "123", "US")
        bid3 = BioPATId(EntityType.PATENT, "123", "EP")

        assert bid1 == bid2
        assert bid1 != bid3

    def test_hash(self):
        """Test BioPATId is hashable."""
        bid1 = BioPATId(EntityType.PATENT, "123", "US")
        bid2 = BioPATId(EntityType.PATENT, "123", "US")

        ids = {bid1, bid2}
        assert len(ids) == 1


class TestEntityResolver:
    """Tests for EntityResolver class."""

    @pytest.fixture
    def resolver(self):
        """Create entity resolver fixture."""
        return EntityResolver(use_rdkit=False)

    def test_resolve_patent_us(self, resolver):
        """Test resolving US patent."""
        bid = resolver.resolve_patent("US10123456")
        assert bid.entity_type == EntityType.PATENT
        assert bid.subtype == "US"
        assert "10123456" in bid.identifier

    def test_resolve_patent_ep(self, resolver):
        """Test resolving EP patent."""
        bid = resolver.resolve_patent("EP 1234567 A1")
        assert bid.entity_type == EntityType.PATENT
        assert bid.subtype == "EP"

    def test_resolve_patent_wo(self, resolver):
        """Test resolving WO patent."""
        bid = resolver.resolve_patent("WO2020/123456")
        assert bid.entity_type == EntityType.PATENT
        assert bid.subtype == "WO"

    def test_resolve_publication_pmid(self, resolver):
        """Test resolving PMID."""
        bid = resolver.resolve_publication("12345678", PublicationSource.PMID)
        assert bid.entity_type == EntityType.PUBLICATION
        assert bid.subtype == "PMID"
        assert bid.identifier == "12345678"

    def test_resolve_publication_doi(self, resolver):
        """Test resolving DOI."""
        bid = resolver.resolve_publication("10.1234/abc.123", PublicationSource.DOI)
        assert bid.entity_type == EntityType.PUBLICATION
        assert bid.subtype == "DOI"
        assert "10.1234" in bid.identifier

    def test_resolve_publication_doi_with_prefix(self, resolver):
        """Test resolving DOI with prefix."""
        bid = resolver.resolve_publication("doi:10.1234/abc.123")
        assert bid.entity_type == EntityType.PUBLICATION
        assert "10.1234" in bid.identifier

    def test_resolve_publication_auto_detect_doi(self, resolver):
        """Test auto-detecting DOI source."""
        bid = resolver.resolve_publication("10.1038/nature12373")
        assert bid.subtype == "DOI"

    def test_resolve_publication_auto_detect_pmid(self, resolver):
        """Test auto-detecting PMID source."""
        bid = resolver.resolve_publication("12345678")
        assert bid.subtype == "PMID"

    def test_resolve_chemical_inchikey(self, resolver):
        """Test resolving chemical by InChIKey."""
        inchikey = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
        bid = resolver.resolve_chemical(inchikey, "inchikey")
        assert bid.entity_type == EntityType.CHEMICAL
        assert bid.identifier == inchikey

    def test_resolve_sequence_amino_acid(self, resolver):
        """Test resolving amino acid sequence."""
        sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH"
        bid = resolver.resolve_sequence(sequence)
        assert bid.entity_type == EntityType.SEQUENCE
        assert bid.subtype == "AA"
        assert len(bid.identifier) == 12  # SHA256 truncated

    def test_resolve_sequence_nucleotide(self, resolver):
        """Test resolving nucleotide sequence."""
        sequence = "ATGCATGCATGCATGCATGCATGCATGC"
        bid = resolver.resolve_sequence(sequence)
        assert bid.entity_type == EntityType.SEQUENCE
        assert bid.subtype == "NT"

    def test_parse_biopat_id_patent(self, resolver):
        """Test parsing BioPAT ID string for patent."""
        parsed = resolver.parse_biopat_id("BP:PAT:US:10123456")
        assert parsed is not None
        assert parsed.entity_type == EntityType.PATENT
        assert parsed.subtype == "US"
        assert parsed.identifier == "10123456"

    def test_parse_biopat_id_chemical(self, resolver):
        """Test parsing BioPAT ID string for chemical."""
        parsed = resolver.parse_biopat_id("BP:CHM:BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
        assert parsed is not None
        assert parsed.entity_type == EntityType.CHEMICAL
        assert "BSYNRYMUTXBXSQ" in parsed.identifier

    def test_parse_biopat_id_invalid(self, resolver):
        """Test parsing invalid BioPAT ID."""
        parsed = resolver.parse_biopat_id("INVALID:ID:123")
        assert parsed is None

    def test_cache_hit(self, resolver):
        """Test resolver caching."""
        bid1 = resolver.resolve_patent("US10123456")
        bid2 = resolver.resolve_patent("US10123456")
        assert bid1 is bid2  # Same object from cache


class TestCreateBiopatId:
    """Tests for create_biopat_id helper."""

    def test_create_patent_id(self):
        """Test creating patent ID."""
        id_str = create_biopat_id(EntityType.PATENT, "10123456", "US")
        assert id_str == "BP:PAT:US:10123456"

    def test_create_chemical_id(self):
        """Test creating chemical ID without subtype."""
        id_str = create_biopat_id(EntityType.CHEMICAL, "ABCDEFGHIJKLMN-OPQRSTUVWX-Y")
        assert id_str == "BP:CHM:ABCDEFGHIJKLMN-OPQRSTUVWX-Y"


class TestIsValidBiopatId:
    """Tests for is_valid_biopat_id helper."""

    def test_valid_patent_id(self):
        """Test valid patent ID."""
        assert is_valid_biopat_id("BP:PAT:US:10123456") is True

    def test_valid_chemical_id(self):
        """Test valid chemical ID."""
        assert is_valid_biopat_id("BP:CHM:INCHIKEY123") is True

    def test_invalid_prefix(self):
        """Test invalid prefix."""
        assert is_valid_biopat_id("XX:PAT:US:123") is False

    def test_invalid_type(self):
        """Test invalid entity type."""
        assert is_valid_biopat_id("BP:XXX:123") is False

    def test_too_few_parts(self):
        """Test ID with too few parts."""
        assert is_valid_biopat_id("BP:PAT") is False


class TestBidirectionalIndex:
    """Tests for BidirectionalIndex class."""

    @pytest.fixture
    def index(self):
        """Create index fixture."""
        idx = BidirectionalIndex()
        idx.add_link("A", "B", {"score": 0.9})
        idx.add_link("A", "C", {"score": 0.8})
        idx.add_link("B", "D")
        return idx

    def test_get_targets(self, index):
        """Test getting targets from source."""
        targets = index.get_targets("A")
        assert targets == {"B", "C"}

    def test_get_sources(self, index):
        """Test getting sources from target."""
        sources = index.get_sources("B")
        assert sources == {"A"}

    def test_get_metadata(self, index):
        """Test getting link metadata."""
        meta = index.get_metadata("A", "B")
        assert meta["score"] == 0.9

    def test_has_link(self, index):
        """Test checking link existence."""
        assert index.has_link("A", "B") is True
        assert index.has_link("A", "D") is False

    def test_remove_link(self, index):
        """Test removing a link."""
        index.remove_link("A", "B")
        assert index.has_link("A", "B") is False
        assert "B" not in index.get_targets("A")
        assert "A" not in index.get_sources("B")

    def test_size(self, index):
        """Test size calculation."""
        assert index.size() == 3

    def test_to_dataframe(self, index):
        """Test converting to DataFrame."""
        df = index.to_dataframe()
        assert len(df) == 3
        assert "source_id" in df.columns
        assert "target_id" in df.columns


class TestSequenceTypeDetection:
    """Tests for sequence type detection."""

    @pytest.fixture
    def resolver(self):
        return EntityResolver(use_rdkit=False)

    def test_detect_amino_acid(self, resolver):
        """Test detecting amino acid sequence."""
        # Contains F, H, W which are AA-only
        seq = "MFHWDEFGHIKLMNPQRSTVWY"
        bid = resolver.resolve_sequence(seq)
        assert bid.subtype == "AA"

    def test_detect_dna(self, resolver):
        """Test detecting DNA sequence."""
        seq = "ATGCATGCATGCATGCATGCATGC"
        bid = resolver.resolve_sequence(seq)
        assert bid.subtype == "NT"

    def test_detect_rna(self, resolver):
        """Test detecting RNA sequence (U instead of T)."""
        seq = "AUGCAUGCAUGCAUGCAUGCAUGC"
        bid = resolver.resolve_sequence(seq)
        assert bid.subtype == "NT"


class TestPublicationSourceDetection:
    """Tests for publication source auto-detection."""

    @pytest.fixture
    def resolver(self):
        return EntityResolver(use_rdkit=False)

    def test_detect_doi_with_slash(self, resolver):
        """Test detecting DOI with slash."""
        bid = resolver.resolve_publication("10.1038/nature12373")
        assert bid.subtype == "DOI"

    def test_detect_doi_url(self, resolver):
        """Test detecting DOI from URL."""
        bid = resolver.resolve_publication("https://doi.org/10.1038/nature12373")
        assert bid.subtype == "DOI"

    def test_detect_pmc(self, resolver):
        """Test detecting PMC ID."""
        bid = resolver.resolve_publication("PMC1234567")
        assert bid.subtype == "PMC"

    def test_detect_openalex(self, resolver):
        """Test detecting OpenAlex ID."""
        bid = resolver.resolve_publication("W2741809807")
        assert bid.subtype == "OA"
