"""Tests for Phase 2 modules."""

import pytest
import polars as pl
from pathlib import Path

from biopat.processing.npl_parser import NPLParser, NPLLinker, ParsedCitation
from biopat.processing.claim_mapper import ClaimMapper, ClaimCitationMapper
from biopat.groundtruth.stratification import DomainStratifier
from biopat.groundtruth.relevance import RelevanceAssigner, RelevanceLevel


class TestNPLParser:
    """Tests for NPL citation parsing."""

    def test_extract_pmid(self):
        """Test PMID extraction."""
        parser = NPLParser()

        assert parser.extract_pmid("PMID: 12345678") == "12345678"
        assert parser.extract_pmid("PubMed: 87654321") == "87654321"
        assert parser.extract_pmid("pubmed.ncbi.nlm.nih.gov/11111111") == "11111111"
        assert parser.extract_pmid("No PMID here") is None

    def test_extract_doi(self):
        """Test DOI extraction."""
        parser = NPLParser()

        assert parser.extract_doi("DOI: 10.1021/jm401234") == "10.1021/jm401234"
        assert parser.extract_doi("doi.org/10.1038/nature12345") == "10.1038/nature12345"
        assert parser.extract_doi("10.1126/science.abc1234") == "10.1126/science.abc1234"
        assert parser.extract_doi("No DOI here") is None

    def test_extract_year(self):
        """Test year extraction."""
        parser = NPLParser()

        assert parser.extract_year("Published in 2015") == 2015
        assert parser.extract_year("J Med Chem, 2018, 61") == 2018
        assert parser.extract_year("1999 journal article") == 1999
        assert parser.extract_year("No year") is None

    def test_parse_citation_with_pmid(self):
        """Test parsing citation with PMID."""
        parser = NPLParser()

        text = "Smith et al., J Med Chem, 2015. PMID: 12345678"
        parsed = parser.parse_citation(text)

        assert parsed.pmid == "12345678"
        assert parsed.year == 2015
        assert parsed.parse_method == "pmid"

    def test_parse_citation_with_doi(self):
        """Test parsing citation with DOI."""
        parser = NPLParser()

        text = "Jones et al., Nature, 2018. DOI: 10.1038/nature12345"
        parsed = parser.parse_citation(text)

        assert parsed.doi == "10.1038/nature12345"
        assert parsed.year == 2018
        assert parsed.parse_method == "doi"


class TestNPLLinker:
    """Tests for NPL citation linking."""

    @pytest.fixture
    def sample_papers(self):
        """Create sample papers DataFrame."""
        return pl.DataFrame({
            "paper_id": ["W1", "W2", "W3"],
            "pmid": ["12345678", "87654321", None],
            "doi": ["10.1021/abc", None, "10.1038/xyz"],
            "title": ["Novel inhibitors for cancer", "Protein structure analysis", "Gene therapy advances"],
        })

    def test_link_by_pmid(self, sample_papers):
        """Test linking by PMID."""
        linker = NPLLinker(sample_papers)

        parsed = ParsedCitation(pmid="12345678", raw_text="test")
        result = linker.link_citation(parsed)

        assert result == "W1"

    def test_link_by_doi(self, sample_papers):
        """Test linking by DOI."""
        linker = NPLLinker(sample_papers)

        parsed = ParsedCitation(doi="10.1038/xyz", raw_text="test")
        result = linker.link_citation(parsed)

        assert result == "W3"

    def test_link_no_match(self, sample_papers):
        """Test when no match found."""
        linker = NPLLinker(sample_papers)

        parsed = ParsedCitation(pmid="99999999", raw_text="test")
        result = linker.link_citation(parsed)

        assert result is None


class TestClaimMapper:
    """Tests for claim mapping."""

    def test_parse_single_claim(self):
        """Test parsing single claim number."""
        mapper = ClaimMapper()

        text = "Claim 1 is rejected under 35 USC 102"
        claims = mapper.parse_claim_numbers(text)

        assert claims == [1]

    def test_parse_claim_range(self):
        """Test parsing claim range."""
        mapper = ClaimMapper()

        text = "Claims 1-5 are rejected under 35 USC 103"
        claims = mapper.parse_claim_numbers(text)

        assert claims == [1, 2, 3, 4, 5]

    def test_parse_claim_list(self):
        """Test parsing comma-separated claims."""
        mapper = ClaimMapper()

        text = "Claims 1, 3, 5, 7 are rejected"
        claims = mapper.parse_claim_numbers(text)

        assert claims == [1, 3, 5, 7]

    def test_parse_mixed_claims(self):
        """Test parsing mixed range and list."""
        mapper = ClaimMapper()

        text = "Claims 1-3, 5, and 10-12 are rejected under 35 USC 102(a)(1)"
        claims = mapper.parse_claim_numbers(text)

        assert claims == [1, 2, 3, 5, 10, 11, 12]

    def test_extract_rejection_type_102(self):
        """Test extracting 102 rejection type."""
        mapper = ClaimMapper()

        text = "rejected under 35 U.S.C. 102(a)(1)"
        assert mapper.extract_rejection_type(text) == "102"

        text = "§102 anticipation rejection"
        assert mapper.extract_rejection_type(text) == "102"

    def test_extract_rejection_type_103(self):
        """Test extracting 103 rejection type."""
        mapper = ClaimMapper()

        text = "rejected under 35 U.S.C. § 103"
        assert mapper.extract_rejection_type(text) == "103"

        text = "would have been obvious under 103"
        assert mapper.extract_rejection_type(text) == "103"


class TestDomainStratifier:
    """Tests for domain stratification."""

    def test_get_ipc3(self):
        """Test IPC3 extraction."""
        stratifier = DomainStratifier()

        ipc3 = stratifier.get_ipc3(["A61K39/395", "C07D213/00"])
        assert "A61K" in ipc3
        assert "C07D" in ipc3

    def test_classify_in_domain(self):
        """Test IN-domain classification."""
        stratifier = DomainStratifier()

        query_ipc = {"A61K"}
        doc_ipc = {"A61K", "A61P"}

        result = stratifier.classify_domain_type(query_ipc, doc_ipc)
        assert result == "IN"

    def test_classify_out_domain(self):
        """Test OUT-domain classification."""
        stratifier = DomainStratifier()

        query_ipc = {"A61K"}
        doc_ipc = {"C07D", "C12N"}

        result = stratifier.classify_domain_type(query_ipc, doc_ipc)
        assert result == "OUT"


class TestGradedRelevance:
    """Tests for graded relevance assignment."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temp directory."""
        return tmp_path

    def test_relevance_levels(self):
        """Test relevance level values."""
        assert RelevanceLevel.NOT_RELEVANT == 0
        assert RelevanceLevel.RELEVANT == 1
        assert RelevanceLevel.HIGHLY_RELEVANT == 2
        assert RelevanceLevel.NOVELTY_DESTROYING == 3

    def test_102_gets_score_3(self, temp_dir):
        """Test that 102 rejections get highest score."""
        assigner = RelevanceAssigner(temp_dir)

        # Citation with 102 rejection
        citations = pl.DataFrame({
            "query_id": ["Q1"],
            "paper_id": ["P1"],
            "confidence": [10],
            "source": ["office_action"],
            "rejection_type": ["102"],
        })

        qrels = assigner.create_graded_qrels_from_merged(citations, save=False)

        assert len(qrels) == 1
        assert qrels["relevance"][0] == 3

    def test_103_gets_score_2(self, temp_dir):
        """Test that 103 rejections get score 2."""
        assigner = RelevanceAssigner(temp_dir)

        citations = pl.DataFrame({
            "query_id": ["Q1"],
            "paper_id": ["P1"],
            "confidence": [10],
            "source": ["office_action"],
            "rejection_type": ["103"],
        })

        qrels = assigner.create_graded_qrels_from_merged(citations, save=False)

        assert len(qrels) == 1
        assert qrels["relevance"][0] == 2

    def test_high_confidence_examiner_gets_score_2(self, temp_dir):
        """Test high-confidence examiner citation gets score 2."""
        assigner = RelevanceAssigner(temp_dir)

        citations = pl.DataFrame({
            "query_id": ["Q1"],
            "paper_id": ["P1"],
            "confidence": [9],
            "source": ["examiner"],
            "rejection_type": [None],
        })

        qrels = assigner.create_graded_qrels_from_merged(citations, save=False)

        assert len(qrels) == 1
        assert qrels["relevance"][0] == 2

    def test_medium_confidence_examiner_gets_score_1(self, temp_dir):
        """Test medium-confidence examiner citation gets score 1."""
        assigner = RelevanceAssigner(temp_dir)

        citations = pl.DataFrame({
            "query_id": ["Q1"],
            "paper_id": ["P1"],
            "confidence": [7],
            "source": ["examiner"],
            "rejection_type": [None],
        })

        qrels = assigner.create_graded_qrels_from_merged(citations, save=False)

        assert len(qrels) == 1
        assert qrels["relevance"][0] == 1

    def test_low_confidence_filtered_out(self, temp_dir):
        """Test low-confidence citations are filtered out."""
        assigner = RelevanceAssigner(temp_dir)

        citations = pl.DataFrame({
            "query_id": ["Q1"],
            "paper_id": ["P1"],
            "confidence": [5],
            "source": ["applicant"],
            "rejection_type": [None],
        })

        qrels = assigner.create_graded_qrels_from_merged(citations, save=False)

        # Should be filtered out (relevance 0)
        assert len(qrels) == 0
