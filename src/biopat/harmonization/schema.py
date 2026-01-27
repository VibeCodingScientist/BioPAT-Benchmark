"""SQL Schema for BioPAT Unified Entity System.

Phase 4.0 (Advanced): Defines SQLAlchemy models for storing and linking
heterogeneous entity types (Patents, Papers, Chemicals, Sequences) in
a unified relational database.

Schema Overview:
- patents: Unified patent records across US/EP/WO jurisdictions
- publications: Scientific papers with PMID/DOI cross-references
- chemicals: Chemical structures with InChIKey, SMILES, properties
- sequences: Biological sequences with type and hash identifiers
- patent_chemical_links: Links patents to chemical structures
- patent_sequence_links: Links patents to biological sequences
- publication_chemical_links: Links papers to chemical structures
- publication_sequence_links: Links papers to biological sequences
- cross_references: Generic entity cross-reference table
"""

import logging
from datetime import datetime
from enum import Enum as PyEnum
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


# Enums for database columns
class JurisdictionEnum(str, PyEnum):
    """Patent jurisdiction types."""
    US = "US"
    EP = "EP"
    WO = "WO"
    JP = "JP"
    CN = "CN"
    KR = "KR"
    GB = "GB"
    DE = "DE"
    OTHER = "OTHER"


class DocumentTypeEnum(str, PyEnum):
    """Document type classification."""
    PATENT_APPLICATION = "application"
    PATENT_GRANT = "grant"
    SCIENTIFIC_PAPER = "paper"
    PREPRINT = "preprint"


class SequenceTypeEnum(str, PyEnum):
    """Biological sequence types."""
    AMINO_ACID = "AA"
    NUCLEOTIDE = "NT"
    UNKNOWN = "UNK"


class LinkSourceEnum(str, PyEnum):
    """Source of entity link."""
    CITATION = "citation"  # Direct citation
    EXAMINER = "examiner"  # Examiner citation
    APPLICANT = "applicant"  # Applicant citation
    EXTRACTED = "extracted"  # Extracted from text
    INFERRED = "inferred"  # Inferred relationship


class Patent(Base):
    """Unified patent record across jurisdictions."""

    __tablename__ = "patents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    biopat_id = Column(String(64), unique=True, nullable=False, index=True)

    # Core identifiers
    jurisdiction = Column(Enum(JurisdictionEnum), nullable=False, index=True)
    doc_number = Column(String(32), nullable=False)
    kind_code = Column(String(8))
    family_id = Column(String(64), index=True)

    # Bibliographic data
    title = Column(Text)
    abstract = Column(Text)
    publication_date = Column(DateTime, index=True)
    filing_date = Column(DateTime)
    priority_date = Column(DateTime, index=True)

    # Classification
    ipc_codes = Column(ARRAY(String), default=list)
    cpc_codes = Column(ARRAY(String), default=list)

    # Document type
    doc_type = Column(Enum(DocumentTypeEnum), default=DocumentTypeEnum.PATENT_APPLICATION)

    # Claims (for text search and analysis)
    claims_text = Column(Text)
    independent_claims = Column(ARRAY(Text), default=list)

    # Metadata
    source = Column(String(32))  # patentsview, epo, wipo
    raw_data = Column(JSONB)  # Store original API response
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    chemical_links = relationship("PatentChemicalLink", back_populates="patent")
    sequence_links = relationship("PatentSequenceLink", back_populates="patent")
    publication_citations = relationship("PatentPublicationCitation", back_populates="patent")

    __table_args__ = (
        UniqueConstraint("jurisdiction", "doc_number", name="uq_patent_jurisdiction_number"),
        Index("ix_patent_ipc", "ipc_codes", postgresql_using="gin"),
    )


class Publication(Base):
    """Scientific publication record."""

    __tablename__ = "publications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    biopat_id = Column(String(64), unique=True, nullable=False, index=True)

    # Core identifiers
    pmid = Column(String(16), unique=True, index=True)
    doi = Column(String(256), unique=True, index=True)
    pmc_id = Column(String(16), unique=True, index=True)
    openalex_id = Column(String(32), index=True)

    # Bibliographic data
    title = Column(Text)
    abstract = Column(Text)
    publication_date = Column(DateTime, index=True)
    journal = Column(String(512))
    volume = Column(String(32))
    issue = Column(String(32))
    pages = Column(String(64))

    # Authors (stored as JSON array)
    authors = Column(JSONB)

    # Document type
    doc_type = Column(Enum(DocumentTypeEnum), default=DocumentTypeEnum.SCIENTIFIC_PAPER)

    # Full text availability
    has_fulltext = Column(Boolean, default=False)
    fulltext_source = Column(String(32))

    # Metadata
    source = Column(String(32))  # openalex, pubmed, semanticscholar
    raw_data = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    chemical_links = relationship("PublicationChemicalLink", back_populates="publication")
    sequence_links = relationship("PublicationSequenceLink", back_populates="publication")
    patent_citations = relationship("PatentPublicationCitation", back_populates="publication")

    __table_args__ = (
        Index("ix_publication_date", "publication_date"),
    )


class Chemical(Base):
    """Chemical structure record."""

    __tablename__ = "chemicals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    biopat_id = Column(String(64), unique=True, nullable=False, index=True)

    # Core identifiers
    inchikey = Column(String(27), unique=True, nullable=False, index=True)
    inchi = Column(Text)
    canonical_smiles = Column(Text, index=True)

    # Alternative identifiers
    pubchem_cid = Column(String(16), index=True)
    chembl_id = Column(String(32), index=True)
    cas_number = Column(String(32))
    drugbank_id = Column(String(16))

    # Names
    preferred_name = Column(String(512))
    synonyms = Column(ARRAY(String), default=list)

    # Molecular properties
    molecular_formula = Column(String(256))
    molecular_weight = Column(Float)
    exact_mass = Column(Float)
    heavy_atom_count = Column(Integer)
    rotatable_bond_count = Column(Integer)
    h_bond_donor_count = Column(Integer)
    h_bond_acceptor_count = Column(Integer)
    logp = Column(Float)
    tpsa = Column(Float)  # Topological Polar Surface Area

    # Fingerprints (stored as binary or base64)
    morgan_fp_2048 = Column(Text)  # Morgan fingerprint radius 2, 2048 bits
    rdkit_fp = Column(Text)  # RDKit fingerprint

    # Source
    source = Column(String(32))  # surechembl, pubchem, extracted
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patent_links = relationship("PatentChemicalLink", back_populates="chemical")
    publication_links = relationship("PublicationChemicalLink", back_populates="chemical")

    __table_args__ = (
        Index("ix_chemical_mw", "molecular_weight"),
    )


class Sequence(Base):
    """Biological sequence record."""

    __tablename__ = "sequences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    biopat_id = Column(String(64), unique=True, nullable=False, index=True)

    # Core identifiers
    sequence_hash = Column(String(64), unique=True, nullable=False, index=True)
    sequence_type = Column(Enum(SequenceTypeEnum), nullable=False, index=True)

    # Sequence data
    sequence = Column(Text, nullable=False)
    length = Column(Integer, nullable=False, index=True)

    # External identifiers
    ncbi_accession = Column(String(32), index=True)
    uniprot_id = Column(String(16), index=True)
    genbank_id = Column(String(32))
    pdb_id = Column(String(8))

    # Annotations
    organism = Column(String(256))
    gene_name = Column(String(128))
    protein_name = Column(String(512))
    description = Column(Text)

    # For patent sequences: SEQ ID NO
    patent_seq_id = Column(String(32))

    # Source
    source = Column(String(32))  # ncbi, uniprot, extracted
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patent_links = relationship("PatentSequenceLink", back_populates="sequence")
    publication_links = relationship("PublicationSequenceLink", back_populates="sequence")

    __table_args__ = (
        Index("ix_sequence_type_length", "sequence_type", "length"),
    )


# Link tables
class PatentChemicalLink(Base):
    """Links patents to chemical structures."""

    __tablename__ = "patent_chemical_links"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patent_id = Column(Integer, ForeignKey("patents.id"), nullable=False, index=True)
    chemical_id = Column(Integer, ForeignKey("chemicals.id"), nullable=False, index=True)

    # Link metadata
    link_source = Column(Enum(LinkSourceEnum), default=LinkSourceEnum.EXTRACTED)
    confidence = Column(Float, default=1.0)
    context = Column(Text)  # Surrounding text where chemical was found
    claim_numbers = Column(ARRAY(Integer), default=list)  # Which claims mention this

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    patent = relationship("Patent", back_populates="chemical_links")
    chemical = relationship("Chemical", back_populates="patent_links")

    __table_args__ = (
        UniqueConstraint("patent_id", "chemical_id", name="uq_patent_chemical"),
    )


class PatentSequenceLink(Base):
    """Links patents to biological sequences."""

    __tablename__ = "patent_sequence_links"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patent_id = Column(Integer, ForeignKey("patents.id"), nullable=False, index=True)
    sequence_id = Column(Integer, ForeignKey("sequences.id"), nullable=False, index=True)

    # Link metadata
    link_source = Column(Enum(LinkSourceEnum), default=LinkSourceEnum.EXTRACTED)
    confidence = Column(Float, default=1.0)
    seq_id_no = Column(String(16))  # SEQ ID NO from patent
    claim_numbers = Column(ARRAY(Integer), default=list)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    patent = relationship("Patent", back_populates="sequence_links")
    sequence = relationship("Sequence", back_populates="patent_links")

    __table_args__ = (
        UniqueConstraint("patent_id", "sequence_id", name="uq_patent_sequence"),
    )


class PublicationChemicalLink(Base):
    """Links publications to chemical structures."""

    __tablename__ = "publication_chemical_links"

    id = Column(Integer, primary_key=True, autoincrement=True)
    publication_id = Column(Integer, ForeignKey("publications.id"), nullable=False, index=True)
    chemical_id = Column(Integer, ForeignKey("chemicals.id"), nullable=False, index=True)

    # Link metadata
    link_source = Column(Enum(LinkSourceEnum), default=LinkSourceEnum.EXTRACTED)
    confidence = Column(Float, default=1.0)
    context = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    publication = relationship("Publication", back_populates="chemical_links")
    chemical = relationship("Chemical", back_populates="publication_links")

    __table_args__ = (
        UniqueConstraint("publication_id", "chemical_id", name="uq_publication_chemical"),
    )


class PublicationSequenceLink(Base):
    """Links publications to biological sequences."""

    __tablename__ = "publication_sequence_links"

    id = Column(Integer, primary_key=True, autoincrement=True)
    publication_id = Column(Integer, ForeignKey("publications.id"), nullable=False, index=True)
    sequence_id = Column(Integer, ForeignKey("sequences.id"), nullable=False, index=True)

    # Link metadata
    link_source = Column(Enum(LinkSourceEnum), default=LinkSourceEnum.EXTRACTED)
    confidence = Column(Float, default=1.0)
    accession_in_text = Column(String(32))  # Accession number as found in text

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    publication = relationship("Publication", back_populates="sequence_links")
    sequence = relationship("Sequence", back_populates="publication_links")

    __table_args__ = (
        UniqueConstraint("publication_id", "sequence_id", name="uq_publication_sequence"),
    )


class PatentPublicationCitation(Base):
    """Direct citation links between patents and publications."""

    __tablename__ = "patent_publication_citations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patent_id = Column(Integer, ForeignKey("patents.id"), nullable=False, index=True)
    publication_id = Column(Integer, ForeignKey("publications.id"), nullable=False, index=True)

    # Citation metadata
    link_source = Column(Enum(LinkSourceEnum), default=LinkSourceEnum.CITATION)
    citation_category = Column(String(8))  # X, Y, A for EP; 102, 103 for US
    relevance_score = Column(Integer)  # 0-3 scale
    claims_affected = Column(ARRAY(Integer), default=list)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    patent = relationship("Patent", back_populates="publication_citations")
    publication = relationship("Publication", back_populates="patent_citations")

    __table_args__ = (
        UniqueConstraint("patent_id", "publication_id", name="uq_patent_publication"),
        Index("ix_citation_relevance", "relevance_score"),
    )


class CrossReference(Base):
    """Generic cross-reference table for entity relationships."""

    __tablename__ = "cross_references"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Source entity
    source_biopat_id = Column(String(64), nullable=False, index=True)
    source_type = Column(String(8), nullable=False)  # PAT, PUB, CHM, SEQ

    # Target entity
    target_biopat_id = Column(String(64), nullable=False, index=True)
    target_type = Column(String(8), nullable=False)

    # Relationship metadata
    relationship_type = Column(String(32), nullable=False)  # cites, contains, similar_to
    confidence = Column(Float, default=1.0)
    extra_data = Column(JSONB)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_crossref_source", "source_biopat_id", "source_type"),
        Index("ix_crossref_target", "target_biopat_id", "target_type"),
        UniqueConstraint(
            "source_biopat_id", "target_biopat_id", "relationship_type",
            name="uq_crossref_relationship"
        ),
    )


# Database connection and session management
class DatabaseManager:
    """Manages database connections and schema operations."""

    def __init__(
        self,
        database_url: str = "sqlite:///biopat.db",
        echo: bool = False,
    ):
        """Initialize database manager.

        Args:
            database_url: SQLAlchemy database URL.
            echo: Whether to echo SQL statements.
        """
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=echo)
        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(self.engine)
        logger.info("Created all database tables")

    def drop_tables(self):
        """Drop all tables in the database."""
        Base.metadata.drop_all(self.engine)
        logger.info("Dropped all database tables")

    def get_session(self):
        """Get a new database session."""
        return self.Session()

    def close(self):
        """Close the database connection."""
        self.engine.dispose()


def get_database_url_from_config(config: dict) -> str:
    """Build database URL from configuration.

    Args:
        config: Configuration dictionary with database settings.

    Returns:
        SQLAlchemy database URL string.
    """
    db_config = config.get("database", {})

    backend = db_config.get("backend", "sqlite")

    if backend == "sqlite":
        path = db_config.get("path", "data/biopat.db")
        return f"sqlite:///{path}"

    elif backend == "postgresql":
        host = db_config.get("host", "localhost")
        port = db_config.get("port", 5432)
        user = db_config.get("user", "biopat")
        password = db_config.get("password", "")
        database = db_config.get("database", "biopat")
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    else:
        raise ValueError(f"Unsupported database backend: {backend}")
