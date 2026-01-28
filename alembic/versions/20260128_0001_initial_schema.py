"""Initial BioPAT v4.0 harmonization schema.

Revision ID: 0001
Revises: None
Create Date: 2026-01-28

This migration creates the complete v4.0 entity harmonization schema with:
- Patents (US/EP/WO jurisdiction support)
- Publications (PMID/DOI/PMC identifiers)
- Chemicals (InChIKey/SMILES/fingerprints)
- Sequences (protein/nucleotide with hashes)
- Link tables for entity relationships
- Cross-reference table for flexible linking
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def get_array_type():
    """Get appropriate array type for the database backend."""
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        return postgresql.ARRAY(sa.String)
    else:
        # SQLite: store as JSON string
        return sa.Text


def get_json_type():
    """Get appropriate JSON type for the database backend."""
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        return postgresql.JSONB
    else:
        # SQLite: use TEXT with JSON
        return sa.Text


def get_int_array_type():
    """Get appropriate integer array type for the database backend."""
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        return postgresql.ARRAY(sa.Integer)
    else:
        return sa.Text


def upgrade() -> None:
    # Create enum types for PostgreSQL
    bind = op.get_bind()
    is_postgres = bind.dialect.name == 'postgresql'

    # Patents table
    op.create_table(
        'patents',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('biopat_id', sa.String(64), nullable=False),
        sa.Column('jurisdiction', sa.String(8), nullable=False),
        sa.Column('doc_number', sa.String(32), nullable=False),
        sa.Column('kind_code', sa.String(8), nullable=True),
        sa.Column('family_id', sa.String(64), nullable=True),
        sa.Column('title', sa.Text(), nullable=True),
        sa.Column('abstract', sa.Text(), nullable=True),
        sa.Column('publication_date', sa.DateTime(), nullable=True),
        sa.Column('filing_date', sa.DateTime(), nullable=True),
        sa.Column('priority_date', sa.DateTime(), nullable=True),
        sa.Column('ipc_codes', get_array_type(), nullable=True),
        sa.Column('cpc_codes', get_array_type(), nullable=True),
        sa.Column('doc_type', sa.String(16), nullable=True),
        sa.Column('claims_text', sa.Text(), nullable=True),
        sa.Column('independent_claims', get_array_type(), nullable=True),
        sa.Column('source', sa.String(32), nullable=True),
        sa.Column('raw_data', get_json_type(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('biopat_id'),
        sa.UniqueConstraint('jurisdiction', 'doc_number', name='uq_patent_jurisdiction_number'),
    )
    op.create_index('ix_patents_biopat_id', 'patents', ['biopat_id'])
    op.create_index('ix_patents_jurisdiction', 'patents', ['jurisdiction'])
    op.create_index('ix_patents_family_id', 'patents', ['family_id'])
    op.create_index('ix_patents_publication_date', 'patents', ['publication_date'])
    op.create_index('ix_patents_priority_date', 'patents', ['priority_date'])

    # Publications table
    op.create_table(
        'publications',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('biopat_id', sa.String(64), nullable=False),
        sa.Column('pmid', sa.String(16), nullable=True),
        sa.Column('doi', sa.String(256), nullable=True),
        sa.Column('pmc_id', sa.String(16), nullable=True),
        sa.Column('openalex_id', sa.String(32), nullable=True),
        sa.Column('title', sa.Text(), nullable=True),
        sa.Column('abstract', sa.Text(), nullable=True),
        sa.Column('publication_date', sa.DateTime(), nullable=True),
        sa.Column('journal', sa.String(512), nullable=True),
        sa.Column('volume', sa.String(32), nullable=True),
        sa.Column('issue', sa.String(32), nullable=True),
        sa.Column('pages', sa.String(64), nullable=True),
        sa.Column('authors', get_json_type(), nullable=True),
        sa.Column('doc_type', sa.String(16), nullable=True),
        sa.Column('has_fulltext', sa.Boolean(), nullable=True),
        sa.Column('fulltext_source', sa.String(32), nullable=True),
        sa.Column('source', sa.String(32), nullable=True),
        sa.Column('raw_data', get_json_type(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('biopat_id'),
        sa.UniqueConstraint('pmid'),
        sa.UniqueConstraint('doi'),
        sa.UniqueConstraint('pmc_id'),
    )
    op.create_index('ix_publications_biopat_id', 'publications', ['biopat_id'])
    op.create_index('ix_publications_pmid', 'publications', ['pmid'])
    op.create_index('ix_publications_doi', 'publications', ['doi'])
    op.create_index('ix_publications_pmc_id', 'publications', ['pmc_id'])
    op.create_index('ix_publications_openalex_id', 'publications', ['openalex_id'])
    op.create_index('ix_publications_publication_date', 'publications', ['publication_date'])

    # Chemicals table
    op.create_table(
        'chemicals',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('biopat_id', sa.String(64), nullable=False),
        sa.Column('inchikey', sa.String(27), nullable=False),
        sa.Column('inchi', sa.Text(), nullable=True),
        sa.Column('canonical_smiles', sa.Text(), nullable=True),
        sa.Column('pubchem_cid', sa.String(16), nullable=True),
        sa.Column('chembl_id', sa.String(32), nullable=True),
        sa.Column('cas_number', sa.String(32), nullable=True),
        sa.Column('drugbank_id', sa.String(16), nullable=True),
        sa.Column('preferred_name', sa.String(512), nullable=True),
        sa.Column('synonyms', get_array_type(), nullable=True),
        sa.Column('molecular_formula', sa.String(256), nullable=True),
        sa.Column('molecular_weight', sa.Float(), nullable=True),
        sa.Column('exact_mass', sa.Float(), nullable=True),
        sa.Column('heavy_atom_count', sa.Integer(), nullable=True),
        sa.Column('rotatable_bond_count', sa.Integer(), nullable=True),
        sa.Column('h_bond_donor_count', sa.Integer(), nullable=True),
        sa.Column('h_bond_acceptor_count', sa.Integer(), nullable=True),
        sa.Column('logp', sa.Float(), nullable=True),
        sa.Column('tpsa', sa.Float(), nullable=True),
        sa.Column('morgan_fp_2048', sa.Text(), nullable=True),
        sa.Column('rdkit_fp', sa.Text(), nullable=True),
        sa.Column('source', sa.String(32), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('biopat_id'),
        sa.UniqueConstraint('inchikey'),
    )
    op.create_index('ix_chemicals_biopat_id', 'chemicals', ['biopat_id'])
    op.create_index('ix_chemicals_inchikey', 'chemicals', ['inchikey'])
    op.create_index('ix_chemicals_canonical_smiles', 'chemicals', ['canonical_smiles'])
    op.create_index('ix_chemicals_pubchem_cid', 'chemicals', ['pubchem_cid'])
    op.create_index('ix_chemicals_chembl_id', 'chemicals', ['chembl_id'])
    op.create_index('ix_chemicals_molecular_weight', 'chemicals', ['molecular_weight'])

    # Sequences table
    op.create_table(
        'sequences',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('biopat_id', sa.String(64), nullable=False),
        sa.Column('sequence_hash', sa.String(64), nullable=False),
        sa.Column('sequence_type', sa.String(8), nullable=False),
        sa.Column('sequence', sa.Text(), nullable=False),
        sa.Column('length', sa.Integer(), nullable=False),
        sa.Column('ncbi_accession', sa.String(32), nullable=True),
        sa.Column('uniprot_id', sa.String(16), nullable=True),
        sa.Column('genbank_id', sa.String(32), nullable=True),
        sa.Column('pdb_id', sa.String(8), nullable=True),
        sa.Column('organism', sa.String(256), nullable=True),
        sa.Column('gene_name', sa.String(128), nullable=True),
        sa.Column('protein_name', sa.String(512), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('patent_seq_id', sa.String(32), nullable=True),
        sa.Column('source', sa.String(32), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('biopat_id'),
        sa.UniqueConstraint('sequence_hash'),
    )
    op.create_index('ix_sequences_biopat_id', 'sequences', ['biopat_id'])
    op.create_index('ix_sequences_sequence_hash', 'sequences', ['sequence_hash'])
    op.create_index('ix_sequences_sequence_type', 'sequences', ['sequence_type'])
    op.create_index('ix_sequences_length', 'sequences', ['length'])
    op.create_index('ix_sequences_ncbi_accession', 'sequences', ['ncbi_accession'])
    op.create_index('ix_sequences_uniprot_id', 'sequences', ['uniprot_id'])

    # Patent-Chemical link table
    op.create_table(
        'patent_chemical_links',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('patent_id', sa.Integer(), nullable=False),
        sa.Column('chemical_id', sa.Integer(), nullable=False),
        sa.Column('link_source', sa.String(16), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('context', sa.Text(), nullable=True),
        sa.Column('claim_numbers', get_int_array_type(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['chemical_id'], ['chemicals.id']),
        sa.ForeignKeyConstraint(['patent_id'], ['patents.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('patent_id', 'chemical_id', name='uq_patent_chemical'),
    )
    op.create_index('ix_patent_chemical_links_patent_id', 'patent_chemical_links', ['patent_id'])
    op.create_index('ix_patent_chemical_links_chemical_id', 'patent_chemical_links', ['chemical_id'])

    # Patent-Sequence link table
    op.create_table(
        'patent_sequence_links',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('patent_id', sa.Integer(), nullable=False),
        sa.Column('sequence_id', sa.Integer(), nullable=False),
        sa.Column('link_source', sa.String(16), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('seq_id_no', sa.String(16), nullable=True),
        sa.Column('claim_numbers', get_int_array_type(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['patent_id'], ['patents.id']),
        sa.ForeignKeyConstraint(['sequence_id'], ['sequences.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('patent_id', 'sequence_id', name='uq_patent_sequence'),
    )
    op.create_index('ix_patent_sequence_links_patent_id', 'patent_sequence_links', ['patent_id'])
    op.create_index('ix_patent_sequence_links_sequence_id', 'patent_sequence_links', ['sequence_id'])

    # Publication-Chemical link table
    op.create_table(
        'publication_chemical_links',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('publication_id', sa.Integer(), nullable=False),
        sa.Column('chemical_id', sa.Integer(), nullable=False),
        sa.Column('link_source', sa.String(16), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('context', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['chemical_id'], ['chemicals.id']),
        sa.ForeignKeyConstraint(['publication_id'], ['publications.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('publication_id', 'chemical_id', name='uq_publication_chemical'),
    )
    op.create_index('ix_publication_chemical_links_publication_id', 'publication_chemical_links', ['publication_id'])
    op.create_index('ix_publication_chemical_links_chemical_id', 'publication_chemical_links', ['chemical_id'])

    # Publication-Sequence link table
    op.create_table(
        'publication_sequence_links',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('publication_id', sa.Integer(), nullable=False),
        sa.Column('sequence_id', sa.Integer(), nullable=False),
        sa.Column('link_source', sa.String(16), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('accession_in_text', sa.String(32), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['publication_id'], ['publications.id']),
        sa.ForeignKeyConstraint(['sequence_id'], ['sequences.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('publication_id', 'sequence_id', name='uq_publication_sequence'),
    )
    op.create_index('ix_publication_sequence_links_publication_id', 'publication_sequence_links', ['publication_id'])
    op.create_index('ix_publication_sequence_links_sequence_id', 'publication_sequence_links', ['sequence_id'])

    # Patent-Publication citation table
    op.create_table(
        'patent_publication_citations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('patent_id', sa.Integer(), nullable=False),
        sa.Column('publication_id', sa.Integer(), nullable=False),
        sa.Column('link_source', sa.String(16), nullable=True),
        sa.Column('citation_category', sa.String(8), nullable=True),
        sa.Column('relevance_score', sa.Integer(), nullable=True),
        sa.Column('claims_affected', get_int_array_type(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['patent_id'], ['patents.id']),
        sa.ForeignKeyConstraint(['publication_id'], ['publications.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('patent_id', 'publication_id', name='uq_patent_publication'),
    )
    op.create_index('ix_patent_publication_citations_patent_id', 'patent_publication_citations', ['patent_id'])
    op.create_index('ix_patent_publication_citations_publication_id', 'patent_publication_citations', ['publication_id'])
    op.create_index('ix_patent_publication_citations_relevance_score', 'patent_publication_citations', ['relevance_score'])

    # Cross-references table
    op.create_table(
        'cross_references',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('source_biopat_id', sa.String(64), nullable=False),
        sa.Column('source_type', sa.String(8), nullable=False),
        sa.Column('target_biopat_id', sa.String(64), nullable=False),
        sa.Column('target_type', sa.String(8), nullable=False),
        sa.Column('relationship_type', sa.String(32), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('extra_data', get_json_type(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('source_biopat_id', 'target_biopat_id', 'relationship_type', name='uq_crossref_relationship'),
    )
    op.create_index('ix_cross_references_source', 'cross_references', ['source_biopat_id', 'source_type'])
    op.create_index('ix_cross_references_target', 'cross_references', ['target_biopat_id', 'target_type'])


def downgrade() -> None:
    op.drop_table('cross_references')
    op.drop_table('patent_publication_citations')
    op.drop_table('publication_sequence_links')
    op.drop_table('publication_chemical_links')
    op.drop_table('patent_sequence_links')
    op.drop_table('patent_chemical_links')
    op.drop_table('sequences')
    op.drop_table('chemicals')
    op.drop_table('publications')
    op.drop_table('patents')
