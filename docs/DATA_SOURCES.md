# BioPAT Data Sources

This document provides detailed information about data sources used in BioPAT.

## Scientific Literature

### PubMed

| Attribute | Value |
|-----------|-------|
| **Provider** | NCBI (National Center for Biotechnology Information) |
| **API** | E-utilities (esearch, efetch) |
| **Coverage** | 35M+ biomedical abstracts |
| **Update Frequency** | Daily |
| **Access** | Free (API key recommended) |
| **Rate Limit** | 3 req/sec (10 with API key) |

**Fields Retrieved**:
- PMID, Title, Abstract
- Authors, Journal, Publication Date
- MeSH Terms, Keywords
- DOI (when available)

### bioRxiv / medRxiv

| Attribute | Value |
|-----------|-------|
| **Provider** | Cold Spring Harbor Laboratory |
| **API** | Content API |
| **Coverage** | 200K+ preprints |
| **Update Frequency** | Daily |
| **Access** | Free |
| **Rate Limit** | Reasonable use |

**Fields Retrieved**:
- DOI, Title, Abstract
- Authors, Posted Date
- Category, Server (biorxiv/medrxiv)

## Patent Databases

### USPTO (PatentsView)

| Attribute | Value |
|-----------|-------|
| **Provider** | United States Patent and Trademark Office |
| **API** | PatentsView API |
| **Coverage** | 8M+ US patents (1976-present) |
| **Update Frequency** | Weekly |
| **Access** | Free (API key optional) |
| **Rate Limit** | 45 req/min |

**Fields Retrieved**:
- Patent Number, Title, Abstract
- Claims (full text)
- Application Date, Grant Date
- IPC/CPC Codes
- Assignee, Inventors

### EPO (Open Patent Services)

| Attribute | Value |
|-----------|-------|
| **Provider** | European Patent Office |
| **API** | Open Patent Services (OPS) |
| **Coverage** | 130M+ patent documents worldwide |
| **Update Frequency** | Weekly |
| **Access** | Free (OAuth required) |
| **Rate Limit** | Based on subscription |

**Fields Retrieved**:
- Publication Number, Title, Abstract
- Claims (when available)
- Application/Publication Dates
- IPC Codes
- Applicants, Inventors

## Chemical Databases

### PubChem

| Attribute | Value |
|-----------|-------|
| **Provider** | NCBI |
| **API** | PUG REST |
| **Coverage** | 100M+ compounds |
| **Update Frequency** | Continuous |
| **Access** | Free |
| **Rate Limit** | 5 req/sec |

**Fields Retrieved**:
- CID (Compound ID)
- Canonical SMILES, InChI, InChIKey
- Molecular Formula, Molecular Weight
- IUPAC Name, Synonyms

### SureChEMBL

| Attribute | Value |
|-----------|-------|
| **Provider** | EMBL-EBI |
| **API** | REST API |
| **Coverage** | 20M+ compounds from patents |
| **Update Frequency** | Monthly |
| **Access** | Free (API key optional) |
| **Rate Limit** | 2 req/sec |

**Fields Retrieved**:
- SureChEMBL ID
- SMILES, InChI, InChIKey
- Patent IDs containing compound
- Extraction method (CHEMICAL_NAME, SMILES, etc.)

## Sequence Databases

### UniProt

| Attribute | Value |
|-----------|-------|
| **Provider** | UniProt Consortium |
| **API** | REST API |
| **Coverage** | 250M+ protein sequences |
| **Update Frequency** | 8 weeks |
| **Access** | Free |
| **Rate Limit** | 100 req/min |

**Fields Retrieved**:
- Accession, Entry Name
- Protein Name, Gene Name
- Organism, Sequence
- Function, Keywords

### NCBI Sequences (GenBank, Protein, Nucleotide)

| Attribute | Value |
|-----------|-------|
| **Provider** | NCBI |
| **API** | E-utilities |
| **Coverage** | 500M+ sequences |
| **Update Frequency** | Daily |
| **Access** | Free (API key recommended) |
| **Rate Limit** | 3 req/sec (10 with API key) |

**Fields Retrieved**:
- Accession, Definition
- Organism, Sequence
- Patent references (for patent sequences)
- Keywords, Features

## Data Quality

### Deduplication

BioPAT deduplicates documents using:
1. **DOI matching** for publications
2. **Patent family** deduplication for patents
3. **InChIKey** for chemicals
4. **Sequence hash** for sequences

### Temporal Filtering

All prior art queries enforce strict temporal constraints:
- Only documents published before the patent's **priority date** are considered
- Publication dates are normalized to ISO 8601 format
- Uncertain dates are handled conservatively

### Data Validation

| Check | Description |
|-------|-------------|
| Schema validation | All records conform to expected schema |
| Required fields | Title, text, and date are mandatory |
| Date parsing | All dates successfully parsed |
| SMILES validation | Valid SMILES syntax (when RDKit available) |
| Sequence validation | Valid amino acid/nucleotide alphabet |

## Data Versioning

Each corpus build includes:
- **Build timestamp**: ISO 8601 datetime
- **Source versions**: API versions and download dates
- **SHA256 checksums**: For all data files
- **Record counts**: By source and type

Example manifest:
```json
{
  "build_date": "2026-01-30T08:00:00Z",
  "sources": {
    "pubmed": {"version": "2026-01", "count": 1000},
    "patents": {"version": "2026-01-28", "count": 500}
  },
  "checksums": {
    "corpus.jsonl": "abc123...",
    "chemicals.jsonl": "def456..."
  }
}
```
