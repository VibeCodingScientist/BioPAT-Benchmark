# BioPAT v4.0: Multi-Modal Retrieval Guide

This guide covers the advanced features introduced in BioPAT v4.0, including trimodal retrieval (text + chemical + sequence), entity harmonization, and database configuration.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Chemical Structure Search](#chemical-structure-search)
4. [Biological Sequence Search](#biological-sequence-search)
5. [Trimodal Retrieval](#trimodal-retrieval)
6. [Entity Harmonization](#entity-harmonization)
7. [Database Configuration](#database-configuration)
8. [Configuration Reference](#configuration-reference)
9. [Troubleshooting](#troubleshooting)

---

## Overview

BioPAT v4.0 extends beyond text-only patent retrieval to support **trimodal search**:

| Modality | What it Searches | Use Case |
|----------|-----------------|----------|
| **Text** | Titles, abstracts, claims | General novelty search |
| **Chemical** | Molecular structures (SMILES/InChIKey) | Small molecule patents |
| **Sequence** | Protein/DNA sequences | Biologics, gene therapy patents |

### Architecture

```
Query (text + SMILES + sequence)
           │
           ├─── Text Index (BM25/Dense) ───┐
           │                               │
           ├─── Chemical Index (FAISS) ────┼──→ Score Fusion ──→ Ranked Results
           │                               │
           └─── Sequence Index (BLAST+) ───┘
```

---

## Installation

### Core Installation

```bash
pip install -e .
```

### Advanced Dependencies (for v4.0 features)

```bash
# Chemical indexing (RDKit + FAISS)
pip install -e ".[advanced]"

# Or install individually:
pip install rdkit faiss-cpu

# For GPU-accelerated chemical search:
pip install faiss-gpu
```

### BLAST+ Installation (for sequence search)

**macOS:**
```bash
brew install blast
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ncbi-blast+
```

**Verify installation:**
```bash
blastp -version
# Output: blastp: 2.x.x+
```

---

## Chemical Structure Search

### Overview

Chemical structure search uses Morgan fingerprints (circular fingerprints) and Tanimoto similarity to find structurally similar compounds in the patent corpus.

### Quick Start

```python
from biopat.processing.chemical_index import (
    ChemicalIndex,
    MorganFingerprintCalculator,
)

# Initialize calculator
calculator = MorganFingerprintCalculator(
    radius=2,       # Morgan radius (2 = ECFP4 equivalent)
    n_bits=2048,    # Fingerprint size
)

# Create index
index = ChemicalIndex(fingerprint_calculator=calculator)

# Add chemicals from corpus
chemicals = [
    {"doc_id": "US10123456", "smiles": "CC1=NC=C(C=C1)C2=CN=C(N2)NC3=CC=CC=C3"},
    {"doc_id": "US10234567", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"},
]

for chem in chemicals:
    index.add_chemical(chem["doc_id"], chem["smiles"])

# Build FAISS index
index.build_index()

# Search
query_smiles = "CC1=NC=C(C=C1)C2=CN=C(N2)NC3=CC=CC(=C3)CF"
results = index.search(query_smiles, top_k=10)

for doc_id, tanimoto_score in results:
    print(f"{doc_id}: {tanimoto_score:.3f}")
```

### Configuration

```yaml
# configs/default.yaml
advanced:
  chemical:
    enabled: true
    fingerprint_type: "morgan"  # morgan or rdkit
    fingerprint_bits: 2048
    morgan_radius: 2
    use_gpu: false
    faiss_index_type: "flat"  # flat, ivf, hnsw

    # Similarity thresholds for relevance tiers
    tanimoto_high: 0.85      # Tier 3: Highly similar
    tanimoto_medium: 0.7     # Tier 2: Moderately similar
    tanimoto_low: 0.5        # Tier 1: Weakly similar
```

### Relevance Mapping

| Tanimoto Score | Relevance Tier | Interpretation |
|----------------|----------------|----------------|
| ≥ 0.85 | 3 (Novelty-destroying) | Near-identical structure |
| 0.70 - 0.85 | 2 (Highly relevant) | Same scaffold, different substituents |
| 0.50 - 0.70 | 1 (Relevant) | Related chemotype |
| < 0.50 | 0 (Not relevant) | Structurally dissimilar |

---

## Biological Sequence Search

### Overview

Sequence search uses BLAST+ to find similar protein or nucleotide sequences in the patent corpus.

### Quick Start

```python
from biopat.processing.sequence_index import (
    BlastDatabaseManager,
    BlastSearcher,
    SequenceIndex,
)

# Create BLAST database from patent sequences
db_manager = BlastDatabaseManager(db_path="data/blast_db")

sequences = [
    {"doc_id": "US10123456", "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK", "type": "AA"},
    {"doc_id": "US10234567", "sequence": "MWLLLLLLLLLPGSSGAAQVQLVQSGAEVKKPGAS", "type": "AA"},
]

db_manager.create_database(sequences, db_type="blastp")

# Search
searcher = BlastSearcher(db_path="data/blast_db", blast_type="blastp")
query_sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"

results = searcher.search(
    query_sequence,
    evalue_threshold=1e-5,
    max_hits=10,
)

for hit in results.hits:
    print(f"{hit.subject_id}: {hit.identity:.1f}% identity, E={hit.evalue:.2e}")
```

### Configuration

```yaml
# configs/default.yaml
advanced:
  sequence:
    enabled: true
    blast_db_path: "data/blast_db"
    blast_type: "blastp"  # blastp (protein) or blastn (nucleotide)
    evalue_threshold: 1e-5

    # Identity thresholds for relevance tiers
    identity_high: 0.95    # Tier 3: Nearly identical
    identity_medium: 0.7   # Tier 2: Homologous
    identity_low: 0.5      # Tier 1: Distantly related
```

### Relevance Mapping

| Sequence Identity | Relevance Tier | Interpretation |
|-------------------|----------------|----------------|
| ≥ 95% | 3 (Novelty-destroying) | Same protein/gene |
| 70% - 95% | 2 (Highly relevant) | Ortholog or close variant |
| 50% - 70% | 1 (Relevant) | Same protein family |
| < 50% | 0 (Not relevant) | No significant homology |

---

## Trimodal Retrieval

### Overview

Trimodal retrieval combines text, chemical, and sequence search with configurable weighting.

### Quick Start

```python
from biopat.evaluation.trimodal_retrieval import (
    TrimodalConfig,
    TrimodalRetriever,
)

# Configure weights
config = TrimodalConfig(
    text_weight=0.5,      # 50% text
    chemical_weight=0.3,  # 30% chemical
    sequence_weight=0.2,  # 20% sequence

    # Significance thresholds
    text_threshold=0.1,
    chemical_threshold=0.5,
    sequence_threshold=0.3,
)

# Initialize retriever with indices
retriever = TrimodalRetriever(
    config=config,
    text_index=bm25_index,       # Your text index
    chemical_index=faiss_index,   # Your FAISS index
    sequence_index=blast_db,      # Your BLAST database
)

# Search
results = retriever.search(
    query_text="kinase inhibitor cancer EGFR",
    query_smiles="CC1=NC=C(C=C1)C2=CN=C(N2)NC3=CC=CC=C3",
    query_sequence=None,  # No sequence query
    top_k=100,
)

for hit in results:
    print(f"{hit.doc_id}: {hit.combined_score:.3f} ({hit.match_type})")
```

### Score Fusion Methods

**Weighted Combination (default):**
```
combined_score = w_text * text_score + w_chem * chem_score + w_seq * seq_score
```

**Reciprocal Rank Fusion (RRF):**
```python
config = TrimodalConfig(
    fusion_method="rrf",
    rrf_k=60,  # RRF smoothing parameter
)
```

### Match Type Analysis

The retriever tracks which modalities contributed to each result:

```python
# Analyze result distribution
match_type_counts = {}
for hit in results:
    mt = hit.match_type
    match_type_counts[mt] = match_type_counts.get(mt, 0) + 1

print(match_type_counts)
# Output: {TEXT_CHEMICAL: 45, TEXT: 30, CHEMICAL: 15, SEQUENCE: 10}
```

---

## Entity Harmonization

### Overview

The entity harmonization layer provides unified identifiers (BioPAT IDs) across heterogeneous data sources.

### BioPAT ID Format

```
BP:{TYPE}:{SUBTYPE}:{ID}

Examples:
- BP:PAT:US:10123456    (US Patent)
- BP:PAT:EP:3000000     (EP Patent)
- BP:PUB:PMID:12345678  (PubMed article)
- BP:PUB:DOI:10.1038/nature12373  (DOI)
- BP:CHM:BSYNRYMUTXBXSQ-UHFFFAOYSA-N  (InChIKey)
- BP:SEQ:AA:a1b2c3d4e5f6  (Protein sequence hash)
```

### Using the Entity Resolver

```python
from biopat.harmonization.entity_resolver import EntityResolver

resolver = EntityResolver()

# Resolve patent IDs
patent_entity = resolver.resolve_patent("US10123456B2")
print(patent_entity.biopat_id)  # BP:PAT:US:10123456

# Resolve publications
pub_entity = resolver.resolve_publication("10.1038/nature12373")
print(pub_entity.biopat_id)  # BP:PUB:DOI:10.1038/nature12373

# Resolve chemicals
chem_entity = resolver.resolve_chemical("CC(=O)OC1=CC=CC=C1C(=O)O")
print(chem_entity.biopat_id)  # BP:CHM:BSYNRYMUTXBXSQ-UHFFFAOYSA-N

# Resolve sequences
seq_entity = resolver.resolve_sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK")
print(seq_entity.biopat_id)  # BP:SEQ:AA:a1b2c3d4...
```

---

## Database Configuration

### SQLite (Default)

```yaml
advanced:
  harmonization:
    enabled: true
    database:
      backend: "sqlite"
      path: "data/biopat.db"
```

### PostgreSQL (Production)

```yaml
advanced:
  harmonization:
    enabled: true
    database:
      backend: "postgresql"
      host: "localhost"
      port: 5432
      user: "biopat"
      password: "${BIOPAT_DB_PASSWORD}"  # Use env var
      database: "biopat"
```

### Running Migrations

```bash
# Initialize database
cd BioPAT-Benchmark
alembic upgrade head

# Check current version
alembic current

# Create new migration (after schema changes)
alembic revision --autogenerate -m "Add new column"
```

### Schema Overview

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│   patents   │────▶│ patent_chemical │◀────│  chemicals  │
│             │     │     _links      │     │             │
└─────────────┘     └─────────────────┘     └─────────────┘
       │                                           │
       │            ┌─────────────────┐            │
       │            │ patent_sequence │            │
       └───────────▶│     _links      │◀───────────┘
                    └─────────────────┘
                            ▲
                            │
                    ┌───────┴───────┐
                    │   sequences   │
                    └───────────────┘

┌─────────────┐     ┌─────────────────────┐     ┌──────────────┐
│ publications│────▶│patent_publication   │◀────│   patents    │
│             │     │    _citations       │     │              │
└─────────────┘     └─────────────────────┘     └──────────────┘
```

---

## Configuration Reference

### Full v4.0 Configuration

```yaml
# configs/v4_production.yaml

phase: "phase1"

paths:
  data_dir: "data"
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  cache_dir: "data/cache"
  benchmark_dir: "data/benchmark"

api:
  patentsview_api_keys:
    - "key1"
    - "key2"
  openalex_email: "your@email.com"
  epo_consumer_key: "${EPO_CONSUMER_KEY}"
  epo_consumer_secret: "${EPO_CONSUMER_SECRET}"

advanced:
  harmonization:
    enabled: true
    database:
      backend: "sqlite"
      path: "data/biopat.db"
    use_rdkit: true
    cache_size: 10000
    extract_doi_from_text: true
    extract_pmid_from_text: true
    extract_sequences_from_claims: true

  chemical:
    enabled: true
    fingerprint_type: "morgan"
    fingerprint_bits: 2048
    morgan_radius: 2
    use_gpu: false
    faiss_index_type: "flat"
    tanimoto_high: 0.85
    tanimoto_medium: 0.7
    tanimoto_low: 0.5

  sequence:
    enabled: true
    blast_db_path: "data/blast_db"
    blast_type: "blastp"
    evalue_threshold: 1e-5
    identity_high: 0.95
    identity_medium: 0.7
    identity_low: 0.5

  # Trimodal retrieval weights
  text_weight: 0.5
  chemical_weight: 0.3
  sequence_weight: 0.2
```

---

## Troubleshooting

### RDKit Not Found

```
Error: RDKit not available
```

**Solution:**
```bash
pip install rdkit
# or
conda install -c conda-forge rdkit
```

### FAISS Not Found

```
Error: FAISS not available
```

**Solution:**
```bash
pip install faiss-cpu
# or for GPU:
pip install faiss-gpu
```

### BLAST+ Not Found

```
Error: blastp command not found
```

**Solution:**
```bash
# macOS
brew install blast

# Ubuntu
sudo apt-get install ncbi-blast+

# Verify
which blastp
```

### Database Migration Errors

```
Error: Can't locate revision
```

**Solution:**
```bash
# Reset migrations
alembic downgrade base
alembic upgrade head
```

### Memory Issues with Large Chemical Index

**Solution:** Use IVF index instead of flat:

```yaml
advanced:
  chemical:
    faiss_index_type: "ivf"  # Approximate but memory-efficient
```

### Slow Sequence Search

**Solution:** Pre-filter by sequence length or use smaller E-value:

```yaml
advanced:
  sequence:
    evalue_threshold: 1e-10  # More stringent
```

---

## Next Steps

1. **Run the benchmark pipeline:**
   ```bash
   python scripts/run_benchmark.py --phase phase1
   ```

2. **Evaluate with trimodal retrieval:**
   ```bash
   python -m biopat.evaluation.trimodal_evaluation --config configs/v4_production.yaml
   ```

3. **Analyze results:**
   ```bash
   python -m biopat.evaluation.error_analysis --output reports/
   ```

For questions or issues, visit: https://github.com/VibeCodingScientist/BioPAT-Benchmark/issues
