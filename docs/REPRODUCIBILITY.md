# Reproducibility Guide

Follow these steps to reproduce the BioPAT benchmark from scratch.

## Prerequisites

- **Python 3.11+** (tested with 3.11 and 3.12)
- **Git**
- **API Keys**:
    - PatentsView API Key (X-Api-Key)
    - NCBI API Key (for PubMed enrichment)
- **Disk Space**: At least 150GB of free space.

## Step 1: Clone and Install

```bash
git clone https://github.com/[org]/biopat.git
cd biopat
python -m venv venv
source venv/activate
pip install -e ".[all]"
```

## Step 2: Configure Environment

Create a `.env` file or export variables:
```bash
export PATENTSVIEW_API_KEY="your_key"
export NCBI_API_KEY="your_key"
export OPENALEX_MAILTO="your_email@example.org"
```

## Step 3: Execute Pipeline

Run the unified build command. This will download the raw data, process claims, link citations, and generate the BEIR-formatted output.

```bash
biopat build --target-patents 2000
```

## Step 4: Run Baselines

Evaluate the generated benchmark against lexical and dense baselines:

```bash
biopat evaluate --all
```

## Reproducibility Infrastructure

BioPAT implements multiple layers of reproducibility guarantees:

### 1. Deterministic Randomness

All random operations use a fixed seed (`REPRODUCIBILITY_SEED = 42`):

```python
from biopat import REPRODUCIBILITY_SEED
from biopat.benchmark import BenchmarkSampler, DatasetSplitter

# Both classes use the fixed seed by default
sampler = BenchmarkSampler()  # seed=42
splitter = DatasetSplitter()   # seed=42
```

If you use the same data sources and configuration, the output `qrels` and `corpus` will be bit-identical.

### 2. Checksum Verification

Every downloaded file is hashed with SHA256 for verification:

```python
from biopat import ChecksumEngine, create_manifest

# Create manifest for tracking
manifest_path = create_manifest(output_dir)
checksum_engine = ChecksumEngine(manifest_path)

# Verify a downloaded file
checksum_engine.verify_checksum(file_path, expected_hash="abc123...")
```

The manifest records:
- Source URLs
- Download timestamps
- File sizes
- SHA256 checksums

### 3. Audit Logging

All API calls are logged for full provenance tracking:

```python
from biopat import AuditLogger

audit_logger = AuditLogger(manifest_path)

# View summary of API calls
print(audit_logger.get_summary())
# {
#   "total_api_calls": 1523,
#   "call_counts_by_endpoint": {...},
#   "services_used": ["patentsview", "openalex", "zenodo"],
#   ...
# }
```

### 4. Dependency Pinning

All dependencies are pinned to specific version ranges in `pyproject.toml`:
- Core dependencies: `polars>=1.0.0,<2.0`, `httpx>=0.27.0,<0.28`, etc.
- Evaluation dependencies: `sentence-transformers>=2.2.0,<4.0`, `faiss-cpu>=1.7.0,<2.0`

To lock exact versions for your environment:
```bash
pip freeze > requirements.lock.txt
```

### 5. Manifest File

After running the pipeline, a `manifest.json` file is generated containing:

```json
{
  "biopat_version": "0.1.0",
  "created_at": "2026-01-27T10:00:00Z",
  "reproducibility_seed": 42,
  "downloads": {
    "_pcs_oa.csv.gz": {
      "source_url": "https://zenodo.org/records/7996195/files/_pcs_oa.csv.gz",
      "sha256": "abc123...",
      "downloaded_at": "2026-01-27T10:05:00Z"
    }
  },
  "api_calls": [...],
  "api_call_counts": {
    "patentsview:patent/": 150,
    "openalex:works": 2000
  }
}
```

## Verifying Reproducibility

To verify your build matches the reference:

1. Compare `manifest.json` checksums against the reference checksums in `docs/DATA_SOURCES.md`
2. Run the test suite: `pytest tests/`
3. Compare output file hashes:
   ```bash
   sha256sum output/corpus.jsonl output/queries.jsonl
   ```
