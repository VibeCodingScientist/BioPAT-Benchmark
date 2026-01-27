# BioPAT Reproducibility Guide (v4.0)

Follow these steps to reproduce the BioPAT benchmark from scratch, including global patent coverage and multi-modal retrieval pipelines.

## 1. Prerequisites

- **Python**: 3.11 or 3.12 (High performance Polars support required).
- **Relational DB**: SQLite (Local) or PostgreSQL (Production).
- **Core Intelligence**:
    - `rdkit`: For chemical structure handling and fingerprinting.
    - `biopython`: For biological sequence extraction.
    - `ncbi-blast+`: Local installation of BLAST required for sequence similarity.
- **API Access**:
    - **PatentsView**: `X-Api-Key` required.
    - **EPO OPS**: Consumer Key and Secret (OAuth 2.0).
    - **WIPO PATENTSCOPE**: API Token.
    - **NCBI**: `NCBI_API_KEY` for high-throughput sequence retrieval.
- **Storage**: At least 300GB of free space (SureChEMBL and BLAST indices).

## 2. Environment Configuration

Create a `.env` file in the root directory:

```bash
# USPTO / PatentsView
export PATENTSVIEW_API_KEY="your_key"

# European Patent Office (EPO)
export EPO_CONSUMER_KEY="your_ops_key"
export EPO_CONSUMER_SECRET="your_ops_secret"

# WIPO / PATENTSCOPE
export WIPO_API_TOKEN="your_token"

# NCBI / PubMed
export NCBI_API_KEY="your_key"

# OpenAlex
export OPENALEX_MAILTO="your_email@example.org"
```

## 3. Installation

```bash
# Clone and install with all multi-modal dependencies
git clone https://github.com/VibeCodingScientist/BioPAT-Benchmark.git
cd BioPAT-Benchmark
python -m venv venv
source venv/activate
pip install -e ".[all]"

# Install local BLAST+ (Platform specific)
brew install blast  # macOS
sudo apt-get install ncbi-blast+  # Linux
```

## 4. Execution Pipeline

### 4.1 Initialize Harmonization Layer
BioPAT v4.0 requires a unified entity database.

```bash
# Initialize SQLite database
biopat db init --backend sqlite --path data/biopat.db
```

### 4.2 Build Unified Corpus (v3.0)
Downloads patents from all three jurisdictions and applies family deduplication.

```bash
biopat build --jurisdiction all --target-size 500000
```

### 4.3 Multi-Modal Enrichment (v4.0)
Extracts molecules and sequences, and builds similarity indices.

```bash
# Extract and index Chemical Structures
biopat hydrate chemical --use-rdkit

# Extract and index Biological Sequences
biopat hydrate sequence --use-blast
```

### 4.4 Run Trimodal Evaluation
```bash
biopat evaluate --trimodal --alpha 0.4 --beta 0.3 --gamma 0.3
```

## 5. Verification

1.  **Checksums**: Compare `manifest.json` against `docs/DATA_SOURCES.md`.
2.  **Audit Logs**: Check `logs/audit.json` for all authenticated API traffic.
3.  **Unit Tests**: All core logic must pass: `pytest tests/`.
