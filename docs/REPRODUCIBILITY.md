# Reproducibility Guide

Follow these steps to reproduce the BioPAT benchmark from scratch.

## Prerequisites

- **Python 3.11+**
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
source venv/bin/activate
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

## Determinism

BioPAT uses a fixed random seed (`42`) for all sampling and splitting. If you use the same data sources and configuration, the output `qrels` and `corpus` will be bit-identical.
