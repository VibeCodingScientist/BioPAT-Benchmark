# BioPAT Reproducibility Guide

This guide provides step-by-step instructions for reproducing BioPAT experiments.

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9+ | 3.11+ |
| RAM | 8 GB | 32 GB |
| Storage | 10 GB | 100 GB |
| GPU | Optional | NVIDIA with 8GB+ VRAM |

### Installation

```bash
# Clone repository
git clone https://github.com/VibeCodingScientist/BioPAT-Benchmark.git
cd BioPAT-Benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install optional dependencies
pip install rdkit           # Chemical search
pip install faiss-cpu       # Fast similarity (or faiss-gpu)
pip install lightgbm        # LambdaMART
pip install xgboost         # XGBoost ranker
```

### API Keys

Set up API keys for higher rate limits:

```bash
# NCBI (PubMed, sequences)
export NCBI_API_KEY=your_key  # Get from: https://www.ncbi.nlm.nih.gov/account/

# USPTO PatentsView
export PATENTSVIEW_API_KEY=your_key  # Get from: https://patentsview.org/

# EPO (optional, for OAuth)
export EPO_CONSUMER_KEY=your_key
export EPO_CONSUMER_SECRET=your_secret

# LLM APIs (for HyDE/reranking)
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
```

## Building a Corpus

### Quick Start

```bash
# Build a small test corpus
python scripts/build_corpus.py \
    --topic "CAR-T cell therapy" \
    --pubmed-limit 100 \
    --patent-limit 50 \
    --output data/test_corpus
```

### Full Corpus Build

```bash
# Build production corpus
python scripts/build_corpus.py \
    --topic "checkpoint inhibitor immunotherapy" \
    --pubmed-limit 5000 \
    --biorxiv-limit 500 \
    --patent-limit 1000 \
    --protein-limit 200 \
    --patent-sequence-limit 100 \
    --date-range "2015-2024" \
    --output data/checkpoint_corpus
```

### Output Structure

```
data/checkpoint_corpus/
├── corpus.jsonl           # All documents
├── corpus_dict.json       # BioPAT format
├── chemicals.jsonl        # Extracted chemicals
├── sequences.jsonl        # Extracted sequences
└── .cache/                # API response cache
```

## Running Experiments

### Test Suite

```bash
# Verify installation
python scripts/test_with_mock_data.py

# Test SOTA components
python scripts/test_sota_components.py

# Test retrieval pipeline
python scripts/test_retrieval_pipeline.py
```

### Benchmark Evaluation

```bash
# Run full benchmark
python scripts/run_benchmark.py \
    --corpus data/corpus.jsonl \
    --queries data/queries.jsonl \
    --methods bm25,dense,hybrid,splade \
    --output results/

# Evaluate specific method
python scripts/run_benchmark.py \
    --corpus data/corpus.jsonl \
    --queries data/queries.jsonl \
    --methods hybrid \
    --dense-model allenai/scibert_scivocab_uncased \
    --fusion rrf \
    --output results/scibert_rrf/
```

### Method-Specific Configurations

#### BM25 Baseline

```bash
python scripts/run_benchmark.py \
    --methods bm25 \
    --bm25-k1 1.5 \
    --bm25-b 0.75
```

#### Dense Retrieval

```bash
python scripts/run_benchmark.py \
    --methods dense \
    --dense-model BAAI/bge-base-en-v1.5 \
    --batch-size 32
```

#### Hybrid (RRF)

```bash
python scripts/run_benchmark.py \
    --methods hybrid \
    --fusion rrf \
    --rrf-k 60
```

#### SPLADE

```bash
python scripts/run_benchmark.py \
    --methods splade \
    --splade-model naver/splade-cocondenser-ensembledistil
```

#### With Reranking

```bash
python scripts/run_benchmark.py \
    --methods hybrid \
    --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --rerank-top-k 100
```

## Reproducing Results

### Expected Metrics

| Method | nDCG@10 | MAP | MRR |
|--------|---------|-----|-----|
| BM25 | 0.45 | 0.38 | 0.52 |
| SciBERT | 0.52 | 0.44 | 0.58 |
| Hybrid (RRF) | 0.58 | 0.49 | 0.64 |
| + Reranking | 0.63 | 0.54 | 0.70 |

*Note: Actual metrics depend on corpus and query set.*

### Verification Checklist

- [ ] All tests pass (`test_with_mock_data.py`)
- [ ] SOTA components verified (`test_sota_components.py`)
- [ ] Retrieval pipeline works (`test_retrieval_pipeline.py`)
- [ ] Corpus builds successfully
- [ ] Metrics within expected range

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ImportError: httpx` | `pip install httpx` |
| `ImportError: rdkit` | `pip install rdkit` |
| `ImportError: faiss` | `pip install faiss-cpu` |
| Rate limit errors | Add API keys to environment |
| Out of memory | Reduce batch size, use CPU |
| CUDA errors | Use `--device cpu` flag |

### Debug Mode

```bash
# Enable verbose logging
python scripts/run_benchmark.py --verbose

# Run with debug output
BIOPAT_DEBUG=1 python scripts/test_with_mock_data.py
```

## Citation

If you use BioPAT in your research, please cite:

```bibtex
@software{biopat2026,
  author    = {BioPAT Contributors},
  title     = {{BioPAT}: Biomedical Patent-Article Trimodal Benchmark},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/VibeCodingScientist/BioPAT-Benchmark}
}
```
