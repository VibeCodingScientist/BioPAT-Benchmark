# BioPAT: Biomedical Patent-Article Trimodal Benchmark

<p align="center">
  <strong>Multi-Modal Retrieval Benchmark for Biomedical Novelty Assessment</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#pipelines">Pipelines</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#benchmarks">Benchmarks</a>
</p>

---

BioPAT constructs a reproducible benchmark for evaluating prior art retrieval in biomedical patents. It connects patent claims to scientific literature using citation evidence from the Reliance on Science (RoS) dataset, then provides a novelty assessment pipeline combining text, chemical, and sequence retrieval.

## Overview

BioPAT has **two pipelines**:

| Pipeline | Purpose | Entry Point |
|----------|---------|-------------|
| **Phase 1** | Benchmark construction from real patent-paper citation data | `biopat.pipeline.Phase1Pipeline` |
| **Novelty Assessment** | End-to-end prior art search + novelty reasoning for a patent | `biopat.pipeline_novelty.NoveltyAssessmentPipeline` |

### Phase 1: Benchmark Construction

Builds a BEIR-compatible benchmark dataset:
1. Downloads patent-paper citation links from Reliance on Science (Zenodo)
2. Fetches patent metadata from USPTO (PatentsView API)
3. Extracts biomedical patent claims (IPC: A61, C07, C12)
4. Fetches cited paper metadata from OpenAlex
5. Creates ground truth relevance judgments with temporal validation
6. Produces train/dev/test splits in BEIR format
7. Runs BM25 baseline evaluation

### Novelty Assessment

Assesses whether a patent claim is novel by:
1. Parsing claims into structured elements (via LLM)
2. Trimodal retrieval: text (SPLADE/ColBERT/dense/hybrid) + chemical fingerprints + sequence alignment
3. Cross-encoder reranking of top candidates
4. LLM-based novelty reasoning (anticipation vs. obviousness)
5. Generating a structured novelty report

## Pipelines

### Retrieval Methods

| Method | Type | Implementation |
|--------|------|----------------|
| **BM25** | Sparse | Custom with k1/b tuning |
| **Dense** | Neural | SciBERT, PubMedBERT, BioBERT, BGE, E5, Contriever |
| **SPLADE** | Learned Sparse | MLM-based term expansion |
| **ColBERT** | Late Interaction | Token-level MaxSim |
| **HyDE** | Query Expansion | LLM-generated hypothetical documents |
| **Hybrid** | Fusion | RRF, linear, convex combination |

### Multi-Modal Search

| Modality | Method | Details |
|----------|--------|---------|
| **Text** | Dense + Sparse + Hybrid | 11 embedding models, BM25, SPLADE |
| **Chemical** | Fingerprints + Neural | Morgan/ECFP4, ChemBERTa, MoLFormer |
| **Sequence** | Alignment + Neural | BLAST+, ESM-2, ProtBERT, ProtT5 |

### Reranking

| Component | Description |
|-----------|-------------|
| **Cross-Encoder** | MS-MARCO, BGE, Jina, LLM-based reranking |
| **Score Fusion** | Weighted trimodal combination with multimodal boost |

## Quick Start

### Installation

```bash
git clone https://github.com/VibeCodingScientist/BioPAT-Benchmark.git
cd BioPAT-Benchmark

python -m venv venv
source venv/bin/activate

# Core install
pip install -e .

# With evaluation models (sentence-transformers, FAISS, torch)
pip install -e ".[evaluation]"

# With chemical structure support (RDKit)
pip install -e ".[advanced]"
```

### Environment Variables

```bash
# API keys (optional, increases rate limits)
export PATENTSVIEW_API_KEYS=your_key        # USPTO patents
export OPENALEX_MAILTO=your_email           # OpenAlex polite pool
export OPENAI_API_KEY=your_key              # HyDE query expansion (optional)
export ANTHROPIC_API_KEY=your_key           # LLM novelty reasoning (optional)
```

### Run the Benchmark Pipeline

```bash
# Via CLI entry point
biopat --config configs/my_config.yaml

# Or from Python
python -c "
from biopat.pipeline import run_phase1
results = run_phase1(config_path='configs/my_config.yaml')
"
```

### Run Tests

```bash
# Core pipeline test with mock data
python scripts/test_with_mock_data.py

# Retrieval pipeline test
python scripts/test_retrieval_pipeline.py

# Unit tests
python -m pytest tests/ -v
```

### Basic Usage — Novelty Assessment

```python
from biopat.pipeline_novelty import NoveltyAssessmentPipeline, PipelineConfig, PatentInput

# Configure
config = PipelineConfig(
    retrieval_method="hybrid",
    text_model="allenai/scibert_scivocab_uncased",
    use_reranker=True,
)

# Create pipeline
pipeline = NoveltyAssessmentPipeline(config)

# Index your corpus
pipeline.index_corpus(corpus_docs, chemicals=chem_data, sequences=seq_data)

# Assess a patent
patent = PatentInput(
    patent_id="US2024123456",
    title="Novel PD-1 antibody composition",
    abstract="The present invention relates to...",
    claims=["A method of treating cancer comprising administering..."],
    smiles="CC(=O)OC1=CC=CC=C1C(=O)O",  # optional
)
report = pipeline.assess(patent)
```

## Architecture

```
BioPAT-Benchmark/
├── src/biopat/
│   ├── pipeline.py              # Phase 1: benchmark construction (8 steps)
│   ├── pipeline_novelty.py      # Novelty assessment pipeline
│   ├── config.py                # Configuration management (YAML + env vars)
│   ├── reproducibility.py       # Checksums, audit logging, seeds
│   │
│   ├── ingestion/               # Data acquisition (3 sources)
│   │   ├── ros.py              # Reliance on Science (Zenodo)
│   │   ├── patentsview.py      # USPTO PatentsView API
│   │   └── openalex.py         # OpenAlex scientific papers
│   │
│   ├── processing/              # Data processing
│   │   ├── patents.py          # Patent filtering, claim extraction
│   │   ├── papers.py           # Paper validation, metadata
│   │   ├── linking.py          # Citation linking
│   │   ├── chemical_index.py   # Morgan fingerprints, FAISS index
│   │   ├── sequence_index.py   # BLAST database, sequence search
│   │   └── ...                 # Patent IDs, NPL parsing, claim mapping
│   │
│   ├── retrieval/               # SOTA retrieval (8 methods)
│   │   ├── dense.py            # Dense embedding retrieval
│   │   ├── hybrid.py           # BM25 + dense fusion (RRF)
│   │   ├── splade.py           # Learned sparse retrieval
│   │   ├── colbert.py          # Late interaction
│   │   ├── hyde.py             # HyDE query expansion
│   │   ├── molecular.py        # Chemical fingerprint search
│   │   ├── sequence.py         # BLAST + PLM sequence search
│   │   └── reranker.py         # Cross-encoder reranking
│   │
│   ├── groundtruth/             # Relevance judgment construction
│   │   ├── relevance.py        # Graded relevance assignment
│   │   ├── temporal.py         # Prior art date validation
│   │   └── stratification.py   # Domain-stratified evaluation
│   │
│   ├── benchmark/               # Output formatting
│   │   ├── sampling.py         # Stratified query sampling
│   │   ├── splits.py           # Train/dev/test splits
│   │   └── beir_format.py      # BEIR JSON Lines output
│   │
│   ├── evaluation/              # Metrics and baselines
│   │   ├── metrics.py          # nDCG, MAP, MRR, P@k, R@k
│   │   ├── bm25.py             # BM25 baseline evaluator
│   │   ├── dense.py            # Dense retrieval evaluator
│   │   ├── hybrid.py           # Hybrid fusion evaluator
│   │   ├── reranker.py         # Reranking evaluator
│   │   ├── ablation.py         # Ablation studies
│   │   ├── error_analysis.py   # Failure analysis
│   │   └── trimodal_retrieval.py # Trimodal fusion evaluator
│   │
│   └── reasoning/               # LLM-powered analysis
│       ├── claim_parser.py     # Patent claim parsing
│       ├── novelty_reasoner.py # Anticipation/obviousness reasoning
│       └── explanation_generator.py # Report generation
│
├── scripts/
│   ├── test_with_mock_data.py       # End-to-end mock test
│   └── test_retrieval_pipeline.py   # Retrieval method tests
│
└── tests/                       # pytest test suite
```

## Benchmarks

### Relevance Grades

| Grade | Legal Standard | Description |
|-------|---------------|-------------|
| **3** | 35 USC §102 / EPO Cat. X | Novelty-destroying anticipation |
| **2** | 35 USC §103 / EPO Cat. Y | Obviousness (with combination) |
| **1** | Background | Relevant context, not prior art |
| **0** | Irrelevant | No relevance to claim |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **nDCG@k** | Normalized Discounted Cumulative Gain (k=10, 100) |
| **MAP** | Mean Average Precision |
| **MRR** | Mean Reciprocal Rank |
| **P@k** | Precision at k |
| **R@k** | Recall at k |

### Data Sources

| Source | API | Content |
|--------|-----|---------|
| **Reliance on Science** | Zenodo | Patent-paper citation links with confidence scores |
| **USPTO** | PatentsView | US biomedical patents with claims |
| **OpenAlex** | REST API | Scientific paper metadata and abstracts |

## Dependencies

**Core** (always required):
`httpx`, `polars`, `pyyaml`, `pydantic`, `diskcache`, `tqdm`, `python-dotenv`, `rank-bm25`

**Evaluation** (optional — for neural retrieval):
`sentence-transformers`, `faiss-cpu`, `torch`

**Advanced** (optional — for chemical structure search):
`rdkit`

## Citation

```bibtex
@software{biopat2026,
  author    = {BioPAT Contributors},
  title     = {{BioPAT}: Biomedical Patent-Article Trimodal Benchmark},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/VibeCodingScientist/BioPAT-Benchmark},
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built for the era of Agentic Science</sub>
</p>
