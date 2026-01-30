# BioPAT: Biomedical Patent-Article Trimodal Benchmark

<p align="center">
  <strong>State-of-the-Art Multi-Modal Retrieval for Biomedical Novelty Assessment</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#documentation">Documentation</a>
</p>

---

BioPAT is a comprehensive benchmark and retrieval framework for evaluating **Agentic Scientists** and **Novelty Agents** in biomedical patent prior art search. It bridges the gap between patent claims, scientific literature, chemical structures, and biological sequences.

## Why BioPAT?

In the era of **Agentic Science**, AI systems generate hypotheses at unprecedented scale. The critical challenge is **Novelty Determination**—distinguishing truly novel discoveries from existing prior art across heterogeneous, multi-modal databases.

| Challenge | BioPAT Solution |
|-----------|----------------|
| **Vocabulary Gap** | Thesaurus expansion (MeSH, ChEBI, DrugBank) + learned sparse retrieval (SPLADE) |
| **Semantic Gap** | Dense retrieval with domain-specific models (SciBERT, PubMedBERT, BioBERT) |
| **Modality Gap** | Trimodal fusion (text + chemical fingerprints + sequence alignment) |
| **Scale** | Efficient indexing (FAISS, inverted indices) with 10+ retrieval methods |
| **Ranking** | Learning-to-Rank (LambdaMART) combining 17+ signals |

## Features

### Retrieval Methods (SOTA Coverage)

| Method | Type | Model/Algorithm |
|--------|------|-----------------|
| **BM25** | Sparse | Custom implementation with k1/b tuning |
| **Dense** | Neural | E5, BGE, SciBERT, PubMedBERT, BioBERT, Contriever |
| **SPLADE** | Learned Sparse | MLM-based term expansion with FLOPS regularization |
| **ColBERT** | Late Interaction | Token-level MaxSim scoring |
| **HyDE** | Query Expansion | LLM-generated hypothetical documents |
| **Hybrid** | Fusion | RRF, linear, convex combination |

### Multi-Modal Search

| Modality | Method | Details |
|----------|--------|---------|
| **Text** | Dense + Sparse + Hybrid | 11 embedding models, BM25, SPLADE |
| **Chemical** | Fingerprints + Neural | Morgan/ECFP4, ChemBERTa, MoLFormer |
| **Sequence** | Alignment + Neural | BLAST+, ESM-2, ProtBERT, ProtT5 |

### Advanced Components

| Component | Description |
|-----------|-------------|
| **Learning-to-Rank** | LambdaMART, XGBoost, RankNet, ListNet ensemble |
| **Biomedical NER** | Chemical, disease, gene/protein extraction |
| **Thesaurus Expansion** | MeSH, ChEBI, DrugBank synonym lookup |
| **Substructure Search** | RDKit-based scaffold and MCS matching |
| **Diversity Reranking** | MMR, xQuAD, patent-family diversification |
| **Cross-Encoder Reranking** | MS-MARCO, BGE, Jina, LLM-based |
| **Domain Adaptation** | Contrastive fine-tuning, knowledge distillation |

### Data Acquisition

| Source | API | Content |
|--------|-----|---------|
| **PubMed** | NCBI E-utilities | 35M+ biomedical articles |
| **bioRxiv/medRxiv** | Cold Spring Harbor | Preprints |
| **USPTO** | PatentsView | US patents with claims |
| **EPO** | Open Patent Services | European patents |
| **UniProt** | REST API | Protein sequences |
| **PubChem** | PUG REST | Chemical structures |
| **NCBI Sequences** | E-utilities | GenBank, patent sequences |
| **SureChEMBL** | REST API | Patent-chemical mappings |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/VibeCodingScientist/BioPAT-Benchmark.git
cd BioPAT-Benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install RDKit for chemical search
pip install rdkit

# Optional: Install FAISS for fast similarity search
pip install faiss-cpu  # or faiss-gpu
```

### Run Tests

```bash
# Core pipeline test
python scripts/test_with_mock_data.py

# SOTA components test
python scripts/test_sota_components.py

# Retrieval pipeline test
python scripts/test_retrieval_pipeline.py
```

### Build a Corpus

```bash
# Set API keys (optional, increases rate limits)
export NCBI_API_KEY=your_ncbi_key
export PATENTSVIEW_API_KEY=your_patentsview_key

# Build corpus on a topic
python scripts/build_corpus.py \
    --topic "CAR-T cell therapy" \
    --pubmed-limit 500 \
    --patent-limit 200 \
    --output data/car_t_corpus
```

### Basic Usage

```python
from biopat.data import DataAcquisition
from biopat.retrieval import (
    create_hybrid_retriever,
    create_reranker,
    create_mmr_diversifier,
)

# Fetch data
async with DataAcquisition() as acq:
    articles = await acq.fetch_pubmed("pembrolizumab melanoma", limit=100)
    patents = await acq.fetch_patents(query="PD-1 antibody", limit=50)

# Create retrieval pipeline
retriever = create_hybrid_retriever(
    dense_model="allenai/scibert_scivocab_uncased",
    fusion_method="rrf"
)

# Search
results = retriever.search("anti-PD-1 immunotherapy for cancer", top_k=100)

# Rerank
reranker = create_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
reranked = reranker.rerank(query, results, top_k=20)

# Diversify
diversifier = create_mmr_diversifier(lambda_param=0.7)
diverse_results = diversifier.rerank(reranked, k=10)
```

## Architecture

```
BioPAT-Benchmark/
├── src/biopat/
│   ├── data/                    # Data acquisition connectors
│   │   ├── pubmed.py           # PubMed/NCBI connector
│   │   ├── patents.py          # USPTO/EPO connectors
│   │   ├── uniprot.py          # Protein sequences
│   │   ├── pubchem.py          # Chemical structures
│   │   ├── ncbi_sequences.py   # GenBank/patent sequences
│   │   ├── surechembl.py       # Patent-chemical mappings
│   │   └── acquisition.py      # Unified interface
│   │
│   ├── retrieval/               # SOTA retrieval methods
│   │   ├── dense.py            # Dense embedding retrieval
│   │   ├── hybrid.py           # BM25 + dense fusion
│   │   ├── splade.py           # Learned sparse retrieval
│   │   ├── colbert.py          # Late interaction
│   │   ├── hyde.py             # Query expansion
│   │   ├── molecular.py        # Chemical fingerprints
│   │   ├── sequence.py         # BLAST + PLM embeddings
│   │   ├── reranker.py         # Cross-encoder reranking
│   │   ├── learning_to_rank.py # LTR framework
│   │   ├── biomedical_ner.py   # Entity extraction
│   │   ├── thesaurus.py        # MeSH/ChEBI expansion
│   │   ├── substructure.py     # Scaffold/MCS search
│   │   └── diversity.py        # MMR/xQuAD reranking
│   │
│   ├── training/                # Domain adaptation
│   │   └── domain_adaptation.py # Fine-tuning utilities
│   │
│   ├── evaluation/              # Metrics and evaluation
│   │   ├── metrics.py          # IR metrics (nDCG, MAP, MRR)
│   │   └── trimodal_retrieval.py # Multi-modal fusion
│   │
│   ├── benchmark/               # Benchmark construction
│   ├── processing/              # Data processing
│   └── harmonization/           # Entity resolution
│
├── scripts/
│   ├── build_corpus.py         # Corpus construction
│   ├── run_benchmark.py        # Benchmark evaluation
│   └── test_*.py               # Test suites
│
└── docs/                        # Documentation
```

## Benchmarks

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **P@k** | Precision at k (k=5, 10, 20, 50, 100) |
| **R@k** | Recall at k |
| **MAP** | Mean Average Precision |
| **nDCG@k** | Normalized Discounted Cumulative Gain |
| **MRR** | Mean Reciprocal Rank |

### Relevance Grades

| Grade | Legal Standard | Description |
|-------|---------------|-------------|
| **3** | §102 / Category X | Novelty-destroying anticipation |
| **2** | §103 / Category Y | Obviousness (with combination) |
| **1** | Background | Relevant context, not prior art |
| **0** | Irrelevant | No relevance to claim |

## Model Support

### Text Embeddings

| Model | Dimension | Domain |
|-------|-----------|--------|
| `allenai/scibert_scivocab_uncased` | 768 | Scientific |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` | 768 | Biomedical |
| `dmis-lab/biobert-base-cased-v1.2` | 768 | Biomedical |
| `BAAI/bge-base-en-v1.5` | 768 | General |
| `BAAI/bge-m3` | 1024 | Multilingual |
| `intfloat/e5-large-v2` | 1024 | General |
| `facebook/contriever` | 768 | General |

### Molecular Embeddings

| Model | Type | Description |
|-------|------|-------------|
| Morgan/ECFP4 | Fingerprint | 2048-bit, radius=2 |
| `seyonec/ChemBERTa-zinc-base-v1` | Transformer | SMILES-based |
| `ibm/MoLFormer-XL-both-10pct` | Transformer | Large-scale |

### Protein Embeddings

| Model | Parameters | Description |
|-------|------------|-------------|
| `facebook/esm2_t33_650M_UR50D` | 650M | SOTA protein LM |
| `Rostlab/prot_bert` | 420M | ProtBERT |
| `Rostlab/prot_t5_xl_uniref50` | 3B | ProtT5 |

## Configuration

### Environment Variables

```bash
# API Keys (optional but recommended)
export NCBI_API_KEY=your_key          # PubMed + NCBI sequences
export PATENTSVIEW_API_KEY=your_key   # USPTO patents
export SURECHEMBL_API_KEY=your_key    # Patent-chemical mappings
export OPENAI_API_KEY=your_key        # HyDE query expansion
export ANTHROPIC_API_KEY=your_key     # LLM reranking
```

### Retrieval Configuration

```python
from biopat.retrieval import HybridConfig

config = HybridConfig(
    fusion_method="rrf",      # "linear", "rrf", "convex"
    sparse_weight=0.4,        # BM25 weight
    dense_weight=0.6,         # Dense weight
    rrf_k=60,                 # RRF parameter
    normalize_scores=True,
    candidate_fusion="union", # "union" or "intersection"
)
```

## Documentation

| Document | Description |
|----------|-------------|
| [METHODOLOGY.md](docs/METHODOLOGY.md) | Benchmark construction methodology |
| [DATA_SOURCES.md](docs/DATA_SOURCES.md) | Data sources and versions |
| [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Reproduction guide |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | API documentation |

## Citation

```bibtex
@software{biopat2026,
  author    = {BioPAT Contributors},
  title     = {{BioPAT}: Biomedical Patent-Article Trimodal Benchmark},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/VibeCodingScientist/BioPAT-Benchmark},
  version   = {4.0}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- NCBI for PubMed and sequence databases
- USPTO and EPO for patent data
- UniProt Consortium for protein sequences
- RDKit community for cheminformatics tools
- Hugging Face for transformer models

---

<p align="center">
  <sub>Built for the era of Agentic Science</sub>
</p>
