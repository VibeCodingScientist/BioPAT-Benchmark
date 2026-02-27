# BioPAT: Biomedical Patent-to-Article Retrieval Benchmark

**A benchmark for evaluating retrieval systems and LLM agents on biomedical patent prior art search.**

BioPAT constructs a large-scale benchmark from real patent-paper citation data, then evaluates how well different retrieval methods — from BM25 baselines to LLM-powered agents — can find relevant prior art across a dual corpus of scientific papers and patents.

---

## Benchmark at a Glance

| Stat | Value |
|------|-------|
| **Queries** | 1,984 patent claims |
| **Documents** | 158,850 scientific papers |
| **Relevance judgments** | 842,170 qrels (764K highly relevant, 78K relevant) |
| **Splits** | Train 586K / Dev 125K / Test 131K |
| **Domain coverage** | A61 (medical), C07 (organic chem), C12 (biochemistry) |
| **Format** | BEIR-compatible (corpus.jsonl, queries.jsonl, qrels/*.tsv) |

Ground truth is derived from real patent examiner citations via the [Reliance on Science](https://zenodo.org/record/3888942) dataset, with temporal validation ensuring papers predate patent priority dates.

## Architecture

```
src/biopat/
├── pipeline.py              # Phase 1: 8-step benchmark construction with checkpoint/resume
├── pipeline_novelty.py      # End-to-end novelty assessment pipeline
├── config.py                # Configuration (paths, APIs, LLM, evaluation)
├── compat.py                # Polars version compatibility layer
├── reproducibility.py       # Checksums and audit logging
│
├── ingestion/               # Data acquisition
│   ├── ros.py               #   Reliance on Science patent-paper citations
│   ├── patentsview.py       #   USPTO PatentsView API (claims, metadata)
│   └── openalex.py          #   OpenAlex API (paper metadata, abstracts)
│
├── processing/              # Data transformation
│   ├── patents.py           #   Patent filtering by IPC, claim extraction
│   ├── papers.py            #   Paper metadata cleaning, abstract validation
│   ├── linking.py           #   Patent-paper citation linking
│   ├── patent_ids.py        #   Patent ID normalization (US/EP/WO)
│   ├── chemical_index.py    #   Morgan fingerprints, FAISS chemical search
│   ├── sequence_index.py    #   BLAST+ sequence alignment search
│   ├── prior_patents.py     #   Prior patent reference filtering
│   ├── international_patents.py  # EP/WO patent support
│   ├── claim_mapper.py      #   Claim-to-citation mapping
│   └── npl_parser.py        #   Non-patent literature reference parsing
│
├── groundtruth/             # Relevance judgment creation
│   ├── relevance.py         #   Graded relevance assignment (0-3 scale)
│   ├── temporal.py          #   Temporal validation (paper before patent)
│   ├── stratification.py    #   Domain-stratified sampling by IPC
│   └── ep_citations.py      #   EP search report category mapping
│
├── benchmark/               # Benchmark formatting
│   ├── beir_format.py       #   BEIR output (corpus.jsonl, queries.jsonl, qrels)
│   ├── splits.py            #   Train/dev/test splitting (stratified)
│   └── sampling.py          #   Uniform and stratified sampling
│
├── evaluation/              # Retrieval evaluation (7 experiment types)
│   ├── llm_evaluator.py     #   LLMBenchmarkRunner: orchestrates all experiments
│   ├── bm25.py              #   BM25 baseline
│   ├── dense.py             #   Dense retrieval (sentence-transformers, FAISS)
│   ├── hybrid.py            #   BM25 + dense fusion (RRF, linear)
│   ├── reranker.py          #   Cross-encoder and LLM listwise reranking
│   ├── metrics.py           #   IR metrics: NDCG, MAP, MRR, P@k, R@k
│   ├── trimodal_retrieval.py#   Text + chemical + sequence fusion
│   ├── agent_retrieval.py   #   Agentic dual-corpus retrieval (Exp 7)
│   ├── agent_metrics.py     #   Agent-specific metrics and refinement curves
│   ├── dual_qrels.py        #   Dual corpus builder, qrel inversion
│   ├── ablation.py          #   Ablation studies (query, doc, domain, temporal, IPC)
│   ├── error_analysis.py    #   Failure categorization and vocabulary analysis
│   ├── statistical_tests.py #   Significance testing
│   └── publication.py       #   Report generation
│
├── retrieval/               # Retrieval methods
│   ├── dense.py             #   Dense retrieval with domain-specific models
│   ├── hybrid.py            #   BM25 + dense hybrid fusion
│   ├── reranker.py          #   Cross-encoder + LLM reranking
│   ├── hyde.py              #   HyDE query expansion via LLM
│   ├── molecular.py         #   Chemical fingerprint retrieval (RDKit)
│   ├── sequence.py          #   Sequence alignment retrieval (BLAST+)
│   ├── splade.py            #   Learned sparse retrieval (SPLADE)
│   └── colbert.py           #   Late-interaction retrieval (ColBERT)
│
├── reasoning/               # LLM-based novelty reasoning
│   ├── claim_parser.py      #   Decompose claims into elements via LLM
│   ├── novelty_reasoner.py  #   Map prior art to claims, assess novelty
│   └── explanation_generator.py  # Generate novelty reports
│
├── novex/                   # NovEx 3-tier prior art discovery benchmark
│   ├── curate.py            #   Statement curation pipeline (3-LLM consensus)
│   ├── annotation.py        #   Multi-LLM annotation protocol (Fleiss' kappa)
│   ├── benchmark.py         #   BEIR-compatible benchmark loader
│   ├── evaluator.py         #   Tier 1/2/3 evaluation harness
│   └── analysis.py          #   Vocabulary gap, cross-domain, agreement analysis
│
└── llm/                     # Unified LLM provider interface
    ├── providers.py         #   OpenAI, Anthropic, Google with consistent API
    └── cost_tracker.py      #   Token tracking, cost estimation, budget enforcement
```

## Pipeline

### Phase 1: Benchmark Construction

The pipeline (`biopat pipeline`) runs 8 steps with checkpoint/resume:

1. **Load RoS** — Download Reliance on Science patent-paper citation data
2. **Fetch patents** — Get patent metadata and claims from PatentsView API
3. **Extract claims** — Parse independent claims, filter by IPC domain
4. **Fetch papers** — Get paper metadata and abstracts from OpenAlex API
5. **Link citations** — Match patents to papers via RoS citation links
6. **Ground truth** — Assign graded relevance (0-3), temporal validation
7. **Split** — Create stratified train/dev/test splits
8. **Format** — Output BEIR-compatible benchmark files

```bash
# Run full pipeline
docker compose run --rm benchmark python -m biopat --config configs/default.yaml

# Or locally
pip install -e .
biopat --config configs/default.yaml
```

### Phase 2: LLM Evaluation (7 Experiments)

| # | Experiment | Type | What it measures |
|---|-----------|------|-----------------|
| 1 | **BM25 baseline** | CPU | Sparse retrieval floor |
| 2 | **Dense baselines** | CPU | Neural embedding ceiling (BGE, MiniLM) |
| 3 | **HyDE expansion** | LLM | Query expansion via hypothetical documents |
| 4 | **Listwise reranking** | LLM | LLM reranking of BM25 top-100 |
| 5 | **Relevance judgment** | LLM | Agreement with silver-standard qrels |
| 6 | **Novelty assessment** | LLM | End-to-end claim→prior art→novelty pipeline |
| 7 | **Agent dual retrieval** | LLM | Iterative agent searching papers + patents |

```bash
# Run all experiments
python scripts/run_experiments.py --config configs/experiments.yaml

# Run only agent experiment
python scripts/run_experiments.py --config configs/experiments_agent.yaml --experiment agent

# Estimate costs without API calls
python scripts/run_experiments.py --config configs/experiments.yaml --dry-run
```

### Experiment 7: Agent-Based Dual Retrieval

The agent is given a scientific claim and must find relevant **papers AND patents** in a dual corpus (~160K docs). It operates in three phases:

1. **Plan** — LLM analyzes the claim, generates up to 5 BM25 search queries
2. **Refine** — LLM reviews results, identifies gaps, optionally searches more
3. **Rank** — LLM produces a final ranked list from accumulated results

Agent-specific metrics include per-doc-type recall (paper vs patent), coverage balance, search efficiency, and refinement gain curves.

## LLM Support

Three providers with a unified interface (`LLMProvider.generate()`):

| Provider | Models | Pricing tracked |
|----------|--------|----------------|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-5.2 | Per-token input/output |
| **Anthropic** | Claude Opus 4.6, Sonnet 4.5, Haiku 4.5 | Per-token input/output |
| **Google** | Gemini 2.5 Pro, Gemini 2.0 Flash | Per-token input/output |

Cost tracking with budget enforcement (`CostTracker`) is built into every experiment.

## Retrieval Methods

| Method | Module | Description |
|--------|--------|-------------|
| **BM25** | `evaluation/bm25.py` | Okapi BM25 with rank-bm25 |
| **Dense** | `retrieval/dense.py` | Sentence-transformers + FAISS indexing |
| **Hybrid** | `retrieval/hybrid.py` | BM25 + dense fusion (RRF, linear) |
| **HyDE** | `retrieval/hyde.py` | LLM-generated hypothetical document expansion |
| **Cross-encoder** | `retrieval/reranker.py` | BAAI/bge-reranker, MS-MARCO rerankers |
| **SPLADE** | `retrieval/splade.py` | Learned sparse retrieval with term expansion |
| **ColBERT** | `retrieval/colbert.py` | Token-level late interaction |
| **Chemical** | `retrieval/molecular.py` | Morgan fingerprints + Tanimoto similarity |
| **Sequence** | `retrieval/sequence.py` | BLAST+ alignment search |
| **Trimodal** | `evaluation/trimodal_retrieval.py` | Text + chemical + sequence fusion |

## Relevance Scale

Graded relevance derived from patent examiner citation behavior:

| Grade | Meaning | Source signal |
|-------|---------|--------------|
| **3** | Novelty-destroying (anticipation) | Not currently assigned automatically |
| **2** | Highly relevant | Front-page citation (examiner-added) |
| **1** | Relevant | In-text citation (applicant-added) |
| **0** | Not relevant | No citation link |

## Quick Start

### Installation

```bash
git clone https://github.com/VibeCodingScientist/BioPAT-Benchmark.git
cd BioPAT-Benchmark

python -m venv venv
source venv/bin/activate

# Core only
pip install -e .

# With evaluation (sentence-transformers, FAISS, PyTorch)
pip install -e ".[evaluation]"

# With LLM experiments (OpenAI, Anthropic, Google)
pip install -e ".[llm]"

# Everything
pip install -e ".[all]"
```

### Environment Variables

```bash
# Data source APIs
export PATENTSVIEW_API_KEY=your_key   # USPTO patent data
export NCBI_API_KEY=your_key          # Optional: PubMed sequences

# LLM providers (for experiments 3-7)
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
export GOOGLE_API_KEY=your_key
```

### Docker

```bash
# Build and run pipeline
docker compose build
docker compose run --rm benchmark

# Run LLM experiments
docker compose run --rm benchmark python scripts/run_experiments.py \
    --config configs/experiments.yaml
```

## Configuration

### `configs/default.yaml` — Pipeline configuration
Controls data paths, API settings, IPC domain filters, and Phase 1 parameters.

### `configs/experiments.yaml` — Full experiment suite
Defines all 7 experiments, model selection, budget limits, and evaluation parameters.

### `configs/experiments_agent.yaml` — Agent experiment only
Focused config for running just the agent dual-retrieval experiment across 3 models.

### `configs/novex.yaml` — NovEx benchmark configuration
Curation targets, annotation protocol models, 3-tier evaluation setup, and budget limits.

### Phase 3: BioPAT-NovEx — Prior Art Discovery Benchmark

**NovEx** extends BioPAT with a 3-tier evaluation framework for prior art discovery across scientific and patent literature:

| Tier | Task | Input | Output | Metrics |
|------|------|-------|--------|---------|
| **1** | Retrieval | Scientific statement | Ranked documents | Recall@k, NDCG@10, MAP, per-doc-type recall |
| **2** | Relevance | Statement + 50 candidates | 0-3 relevance grade | Cohen's kappa, Kendall's tau, accuracy, MAE |
| **3** | Novelty | Statement + prior art set | NOVEL / ANTICIPATED / PARTIAL | Per-class F1, accuracy, macro-F1 |

100 expert-curated statements (stratified: 35 multi-patent examiner, 25 single-patent, 15 cross-domain, 15 applicant-only, 10 negative controls), evaluated with multi-LLM annotation protocol (GPT-5.2 + Claude Sonnet 4.6 + Gemini 3 Pro consensus).

```bash
# Step 1: Curate statements (3-LLM extraction + quality filter)
python scripts/curate_statements.py --config configs/novex.yaml

# Step 2: Run all evaluations (8 retrieval methods × 3 LLMs × 3 tiers)
python scripts/run_novex.py --config configs/novex.yaml

# Step 3: Generate paper figures and tables
python scripts/analyze_novex.py --config configs/novex.yaml

# Single tier/method
python scripts/run_novex.py --tier 1 --method bm25
python scripts/run_novex.py --tier 2 --model gpt-4o
python scripts/run_novex.py --dry-run  # Cost estimate only
```

**NovEx architecture:**
```
src/biopat/novex/
├── curate.py       # Statement selection, 3-LLM extraction, ground truth
├── annotation.py   # Multi-LLM relevance judgment protocol (Fleiss' kappa)
├── benchmark.py    # BEIR-compatible loader with domain/category/difficulty filters
├── evaluator.py    # Tier 1/2/3 evaluation harness
└── analysis.py     # Vocabulary gap, cross-domain, agreement analysis, LaTeX tables
```

## Results

### Phase 1: BM25 Baseline (Test Split)

| Metric | Score |
|--------|-------|
| Precision@10 | 0.257 |
| Recall@10 | 0.007 |
| Precision@50 | 0.167 |
| Recall@50 | 0.022 |
| Precision@100 | 0.131 |
| Recall@100 | 0.034 |

### NovEx Reverse: Novelty Assessment (100 Patents)

Three frontier LLMs independently assessed each patent claim against retrieved prior art, with majority-vote consensus:

| Model | Anticipated | Partially Anticipated | Novel |
|-------|------------|----------------------|-------|
| GPT-5.2 | 58 | 27 | — |
| Claude Sonnet 4.6 | 61 | 21 | 3 |
| Gemini 3 Pro | 58 | 13 | 10 |
| **Consensus** | **61 (61%)** | **21 (21%)** | **18 (18%)** |

**Inter-model agreement**: 69% unanimous, 15% majority, 15% override, 1% no consensus.

Domain distribution: A61 (41), C07 (33), C12 (26).

### NovEx Tier 1: Relevance Grading

99 queries evaluated with 1,483 total relevance judgments:
- Grade 1 (relevant): 671
- Grade 2 (highly relevant): 393
- Grade 3 (anticipation): 419
- Mean judgments per query: 15.0

### API Costs

| Task | Cost | Calls | Tokens |
|------|------|-------|--------|
| Relevance grading (Tier 1) | $27.49 | 12,175 | 6.1M |
| Novelty assessment (Tier 3) | $2.50 | 255 | 418K |
| **Total** | **$29.98** | **12,430** | **6.5M** |

Pipeline runtime: ~13.7 hours (Feb 25–26, 2026).

## Project Status

- **Phase 1 pipeline**: Complete (842K qrels, 1,984 queries, 158K docs)
- **BM25 baseline**: Complete (P@10=0.257, R@100=0.034)
- **LLM evaluation framework**: 7 experiment types implemented
- **Agent dual retrieval**: Implemented with dual corpus (papers + patents)
- **NovEx benchmark**: Complete — 100 patent claims assessed, 3-model consensus novelty verdicts

## Dependencies

**Core**: httpx, polars, pyyaml, pydantic, diskcache, tqdm, rank-bm25

**Evaluation**: sentence-transformers, faiss-cpu, torch

**LLM**: openai, anthropic, google-genai, scipy

**Advanced**: rdkit (chemical structures)

Python 3.11–3.12 required.

## Citation

```bibtex
@software{biopat2026,
  author    = {BioPAT Contributors},
  title     = {{BioPAT}: Biomedical Patent-to-Article Retrieval Benchmark},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/VibeCodingScientist/BioPAT-Benchmark}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
