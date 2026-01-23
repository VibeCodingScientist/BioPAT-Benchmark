# Phase 3 Implementation Plan: Comprehensive Baselines

## Overview

Phase 3 runs comprehensive baseline evaluations, performs ablation studies, and conducts error analysis. The goal is to characterize benchmark difficulty and establish reference performance levels.

## Baseline Models

| Model | Type | Rationale |
|-------|------|-----------|
| BM25 | Lexical | Strong sparse baseline; competitive in technical domains |
| Contriever | Dense | State-of-art general dense retriever; unsupervised pretraining |
| SPECTER2 | Dense | Scientific paper embeddings; citation-trained |
| PubMedBERT | Dense | Biomedical domain-specific; trained on PubMed |
| GTR-T5-XL | Dense | Large-scale dense retriever; good zero-shot |
| BM25 + CE | Hybrid | BM25 retrieval + cross-encoder reranking |
| ColBERT | Late Interaction | Token-level matching; good for long documents |

## Tasks

### Task 3.1: Embedding Infrastructure
- Set up embedding generation pipeline
- Implement vector search with FAISS
- Create model registry for baseline models
- Handle GPU/CPU detection automatically

### Task 3.2: Dense Retrieval Baselines
- Implement generic dense retriever class
- Add Contriever baseline
- Add SPECTER2 baseline
- Add PubMedBERT baseline
- Add GTR-T5-XL baseline
- Store results in TREC format

### Task 3.3: Hybrid and Reranking Methods
- Implement Reciprocal Rank Fusion (RRF)
- Implement score-based fusion
- Add cross-encoder reranking
- Support BM25 + dense hybrid

### Task 3.4: Ablation Studies
- Query representation ablation (claim only vs claim + abstract)
- Document representation ablation (title + abstract vs title only)
- Domain ablation (IN-domain vs OUT-domain)
- Temporal ablation (recent vs older patents)
- IPC subclass performance analysis

### Task 3.5: Error Analysis
- Implement error categorization
- Vocabulary mismatch detection
- Cross-domain failure analysis
- Generate error analysis reports

## File Structure

```
src/biopat/
├── evaluation/
│   ├── __init__.py         # MODIFIED: Add new exports
│   ├── bm25.py             # EXISTS: BM25 baseline (enhance)
│   ├── dense.py            # NEW: Dense retrieval baselines
│   ├── hybrid.py           # NEW: Hybrid methods and fusion
│   ├── reranker.py         # NEW: Cross-encoder reranking
│   ├── metrics.py          # EXISTS: Metrics (enhance)
│   ├── ablation.py         # NEW: Ablation studies
│   └── error_analysis.py   # NEW: Error analysis
├── pipeline_phase3.py      # NEW: Phase 3 pipeline

tests/
└── test_phase3.py          # NEW: Phase 3 tests
```

## Dependencies

New dependencies needed for Phase 3:
- sentence-transformers: For dense embeddings
- faiss-cpu or faiss-gpu: For vector search
- torch: Backend for transformers

## Implementation Notes

1. **Model Loading**: Use sentence-transformers for most models
2. **Memory Management**: Process embeddings in batches for large corpora
3. **Caching**: Cache embeddings to disk to avoid recomputation
4. **GPU Support**: Auto-detect GPU and use if available
5. **TREC Format**: Save all results in standard TREC format for pytrec_eval

## Evaluation Metrics

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| NDCG@k | Normalized Discounted Cumulative Gain | Primary metric; rewards graded relevance |
| Recall@k | Proportion of relevant docs in top k | Coverage metric; critical for legal |
| MAP | Mean Average Precision | Overall ranking quality |
| MRR | Mean Reciprocal Rank | Position of first relevant document |
| P@k | Precision at rank k | Proportion of top k that are relevant |
