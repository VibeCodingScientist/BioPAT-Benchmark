# BioPAT-Benchmark Phase 3 Implementation Plan

Phase 3 establishes the **Comprehensive Evaluation Framework**. It focuses on building the infrastructure to run multiple retrieval baselines (lexical, dense, hybrid), performing ablation studies, and computing detailed metrics to characterize benchmark difficulty.

## User Review Required

> [!IMPORTANT]
> - **GPU Resources**: While the *code* can be written on any system, running dense retrieval baselines (SPECTER2, Contriever) on the full corpus will benefit significantly from a GPU (CUDA or Metal).
> - **Model Downloads**: This phase will require downloading pre-trained models from HuggingFace (e.g., `allenai/specter2`, `sentence-transformers/all-mpnet-base-v2`).

## Proposed Changes

### 1. Evaluation Infrastructure
#### [NEW] [embeddings.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/evaluation/embeddings.py)
- Logic to generate document and query embeddings using `sentence-transformers`.
- Support for batch processing and persistent storage of embeddings (Parquet).

#### [NEW] [retriever.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/evaluation/retriever.py)
- Unified interface for different retrieval methods:
    - `LexicalRetriever` (BM25)
    - `DenseRetriever` (FAISS inner product search)
    - `HybridRetriever` (Reciprocal Rank Fusion - RRF)

### 2. Baseline Models
#### [NEW] [dense.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/evaluation/dense.py)
- Configurations for specter2, contriever, and PubMedBERT.
- Integration with FAISS for fast similarity search.

#### [NEW] [cross_encoder.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/evaluation/cross_encoder.py)
- Reranking infrastructure using cross-encoders (e.g., `cross-encoder/ms-marco-MiniLM-L-12-v2`).

### 3. Analysis & Reporting
#### [NEW] [analysis.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/evaluation/analysis.py)
- Logic for ablation studies:
    - Performance by IPC subclass.
    - Performance by relevance tier (N@10 for score 3 vs score 2).
    - IN-domain vs. OUT-domain comparison.

#### [NEW] [report.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/evaluation/report.py)
- Generator for standard TREC-style result files and summary tables (NDCG@10, Recall@100, etc.).

## Verification Plan

### Automated Tests
- **Retriever Interface Test**: Ensure all retrievers return results in the same format.
- **RRF Test**: Verify the reciprocal rank fusion logic on a small sample of ranked lists.
- **Metrics Test**: Use a toy qrel and run file to verify that NDCG and Recall calculations match expected values.

### Manual Verification
- Verify that embedding generation correctly utilizes the GPU if available.
- Inspect the ablation report to ensure all dimensions (IPC, Domain, Relevance) are present.
