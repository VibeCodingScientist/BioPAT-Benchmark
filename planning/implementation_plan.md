# BioPAT-Benchmark Phase 1 Implementation Plan

This plan outlines the steps to implement Phase 1 of the BioPAT (Biomedical Patent-to-Article Retrieval) benchmark. The goal of this phase is to create a working benchmark using citation-based ground truth from the Reliance on Science (RoS) dataset.

## User Review Required

> [!IMPORTANT]
> - **API Keys Required**: To execute this plan, API keys for PatentsView and NCBI (PubMed) are needed. These should be set in the environment or fixed in `configs/default.yaml`.
> - **Data Storage**: The raw Reliance on Science dataset and USPTO bulk data can be large (~GBs). Ensure sufficient disk space in the `data/` directory.

## Proposed Changes

### 1. Ingestion Layer
Implementation of API clients and data loaders.

#### [NEW] [patentsview.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/ingestion/patentsview.py)
- Async client for PatentsView API.
- Support for filtering by IPC codes and retrieving patent metadata/claims.
- Integration with `diskcache` for response caching.

#### [NEW] [openalex.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/ingestion/openalex.py)
- Client for OpenAlex API.
- Batch retrieval of paper metadata.
- Abstract reconstruction from inverted index.

#### [NEW] [ros.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/ingestion/ros.py)
- Loader for the Reliance on Science (RoS) dataset (CSV/Parquet).

---

### 2. Processing Layer
Transformation and cleaning of raw data.

#### [NEW] [patents.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/processing/patents.py)
- Logic to filter biomedical patents.
- Parser for claim text, distinguishing between independent and dependent claims.
- Handling of truncated claims via USPTO bulk data fallback.

#### [NEW] [papers.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/processing/papers.py)
- Cleaning and formatting of OpenAlex paper metadata.
- Management of paper corpus.

---

### 3. Ground Truth & Benchmark Layer
Constructing relevance judgments and final output.

#### [NEW] [relevance.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/groundtruth/relevance.py)
- Logic to assign binary relevance based on RoS confidence.
- Implementation of the **Hard Temporal Constraint** (paper date < patent priority date).

#### [NEW] [beir_format.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/benchmark/beir_format.py)
- Exporter to BEIR-compatible JSONL and TSV formats.
- Stratified splitting (Train/Dev/Test) by IPC subclass.

---

### 4. Evaluation Layer
Baselines and metrics.

#### [NEW] [bm25.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/evaluation/bm25.py)
- Implementation of BM25 baseline using `beir` and `pyserini` or similar.

---

### 5. CLI Scripts
Entry points for the pipeline.

#### [NEW] [main.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/scripts/main.py)
- Unified CLI to run ingestion, processing, and benchmark generation.

## Verification Plan

### Automated Tests
- **Unit Tests**: Add tests for each module in `tests/`.
  - `test_ingestion.py`: Mock API responses and verify parsing.
  - `test_processing.py`: Verify claim extraction and filtering logic.
  - `test_groundtruth.py`: **CRITICAL**: Verify that no positive relevance judgment violates the temporal constraint.
- **Integration Test**: Run a "mini" pipeline with a small sample of patents.

### Manual Verification
- Inspect `data/benchmark/queries.jsonl` to ensure claim text is clean and complete.
- Verify BEIR formatting using the `beir` library's data loader.
