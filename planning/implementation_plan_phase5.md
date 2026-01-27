# BioPAT-Benchmark Phase 5 Implementation Plan: Full Novelty Benchmark (v2.0)

Phase 5 extends BioPAT from paper-only retrieval to **Full Prior Art Retrieval**. This involves incorporating prior patents into the corpus, matching the real-world workflow of patent examiners who search both scientific literature and earlier patents.

## User Review Required

> [!IMPORTANT]
> - **Corpus Expansion**: The corpus will grow from ~200K to ~500K documents. This will increase retrieval time and indexing memory requirements (FAISS/BM25).
> - **Patent Representation**: For the patent corpus, documents will consist of **Abstract + first Independent Claim** to provide a comparable textual length to scientific abstracts.

## Proposed Changes

### 1. Schema Extensions
#### [MODIFY] [benchmark/beir_format.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/benchmark/beir_format.py)
- Addition of `doc_type` field (`paper` | `patent`) and `date` field to the corpus entries.
- Logic to combine patent abstracts and claims for corpus text.

### 2. Processing Layer (New Module)
#### [NEW] [processing/prior_patents.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/processing/prior_patents.py)
- Logic to identify prior patent candidates based on:
    - Office Action patent citations (Gold Standard).
    - Query-cited patents (Applicant prior art).
    - Same-IPC patents (Subclass hard negatives).
- Temporal filtering to ensure candidates predate the query priority date.

### 3. Ingestion Layer (Extension)
#### [MODIFY] [ingestion/patentsview.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/ingestion/patentsview.py)
- Support for batch retrieval of patent metadata specifically for corpus assembly (fetching titles, abstracts, and claims in bulk).

### 4. Ground Truth & Evaluation
#### [MODIFY] [groundtruth/relevance.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/groundtruth/relevance.py)
- Removal of the NPL-only filter to keep both paper and patent citations.
- Unified relevance assignment logic for all document types.

#### [MODIFY] [evaluation/metrics.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/evaluation/metrics.py)
- Multi-dimensional reporting:
    - **Overall Metrics**: Combined performance.
    - **Papers Only**: Retrieval performance on the literature subset.
    - **Patents Only**: Retrieval performance on the patent subset.

### 5. Configuration (Extension)
#### [MODIFY] [configs/default.yaml](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/configs/default.yaml)
- Addition of `corpus` toggles (`include_papers`, `include_patents`) and patent sampling limits.

## Verification Plan

### Automated Tests
- **Schema Validation**: Ensure `corpus.jsonl` contains the `doc_type` field for all entries.
- **Merge Integrity**: Verify that paper and patent IDs do not collide in the merged corpus.
- **Metric Breakdown Test**: Verify that the evaluation report correctly splits metrics by document type.

### Manual Verification
- Inspect patent corpus entries to ensure the "Abstract + Claim 1" join looks clean.
- Verify that Phase 1/2 behavior can be reproduced by setting `include_patents: false`.
