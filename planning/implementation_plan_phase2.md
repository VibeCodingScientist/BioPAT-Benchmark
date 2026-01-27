# BioPAT-Benchmark Phase 2 Implementation Plan

Phase 2 transitions BioPAT from a citation-based benchmark to an **examiner-validated gold standard**. By integrating the USPTO Office Action Research Dataset, we gain access to explicit rejections under §102 (novelty) and §103 (obviousness), enabling graded relevance and claim-level precision.

## User Review Required

> [!WARNING]
> - **Data Volume**: The USPTO Office Action dataset is ~15GB compressed. Ensure at least 50GB of free disk space in `data/raw`.
> - **Accuracy Trade-off**: Linking unstructured NPL citations from Office Actions to OpenAlex IDs is probabilistic. We target >75% accuracy; lower-quality links will be discarded to maintain "gold standard" status.

## Proposed Changes

### 1. Ingestion Layer (Extension)
#### [NEW] [office_action.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/ingestion/office_action.py)
- Logic to download and parse the 15GB USPTO Office Action dataset.
- Optimized DuckDB ingestion for large-scale CSV/JSON tables.

### 2. Processing Layer (Advanced)
#### [NEW] [linking.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/processing/linking.py) (Phase 2 Upgrade)
- Implementation of an NPL parser (regex + fuzzy matching).
- Cascading lookup: DOI/PMID -> Title/Author Match -> OpenAlex Search.

#### [NEW] [claims_mapping.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/processing/claims_mapping.py)
- Logic to extract specific claim numbers from rejection text (e.g., "Claims 1-5 and 8 are rejected...").
- Transformation of patent-level citations into claim-level query-document pairs.

### 3. Ground Truth Layer (Graded)
#### [MODIFY] [relevance.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/groundtruth/relevance.py)
- Upgrade from binary (0/1) to 4-tier graded relevance (0-3):
    - **3**: §102 Novelty-destroying (Examiner validated)
    - **2**: §103 Obviousness OR High-confidence examiner citation
    - **1**: Background/Applicant-cited
    - **0**: Post-priority date (Hard constraint) OR irrelevant

### 4. Benchmark Layer (Sampling)
#### [MODIFY] [sampling.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/benchmark/sampling.py)
- Expansion to target 8,000 - 12,000 queries.
- Implementation of IN-domain vs. OUT-domain stratification logic based on IPC subclass overlap.

## Verification Plan

### Automated Tests
- **NPL Linking Test**: Verify linking on a set of 100 known-good Office Action citations.
- **Claim Parsing Test**: Test extraction logic against various text patterns (ranges, lists, singulars).
- **Graded Relevance Test**: Ensure §102 rejections consistently map to score 3.

### Manual Verification
- Manually inspect a random sample of 50 (Claim, Paper) pairs to verify that the paper actually mentions the claimed concepts.
- Validate that the final `qrels.tsv` contains all 4 relevance levels.
