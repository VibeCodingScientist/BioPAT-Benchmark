# BioPAT-Benchmark Phase 6 Implementation Plan: International Patent Coverage (v3.0)

Phase 6 upgrades BioPAT from a US-centric benchmark to a **Global Prior Art Retrieval Benchmark**. This involves integrating European (EP) and PCT (WO) patents into the corpus and ground truth, providing ~85% total biomedical patent coverage.

## User Review Required

> [!IMPORTANT]
> - **API Credentials**: This phase requires new API keys: EPO OPS (Consumer Key/Secret) and WIPO PATENTSCOPE.
> - **Corpus Size**: The merged corpus will grow to ~500K-1M documents. Check disk space (~300GB recommended for production).
> - **Translations**: While the priority is English text, some international abstracts/claims may require machine translation or fallback logic.

## Proposed Changes

### 1. Ingestion Layer (New Clients)
#### [NEW] [ingestion/epo.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/ingestion/epo.py)
- EPO Open Patent Services (OPS) client with OAuth 2.0 flow.
- Methods for bibliographic search, publication retrieval, and citation extraction.

#### [NEW] [ingestion/wipo.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/ingestion/wipo.py)
- WIPO PATENTSCOPE client for PCT (WO) application retrieval.

#### [NEW] [ingestion/bigquery.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/ingestion/bigquery.py) (Optional Fallback)
- Google Cloud BigQuery client for consolidated multi-jurisdiction data access.

### 2. Processing Layer
#### [NEW] [processing/patent_ids.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/processing/patent_ids.py)
- Normalization logic to handle disparate ID formats (e.g., `EP 1234567 A1` vs `WO2020123456`).

#### [NEW] [processing/international_patents.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/processing/international_patents.py)
- Logic to build the international corpus, including patent family mapping.

### 3. Ground Truth & Evaluation
#### [NEW] [groundtruth/ep_citations.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/groundtruth/ep_citations.py)
- Parser for EP search report citations (Categories X, Y, A).

#### [MODIFY] [evaluation/metrics.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/evaluation/metrics.py)
- Updated reporting to include **Per-Jurisdiction metrics** (US, EP, WO).

### 4. Configuration
#### [MODIFY] [configs/default.yaml](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/configs/default.yaml)
- Addition of EPO/WIPO API keys and jurisdiction toggles.

## Verification Plan

### Automated Tests
- **API Connectivity**: Verify OAuth token acquisition for EPO.
- **Normalization Test**: Ensure variety of ID strings resolve to bit-identical normalized keys.
- **Categorization Test**: Verify EP Category 'X' correctly maps to relevance score 3.

### Manual Verification
- **Translation Audit**: Sample WO corpus entries to ensure English text is correctly extracted from multilingual records.
- **Family Integrity**: Verify that adding an EP family member doesn't create redundant qrels if the US version is already present.
