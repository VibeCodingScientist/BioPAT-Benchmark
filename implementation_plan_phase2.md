# Phase 2: Examiner-Grade Ground Truth Implementation Plan

## Overview

Phase 2 enhances the BioPAT benchmark with USPTO Office Action data, providing examiner-validated relevance judgments with graded relevance (0-3) and claim-level mapping.

## Objectives

| Attribute | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Ground Truth Source | RoS citations only | RoS + Office Actions |
| Relevance Scale | Binary (0/1) | Graded (0/1/2/3) |
| Rejection Type Info | Not available | §102/§103 distinguished |
| Claim Mapping | Patent-level | Claim-level from OA |
| Query Count | ~5,000 | 8,000-12,000 |

## Relevance Tiers

| Score | Label | Criteria |
|-------|-------|----------|
| 3 | novelty_destroying | Examiner cited under §102 (anticipation) |
| 2 | highly_relevant | Examiner cited under §103 OR RoS examiner confidence >= 9 |
| 1 | relevant | RoS examiner confidence 7-8 OR applicant confidence >= 8 |
| 0 | not_relevant | No link, low confidence, or temporal violation |

## Implementation Tasks

### Task 2.1: Download Office Action Dataset (2 days)
- Download USPTO Office Action Research Dataset from USPTO (~15GB)
- Parse key tables: office_actions, rejections, citations
- Store in Parquet format

### Task 2.2: Parse NPL Citations from Office Actions (5 days)
- Extract NPL citations in various formats (structured, semi-structured, unstructured)
- Link to papers via PMID, DOI, title matching
- Target >75% linking accuracy

### Task 2.3: Map Rejections to Claims (3 days)
- Parse claim numbers from rejection text (ranges, lists)
- Extract rejection type (102/103)
- Create claim-citation mapping

### Task 2.4: Implement Graded Relevance (2 days)
- Upgrade binary relevance to 4-tier scale
- Apply relevance assignment logic based on source and type
- Enforce temporal constraint

### Task 2.5: Expand and Rebalance Corpus (3 days)
- Add papers from Office Actions not in Phase 1
- Implement hard negative strategies:
  - BM25 negatives
  - Concept negatives
  - Journal negatives
  - Temporal negatives

### Task 2.6: Domain Stratification Analysis (2 days)
- Implement IN-domain vs OUT-domain classification
- Map OpenAlex concepts to IPC codes
- Create stratified evaluation splits

## Acceptance Criteria

1. Graded relevance implemented with clear tier definitions
2. NPL citation linking achieves >75% accuracy
3. At least 500 queries have §102 (novelty-destroying) ground truth
4. IN-domain and OUT-domain splits created
5. Updated BM25 baseline with NDCG metrics

## Data Sources

- **Office Action Dataset**: https://www.uspto.gov/ip-policy/economic-research/research-datasets/office-action-research-dataset-patents
- Format: CSV/JSON, ~15GB compressed
- Coverage: 4.4 million Office Actions (2008-2017)
- Key fields: application_id, rejection_type, cited_ref_type, cited_ref_text, rejected_claims
