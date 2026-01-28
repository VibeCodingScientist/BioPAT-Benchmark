# BioPAT Methodology (v4.0)

This document outlines the construction methodology for the BioPAT benchmark, which has evolved from a text-only retrieval task to a **Global Multi-Modal Prior Art Discovery** system specifically designed to evaluate **Agentic Novelty Determination**. 

In an era where AI agents generate hypotheses at scale, BioPAT serves as the definitive audit layer for validating claimed innovations.

## 1. Scope & Jurisdiction

BioPAT v4.0 provides ~80% total biomedical patent coverage by integrating data from:
- **United States (USPTO)**: Primary source for examiner rejections.
- **Europe (EPO)**: Integration of EP search reports and "X/Y/A" categorical relevance.

## 2. Selection Criteria (The "Gold Set")

Patents are selected for the query set based on:
1.  **Domain**: IPC codes in `A61` (Medical), `C07` (Organic Chem), `C12` (Biotech).
2.  **Evidence Density**: Minimum 3 high-confidence NPL citations OR at least 2 structural entity mentions (Chemical/Sequence).
3.  **Temporal Integrity**: Strict adherence to priority dates to prevent leakage from future publications.

## 3. Graded Relevance Framework

BioPAT uses a unified 0-3 grading system across all modalities:

### 3.1 Text & Metadata (Legal Evidence)
- **Score 3 (Anticipation)**: USPTO §102 rejection OR EPO Category 'X' citation.
- **Score 2 (Obviousness)**: USPTO §103 rejection OR EPO Category 'Y' citation.
- **Score 1 (Background)**: Mentioned in Background, IDS, or EPO Category 'A'.

### 3.2 Chemical Modality (Structural Similarity)
- **Score 3 (Exact)**: Identical InChIKey match in paper/prior patent.
- **Score 2 (High Sim)**: Tanimoto coefficient ≥ 0.85 (Morgan Fingerprint).
- **Score 1 (Med Sim)**: Tanimoto coefficient 0.70 - 0.85.

### 3.3 Sequence Modality (Biotech Identity)
- **Score 3 (Near-Identity)**: BLAST identity ≥ 95% + high query coverage.
- **Score 2 (Homolog)**: BLAST identity 85% - 95%.
- **Score 1 (Related)**: BLAST identity 70% - 85%.

## 4. Multi-Modal Retrieval Strategy

BioPAT evaluates systems on their ability to fuse independent retrieval signals:
- **Lexical/Semantic**: BM25 + Dense embedding of claim text.
- **Structural**: Subgraph and similarity matching of extracted chemical SMILES.
- **Biological**: Sequence alignment (BLAST) of protein/DNA fragments.

Total Prior Art Score ($S$) is calculated as a weighted fusion:
$$S = \alpha \cdot \text{Text} + \beta \cdot \text{Chem} + \gamma \cdot \text{Seq}$$

## 5. Performance Metrics

- **NDCG@k (Primary)**: Assesses the ranking of anticipation references at the top.
- **Recall@100**: Measures the safety margin of the search (exhaustiveness).
- **Cross-Jurisdiction Stability**: Measures if a model performance is uniform across US/EP docs.
