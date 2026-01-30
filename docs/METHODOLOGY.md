# BioPAT Methodology

This document describes the methodology for constructing and evaluating the BioPAT benchmark.

## Overview

BioPAT evaluates retrieval systems on their ability to find relevant prior art for biomedical patent claims. The benchmark simulates the task of a patent examiner or novelty agent searching for evidence that could affect the patentability of a claim.

## Task Definition

**Input**: A patent claim (text) with optional chemical structures (SMILES) and biological sequences.

**Output**: A ranked list of documents (scientific articles and patents) that are relevant to the claim.

**Goal**: Maximize retrieval of novelty-destroying prior art while minimizing false positives.

## Corpus Construction

### Data Sources

| Source | Type | Coverage |
|--------|------|----------|
| PubMed | Articles | 35M+ biomedical abstracts |
| bioRxiv/medRxiv | Preprints | Life sciences preprints |
| USPTO | Patents | US patents (full text + claims) |
| EPO | Patents | European patents |
| UniProt | Sequences | 250M+ protein sequences |
| PubChem | Chemicals | 100M+ compounds |

### Preprocessing

1. **Text Normalization**: Lowercasing, punctuation removal, whitespace normalization
2. **Chemical Extraction**: SMILES extraction from patent claims using NER
3. **Sequence Extraction**: Protein/nucleotide sequences from sequence listings
4. **Date Normalization**: ISO 8601 format for temporal filtering

### Entity Harmonization

BioPAT uses a unified identifier scheme (BioPAT IDs) to link entities across sources:

```
BP:PAT:US:10000001    # US Patent
BP:PUB:DOI:10.1000/x  # Publication
BP:CHM:SCHEMBL123456  # Chemical
BP:SEQ:AA:abc123      # Protein sequence
```

## Query Construction

### Claim-Level Queries

Each query represents a single patent claim, which is the legal unit of novelty:

```python
{
    "query_id": "US10000001-C1",
    "claim_text": "A method of treating melanoma comprising administering pembrolizumab",
    "claim_number": 1,
    "patent_id": "US10000001",
    "priority_date": "2015-01-15",
    "chemicals": ["CC(C)CC1=CC=C(C=C1)..."],  # Optional SMILES
    "sequences": ["MVLSPADKTNVKAAWGKVG..."],   # Optional sequences
}
```

### Query Categories

| Category | Description | Example |
|----------|-------------|---------|
| Composition | Drug formulations | "A pharmaceutical composition comprising..." |
| Method | Treatment methods | "A method of treating X comprising..." |
| Use | Medical uses | "Use of compound X for treating..." |
| Compound | Novel molecules | "A compound of formula I wherein..." |

## Relevance Judgment

### Grading Scale

| Grade | Label | Legal Standard | Description |
|-------|-------|---------------|-------------|
| 3 | Anticipation | §102 / X | Single reference destroys novelty |
| 2 | Obviousness | §103 / Y | Obvious in combination |
| 1 | Background | - | Related but not prior art |
| 0 | Irrelevant | - | No relevance |

### Annotation Guidelines

1. **Temporal Constraint**: Only documents published before the priority date qualify as prior art
2. **Technical Scope**: Document must relate to the same technical field
3. **Specific Disclosure**: Higher grades require specific disclosure of claim elements
4. **Enablement**: Prior art must enable the claimed invention

## Evaluation Protocol

### Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| P@k | TP@k / k | Precision at cutoff k |
| R@k | TP@k / Total Relevant | Recall at cutoff k |
| MAP | Mean of AP per query | Mean Average Precision |
| nDCG@k | DCG@k / IDCG@k | Normalized Discounted Cumulative Gain |
| MRR | 1 / rank of first relevant | Mean Reciprocal Rank |

### Multi-Modal Evaluation

For trimodal retrieval, we evaluate each modality separately and combined:

1. **Text-only**: Standard lexical/semantic retrieval
2. **Chemical-augmented**: Text + Tanimoto similarity
3. **Sequence-augmented**: Text + BLAST alignment
4. **Trimodal**: Full fusion of all modalities

### Significance Testing

- Paired t-test for metric comparisons
- Bootstrap confidence intervals (1000 iterations)
- Effect size (Cohen's d) for practical significance

## Baseline Systems

### Sparse Retrieval

| System | Configuration |
|--------|---------------|
| BM25 | k1=1.5, b=0.75 |
| TF-IDF | L2 normalization |

### Dense Retrieval

| System | Model |
|--------|-------|
| SciBERT | allenai/scibert_scivocab_uncased |
| PubMedBERT | microsoft/BiomedNLP-PubMedBERT |
| BioBERT | dmis-lab/biobert-base-cased-v1.2 |

### Hybrid Systems

| System | Fusion |
|--------|--------|
| BM25 + Dense | RRF (k=60) |
| SPLADE + Dense | Linear (0.5/0.5) |

## Reproducibility

All experiments can be reproduced using:

```bash
# Run benchmark with default settings
python scripts/run_benchmark.py \
    --corpus data/corpus.jsonl \
    --queries data/queries.jsonl \
    --output results/

# Compare methods
python scripts/evaluate.py \
    --results results/ \
    --metrics ndcg@10,map,mrr
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for detailed instructions.
