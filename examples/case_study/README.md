# BioPAT Case Study: Cancer Immunotherapy Patents

This case study demonstrates the BioPAT benchmark data format using a realistic mock dataset focused on **cancer immunotherapy patents** and their cited scientific literature.

## Overview

The dataset simulates prior art search for 10 patent queries against a corpus of 50 scientific articles. It demonstrates:

- **Trimodal data**: Text descriptions, chemical structures (SMILES), and protein sequences
- **Graded relevance**: 4-level relevance scale (0-3) based on citation type and similarity
- **Temporal ordering**: Patents cite articles published before the priority date

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Queries (Patents) | 10 |
| Corpus (Articles) | 50 |
| Total Judgments | 85 |
| Avg. Relevant/Query | 8.5 |

### Relevance Distribution

| Score | Meaning | Count |
|-------|---------|-------|
| 3 | Novelty-destroying (examiner cited) | 12 |
| 2 | Highly relevant (applicant cited) | 28 |
| 1 | Relevant (semantic similarity) | 45 |
| 0 | Not relevant | (implicit) |

## File Formats

### corpus.jsonl
```json
{"_id": "W2100000001", "title": "...", "text": "...", "metadata": {...}}
```

Each document contains:
- `_id`: OpenAlex Work ID
- `title`: Article title
- `text`: Abstract text
- `metadata`: Additional fields (DOI, journal, publication date, MeSH terms)

### queries.jsonl
```json
{"_id": "US10500001", "text": "...", "metadata": {...}}
```

Each query contains:
- `_id`: Patent ID
- `text`: Combined title + abstract + claims
- `metadata`: Priority date, IPC codes, chemicals (SMILES), sequences

### qrels.tsv
```
query_id    doc_id    relevance
US10500001  W2100000001    3
```

Tab-separated relevance judgments (TREC format).

## Loading the Data

```python
import json

# Load corpus
corpus = {}
with open("corpus.jsonl") as f:
    for line in f:
        doc = json.loads(line)
        corpus[doc["_id"]] = doc

# Load queries
queries = {}
with open("queries.jsonl") as f:
    for line in f:
        q = json.loads(line)
        queries[q["_id"]] = q

# Load qrels
qrels = {}
with open("qrels.tsv") as f:
    next(f)  # Skip header
    for line in f:
        qid, did, rel = line.strip().split("\t")
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][did] = int(rel)

print(f"Corpus: {len(corpus)} documents")
print(f"Queries: {len(queries)} patents")
print(f"Judgments: {sum(len(v) for v in qrels.values())}")
```

## Example Query

**Patent US10500001**: *Anti-PD-1 antibody for treating melanoma*

```
Title: Humanized anti-PD-1 monoclonal antibody and methods of use

Abstract: The present invention relates to humanized monoclonal antibodies
that specifically bind to human PD-1 (programmed death-1) receptor...

Claims: 1. A humanized monoclonal antibody comprising a heavy chain variable
region with CDR sequences SEQ ID NO:1, SEQ ID NO:2, and SEQ ID NO:3...
```

**Relevant Prior Art (Score=3)**:
- W2100000005: "PD-1 blockade enhances anti-tumor immunity in melanoma models"

**Chemical Structure** (from patent):
```
SMILES: CC(C)CC1=CC=C(C=C1)C(C)C(=O)NC2=CC=C(C=C2)NC(=O)C3=CC=NC=C3
```

**Protein Sequence** (from patent):
```
>SEQ_001 Anti-PD-1 Heavy Chain Variable Region
QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGEINPSNGGTNFNEKFKS
```

## Evaluation

Run evaluation using the provided script:

```bash
python evaluate_case_study.py
```

Expected output:
```
BioPAT Case Study Evaluation
============================
NDCG@10: 0.7234
P@10: 0.6500
R@100: 0.8941
MRR: 0.8167
```

## Use Cases

This case study demonstrates:

1. **Text-based retrieval**: Finding articles by semantic similarity to patent text
2. **Chemical structure search**: Finding articles describing similar compounds
3. **Sequence similarity**: Finding articles about related proteins/antibodies
4. **Citation verification**: Validating that examiner/applicant citations are retrieved

## Notes

- All data is synthetic/mock for demonstration purposes
- Real BioPAT datasets contain thousands of patents and millions of articles
- Chemical SMILES and protein sequences are simplified examples
