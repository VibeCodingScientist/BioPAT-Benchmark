# BioPAT-Benchmark: Advanced Phases Implementation Plan (v3.1 to v4.0)

This plan outlines the architectural leap from a text-and-metadata benchmark to a **Multi-Modal Retrieval Benchmark**. We will harmonize heterogeneous data types (Chemicals, Sequences, Papers, Patents) to enable searching across structural and biological similarities.

## User Review Required

> [!IMPORTANT]
> - **Data Volume**: SureChEMBL (~50GB) and NCBI BLAST databases (~100GB+) will significantly increase storage requirements (~500GB+ recommended).
> - **Compute Resources**: Building FAISS indices for 20M+ chemical structures and running BLAST searches is compute-intensive (GPU recommended for FAISS).
> - **Orchestration**: This phase introduces SQL storage (PostgreSQL recommended) to manage complex linking tables between entities.

---

## 1. Data Harmonization Layer (Foundation)

Link disparate identifiers across US, EP, WO, and biomedical databases into a unified BioPAT ID system.

### [NEW] [harmonization/entity_resolver.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/harmonization/entity_resolver.py)
- Unified ID generation: `BP:PAT:{ID}`, `BP:PUB:{ID}`, `BP:CHM:{InChIKey}`, `BP:SEQ:{Hash}`.
- Cross-source resolution: Mapping SMILES to InChIKey, PMID to DOI, etc.

### [NEW] Schema Implementation
- Create unified tables for Patents, Papers, Chemicals, and Sequences.
- Implement linking tables: `patent_chemical_links`, `patent_sequence_links`.

---

## 2. Phase 9: Chemical Structure Matching (v3.1)

Integrate structural similarity search into the retrieval pipeline.

### [DONE] [ingestion/chembl.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/ingestion/chembl.py)
- Official `chembl-webresource-client` wrapper.
- Logic for bioactivity mapping and target resolution.

### [DONE] [ingestion/pubchem.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/ingestion/pubchem.py)
- PUG-REST API client wrapper.
- Logic for Compound ID (CID) resolution and PubMed cross-reference retrieval.

### [DONE] [ingestion/surechembl.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/ingestion/surechembl.py)
- SureChEMBL V3 REST API client wrapper.
- High-performance mapping between patents and chemical structures.

### [DONE] [processing/chemical_index.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/processing/chemical_index.py)
- Morgan/RDKit fingerprint computation.
- **FAISS IndexFlatIP** for Tanimoto similarity approximation via L2-normalized inner product.

### [DONE] [evaluation/hybrid_retrieval.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/evaluation/hybrid_retrieval.py)
- **Alpha-Weighted Hybrid Retriever**: `score = α * text_score + (1 - α) * chemical_score`.

---

## 3. Phase 10: Biological Sequence Similarity (v3.2)

Enable sequence-level prior art findability for biotechnology/antibody patents.

### [DONE] [ingestion/uniprot.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/ingestion/uniprot.py)
- UniProt REST API client wrapper.
- Logic for resolving protein accessions and fetching FASTA sequences.

### [DONE] [ingestion/patent_sequences.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/ingestion/patent_sequences.py)
- NCBI Patent Sequence Database integration via `Biopython.Entrez`.
- Regex extraction of `SEQ ID NO` references from claim text.

### [DONE] [processing/sequence_index.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/processing/sequence_index.py)
- Local **BLAST+** database management (`makeblastdb`).
- FASTA management for combined patent/paper sequence databases.

### [DONE] [evaluation/sequence_search.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/evaluation/sequence_search.py)
- Wrapper for `blastp` (protein) and `blastn` (nucleotide) execution and XML result parsing.

---

## 4. Phase 11: BioPAT v4.0 Trimodal Integration [DONE]

Unification of all retrieval signals into a single "Prior Art Score".

### [DONE] [evaluation/trimodal_retrieval.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/evaluation/trimodal_retrieval.py)
- **Weighted Trimodal Retriever**: Combines Text, Chemical Structure, and Sequence Similarity.
- Support for modality-specific thresholds (e.g., identity >= 0.7 for sequences).

### [MODIFY] Ground Truth
- Extension of `qrels` to include `evidence.match_type` (text, chemical, sequence).

---

## Verification Plan

### Automated Tests
- **Fingerprint Consistency**: Ensure same molecule yields same Morgan/FAISS score.
- **BLAST Wrapper**: Test local `blastp` execution against controlled dummy FASTA.
- **Normalization**: Verify `BP:PAT` format across all jurisdictions.

### Manual Verification
- **Cross-Modality Audit**: Verify cases where a text search fails due to jargon, but InChIKey or BLAST finds a 95%+ match in a paper.
- **Performance Benchmarking**: Measure retrieval latency of Trimodal vs Text-Only (Target: <2s per query).
