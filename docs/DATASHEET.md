# BioPAT Datasheet (v4.0)

Following the "Datasheets for Datasets" framework (Gebru et al., 2018).

## 1. Motivation
- **Why was the dataset created?** To benchmark information retrieval in the high-stakes biomedical domain, where valid prior art may be hidden in text, chemical formulas, or biological sequences across global jurisdictions.
- **Who created it?** VibeCodingScientist and the BioPAT-Benchmark contributors.

## 2. Composition
- **What do instances represent?**
    - **Queries**: Independent patent claims from USPTO, EPO, and WIPO.
    - **Documents**: Scientific paper abstracts (OpenAlex) and prior patent descriptions.
    - **Entities**: Extracted chemical structures (SMILES) and biological sequences (AA/NT).
- **How many instances are there?**
    - ~25,000 High-confidence queries.
    - ~1,000,000 Corpus documents.
    - ~500,000 Harmonized chemical/sequence entities.
- **Is there a label?** Yes, graded relevance (0-3) based on examiner novelty §102/Category X, obviousness §103/Category Y, and structural similarity (Tanimoto/BLAST).

## 3. Collection Process
- **How was the data collected?** Federated API polling + Bulk data ingestion.
    - Patent Meta: PatentsView, EPO OPS, WIPO.
    - Legal Evidence: USPTO OARD, EP Search Reports.
    - Entities: SureChEMBL, UniProt, NCBI.
- **Who was involved?** An automated, reproducible pipeline audit-trailed by hashes and manifests.

## 4. Preprocessing/Cleaning/Labeling
- **Entities**: Canonicalization via RDKit (InChIKey generation) and sequence hashing.
- **Temporal Constraint**: Strict global filter ensuring all prior art ($d$) predates patent priority ($p$): $\text{PubDate}(d) < \text{PriorityDate}(p)$.
- **Deduplication**: Multi-jurisdictional patent family resolution to avoid repetitive qrels.

## 5. Distribution
- **How is it distributed?** Code via GitHub; Dataset via Zenodo/HuggingFace.
- **License**: CC-BY-SA 4.0.
- **DOI**: `[PENDING v4.0 FINAL RELEASE]`
