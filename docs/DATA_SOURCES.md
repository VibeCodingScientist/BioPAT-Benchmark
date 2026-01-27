# BioPAT Data Sources (v4.0)

This document tracks all data sources, versions, and access requirements used in the construction and maintenance of the BioPAT Multi-Modal benchmark.

## 1. Global Patent Sources

### 1.1 USPTO (United States)
- **Primary Access**: PatentsView API (2026-Q1 release).
- **Ground Truth**: USPTO Office Action Research Dataset (OARD).
- **Scope**: Direct examiner rejections (§102/§103).

### 1.2 EPO OPS (Europe)
- **Primary Access**: Open Patent Services (OPS) v3.2 API.
- **Ground Truth**: EP Search Reports (Register Service).
- **Scope**: Categorical relevance (X, Y, A).

### 1.3 WIPO PATENTSCOPE (International)
- **Primary Access**: PATENTSCOPE JSON/XML Interface.
- **Ground Truth**: PCT International Search Reports (ISR).
- **Scope**: Global novelty retrieval.

## 2. Literature & Linking

### 2.1 OpenAlex
- **Access**: REST API (mailto: enabled).
- **Scope**: Comprehensive corpus of scientific works, affiliations, and MeSH terms.

### 2.2 Reliance on Science (PCS)
- **URL**: [https://zenodo.org/records/7996195](https://zenodo.org/records/7996195)
- **Version**: v2023.1 (Marx & Fuegi).
- **Rationale**: Seed citations for determining initial relevance confidence.

### 2.3 CrossRef
- **Access**: DOI REST API.
- **Scope**: Entity resolution between DOIs and other publication ids.

## 3. Multi-Modal Entity Sources (v4.0)

### 3.1 SureChEMBL (Chemical)
- **URL**: [https://www.surechembl.org/](https://www.surechembl.org/)
- **Access**: Quarterly Bulk Dumps (FTP).
- **Scope**: ~20M Chemical structures extracted from patent text.

### 3.2 PubChem (Chemical-Literature)
- **Access**: PUG-REST API.
- **Scope**: Chemical properties and PubMed-to-Structure cross-references.

### 3.3 UniProt / NCBI (Biological Sequences)
- **UniProt**: REST API for protein sequence annotations and cross-refs.
- **NCBI nuccore**: Patent Sequence Database via Entrez API.
- **Scope**: SEQ ID NO extraction and alignment (BLAST).

---

## Data Inventory Manifest

| Dataset | Format | Version | Size (est) |
|---------|--------|---------|------------|
| PatentsView | TSV/JSON | 2026-Q1 | 150GB |
| Office Actions | XML/CSV | 2017 Rel | 20GB |
| Reliance on Science | CSV.gz | v2023 | 5GB |
| SureChEMBL | SDF/TXT | Quarterly | 60GB |
| UniProt/SwissProt | FASTA | 2026_01 | 1GB |
| BioPAT Index | FAISS/SQL| 0.4.x | 200GB |
