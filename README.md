# BioPAT: Biomedical Patent-to-Article & Patent-to-Patent Retrieval Benchmark

BioPAT is the first dedicated benchmark dataset designed to evaluate information retrieval systems in the specialized task of finding relevant **scientific literature and earlier patents** given biomedical patent claims.

## What is BioPAT?

In the pharmaceutical and biotechnology industries, prior art search is critical. Patent examiners and researchers must determine if a claimed invention is truly novel by searching the vast body of scientific literature **and** earlier patents. While previous benchmarks focused on either patent-to-patent (CLEF-IP) or general literature retrieval (BEIR), BioPAT fills the critical gap for **Full Prior Art retrieval**, where patent claims (queries) are matched against a comprehensive, heterogeneous corpus of both research papers and patents.

## Why BioPAT?

- **Real-World Fidelity**: Mirrors the actual workflow of patent examiners who must search multiple source types to assess legal novelty.
- **The NPL & Patent Gap**: Biomedical patents require a unique cross-domain retrieval capability to bridge legal jargon, chemical indexing, and scientific terminology.
- **High Stakes**: Patent validity directly impacts billion-dollar drug lifecycles; missing a single piece of prior art in either papers or patents can be catastrophic.
- **Domain Difficulty**: Provides a rigorous test for systems attempting to generalize across scientific disciplines and legal domains.

## Key Features

- **Claim-Level Granularity**: BioPAT assesses relevance at the **claim level**, matching the actual unit of legal novelty.
- **Tri-Modal Retrieval (v4.0)**: Supports simultaneous searching across **Text** (lexical/dense), **Chemical Structures** (Morgan/FAISS), and **Biological Sequences** (BLAST).
- **Global Coverage**: Fully normalized data from **USPTO**, **EPO**, and **WIPO**, mirroring real-world patent family searches.
- **Graded Relevance (0-3)**: Reflects legal certainty, from background context (§103/Category Y) to novelty-destroying anticipation (§102/Category X).
- **Hard Temporal Constraints**: Strictly enforces priority-date filtering to prevent citation leakage.
- **API-First & Reproducible**: Fully automated pipeline with built-in audit logging and SHA256 checksumming.

## BioPAT v4.0: Global Multi-Modal Retrieval

BioPAT v4.0 is a **Global Discovery Engine**, enabling researchers to search for prior art across disparate data modalities.

| Attribute | v1.0 | v2.0 | v3.0 | **v4.0 (Advanced)** |
|-----------|------|------|------|--------------------|
| **Jurisdictions** | US only | US only | Global (US+EP+WO) | **Global + Data Linking** |
| **Corpus** | Papers | Papers + Patents | Papers + Intl Patents | **Unified Multi-Modal Corpus**|
| **Modality** | Text only | Text only | Text only | **Trimodal (Text + Chem + Seq)**|
| **Search Engine**| BM25 | BM25 + Dense | Hybrid Text | **FAISS + BLAST + Weighted Fusion**|
| **Status** | Stable | Stable | Stable | **CODE COMPLETE** |

## Project Roadmap

The BioPAT codebase is structured for iterative expansion, transitioning from a text-based benchmark to a full multi-modal discovery engine.

- **Phases 1-5**: [CORE READY] v2.0: Dual-Corpus search (US Patents + Literature).
- **Phase 6-7**: [CORE READY] v3.0: Global Expansion (EPO & WIPO) and production-scale retrieval (~1M docs).
- **Phases 8-11**: [CODE COMPLETE] v4.0: Multi-Modal Discovery - Chemical (Morgan/FAISS), Sequence (BLAST+), and Tri-modal fusion.
- **Future**: [PLANNED] **Public Release** via HuggingFace/Zenodo.
- **Future**: [PLANNED] **Public Release** via HuggingFace/Zenodo.

## Quick Start

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for installation and build instructions.

## Documentation

- [Methodology](docs/METHODOLOGY.md): Detailed construction process and relevance definitions.
- [Data Sources](docs/DATA_SOURCES.md): Inventory of sources, versions, and checksums.
- [Reproducibility](docs/REPRODUCIBILITY.md): Step-by-step guide to reproducing the benchmark.
- [Datasheet](docs/DATASHEET.md): Gebru et al. standard documentation.

## Citation

```bibtex
@dataset{biopat2026,
  author    = {[Your Name]},
  title     = {{BioPAT}: A Biomedical Patent-to-Article Retrieval Benchmark},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/VibeCodingScientist/BioPAT-Benchmark}
}
```
