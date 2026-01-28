# BioPAT: The Novelty Agent Audit Framework (v4.0)

BioPAT is a specialized benchmark designed to evaluate **Agentic Scientists** and **Novelty Agents** in their ability to determine true biomedical novelty across global scientific literature and patent databases.

## Why BioPAT?

In the current era of **Agentic Science**, AI can generate an abundance of new hypotheses and "innovative" claims at scale. However, the critical bottleneck is **Novelty Determination**: filtering the truly unique from the already known. 

Patent examiners and researchers have long faced this challenge, but as autonomous agents begin to lead discovery, we require a rigorous, multi-modal audit to test if these agents can reliably find the "novelty-killing" evidence hidden in massive, heterogeneous datasets.

- **Agentic Evaluation**: Specifically designed to test the reasoning of autonomous discovery agents.
- **Hypothesis Abundance**: Provides the filter required to validate AI-generated biomedical breakthroughs.
- **The Tri-Modal Gap**: Testing if an agent can bridge the gap between legal "claim language" and the underlying reality of **Chemical Structures** and **Biological Sequences**.
- **High Stakes**: In pharma, missing a single piece of prior art invalidates billion-dollar lifecycles. BioPAT is the stress test for this "Zero-Failure" requirement.

## Key Features

- **Claim-Level Granularity**: BioPAT assesses relevance at the **claim level**, matching the actual unit of legal novelty.
- **Tri-Modal Retrieval (v4.0)**: Supports simultaneous searching across **Text** (lexical/dense), **Chemical Structures** (Morgan/FAISS), and **Biological Sequences** (BLAST).
- **Global Coverage**: Fully normalized data from **USPTO** and **EPO**, mirroring real-world patent family searches.
- **Graded Relevance (0-3)**: Reflects legal certainty, from background context (§103/Category Y) to novelty-destroying anticipation (§102/Category X).
- **Hard Temporal Constraints**: Strictly enforces priority-date filtering to prevent citation leakage.
- **API-First & Reproducible**: Fully automated pipeline with built-in audit logging and SHA256 checksumming.

## BioPAT v4.0: Global Multi-Modal Retrieval

BioPAT v4.0 is a **Global Discovery Engine**, enabling researchers to search for prior art across disparate data modalities.

| Attribute | v1.0 | v2.0 | v3.0 | **v4.0 (Advanced)** |
|-----------|------|------|------|--------------------|
| **Jurisdictions** | US only | US only | Global (US+EP) | **Global + Data Linking** |
| **Corpus** | Papers | Papers + Patents | Papers + Intl Patents | **Unified Multi-Modal Corpus**|
| **Modality** | Text only | Text only | Text only | **Trimodal (Text + Chem + Seq)**|
| **Search Engine**| BM25 | BM25 + Dense | Hybrid Text | **FAISS + BLAST + Weighted Fusion**|
| **Status** | Stable | Stable | Stable | **CODE COMPLETE** |

## Project Roadmap

The BioPAT codebase is structured for iterative expansion, transitioning from a text-based benchmark to a full multi-modal discovery engine.

- **Phases 1-5**: [CORE READY] v2.0: Dual-Corpus search (US Patents + Literature).
- **Phase 6-7**: [CORE READY] v3.0: Global Expansion (EPO) and production-scale retrieval (~1M docs).
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
