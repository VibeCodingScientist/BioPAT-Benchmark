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

- **Claim-Level Granularity**: Unlike full-document benchmarks, BioPAT assesses relevance at the **claim level**, matching the actual unit of legal novelty.
- **Graded Relevance (0-3)**: Moves beyond binary "relevant/not-relevant" to reflect real-world prior art rejections (§102 Novelty-destroying vs. §103 Obviousness).
- **Hard Temporal Constraints**: Strictly enforces that prior art must predate the patent's priority date to be valid.
- **API-First & Reproducible**: Built using public APIs (PatentsView, OpenAlex) with full SHA256 checksumming and deterministic sampling for academic audit.

## BioPAT v2.0: Full Novelty Search

BioPAT is evolving from a literature-only benchmark to a **Full Prior Art retrieval benchmark**, matching the real-world complexity of patent examination.

| Attribute | v1.0 (Phases 1-4) | v2.0 (Phase 5) | v3.0 (Phase 6+) |
|-----------|-------------------|-----------------|-----------------|
| **Jurisdictions** | US only | US only | **Global (US+EP+WO)** |
| **Corpus** | papers | papers + patents | papers + intl patents |
| **Ground Truth**| NPL citations | NPL + US patents | NPL + Global citations |
| **Task** | Patent-to-Literature | Full Prior Art | **Global Novelty retrieval** |
| **Size** | ~200K | ~500K | **~1M documents** |

## Project Roadmap

- **Phases 1-3**: [DONE] Baseline implementation, Examiner-grade ground truth, and Evaluation framework.
- **Phases 4-5**: [DONE] **BioPAT v2.0: Full Novelty Search** - Expansion to include US patents.
- **Phase 6**: [IN PROGRESS] **BioPAT v3.0: Global Prior Art** - International coverage (EP, WO) and multi-jurisdictional evaluation.
- **Phase 7**: [PLANNED] Formal academic publication and public dataset release (HuggingFace & Zenodo).

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
