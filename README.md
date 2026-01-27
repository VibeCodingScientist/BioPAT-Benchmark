# BioPAT: Biomedical Patent-to-Article Retrieval Benchmark

BioPAT is the first dedicated benchmark dataset designed to evaluate information retrieval systems in the specialized task of finding relevant scientific literature given biomedical patent claims.

## What is BioPAT?

In the pharmaceutical and biotechnology industries, prior art search is critical. Patent examiners and researchers must determine if a claimed invention is truly novel by searching both earlier patents and the vast body of scientific literature. While benchmarks exist for patent-to-patent retrieval (CLEF-IP) and general literature retrieval (BEIR), BioPAT fills the critical gap for **cross-domain retrieval** where patent claims (queries) must be matched against scientific research papers (corpus).

## Why BioPAT?

- **The NPL Gap**: Biomedical patents cite non-patent literature (NPL) at 3-5x the rate of other domains.
- **High Stakes**: Patent validity directly impacts billion-dollar drug lifecycles and global access to medicine.
- **Domain Difficulty**: Bridging the linguistic gap between legal patent jargon and scientific terminology is a unique challenge for AI and IR systems.

## Key Features

- **Claim-Level Granularity**: Unlike full-document benchmarks, BioPAT assesses relevance at the **claim level**, matching the actual unit of legal novelty.
- **Graded Relevance (0-3)**: Moves beyond binary "relevant/not-relevant" to reflect real-world prior art rejections (§102 Novelty-destroying vs. §103 Obviousness).
- **Hard Temporal Constraints**: Strictly enforces that prior art must predate the patent's priority date to be valid.
- **API-First & Reproducible**: Built using public APIs (PatentsView, OpenAlex) with full SHA256 checksumming and deterministic sampling for academic audit.

## Project Roadmap

- **Phase 1**: [DONE] Minimum Viable Benchmark using citation-based ground truth.
- **Phase 2**: [DONE] Enhanced ground truth using expert examiner judgments from USPTO Office Actions.
- **Phase 3**: [DONE] Comprehensive baseline evaluation with Lexical, Dense, and Hybrid models.
- **Phase 4**: [PLANNED] Formal academic publication and public dataset release.

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
