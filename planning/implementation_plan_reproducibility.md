# BioPAT-Benchmark: Open Science & Reproducibility Implementation Plan

This plan addresses the requirements for transparency, reproducibility, and academic rigor. It ensures that the benchmark can be audited and reproduced by third-party researchers.

## User Review Required

> [!IMPORTANT]
> - **License Choice**: The user specified CC-BY-NC-SA 4.0 or CC-BY 4.0. Both allow research use, but NC (Non-Commercial) restricts commercial exploitation. I will initialize with CC-BY-NC-SA 4.0 as per the detailed requirement.
> - **DOI / Zenodo**: The citation format includes placeholder DOIs. Permanent DOIs must be obtained after the project is archived on Zenodo.

## Proposed Changes

### 1. Repository Structure
We will align the project structure with the documentation standards.

#### [NEW] [docs/DATA_SOURCES.md](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/docs/DATA_SOURCES.md)
Detailed inventory of all data sources, versions, and checksums.

#### [NEW] [docs/METHODOLOGY.md](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/docs/METHODOLOGY.md)
Methodological detail matching future paper sections.

#### [NEW] [docs/REPRODUCIBILITY.md](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/docs/REPRODUCIBILITY.md)
Clear, step-by-step instructions for reproducing the benchmark from scratch.

#### [NEW] [docs/DATASHEET.md](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/docs/DATASHEET.md)
Standardized dataset documentation following Gebru et al.

#### [NEW] [CITATION.cff](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/CITATION.cff)
Machine-readable citation file.

#### [NEW] [CONTRIBUTING.md](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/CONTRIBUTING.md)
Guidelines for external contributors.

#### [NEW] [CHANGELOG.md](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/CHANGELOG.md)
Version history and major methodology changes.

---

### 2. Code Modifications (Reproducibility)
#### [MODIFY] [ingestion/patentsview.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/ingestion/patentsview.py) & Others
- Implement `log_download` with SHA256 checksumming.
- Ensure all API calls are logged with timestamps and counts.

#### [MODIFY] [benchmark/sampling.py](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/src/biopat/benchmark/sampling.py)
- Enforce fixed random seeds (e.g., `seed=42`) for all sampling and splitting logic.
- Ensure deterministic stratification.

---

### 3. Documentation Updates
#### [MODIFY] [README.md](file:///Users/LV/Library/Mobile%20Documents/com~apple~CloudDocs/VibeCoding/BioPAT-Benchmark/README.md)
Update with quick start, citation (BibTeX), and links to the new `docs/` folder.

## Verification Plan

### Automated Tests
- **Checksum Verification**: Test that re-downloading a known file produces the same checksum as recorded in `data/manifest.json`.
- **Determinism Test**: Run sampling twice with the same seed and verify that the output splits are identical.

### Manual Verification
- **Handoff Verification**: Run `pip install -e .` in a fresh environment to ensure all scripts work.
- **Documentation Audit**: Verify that all placeholders in `docs/*.md` are ready for final population during the production run.
