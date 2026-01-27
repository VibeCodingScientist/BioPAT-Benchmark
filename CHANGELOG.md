# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2026-01-27

### Added
- **Checksum Engine**: New `ChecksumEngine` class in `reproducibility.py` that computes and records SHA256 hashes for all downloaded files. Integrated into RoS and Office Action loaders.
- **Audit Logging**: New `AuditLogger` class that tracks API calls with timestamps, parameters, and response counts. Integrated into PatentsView, OpenAlex, RoS, and Office Action clients.
- **Strict Determinism**: Added `REPRODUCIBILITY_SEED` constant (42) used by `BenchmarkSampler` and `DatasetSplitter`. Warnings logged when non-standard seeds are used.
- **Manifest Support**: New `create_manifest()` function generates `manifest.json` files tracking downloads, checksums, and API calls for full reproducibility auditing.

### Changed
- **Dependency Pinning**: Updated `pyproject.toml` with specific version ranges for all dependencies to prevent future breakage (e.g., `polars>=1.0.0,<2.0`).
- **Ingestion Clients**: Updated constructors for `RelianceOnScienceLoader`, `OfficeActionLoader`, `PatentsViewClient`, and `OpenAlexClient` to accept optional `checksum_engine` and `audit_logger` parameters.

### Fixed
- Duplicate type annotations in `patentsview.py` and `openalex.py`.

## [0.1.0] - 2026-01-27

### Added
- Phase 1: Minimum Viable Benchmark infrastructure.
- Phase 2: Examiner-grade ground truth logic.
- Phase 3: Comprehensive evaluation framework (Dense/Hybrid).
- Open Science & Reproducibility documentation.
- Python 3.12 environment setup and verification.
- Initial project skeletons and test suite passing (76/76).
