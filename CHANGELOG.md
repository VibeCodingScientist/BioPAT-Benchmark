# Changelog

All notable changes to this project will be documented in this file.

## [4.0.0] - 2026-01-30

### Added - Complete SOTA Retrieval Framework

#### Retrieval Methods
- **SPLADE**: Learned sparse retrieval with MLM-based term expansion
- **ColBERT**: Token-level late interaction with MaxSim scoring
- **HyDE**: LLM-based hypothetical document expansion (OpenAI/Anthropic)

#### Learning-to-Rank
- **LambdaMARTRanker**: Gradient boosted trees with NDCG optimization (LightGBM)
- **XGBoostRanker**: Pairwise/listwise ranking objectives
- **RankNetModel**: Neural pairwise ranking
- **ListNetModel**: Neural listwise ranking with cross-entropy
- **EnsembleLTR**: Combined model ensemble
- **RankingFeatures**: 17-feature extraction for combining retrieval signals

#### Biomedical NLP
- **BiomedicalNER**: Chemical, disease, gene/protein entity extraction
- **ExtractedClaim**: Patent claim structure extraction
- **EntityLinker**: Knowledge base linking (MeSH, DrugBank, UniProt)
- **QueryEntityExpander**: Synonym-based query expansion

#### Thesaurus Integration
- **MeSHThesaurus**: Medical Subject Headings lookup
- **ChEBIThesaurus**: Chemical Entities of Biological Interest
- **DrugBankThesaurus**: Drug synonym database
- **UnifiedThesaurus**: Combined expansion interface

#### Molecular Search
- **SubstructureSearcher**: RDKit-based substructure matching
- **ScaffoldSearcher**: Bemis-Murcko scaffold extraction and matching
- **MCSSearcher**: Maximum Common Substructure search
- **MolecularDescriptorCalculator**: Drug-likeness filters (Lipinski, Veber, Ghose)

#### Diversity Reranking
- **MMRDiversifier**: Maximal Marginal Relevance
- **XQuADDiversifier**: Query aspect diversification
- **ClusterDiversifier**: Clustering-based diversification
- **PatentDiversifier**: Patent family and assignee constraints

#### Training & Adaptation
- **DenseRetrieverTrainer**: Contrastive fine-tuning with in-batch negatives
- **CrossEncoderTrainer**: Cross-encoder reranker fine-tuning
- **KnowledgeDistillation**: Teacher-student model distillation

#### Data Acquisition
- **NCBISequenceConnector**: GenBank, Protein, Nucleotide databases
- **SureChEMBLConnector**: Patent-chemical mappings
- Extended **DataAcquisition** with patent sequences and chemicals

### Changed
- Complete README rewrite with comprehensive documentation
- Updated documentation (METHODOLOGY, DATA_SOURCES, REPRODUCIBILITY, API_REFERENCE)
- Removed phase-specific planning files and pipelines

### Removed
- Phase-specific planning documents (`planning/` directory)
- Phase-specific pipelines (`pipeline_phase2.py`, `pipeline_phase3.py`)
- Phase-specific tests (`test_phase2.py`, `test_phase3.py`)
- Alembic database migrations (not needed for benchmark)

## [0.4.1] - 2026-01-28

### Changed
- **WIPO Removal**: Removed active WIPO integration and PATENTSCOPE client. BioPAT now focuses on USPTO and EPO for global patent coverage.
- **PatentsView Multi-Key Support**: Enhanced `PatentsViewClient` to support a pool of API keys with individual rate limiting and rotation, significantly increasing throughput for large-scale ingestion.
- **Configuration**: Updated `ApiConfig` to accept `patentsview_keys` as a list and removed `wipo_api_token`.

## [0.4.0] - 2026-01-27

### Added - Phase 4.0: Data Harmonization Layer (v4.0 Foundation)
- **EntityResolver**: New `EntityResolver` class in `harmonization/entity_resolver.py` for unified ID generation across entity types (Patents, Publications, Chemicals, Sequences). Supports BioPAT ID format: `BP:{TYPE}:{SUBTYPE}:{ID}`.
- **SQL Schema**: New SQLAlchemy models in `harmonization/schema.py` for `Patent`, `Publication`, `Chemical`, `Sequence` entities with comprehensive linking tables (`patent_chemical_links`, `patent_sequence_links`, etc.).
- **EntityLinker**: New `EntityLinker` class in `harmonization/linker.py` for extracting and linking cross-references (DOIs, PMIDs, sequence accessions) from patent/paper text.
- **BidirectionalIndex**: In-memory bidirectional index for fast entity cross-reference lookups with DataFrame import/export.
- **CrossReferenceManager**: High-level manager for bulk cross-reference operations with database persistence.
- **Advanced Configuration**: New `AdvancedConfig`, `HarmonizationConfig`, `ChemicalConfig`, and `SequenceConfig` classes for v4.0 features.

### Changed
- **Config**: Extended `BioPatConfig` with `advanced` section containing harmonization, chemical, and sequence settings.
- **default.yaml**: Added comprehensive v4.0 configuration options for database, chemical fingerprints, and sequence matching.

## [0.3.0] - 2026-01-27

### Added - Phase 6: International Patent Coverage (v3.0)
- **EPO OPS Client**: New `EPOClient` class in `ingestion/epo.py` with OAuth 2.0 authentication for European Patent Office data access. Supports publication retrieval, bibliographic search, citation extraction, and search report parsing.
- **WIPO PATENTSCOPE Client**: New `WIPOClient` class in `ingestion/wipo.py` for PCT/WO patent data access. Supports search, publication retrieval, and International Search Report citation extraction.
- **Patent ID Normalization**: New `patent_ids.py` module in `processing/` with `normalize_patent_id()`, `ParsedPatentId`, `PatentIdNormalizer`, and jurisdiction detection for consistent patent ID handling across US, EP, WO, and other jurisdictions.
- **EP Search Report Parser**: New `ep_citations.py` module in `groundtruth/` with `EPSearchReportParser` class for extracting citations from European search reports and mapping EP categories (X, Y, A) to relevance scores.
- **International Corpus Assembly**: New `international_patents.py` module in `processing/` with `InternationalCorpusBuilder` class for merging US, EP, and WO patents into a unified corpus with family deduplication.
- **Per-Jurisdiction Metrics**: New `compute_metrics_by_jurisdiction()` and `compute_cross_jurisdiction_analysis()` methods in `MetricsComputer` for breakdown reporting by patent jurisdiction.
- **Phase 6 Configuration**: New `Phase6Config` and `JurisdictionConfig` classes with options for `include_us`, `include_ep`, `include_wo`, `deduplicate_families`, and publication date filtering.

### Changed
- **ApiConfig**: Extended with `epo_consumer_key`, `epo_consumer_secret`, and `wipo_api_token` fields for international API access.
- **Config**: Extended `BioPatConfig` with `phase6` settings for v3.0 features.
- **Metrics**: Added `JURISDICTION_US`, `JURISDICTION_EP`, `JURISDICTION_WO`, and `ALL_JURISDICTIONS` constants.
- **Evaluation Reports**: New `format_jurisdiction_report()` and `format_full_international_report()` methods for comprehensive v3.0 reporting.

## [0.2.0] - 2026-01-27

### Added - Phase 5: Full Novelty Benchmark (v2.0)
- **Dual-Corpus Support**: Extended BEIR formatter with `doc_type` field (`paper` | `patent`) and `date` field for corpus entries.
- **Prior Patent Selection**: New `PriorPatentSelector` class in `processing/prior_patents.py` for identifying prior patent candidates from Office Action citations, applicant citations, and IPC-based hard negatives.
- **Patent Corpus Assembly**: New `get_patents_for_corpus()` and `search_patents_by_ids()` methods in `PatentsViewClient` for efficient batch retrieval.
- **Dual-Corpus Relevance**: New `assign_dual_corpus_relevance()` and `create_dual_corpus_qrels()` methods in `RelevanceAssigner` for unified relevance assignment across document types.
- **Cross-Type Metrics**: New `compute_metrics_by_doc_type()` and `compute_cross_type_retrieval_metrics()` methods in `MetricsComputer` for multi-dimensional evaluation reporting.
- **Phase 5 Configuration**: New `Phase5Config` and `CorpusConfig` classes with options for `include_papers`, `include_patents`, `max_prior_patents`, and IPC negative sampling.

### Changed
- **BEIRFormatter**: Added `format_dual_corpus()` method and `create_patent_corpus_text()` static method for patent corpus text generation (abstract + first independent claim).
- **Config**: Extended `BioPatConfig` with `phase5` settings for v2.0 features.
- **Validation**: Updated `validate_output()` to count papers vs patents and detect dual-corpus benchmarks.

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
