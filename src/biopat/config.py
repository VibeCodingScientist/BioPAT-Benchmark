"""BioPAT configuration management."""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
import yaml
import os
from pathlib import Path

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class PathsConfig(BaseModel):
    """Configuration for data paths."""
    data_dir: Path = Field(default=Path("data"))
    raw_dir: Path = Field(default=Path("data/raw"))
    processed_dir: Path = Field(default=Path("data/processed"))
    cache_dir: Path = Field(default=Path("data/cache"))
    benchmark_dir: Path = Field(default=Path("data/benchmark"))

    class Config:
        arbitrary_types_allowed = True

    def create_dirs(self):
        """Create all data directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)


class ApiConfig(BaseModel):
    """Configuration for API credentials."""
    patentsview_keys: List[str] = Field(default_factory=list)
    ncbi_key: Optional[str] = Field(default=None)
    openalex_mailto: Optional[str] = Field(default=None)
    use_bulk_data: bool = Field(default=True)
    # Phase 6 (v3.0): International patent APIs
    epo_consumer_key: Optional[str] = Field(default=None)
    epo_consumer_secret: Optional[str] = Field(default=None)

    def __init__(self, **data):
        # Map user-friendly names from YAML to internal field names
        field_map = {
            "patentsview_api_key": "patentsview_keys", # Map single key to list
            "patentsview_api_keys": "patentsview_keys", # Map list of keys
            "ncbi_api_key": "ncbi_key",
            "openalex_email": "openalex_mailto"
        }
        for yaml_key, internal_key in field_map.items():
            if yaml_key in data and internal_key not in data:
                # If it's a single key for patentsview_keys, convert to list
                if internal_key == "patentsview_keys" and isinstance(data[yaml_key], str):
                    data[internal_key] = [data.pop(yaml_key)]
                else:
                    data[internal_key] = data.pop(yaml_key)

        super().__init__(**data)
        # Load from environment variables if not set
        if not self.patentsview_keys:
            keys_str = os.environ.get("PATENTSVIEW_API_KEYS")
            if keys_str:
                object.__setattr__(self, 'patentsview_keys', [k.strip() for k in keys_str.split(",")])
            elif os.environ.get("PATENTSVIEW_API_KEY"):
                # Legacy single key support
                object.__setattr__(self, 'patentsview_keys', [os.environ.get("PATENTSVIEW_API_KEY")])

        if self.ncbi_key is None:
            object.__setattr__(self, 'ncbi_key', os.environ.get("NCBI_API_KEY"))
        if self.openalex_mailto is None:
            object.__setattr__(self, 'openalex_mailto', os.environ.get("OPENALEX_MAILTO"))
        # Phase 6: EPO credentials
        if self.epo_consumer_key is None:
            object.__setattr__(self, 'epo_consumer_key', os.environ.get("EPO_CONSUMER_KEY"))
        if self.epo_consumer_secret is None:
            object.__setattr__(self, 'epo_consumer_secret', os.environ.get("EPO_CONSUMER_SECRET"))


class Phase1Config(BaseModel):
    """Configuration for Phase 1 benchmark construction."""
    target_patent_count: int = 2000
    target_query_count: int = 5000
    min_citations: int = 3
    ros_confidence_threshold: int = 8
    ipc_prefixes: List[str] = ["A61", "C07", "C12"]
    seed: int = 42


class CorpusConfig(BaseModel):
    """Configuration for dual-corpus (v2.0)."""
    include_papers: bool = True
    include_patents: bool = False
    max_prior_patents: Optional[int] = None
    include_ipc_negatives: bool = False
    max_ipc_negatives_per_query: int = 10


class Phase5Config(BaseModel):
    """Configuration for Phase 5 (v2.0) Full Novelty Benchmark."""
    enabled: bool = False
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    target_corpus_size: int = 500000
    seed: int = 42


class JurisdictionConfig(BaseModel):
    """Configuration for jurisdiction inclusion (v3.0)."""
    include_us: bool = True
    include_ep: bool = True
    include_wo: bool = False
    max_patents_per_jurisdiction: Optional[int] = None


class Phase6Config(BaseModel):
    """Configuration for Phase 6 (v3.0) International Patent Coverage."""
    enabled: bool = False
    jurisdictions: JurisdictionConfig = Field(default_factory=JurisdictionConfig)
    # IPC filtering for biomedical domain
    ipc_prefixes: List[str] = ["A61", "C07", "C12"]
    # Publication date range (YYYYMMDD format for EPO, YYYY-MM-DD for WIPO)
    publication_date_from: Optional[str] = None
    publication_date_to: Optional[str] = None
    # Family deduplication: avoid including US+EP+WO versions of same patent
    deduplicate_families: bool = True
    # Target corpus size (~500K-1M for v3.0)
    target_corpus_size: int = 1000000
    # Random seed
    seed: int = 42


class DatabaseConfig(BaseModel):
    """Configuration for SQL database (v4.0 harmonization)."""
    backend: str = "sqlite"  # sqlite or postgresql
    path: str = "data/biopat.db"  # For SQLite
    host: str = "localhost"  # For PostgreSQL
    port: int = 5432
    user: str = "biopat"
    password: str = ""
    database: str = "biopat"


class HarmonizationConfig(BaseModel):
    """Configuration for data harmonization (v4.0)."""
    enabled: bool = False
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    # Entity resolution settings
    use_rdkit: bool = True  # Use RDKit for chemical canonicalization
    cache_size: int = 10000  # Resolution cache size
    # Cross-reference extraction
    extract_doi_from_text: bool = True
    extract_pmid_from_text: bool = True
    extract_sequences_from_claims: bool = True


class ChemicalConfig(BaseModel):
    """Configuration for chemical structure matching (v3.1)."""
    enabled: bool = False
    # SureChEMBL data path
    surechembl_path: Optional[str] = None
    # Fingerprint settings
    fingerprint_type: str = "morgan"  # morgan or rdkit
    fingerprint_bits: int = 2048
    morgan_radius: int = 2
    # FAISS index settings
    use_gpu: bool = False
    faiss_index_type: str = "flat"  # flat, ivf, hnsw
    # Similarity thresholds for ground truth
    tanimoto_high: float = 0.85  # High similarity
    tanimoto_medium: float = 0.7  # Medium similarity
    tanimoto_low: float = 0.5  # Low similarity


class SequenceConfig(BaseModel):
    """Configuration for biological sequence matching (v3.2)."""
    enabled: bool = False
    # BLAST+ settings
    blast_db_path: Optional[str] = None
    blast_type: str = "blastp"  # blastp or blastn
    evalue_threshold: float = 1e-5
    # Sequence identity thresholds for ground truth
    identity_high: float = 0.95  # High identity
    identity_medium: float = 0.7  # Medium identity
    identity_low: float = 0.5  # Low identity


class AdvancedConfig(BaseModel):
    """Configuration for Advanced Phases (v4.0)."""
    harmonization: HarmonizationConfig = Field(default_factory=HarmonizationConfig)
    chemical: ChemicalConfig = Field(default_factory=ChemicalConfig)
    sequence: SequenceConfig = Field(default_factory=SequenceConfig)
    # Trimodal retrieval weights
    text_weight: float = 0.5
    chemical_weight: float = 0.3
    sequence_weight: float = 0.2


class BioPatConfig(BaseModel):
    """Main configuration for BioPAT benchmark."""
    phase: str = "phase1"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    phase1: Phase1Config = Field(default_factory=Phase1Config)
    phase5: Phase5Config = Field(default_factory=Phase5Config)
    phase6: Phase6Config = Field(default_factory=Phase6Config)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)

    @classmethod
    def load(cls, config_path: str = "configs/default.yaml") -> "BioPatConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            Loaded configuration.
        """
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        return cls()

if __name__ == "__main__":
    config = BioPatConfig()
    config.paths.create_dirs()
    print(f"Configuration loaded for phase: {config.phase}")
    print(f"Data directory: {config.paths.data_dir}")
