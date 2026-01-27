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
    patentsview_key: Optional[str] = Field(default=None)
    patentsview_key_2: Optional[str] = Field(default=None)
    ncbi_key: Optional[str] = Field(default=None)
    openalex_mailto: Optional[str] = Field(default=None)
    use_bulk_data: bool = Field(default=True)

    def __init__(self, **data):
        # Map user-friendly names from YAML to internal field names
        field_map = {
            "patentsview_api_key": "patentsview_key",
            "patentsview_api_key_2": "patentsview_key_2",
            "ncbi_api_key": "ncbi_key",
            "openalex_email": "openalex_mailto"
        }
        for yaml_key, internal_key in field_map.items():
            if yaml_key in data and internal_key not in data:
                data[internal_key] = data.pop(yaml_key)

        super().__init__(**data)
        # Load from environment variables if not set
        if self.patentsview_key is None:
            object.__setattr__(self, 'patentsview_key', os.environ.get("PATENTSVIEW_API_KEY"))
        if self.ncbi_key is None:
            object.__setattr__(self, 'ncbi_key', os.environ.get("NCBI_API_KEY"))
        if self.openalex_mailto is None:
            object.__setattr__(self, 'openalex_mailto', os.environ.get("OPENALEX_MAILTO"))


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


class BioPatConfig(BaseModel):
    """Main configuration for BioPAT benchmark."""
    phase: str = "phase1"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    phase1: Phase1Config = Field(default_factory=Phase1Config)
    phase5: Phase5Config = Field(default_factory=Phase5Config)

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
