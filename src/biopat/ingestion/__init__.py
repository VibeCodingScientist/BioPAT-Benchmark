"""Data ingestion modules for BioPAT."""

from .ros import RelianceOnScienceLoader
from .patentsview import PatentsViewClient
from .openalex import OpenAlexClient

# Re-export reproducibility utilities for convenience
from biopat.reproducibility import ChecksumEngine, AuditLogger

__all__ = [
    "RelianceOnScienceLoader",
    "PatentsViewClient",
    "OpenAlexClient",
    "ChecksumEngine",
    "AuditLogger",
]
