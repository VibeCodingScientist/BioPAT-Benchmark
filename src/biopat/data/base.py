"""Base classes for data connectors.

Provides common functionality for all data source connectors:
- Rate limiting
- Caching
- Error handling
- Async HTTP client management
"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import httpx, fall back gracefully
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available. Install with: pip install httpx")


@dataclass
class ConnectorConfig:
    """Configuration for data connectors."""

    api_key: Optional[str] = None
    timeout: float = 30.0
    rate_limit: float = 3.0  # requests per second
    cache_dir: Optional[Path] = None
    cache_ttl: int = 86400  # 24 hours


@dataclass
class Document:
    """Standardized document from any source."""

    id: str
    source: str  # pubmed, biorxiv, patent, uniprot, etc.
    title: str
    text: str  # abstract or full text

    # Optional metadata
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None

    # Domain-specific fields
    smiles: Optional[str] = None  # Chemical structure
    sequence: Optional[str] = None  # Protein/nucleotide sequence
    sequence_type: Optional[str] = None  # protein, nucleotide

    # Patent-specific
    claims: List[str] = field(default_factory=list)
    ipc_codes: List[str] = field(default_factory=list)
    assignees: List[str] = field(default_factory=list)
    filing_date: Optional[str] = None

    # Raw data for debugging
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "text": self.text,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "url": self.url,
            "smiles": self.smiles,
            "sequence": self.sequence,
            "sequence_type": self.sequence_type,
            "claims": self.claims,
            "ipc_codes": self.ipc_codes,
            "assignees": self.assignees,
            "filing_date": self.filing_date,
            "metadata": self.metadata,
        }

    def to_corpus_entry(self) -> Dict[str, Any]:
        """Convert to BioPAT corpus format."""
        return {
            "title": self.title,
            "text": self.text,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "url": self.url,
            "source": self.source,
        }


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rate: float = 3.0):
        """
        Initialize rate limiter.

        Args:
            rate: Maximum requests per second
        """
        self.rate = rate
        self.min_interval = 1.0 / rate if rate > 0 else 0
        self._lock = asyncio.Lock()
        self._last_request = 0.0

    async def acquire(self) -> None:
        """Wait until a request can be made."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last_request = time.monotonic()


class DiskCache:
    """Simple disk-based cache."""

    def __init__(self, cache_dir: Path, ttl: int = 86400):
        """
        Initialize disk cache.

        Args:
            cache_dir: Directory for cache files
            ttl: Time-to-live in seconds
        """
        self.cache_dir = cache_dir
        self.ttl = ttl
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        """Get cache file path for key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        path = self._get_path(key)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            cached_time = data.get("_cached_at", 0)
            if time.time() - cached_time > self.ttl:
                path.unlink()
                return None
            return data.get("value")
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        path = self._get_path(key)
        try:
            data = {"value": value, "_cached_at": time.time()}
            path.write_text(json.dumps(data))
        except Exception as e:
            logger.warning(f"Failed to cache {key}: {e}")


class BaseConnector(ABC):
    """Base class for all data connectors."""

    def __init__(self, config: Optional[ConnectorConfig] = None):
        """
        Initialize connector.

        Args:
            config: Connector configuration
        """
        self.config = config or ConnectorConfig()
        self._client: Optional["httpx.AsyncClient"] = None
        self._rate_limiter = RateLimiter(self.config.rate_limit)

        # Setup cache
        self._cache: Optional[DiskCache] = None
        if self.config.cache_dir:
            cache_path = self.config.cache_dir / self.source_name
            self._cache = DiskCache(cache_path, self.config.cache_ttl)

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return source name for this connector."""
        pass

    async def _get_client(self) -> "httpx.AsyncClient":
        """Get or create HTTP client."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required. Install with: pip install httpx")

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.config.timeout)
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _cache_get(self, key: str) -> Optional[Any]:
        """Get from cache."""
        if self._cache:
            return self._cache.get(key)
        return None

    def _cache_set(self, key: str, value: Any) -> None:
        """Set in cache."""
        if self._cache:
            self._cache.set(key, value)

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 100,
        **kwargs,
    ) -> List[Document]:
        """
        Search for documents.

        Args:
            query: Search query
            limit: Maximum results
            **kwargs: Source-specific parameters

        Returns:
            List of Document objects
        """
        pass

    async def health_check(self) -> bool:
        """Check if the data source is accessible."""
        return True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
