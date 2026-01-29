"""bioRxiv/medRxiv Connector for preprints.

Provides access to bioRxiv and medRxiv preprint servers.
Useful for finding cutting-edge research not yet in PubMed.

API Documentation: https://api.biorxiv.org/
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from biopat.data.base import BaseConnector, ConnectorConfig, Document

logger = logging.getLogger(__name__)

# Try to import httpx
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class BioRxivConnector(BaseConnector):
    """Connector for bioRxiv/medRxiv API.

    Note: bioRxiv API supports date ranges, not full-text search.
    We fetch recent preprints and filter client-side.

    Example:
        ```python
        async with BioRxivConnector() as biorxiv:
            # Search recent preprints
            preprints = await biorxiv.search(
                "CRISPR gene therapy",
                limit=50,
                date_range="2024-01-01/2024-12-31"
            )
        ```
    """

    BIORXIV_URL = "https://api.biorxiv.org/details/biorxiv"
    MEDRXIV_URL = "https://api.biorxiv.org/details/medrxiv"

    def __init__(
        self,
        server: str = "biorxiv",  # "biorxiv" or "medrxiv"
        config: Optional[ConnectorConfig] = None,
    ):
        """
        Initialize bioRxiv connector.

        Args:
            server: "biorxiv" or "medrxiv"
            config: Connector configuration
        """
        if config is None:
            config = ConnectorConfig(rate_limit=5.0)
        super().__init__(config)

        self.server = server
        self.base_url = self.MEDRXIV_URL if server == "medrxiv" else self.BIORXIV_URL

    @property
    def source_name(self) -> str:
        return self.server

    def _build_interval(self, date_range: Optional[str]) -> str:
        """Build date interval for API."""
        if not date_range:
            return "30d"  # Default: last 30 days

        # Accept YYYY-YYYY -> YYYY-01-01/YYYY-12-31
        match = re.match(r"^(\d{4})-(\d{4})$", date_range)
        if match:
            start_year, end_year = match.groups()
            return f"{start_year}-01-01/{end_year}-12-31"

        # Accept full date range
        if re.match(r"^\d{4}-\d{2}-\d{2}/\d{4}-\d{2}-\d{2}$", date_range):
            return date_range

        return "30d"

    async def search(
        self,
        query: str,
        limit: int = 100,
        date_range: Optional[str] = None,
        **kwargs,
    ) -> List[Document]:
        """
        Search bioRxiv/medRxiv preprints.

        Args:
            query: Search query (filtered client-side)
            limit: Maximum results
            date_range: Date filter (e.g., "2020-2024" or "2024-01-01/2024-06-30")

        Returns:
            List of Document objects
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required. Install with: pip install httpx")

        interval = self._build_interval(date_range)

        # Fetch preprints
        items, error = await self._fetch_results(interval, limit * 5)  # Fetch extra for filtering
        if error:
            logger.error(f"bioRxiv search failed: {error}")
            return []

        # Filter by query
        filtered = self._filter_results(items, query)

        # Convert to documents
        documents = []
        for item in filtered[:limit]:
            doc = self._to_document(item)
            if doc:
                documents.append(doc)

        return documents

    async def get_recent(
        self,
        days: int = 30,
        limit: int = 100,
    ) -> List[Document]:
        """
        Get recent preprints.

        Args:
            days: Number of days to look back
            limit: Maximum results

        Returns:
            List of Document objects
        """
        interval = f"{days}d"
        items, _ = await self._fetch_results(interval, limit)

        documents = []
        for item in items[:limit]:
            doc = self._to_document(item)
            if doc:
                documents.append(doc)

        return documents

    async def _fetch_results(
        self,
        interval: str,
        limit: int,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Fetch results from API."""
        per_page = 100
        max_pages = 5
        items: List[Dict[str, Any]] = []

        try:
            client = await self._get_client()
            cursor = 0

            for _ in range(max_pages):
                await self._rate_limiter.acquire()

                url = f"{self.base_url}/{interval}/{cursor}"
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()

                collection = data.get("collection", []) or []
                if not collection:
                    break

                items.extend(collection)

                if len(collection) < per_page or len(items) >= limit:
                    break

                cursor += per_page

        except Exception as e:
            return [], str(e)

        return items, None

    def _filter_results(
        self,
        items: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """Filter results by query terms."""
        if not query:
            return items

        # Tokenize query
        tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z0-9-]+", query)
            if len(token) > 2
        ]
        if not tokens:
            return items

        # Score and filter
        scored: List[Tuple[int, Dict[str, Any]]] = []
        for item in items:
            title = (item.get("title") or "").lower()
            abstract = (item.get("abstract") or "").lower()
            authors = (item.get("authors") or "").lower()
            haystack = f"{title} {abstract} {authors}"

            score = sum(1 for token in tokens if token in haystack)
            if score:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored]

    def _to_document(self, item: Dict[str, Any]) -> Optional[Document]:
        """Convert API item to Document."""
        doi = item.get("doi")
        title = item.get("title") or "Untitled"
        authors_raw = item.get("authors") or ""
        authors = [a.strip() for a in authors_raw.split(";") if a.strip()]
        abstract = item.get("abstract")

        # Parse year
        year = None
        date_str = item.get("date")
        if date_str:
            match = re.search(r"(\d{4})", date_str)
            if match:
                year = int(match.group(1))

        # Build URL
        if doi:
            url = f"https://www.{self.server}.org/content/{doi}"
            doc_id = f"doi:{doi}"
        else:
            url = ""
            doc_id = f"{self.server}:{title[:32]}"

        return Document(
            id=doc_id,
            source=self.server,
            title=title,
            text=abstract or "",
            authors=authors,
            year=year,
            doi=doi,
            url=url,
            metadata={
                "category": item.get("category"),
                "date": date_str,
            },
        )

    async def health_check(self) -> bool:
        """Check if bioRxiv API is accessible."""
        try:
            client = await self._get_client()
            resp = await client.get(f"{self.base_url}/1/0", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False


def create_biorxiv_connector(
    server: str = "biorxiv",
    cache_dir: Optional[str] = None,
) -> BioRxivConnector:
    """Factory function for bioRxiv connector.

    Args:
        server: "biorxiv" or "medrxiv"
        cache_dir: Directory for caching

    Returns:
        Configured BioRxivConnector
    """
    from pathlib import Path

    config = ConnectorConfig(
        cache_dir=Path(cache_dir) if cache_dir else None,
    )
    return BioRxivConnector(server=server, config=config)
