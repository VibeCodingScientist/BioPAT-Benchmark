"""PubMed/NCBI Connector for scientific literature.

Provides access to PubMed articles via the NCBI E-utilities API.
Supports searching, fetching metadata, and extracting abstracts.

API Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""

import asyncio
import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from biopat.data.base import BaseConnector, ConnectorConfig, Document

logger = logging.getLogger(__name__)

# Try to import httpx
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class PubMedConnector(BaseConnector):
    """Connector for PubMed/NCBI E-utilities API.

    Example:
        ```python
        async with PubMedConnector() as pubmed:
            articles = await pubmed.search(
                "CRISPR gene therapy",
                limit=100,
                date_range="2020-2024"
            )
            for article in articles:
                print(f"{article.title} ({article.year})")
        ```
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # Biomedical term mapping for query optimization
    BIOMEDICAL_TERMS = {
        "crispr": "CRISPR",
        "cas9": "Cas9",
        "gene editing": '"gene editing"',
        "gene therapy": '"gene therapy"',
        "car-t": "CAR-T",
        "checkpoint inhibitor": '"checkpoint inhibitor"',
        "pd-1": "PD-1",
        "pd-l1": "PD-L1",
        "antibody": "antibody",
        "monoclonal": "monoclonal",
        "immunotherapy": "immunotherapy",
        "cancer": "cancer",
        "tumor": "tumor",
        "clinical trial": '"clinical trial"',
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ConnectorConfig] = None,
    ):
        """
        Initialize PubMed connector.

        Args:
            api_key: NCBI API key (increases rate limit from 3 to 10 req/sec)
            config: Connector configuration
        """
        if config is None:
            config = ConnectorConfig(api_key=api_key)
        elif api_key:
            config.api_key = api_key

        # NCBI allows 3 req/sec without key, 10 with key
        if config.api_key:
            config.rate_limit = 10.0
        else:
            config.rate_limit = 3.0

        super().__init__(config)
        self._last_request_ts = 0.0
        self._rate_lock = asyncio.Lock()

    @property
    def source_name(self) -> str:
        return "pubmed"

    async def _throttle(self) -> None:
        """Enforce rate limiting."""
        rate = self.config.rate_limit
        min_interval = 1.0 / rate if rate > 0 else 0

        async with self._rate_lock:
            elapsed = time.monotonic() - self._last_request_ts
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            self._last_request_ts = time.monotonic()

    async def _get(
        self,
        url: str,
        params: Dict[str, Any],
        retries: int = 3,
    ) -> "httpx.Response":
        """Make rate-limited GET request with retries."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required. Install with: pip install httpx")

        # Add API key if available
        if self.config.api_key:
            params["api_key"] = self.config.api_key

        client = await self._get_client()
        last_exc: Optional[Exception] = None

        for attempt in range(retries):
            await self._throttle()

            try:
                resp = await client.get(url, params=params)

                # Handle rate limiting
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After", str(2 ** attempt))
                    wait = float(retry_after) if retry_after.isdigit() else (2 ** attempt)
                    logger.warning(f"PubMed rate limited. Waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp

            except Exception as e:
                last_exc = e
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)

        if last_exc:
            raise last_exc
        raise RuntimeError("Request failed")

    def _build_query(self, query: str) -> str:
        """Build optimized PubMed query from natural language."""
        query_lower = query.lower()
        terms = []

        # Extract known biomedical terms
        for phrase, pubmed_term in self.BIOMEDICAL_TERMS.items():
            if phrase in query_lower:
                terms.append(pubmed_term)

        # Extract gene-like tokens (all caps with digits)
        gene_like = re.findall(r'\b[A-Z]{2,}\d*\b', query)
        terms.extend(gene_like)

        # If no specific terms found, use query words
        if not terms:
            words = re.findall(r'\b[a-zA-Z0-9-]+\b', query)
            terms = [w for w in words if len(w) > 2][:4]

        # Deduplicate while preserving order
        seen = set()
        unique_terms = []
        for t in terms:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique_terms.append(t)

        # Limit to avoid over-restriction
        unique_terms = unique_terms[:5]

        if not unique_terms:
            return query

        # Build query with Title/Abstract field tags
        tagged = [f'({term}[Title/Abstract])' for term in unique_terms]
        pubmed_query = " AND ".join(tagged)

        logger.debug(f"PubMed query: '{query}' -> '{pubmed_query}'")
        return pubmed_query

    async def search(
        self,
        query: str,
        limit: int = 100,
        date_range: Optional[str] = None,
        **kwargs,
    ) -> List[Document]:
        """
        Search PubMed for articles.

        Args:
            query: Search query (natural language or PubMed syntax)
            limit: Maximum number of results
            date_range: Optional date filter (e.g., "2020-2024")

        Returns:
            List of Document objects
        """
        # Check cache
        cache_key = f"search:{query}:{limit}:{date_range}"
        cached = self._cache_get(cache_key)
        if cached:
            return [Document(**d) for d in cached]

        # Build query
        pubmed_query = self._build_query(query)

        # Build date parameters
        date_params = {}
        if date_range:
            match = re.match(r'(\d{4})-(\d{4})', date_range)
            if match:
                start_year, end_year = match.groups()
                date_params = {
                    "mindate": f"{start_year}/01/01",
                    "maxdate": f"{end_year}/12/31",
                    "datetype": "pdat",
                }

        try:
            # Step 1: Search for PMIDs
            search_params = {
                "db": "pubmed",
                "term": pubmed_query,
                "retmax": limit,
                "retmode": "json",
                **date_params,
            }

            resp = await self._get(f"{self.BASE_URL}/esearch.fcgi", search_params)
            data = resp.json()

            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            # Step 2: Fetch metadata
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml",
            }

            fetch_resp = await self._get(f"{self.BASE_URL}/efetch.fcgi", fetch_params)
            documents = self._parse_xml(fetch_resp.content)

            # Cache results
            self._cache_set(cache_key, [d.to_dict() for d in documents])

            return documents

        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []

    async def fetch_by_pmids(self, pmids: List[str]) -> List[Document]:
        """
        Fetch articles by PMID.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of Document objects
        """
        if not pmids:
            return []

        try:
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            }

            resp = await self._get(f"{self.BASE_URL}/efetch.fcgi", fetch_params)
            return self._parse_xml(resp.content)

        except Exception as e:
            logger.error(f"PubMed fetch failed: {e}")
            return []

    def _parse_xml(self, content: bytes) -> List[Document]:
        """Parse PubMed XML response."""
        documents = []

        try:
            root = ET.fromstring(content)

            for article in root.findall(".//PubmedArticle"):
                try:
                    doc = self._parse_article(article)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to parse article: {e}")

        except Exception as e:
            logger.error(f"Failed to parse PubMed XML: {e}")

        return documents

    def _parse_article(self, article: ET.Element) -> Optional[Document]:
        """Parse single PubMed article."""
        medline = article.find("MedlineCitation")
        if medline is None:
            return None

        pmid_elem = medline.find("PMID")
        if pmid_elem is None or not pmid_elem.text:
            return None
        pmid = pmid_elem.text

        article_data = medline.find("Article")
        if article_data is None:
            return None

        # Title
        title_elem = article_data.find("ArticleTitle")
        title = title_elem.text.strip() if title_elem is not None and title_elem.text else "No title"

        # Abstract
        abstract_elem = article_data.find(".//Abstract/AbstractText")
        abstract = abstract_elem.text if abstract_elem is not None else ""

        # Authors
        authors = []
        author_list = article_data.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                last = author.find("LastName")
                initials = author.find("Initials")
                if last is not None and last.text:
                    name = last.text
                    if initials is not None and initials.text:
                        name += f" {initials.text}"
                    authors.append(name)

        # Journal
        journal_elem = article_data.find("Journal/Title")
        journal = journal_elem.text if journal_elem is not None else None

        # Year
        year = None
        pub_date = article_data.find("Journal/JournalIssue/PubDate/Year")
        if pub_date is not None and pub_date.text:
            match = re.search(r'\d{4}', pub_date.text)
            if match:
                year = int(match.group(0))

        # DOI
        doi = None
        for eloc in article_data.findall("ELocationID"):
            if eloc.get("EIdType") == "doi":
                doi = eloc.text
                break

        return Document(
            id=f"pmid:{pmid}",
            source="pubmed",
            title=title,
            text=abstract or "",
            authors=authors,
            year=year,
            doi=doi,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            metadata={"journal": journal, "pmid": pmid},
        )

    async def health_check(self) -> bool:
        """Check if PubMed API is accessible."""
        try:
            params = {"db": "pubmed", "term": "test", "retmax": 1, "retmode": "json"}
            client = await self._get_client()
            resp = await client.get(f"{self.BASE_URL}/esearch.fcgi", params=params, timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False


def create_pubmed_connector(
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> PubMedConnector:
    """Factory function for PubMed connector.

    Args:
        api_key: NCBI API key (optional, increases rate limit)
        cache_dir: Directory for caching responses

    Returns:
        Configured PubMedConnector
    """
    from pathlib import Path

    config = ConnectorConfig(
        api_key=api_key,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )
    return PubMedConnector(config=config)
