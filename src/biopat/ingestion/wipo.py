"""WIPO PATENTSCOPE client.

Phase 6 (v3.0): Provides access to World Intellectual Property Organization
(WIPO) PATENTSCOPE data for international PCT patent coverage in BioPAT.

PATENTSCOPE API documentation: https://patentscope.wipo.int/search/help/en/webServices.jsf
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
import polars as pl
from diskcache import Cache

from biopat.reproducibility import AuditLogger

logger = logging.getLogger(__name__)

# WIPO PATENTSCOPE API endpoints
WIPO_API_BASE = "https://patentscope.wipo.int/search/webservices/rest/v1"

# Rate limits: Registered users typically get 2 req/sec
DEFAULT_RATE_LIMIT = 120  # per minute


class WIPOAuthError(Exception):
    """Exception raised for WIPO authentication failures."""
    pass


class WIPOClient:
    """Async client for WIPO PATENTSCOPE API.

    Supports searching PCT (WO) applications and retrieving publication data.
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        rate_limit: int = DEFAULT_RATE_LIMIT,
        audit_logger: Optional[AuditLogger] = None,
    ):
        """Initialize WIPO PATENTSCOPE client.

        Args:
            api_token: WIPO API access token.
            cache_dir: Directory for caching responses.
            rate_limit: Maximum requests per minute.
            audit_logger: Optional audit logger for tracking API calls.
        """
        self.api_token = api_token
        self.rate_limit = rate_limit
        self.cache = Cache(str(cache_dir / "wipo")) if cache_dir else None
        self._semaphore = asyncio.Semaphore(rate_limit)
        self._request_times: List[float] = []
        self.audit_logger = audit_logger

    def _get_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    async def _rate_limit_wait(self):
        """Enforce rate limiting."""
        import time
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]

        if len(self._request_times) >= self.rate_limit:
            sleep_time = 60 - (now - self._request_times[0])
            if sleep_time > 0:
                logger.debug(f"WIPO rate limiting: sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)

        self._request_times.append(time.time())

    async def _make_request(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        params: Optional[dict] = None,
        method: str = "GET",
    ) -> Dict[str, Any]:
        """Make rate-limited API request."""
        async with self._semaphore:
            await self._rate_limit_wait()

            url = f"{WIPO_API_BASE}/{endpoint}"

            try:
                response = await client.request(
                    method,
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=60.0,
                )
                response.raise_for_status()
                result = response.json()

                # Log API call
                if self.audit_logger:
                    self.audit_logger.log_api_call(
                        service="wipo",
                        endpoint=endpoint,
                        method=method,
                        params=params,
                        response_status=response.status_code,
                    )

                return result

            except httpx.HTTPStatusError as e:
                logger.error(f"WIPO HTTP error {e.response.status_code}: {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"WIPO request failed: {e}")
                raise

    async def search_pct(
        self,
        query: str,
        start: int = 0,
        rows: int = 100,
        sort: str = "PD_D",
    ) -> Dict[str, Any]:
        """Search PCT applications using PATENTSCOPE query.

        Args:
            query: Search query string.
            start: Starting offset for results.
            rows: Number of results to return.
            sort: Sort order (PD_D = publication date descending).

        Returns:
            Search results with application references.
        """
        endpoint = "search"
        params = {
            "query": query,
            "start": start,
            "rows": rows,
            "sort": sort,
        }

        async with httpx.AsyncClient() as client:
            return await self._make_request(client, endpoint, params=params)

    async def get_publication(
        self,
        publication_number: str,
        data_type: str = "biblio",
    ) -> Optional[Dict[str, Any]]:
        """Retrieve PCT publication data by number.

        Args:
            publication_number: WO publication number (e.g., "WO2020123456").
            data_type: Data type - "biblio", "abstract", "claims", "description".

        Returns:
            Publication data dict or None if not found.
        """
        # Normalize publication number
        pub_num = publication_number.replace(" ", "").replace("/", "").upper()
        if not pub_num.startswith("WO"):
            pub_num = f"WO{pub_num}"

        cache_key = f"pub:{pub_num}:{data_type}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        endpoint = f"publication/{pub_num}/{data_type}"

        async with httpx.AsyncClient() as client:
            try:
                result = await self._make_request(client, endpoint)

                if self.cache and result:
                    self.cache[cache_key] = result

                return result
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return None
                raise

    async def get_application_details(
        self,
        application_number: str,
    ) -> Optional[Dict[str, Any]]:
        """Get detailed PCT application information.

        Args:
            application_number: PCT application number.

        Returns:
            Application details or None if not found.
        """
        app_num = application_number.replace(" ", "").replace("/", "").upper()

        cache_key = f"app:{app_num}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        endpoint = f"application/{app_num}"

        async with httpx.AsyncClient() as client:
            try:
                result = await self._make_request(client, endpoint)

                if self.cache and result:
                    self.cache[cache_key] = result

                return result
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return None
                raise

    async def get_citations(
        self,
        publication_number: str,
    ) -> Dict[str, Any]:
        """Get citations from International Search Report.

        Args:
            publication_number: WO publication number.

        Returns:
            Citation data from the search report.
        """
        pub_num = publication_number.replace(" ", "").replace("/", "").upper()
        if not pub_num.startswith("WO"):
            pub_num = f"WO{pub_num}"

        cache_key = f"citations:{pub_num}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        # Citations are part of the ISR (International Search Report)
        endpoint = f"publication/{pub_num}/citations"

        async with httpx.AsyncClient() as client:
            try:
                result = await self._make_request(client, endpoint)

                if self.cache and result:
                    self.cache[cache_key] = result

                return result
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return {"citations": []}
                raise

    async def get_biomedical_pct(
        self,
        ipc_classes: List[str] = ["A61", "C07", "C12"],
        limit: Optional[int] = None,
        publication_date_from: Optional[str] = None,
        publication_date_to: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for biomedical PCT applications by IPC class.

        Args:
            ipc_classes: List of IPC class prefixes.
            limit: Maximum number of results.
            publication_date_from: Start date (YYYY-MM-DD).
            publication_date_to: End date (YYYY-MM-DD).

        Returns:
            List of PCT application data.
        """
        # Build PATENTSCOPE query
        ipc_query = " OR ".join(f"IC:{ipc}*" for ipc in ipc_classes)
        query = f"({ipc_query})"

        if publication_date_from:
            query += f" AND PD:[{publication_date_from} TO *]"
        if publication_date_to:
            query += f" AND PD:[* TO {publication_date_to}]"

        results = []
        start = 0
        batch_size = 100

        async with httpx.AsyncClient() as client:
            while True:
                try:
                    response = await self._make_request(
                        client,
                        "search",
                        params={
                            "query": query,
                            "start": start,
                            "rows": batch_size,
                        },
                    )

                    # Parse results
                    batch = self._parse_search_results(response)

                    if not batch:
                        break

                    results.extend(batch)

                    if limit and len(results) >= limit:
                        results = results[:limit]
                        break

                    # Check if more results available
                    total_count = response.get("numFound", 0)
                    if start + batch_size >= total_count:
                        break

                    start += batch_size

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        break
                    raise

        logger.info(f"Retrieved {len(results)} WO biomedical patents")
        return results

    def _parse_search_results(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse search results from WIPO response.

        Args:
            response: Raw WIPO API response.

        Returns:
            List of parsed patent records.
        """
        results = []

        try:
            docs = response.get("results", [])
            if not isinstance(docs, list):
                docs = [docs]

            for doc in docs:
                results.append({
                    "publication_number": doc.get("publicationNumber", ""),
                    "application_number": doc.get("applicationNumber", ""),
                    "title": doc.get("title", ""),
                    "abstract": doc.get("abstract", ""),
                    "publication_date": doc.get("publicationDate", ""),
                    "applicants": doc.get("applicants", []),
                    "inventors": doc.get("inventors", []),
                    "ipc_classes": doc.get("ipcClasses", []),
                })

        except (KeyError, TypeError) as e:
            logger.warning(f"Failed to parse WIPO results: {e}")

        return results

    async def get_patents_batch(
        self,
        publication_numbers: List[str],
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """Fetch multiple PCT publications by number.

        Args:
            publication_numbers: List of WO publication numbers.
            show_progress: Show progress bar.

        Returns:
            List of publication data dicts.
        """
        results = []

        async with httpx.AsyncClient() as client:
            if show_progress:
                from tqdm.asyncio import tqdm_asyncio
                iterator = tqdm_asyncio(
                    publication_numbers, desc="Fetching WO patents"
                )
            else:
                iterator = publication_numbers

            for pub_num in iterator:
                try:
                    data = await self.get_publication(pub_num, "biblio")
                    if data:
                        results.append(data)
                except Exception as e:
                    logger.warning(f"Failed to fetch {pub_num}: {e}")

        return results

    def patents_to_dataframe(self, patents: List[Dict[str, Any]]) -> pl.DataFrame:
        """Convert patent data to Polars DataFrame.

        Args:
            patents: List of patent data dicts from WIPO.

        Returns:
            DataFrame with standardized patent data.
        """
        records = []

        for p in patents:
            try:
                # Handle both search results and individual publication data
                pub_num = p.get("publication_number", p.get("publicationNumber", ""))

                # Extract title (may be in different languages)
                title = p.get("title", "")
                if isinstance(title, dict):
                    title = title.get("en", title.get("EN", list(title.values())[0] if title else ""))
                elif isinstance(title, list):
                    # Find English title
                    for t in title:
                        if isinstance(t, dict) and t.get("lang", "").lower() == "en":
                            title = t.get("text", "")
                            break
                    else:
                        title = title[0] if title else ""

                # Extract abstract
                abstract = p.get("abstract", "")
                if isinstance(abstract, dict):
                    abstract = abstract.get("en", abstract.get("EN", list(abstract.values())[0] if abstract else ""))
                elif isinstance(abstract, list):
                    for a in abstract:
                        if isinstance(a, dict) and a.get("lang", "").lower() == "en":
                            abstract = a.get("text", "")
                            break
                    else:
                        abstract = abstract[0] if abstract else ""

                # Extract dates
                pub_date = p.get("publication_date", p.get("publicationDate", ""))
                priority_date = p.get("priority_date", p.get("priorityDate", ""))

                # Extract IPC codes
                ipc_codes = p.get("ipc_classes", p.get("ipcClasses", []))
                if isinstance(ipc_codes, str):
                    ipc_codes = [ipc_codes]

                records.append({
                    "patent_id": pub_num,
                    "jurisdiction": "WO",
                    "title": title if isinstance(title, str) else str(title),
                    "abstract": abstract if isinstance(abstract, str) else str(abstract),
                    "publication_date": pub_date,
                    "priority_date": priority_date,
                    "ipc_codes": ipc_codes,
                })

            except Exception as e:
                logger.warning(f"Failed to parse WO patent: {e}")
                continue

        return pl.DataFrame(records)

    async def get_isr_citations(
        self,
        publication_number: str,
    ) -> List[Dict[str, Any]]:
        """Get International Search Report citations with categories.

        The ISR contains examiner citations with relevance categories
        (X = particularly relevant, Y = relevant in combination, A = background).

        Args:
            publication_number: WO publication number.

        Returns:
            List of citation dicts with category information.
        """
        citations_data = await self.get_citations(publication_number)

        citations = []
        raw_citations = citations_data.get("citations", [])

        for cit in raw_citations:
            citations.append({
                "cited_document": cit.get("documentNumber", ""),
                "category": cit.get("category", "A"),  # X, Y, A, etc.
                "claims_affected": cit.get("relevantClaims", []),
                "citation_type": "patent" if self._is_patent_number(cit.get("documentNumber", "")) else "npl",
            })

        return citations

    def _is_patent_number(self, doc_id: str) -> bool:
        """Check if a document ID appears to be a patent number."""
        if not doc_id:
            return False
        doc_id = doc_id.upper().replace(" ", "")
        # Common patent prefixes
        patent_prefixes = ["US", "EP", "WO", "JP", "CN", "KR", "GB", "DE", "FR", "CA", "AU"]
        return any(doc_id.startswith(prefix) for prefix in patent_prefixes)
