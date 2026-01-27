"""EPO Open Patent Services (OPS) client.

Phase 6 (v3.0): Provides access to European Patent Office data for
international patent coverage in the BioPAT benchmark.

EPO OPS API documentation: https://developers.epo.org/
"""

import asyncio
import base64
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
import polars as pl
from diskcache import Cache

from biopat.reproducibility import AuditLogger

logger = logging.getLogger(__name__)

# EPO OPS API endpoints
EPO_AUTH_URL = "https://ops.epo.org/3.2/auth/accesstoken"
EPO_API_BASE = "https://ops.epo.org/3.2/rest-services"

# Rate limits: Anonymous = 4 req/min, Registered = 20 req/min
DEFAULT_RATE_LIMIT = 20


class EPOAuthError(Exception):
    """Exception raised for EPO authentication failures."""
    pass


class EPOClient:
    """Async client for EPO Open Patent Services (OPS) API.

    Uses OAuth 2.0 client credentials flow for authentication.
    Supports bibliographic search, publication retrieval, and citation extraction.
    """

    def __init__(
        self,
        consumer_key: Optional[str] = None,
        consumer_secret: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        rate_limit: int = DEFAULT_RATE_LIMIT,
        audit_logger: Optional[AuditLogger] = None,
    ):
        """Initialize EPO OPS client.

        Args:
            consumer_key: EPO OPS consumer key.
            consumer_secret: EPO OPS consumer secret.
            cache_dir: Directory for caching responses.
            rate_limit: Maximum requests per minute.
            audit_logger: Optional audit logger for tracking API calls.
        """
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.rate_limit = rate_limit
        self.cache = Cache(str(cache_dir / "epo")) if cache_dir else None
        self._semaphore = asyncio.Semaphore(rate_limit)
        self._request_times: List[float] = []
        self.audit_logger = audit_logger

        # OAuth token management
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None

    async def _get_access_token(self) -> str:
        """Acquire or refresh OAuth 2.0 access token.

        Returns:
            Valid access token string.

        Raises:
            EPOAuthError: If authentication fails.
        """
        # Return cached token if still valid
        if self._access_token and self._token_expires:
            if datetime.now() < self._token_expires - timedelta(minutes=1):
                return self._access_token

        if not self.consumer_key or not self.consumer_secret:
            raise EPOAuthError(
                "EPO consumer_key and consumer_secret required for authenticated access"
            )

        # Build Basic auth header
        credentials = f"{self.consumer_key}:{self.consumer_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    EPO_AUTH_URL,
                    headers={
                        "Authorization": f"Basic {encoded}",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    data={"grant_type": "client_credentials"},
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

                self._access_token = data["access_token"]
                # Token typically valid for 20 minutes
                expires_in = int(data.get("expires_in", 1200))
                self._token_expires = datetime.now() + timedelta(seconds=expires_in)

                logger.info(f"EPO OAuth token acquired, expires in {expires_in}s")
                return self._access_token

            except httpx.HTTPStatusError as e:
                raise EPOAuthError(f"EPO authentication failed: {e.response.text}")
            except Exception as e:
                raise EPOAuthError(f"EPO authentication error: {e}")

    def _get_headers(self, access_token: str) -> Dict[str, str]:
        """Build request headers with OAuth token."""
        return {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }

    async def _rate_limit_wait(self):
        """Enforce rate limiting."""
        import time
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]

        if len(self._request_times) >= self.rate_limit:
            sleep_time = 60 - (now - self._request_times[0])
            if sleep_time > 0:
                logger.debug(f"EPO rate limiting: sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)

        self._request_times.append(time.time())

    async def _make_request(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        params: Optional[dict] = None,
        method: str = "GET",
    ) -> Dict[str, Any]:
        """Make authenticated, rate-limited API request."""
        async with self._semaphore:
            await self._rate_limit_wait()

            access_token = await self._get_access_token()
            url = f"{EPO_API_BASE}/{endpoint}"

            try:
                response = await client.request(
                    method,
                    url,
                    headers=self._get_headers(access_token),
                    params=params,
                    timeout=60.0,
                )
                response.raise_for_status()

                # Handle XML or JSON response
                content_type = response.headers.get("content-type", "")
                if "json" in content_type:
                    result = response.json()
                else:
                    # Return raw text for XML parsing
                    result = {"_raw": response.text, "_content_type": content_type}

                # Log API call
                if self.audit_logger:
                    self.audit_logger.log_api_call(
                        service="epo",
                        endpoint=endpoint,
                        method=method,
                        params=params,
                        response_status=response.status_code,
                    )

                return result

            except httpx.HTTPStatusError as e:
                logger.error(f"EPO HTTP error {e.response.status_code}: {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"EPO request failed: {e}")
                raise

    async def get_publication(
        self,
        publication_number: str,
        endpoint_type: str = "biblio",
    ) -> Optional[Dict[str, Any]]:
        """Retrieve publication data by number.

        Args:
            publication_number: EP publication number (e.g., "EP1234567").
            endpoint_type: Data type - "biblio", "abstract", "claims", "fulltext".

        Returns:
            Publication data dict or None if not found.
        """
        # Normalize publication number
        pub_num = publication_number.replace(" ", "").upper()
        if not pub_num.startswith("EP"):
            pub_num = f"EP{pub_num}"

        cache_key = f"pub:{pub_num}:{endpoint_type}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        endpoint = f"published-data/publication/epodoc/{pub_num}/{endpoint_type}"

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

    async def search_publications(
        self,
        query: str,
        range_begin: int = 1,
        range_end: int = 100,
    ) -> Dict[str, Any]:
        """Search publications using CQL query.

        Args:
            query: CQL query string (e.g., "ti=drug AND ic=A61K").
            range_begin: Start of result range (1-indexed).
            range_end: End of result range.

        Returns:
            Search results with publication references.
        """
        endpoint = "published-data/search/biblio"
        params = {
            "q": query,
            "Range": f"{range_begin}-{range_end}",
        }

        async with httpx.AsyncClient() as client:
            return await self._make_request(client, endpoint, params=params)

    async def get_citations(
        self,
        publication_number: str,
    ) -> Dict[str, Any]:
        """Get citations for a publication.

        Args:
            publication_number: EP publication number.

        Returns:
            Citation data including forward and backward citations.
        """
        pub_num = publication_number.replace(" ", "").upper()
        if not pub_num.startswith("EP"):
            pub_num = f"EP{pub_num}"

        cache_key = f"citations:{pub_num}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        endpoint = f"published-data/publication/epodoc/{pub_num}/biblio"

        async with httpx.AsyncClient() as client:
            result = await self._make_request(client, endpoint)

            if self.cache and result:
                self.cache[cache_key] = result

            return result

    async def get_search_report(
        self,
        application_number: str,
    ) -> Optional[Dict[str, Any]]:
        """Get European search report for an application.

        The search report contains examiner citations with relevance categories
        (X = particularly relevant, Y = relevant in combination, A = background).

        Args:
            application_number: EP application number.

        Returns:
            Search report data or None if not available.
        """
        app_num = application_number.replace(" ", "").upper()
        if not app_num.startswith("EP"):
            app_num = f"EP{app_num}"

        cache_key = f"search_report:{app_num}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        # Search reports are in the register service
        endpoint = f"register/search/biblio"
        params = {"q": f"ap={app_num}"}

        async with httpx.AsyncClient() as client:
            try:
                result = await self._make_request(client, endpoint, params=params)

                if self.cache and result:
                    self.cache[cache_key] = result

                return result
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return None
                raise

    async def get_biomedical_patents(
        self,
        ipc_classes: List[str] = ["A61", "C07", "C12"],
        limit: Optional[int] = None,
        publication_date_from: Optional[str] = None,
        publication_date_to: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for biomedical patents by IPC class.

        Args:
            ipc_classes: List of IPC class prefixes.
            limit: Maximum number of results.
            publication_date_from: Start date (YYYYMMDD).
            publication_date_to: End date (YYYYMMDD).

        Returns:
            List of patent bibliographic data.
        """
        # Build CQL query
        ipc_query = " OR ".join(f"ic={ipc}" for ipc in ipc_classes)
        query = f"({ipc_query})"

        if publication_date_from:
            query += f" AND pd>={publication_date_from}"
        if publication_date_to:
            query += f" AND pd<={publication_date_to}"

        results = []
        range_begin = 1
        batch_size = 100

        async with httpx.AsyncClient() as client:
            while True:
                range_end = range_begin + batch_size - 1

                try:
                    response = await self._make_request(
                        client,
                        "published-data/search/biblio",
                        params={
                            "q": query,
                            "Range": f"{range_begin}-{range_end}",
                        },
                    )

                    # Parse results
                    # Note: Actual parsing depends on response format
                    batch = self._parse_search_results(response)

                    if not batch:
                        break

                    results.extend(batch)

                    if limit and len(results) >= limit:
                        results = results[:limit]
                        break

                    range_begin = range_end + 1

                except httpx.HTTPStatusError as e:
                    if "no results found" in str(e.response.text).lower():
                        break
                    raise

        logger.info(f"Retrieved {len(results)} EP biomedical patents")
        return results

    def _parse_search_results(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse search results from EPO response.

        Args:
            response: Raw EPO API response.

        Returns:
            List of parsed patent records.
        """
        # Handle XML response
        if "_raw" in response:
            return self._parse_xml_results(response["_raw"])

        # Handle JSON response
        results = []
        try:
            # Navigate EPO response structure
            search_result = response.get("ops:world-patent-data", {})
            search_result = search_result.get("ops:biblio-search", {})
            search_result = search_result.get("ops:search-result", {})

            documents = search_result.get("ops:publication-reference", [])
            if not isinstance(documents, list):
                documents = [documents]

            for doc in documents:
                doc_id = doc.get("document-id", {})
                results.append({
                    "country": doc_id.get("country", {}).get("$", ""),
                    "doc_number": doc_id.get("doc-number", {}).get("$", ""),
                    "kind": doc_id.get("kind", {}).get("$", ""),
                })

        except (KeyError, TypeError) as e:
            logger.warning(f"Failed to parse EPO results: {e}")

        return results

    def _parse_xml_results(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse XML search results.

        Args:
            xml_text: Raw XML response text.

        Returns:
            List of parsed patent records.
        """
        try:
            from lxml import etree
            root = etree.fromstring(xml_text.encode())

            results = []
            # EPO XML namespaces
            ns = {
                "ops": "http://ops.epo.org",
                "epo": "http://www.epo.org/exchange",
            }

            for pub_ref in root.findall(".//ops:publication-reference", ns):
                doc_id = pub_ref.find(".//epo:document-id", ns)
                if doc_id is not None:
                    results.append({
                        "country": doc_id.findtext("epo:country", "", ns),
                        "doc_number": doc_id.findtext("epo:doc-number", "", ns),
                        "kind": doc_id.findtext("epo:kind", "", ns),
                    })

            return results

        except Exception as e:
            logger.warning(f"Failed to parse EPO XML: {e}")
            return []

    async def get_patents_batch(
        self,
        publication_numbers: List[str],
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """Fetch multiple patents by publication number.

        Args:
            publication_numbers: List of EP publication numbers.
            show_progress: Show progress bar.

        Returns:
            List of patent data dicts.
        """
        results = []

        async with httpx.AsyncClient() as client:
            if show_progress:
                from tqdm.asyncio import tqdm_asyncio
                iterator = tqdm_asyncio(
                    publication_numbers, desc="Fetching EP patents"
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
            patents: List of patent data dicts from EPO.

        Returns:
            DataFrame with standardized patent data.
        """
        records = []

        for p in patents:
            # Extract from EPO response structure
            try:
                biblio = p.get("ops:world-patent-data", {})
                biblio = biblio.get("exchange-documents", {})
                biblio = biblio.get("exchange-document", {})

                if isinstance(biblio, list):
                    biblio = biblio[0]

                # Extract publication info
                pub_ref = biblio.get("@country", "") + biblio.get("@doc-number", "")

                # Extract bibliographic data
                bib_data = biblio.get("bibliographic-data", {})

                # Title
                title = ""
                invention_title = bib_data.get("invention-title", [])
                if isinstance(invention_title, list):
                    for t in invention_title:
                        if t.get("@lang") == "en":
                            title = t.get("$", "")
                            break
                elif isinstance(invention_title, dict):
                    title = invention_title.get("$", "")

                # Abstract
                abstract = ""
                abs_data = biblio.get("abstract", [])
                if isinstance(abs_data, list):
                    for a in abs_data:
                        if a.get("@lang") == "en":
                            abstract = a.get("p", {}).get("$", "")
                            break

                # IPC codes
                ipc_codes = []
                class_data = bib_data.get("classifications-ipcr", {})
                class_list = class_data.get("classification-ipcr", [])
                if not isinstance(class_list, list):
                    class_list = [class_list]
                for c in class_list:
                    if "text" in c:
                        ipc_codes.append(c["text"].get("$", ""))

                # Dates
                pub_date = bib_data.get("publication-reference", {})
                pub_date = pub_date.get("document-id", {})
                pub_date = pub_date.get("date", {}).get("$", "")

                priority_date = ""
                priority_claims = bib_data.get("priority-claims", {})
                priority_list = priority_claims.get("priority-claim", [])
                if not isinstance(priority_list, list):
                    priority_list = [priority_list]
                if priority_list:
                    priority_date = priority_list[0].get("date", {}).get("$", "")

                records.append({
                    "patent_id": pub_ref,
                    "jurisdiction": "EP",
                    "title": title,
                    "abstract": abstract,
                    "publication_date": pub_date,
                    "priority_date": priority_date,
                    "ipc_codes": ipc_codes,
                })

            except Exception as e:
                logger.warning(f"Failed to parse EP patent: {e}")
                continue

        return pl.DataFrame(records)
