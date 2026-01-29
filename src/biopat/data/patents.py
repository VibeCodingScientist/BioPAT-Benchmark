"""Patent Data Connectors for USPTO and EPO.

Provides access to patent databases:
- USPTO PatentsView API (US patents)
- EPO Open Patent Services (European patents)

Supports fetching patent metadata, claims, and classifications.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from biopat.data.base import BaseConnector, ConnectorConfig, Document

logger = logging.getLogger(__name__)

# Try to import httpx
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# PatentsView API
PATENTSVIEW_BASE_URL = "https://search.patentsview.org/api/v1"
PATENTSVIEW_RATE_LIMIT = 45  # Requests per minute with API key

# Biomedical IPC codes
BIOMEDICAL_IPC_CODES = [
    "A61K",  # Pharmaceutical preparations
    "A61P",  # Therapeutic activity
    "C07D",  # Heterocyclic compounds
    "C07K",  # Peptides
    "C12N",  # Microorganisms, enzymes
    "C12Q",  # Measuring/testing with enzymes
    "G01N",  # Investigating materials
]


@dataclass
class PatentClaim:
    """Parsed patent claim."""
    claim_number: int
    claim_text: str
    is_independent: bool
    depends_on: Optional[int] = None


class PatentsViewConnector(BaseConnector):
    """Connector for USPTO PatentsView API.

    Example:
        ```python
        async with PatentsViewConnector(api_key="your-key") as pv:
            # Search biomedical patents
            patents = await pv.search_biomedical(
                ipc_codes=["A61K", "C07K"],
                date_range="2020-2024",
                limit=500
            )

            # Get specific patent
            patent = await pv.get_patent("US10500001B2")
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ConnectorConfig] = None,
    ):
        """
        Initialize PatentsView connector.

        Args:
            api_key: PatentsView API key (increases rate limit)
            config: Connector configuration
        """
        if config is None:
            config = ConnectorConfig(api_key=api_key)
        elif api_key:
            config.api_key = api_key

        config.rate_limit = PATENTSVIEW_RATE_LIMIT / 60  # Convert to per-second
        super().__init__(config)

    @property
    def source_name(self) -> str:
        return "patentsview"

    def _get_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["X-Api-Key"] = self.config.api_key
        return headers

    def _default_fields(self) -> List[str]:
        """Default fields for patent retrieval."""
        return [
            "patent_id",
            "patent_date",
            "patent_title",
            "patent_abstract",
            "claims",
            "application",
            "ipc_current",
            "cpc_current",
            "assignees_at_grant",
            "inventors",
        ]

    async def search(
        self,
        query: str,
        limit: int = 100,
        **kwargs,
    ) -> List[Document]:
        """
        Search patents by text query.

        Args:
            query: Search query (title/abstract)
            limit: Maximum results

        Returns:
            List of Document objects
        """
        # Build text search query
        pv_query = {
            "_or": [
                {"_text_any": {"patent_title": query}},
                {"_text_any": {"patent_abstract": query}},
            ]
        }
        return await self._search_with_query(pv_query, limit)

    async def search_biomedical(
        self,
        ipc_codes: Optional[List[str]] = None,
        date_range: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> List[Document]:
        """
        Search biomedical patents by IPC classification.

        Args:
            ipc_codes: IPC code prefixes (default: biomedical codes)
            date_range: Date filter (e.g., "2020-2024")
            limit: Maximum results

        Returns:
            List of Document objects
        """
        if ipc_codes is None:
            ipc_codes = BIOMEDICAL_IPC_CODES

        # Build IPC query
        ipc_conditions = [
            {"_begins": {"ipc_current.ipc_class": code}}
            for code in ipc_codes
        ]

        conditions = [{"_or": ipc_conditions}]

        # Add date filter
        if date_range:
            match = re.match(r'(\d{4})-(\d{4})', date_range)
            if match:
                start_year, end_year = match.groups()
                conditions.append({
                    "_gte": {"application.application_date": f"{start_year}-01-01"}
                })
                conditions.append({
                    "_lte": {"application.application_date": f"{end_year}-12-31"}
                })

        pv_query = {"_and": conditions} if len(conditions) > 1 else conditions[0]
        return await self._search_with_query(pv_query, limit)

    async def _search_with_query(
        self,
        query: Dict[str, Any],
        limit: int,
    ) -> List[Document]:
        """Execute search with PatentsView query."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required. Install with: pip install httpx")

        documents = []
        cursor: Optional[str] = None
        batch_size = min(100, limit)

        try:
            client = await self._get_client()

            while len(documents) < limit:
                await self._rate_limiter.acquire()

                body: Dict[str, Any] = {
                    "q": query,
                    "f": self._default_fields(),
                    "o": {"size": batch_size},
                }
                if cursor:
                    body["o"]["after"] = cursor

                resp = await client.post(
                    f"{PATENTSVIEW_BASE_URL}/patent/",
                    headers=self._get_headers(),
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()

                patents = data.get("patents", [])
                if not patents:
                    break

                for p in patents:
                    doc = self._parse_patent(p)
                    if doc:
                        documents.append(doc)

                cursor = data.get("cursor")
                if not cursor or len(documents) >= limit:
                    break

        except Exception as e:
            logger.error(f"PatentsView search failed: {e}")

        return documents[:limit]

    async def get_patent(self, patent_id: str) -> Optional[Document]:
        """
        Get single patent by ID.

        Args:
            patent_id: USPTO patent ID (e.g., "US10500001B2" or "10500001")

        Returns:
            Document or None
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required")

        # Normalize patent ID
        patent_id = patent_id.lstrip("US").rstrip("B1B2A1A2")

        try:
            await self._rate_limiter.acquire()
            client = await self._get_client()

            params = {"f": json.dumps(self._default_fields())}

            resp = await client.get(
                f"{PATENTSVIEW_BASE_URL}/patent/{patent_id}",
                headers=self._get_headers(),
                params=params,
            )

            if resp.status_code == 404:
                return None

            resp.raise_for_status()
            data = resp.json()

            return self._parse_patent(data)

        except Exception as e:
            logger.error(f"Failed to get patent {patent_id}: {e}")
            return None

    async def get_patents_batch(
        self,
        patent_ids: List[str],
    ) -> List[Document]:
        """
        Get multiple patents by ID.

        Args:
            patent_ids: List of patent IDs

        Returns:
            List of Document objects
        """
        if not patent_ids:
            return []

        # Build OR query
        query = {"_or": [{"patent_id": pid} for pid in patent_ids]}

        return await self._search_with_query(query, len(patent_ids))

    def _parse_patent(self, data: Dict[str, Any]) -> Optional[Document]:
        """Parse patent data into Document."""
        if not data:
            return None

        patent_id = data.get("patent_id", "")
        if not patent_id:
            return None

        # Parse dates
        grant_date = data.get("patent_date")
        filing_date = None
        application = data.get("application")
        if application:
            app = application[0] if isinstance(application, list) else application
            filing_date = app.get("application_date")

        # Parse IPC codes
        ipc_codes = []
        for ipc in data.get("ipc_current") or []:
            if "ipc_class" in ipc:
                ipc_codes.append(ipc["ipc_class"])

        # Parse assignees
        assignees = []
        for a in data.get("assignees_at_grant") or []:
            name = a.get("assignee_organization") or a.get("assignee_individual_name_last")
            if name:
                assignees.append(name)

        # Parse inventors
        inventors = []
        for inv in data.get("inventors") or []:
            first = inv.get("inventor_name_first", "")
            last = inv.get("inventor_name_last", "")
            if first or last:
                inventors.append(f"{first} {last}".strip())

        # Parse claims
        claims = []
        for c in data.get("claims") or []:
            claim_text = c.get("claim_text", "")
            if claim_text:
                claims.append(claim_text)

        # Extract year
        year = None
        if grant_date:
            match = re.search(r'(\d{4})', grant_date)
            if match:
                year = int(match.group(1))

        return Document(
            id=f"patent:{patent_id}",
            source="patent",
            title=data.get("patent_title", ""),
            text=data.get("patent_abstract", ""),
            authors=inventors,
            year=year,
            claims=claims,
            ipc_codes=ipc_codes,
            assignees=assignees,
            filing_date=filing_date,
            url=f"https://patents.google.com/patent/US{patent_id}",
            metadata={
                "patent_id": patent_id,
                "grant_date": grant_date,
                "filing_date": filing_date,
            },
        )

    async def health_check(self) -> bool:
        """Check if PatentsView API is accessible."""
        try:
            client = await self._get_client()
            resp = await client.get(
                f"{PATENTSVIEW_BASE_URL}/patent/10000000",
                headers=self._get_headers(),
                timeout=5.0,
            )
            return resp.status_code in [200, 404]
        except Exception:
            return False


class EPOConnector(BaseConnector):
    """Connector for EPO Open Patent Services.

    Requires registration at: https://developers.epo.org/

    Example:
        ```python
        async with EPOConnector(
            consumer_key="your-key",
            consumer_secret="your-secret"
        ) as epo:
            patents = await epo.search_by_ipc(["A61K"], limit=100)
        ```
    """

    AUTH_URL = "https://ops.epo.org/3.2/auth/accesstoken"
    API_BASE = "https://ops.epo.org/3.2/rest-services"

    def __init__(
        self,
        consumer_key: Optional[str] = None,
        consumer_secret: Optional[str] = None,
        config: Optional[ConnectorConfig] = None,
    ):
        """
        Initialize EPO connector.

        Args:
            consumer_key: EPO OPS consumer key
            consumer_secret: EPO OPS consumer secret
            config: Connector configuration
        """
        if config is None:
            config = ConnectorConfig(rate_limit=20 / 60)  # 20 req/min
        super().__init__(config)

        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None

    @property
    def source_name(self) -> str:
        return "epo"

    async def _get_access_token(self) -> str:
        """Acquire OAuth access token."""
        import base64
        from datetime import timedelta

        # Return cached token if valid
        if self._access_token and self._token_expires:
            if datetime.now() < self._token_expires - timedelta(minutes=1):
                return self._access_token

        if not self.consumer_key or not self.consumer_secret:
            raise ValueError("EPO consumer_key and consumer_secret required")

        credentials = f"{self.consumer_key}:{self.consumer_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()

        client = await self._get_client()

        resp = await client.post(
            self.AUTH_URL,
            headers={
                "Authorization": f"Basic {encoded}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials"},
        )
        resp.raise_for_status()
        data = resp.json()

        self._access_token = data["access_token"]
        expires_in = int(data.get("expires_in", 1200))
        self._token_expires = datetime.now() + timedelta(seconds=expires_in)

        return self._access_token

    async def search(
        self,
        query: str,
        limit: int = 100,
        **kwargs,
    ) -> List[Document]:
        """
        Search EPO patents by CQL query.

        Args:
            query: CQL query string
            limit: Maximum results

        Returns:
            List of Document objects
        """
        # This is a placeholder - EPO search requires more complex implementation
        logger.warning("EPO text search not fully implemented")
        return []

    async def search_by_ipc(
        self,
        ipc_codes: List[str],
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 100,
    ) -> List[Document]:
        """
        Search patents by IPC classification.

        Args:
            ipc_codes: List of IPC prefixes
            date_from: Start date (YYYYMMDD)
            date_to: End date (YYYYMMDD)
            limit: Maximum results

        Returns:
            List of Document objects
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required")

        # Build CQL query
        ipc_query = " OR ".join(f"ic={ipc}" for ipc in ipc_codes)
        query = f"({ipc_query})"

        if date_from:
            query += f" AND pd>={date_from}"
        if date_to:
            query += f" AND pd<={date_to}"

        try:
            access_token = await self._get_access_token()
            client = await self._get_client()

            pub_numbers = []
            range_begin = 1
            batch_size = 100

            while len(pub_numbers) < limit:
                await self._rate_limiter.acquire()

                range_end = range_begin + batch_size - 1
                params = {
                    "q": query,
                    "Range": f"{range_begin}-{range_end}",
                }

                resp = await client.get(
                    f"{self.API_BASE}/published-data/search/biblio",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/json",
                    },
                    params=params,
                )

                if resp.status_code == 404:
                    break

                resp.raise_for_status()
                data = resp.json()

                # Parse publication numbers
                batch = self._parse_search_results(data)
                if not batch:
                    break

                pub_numbers.extend(batch)
                range_begin = range_end + 1

                if len(batch) < batch_size:
                    break

            # Fetch full data for each patent
            documents = []
            for pub_num in pub_numbers[:limit]:
                doc = await self._get_publication(pub_num)
                if doc:
                    documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"EPO search failed: {e}")
            return []

    async def _get_publication(self, pub_number: str) -> Optional[Document]:
        """Fetch publication data."""
        try:
            access_token = await self._get_access_token()
            client = await self._get_client()

            await self._rate_limiter.acquire()

            resp = await client.get(
                f"{self.API_BASE}/published-data/publication/epodoc/{pub_number}/biblio",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                },
            )

            if resp.status_code == 404:
                return None

            resp.raise_for_status()
            data = resp.json()

            return self._parse_biblio(data, pub_number)

        except Exception as e:
            logger.warning(f"Failed to fetch {pub_number}: {e}")
            return None

    def _parse_search_results(self, data: Dict[str, Any]) -> List[str]:
        """Parse search results for publication numbers."""
        results = []
        try:
            search_result = data.get("ops:world-patent-data", {})
            search_result = search_result.get("ops:biblio-search", {})
            search_result = search_result.get("ops:search-result", {})

            pub_refs = search_result.get("ops:publication-reference", [])
            if not isinstance(pub_refs, list):
                pub_refs = [pub_refs]

            for ref in pub_refs:
                doc_id = ref.get("document-id", {})
                if isinstance(doc_id, list):
                    doc_id = doc_id[0]

                country = doc_id.get("country", {}).get("$", "")
                doc_num = doc_id.get("doc-number", {}).get("$", "")
                kind = doc_id.get("kind", {}).get("$", "")

                if doc_num:
                    results.append(f"{country}{doc_num}{kind}")

        except Exception as e:
            logger.warning(f"Failed to parse EPO search results: {e}")

        return results

    def _parse_biblio(self, data: Dict[str, Any], pub_number: str) -> Optional[Document]:
        """Parse bibliographic data."""
        try:
            world_patent = data.get("ops:world-patent-data", {})
            exchange_docs = world_patent.get("exchange-documents", {})
            exchange_doc = exchange_docs.get("exchange-document", {})

            if isinstance(exchange_doc, list):
                exchange_doc = exchange_doc[0]

            if not exchange_doc:
                return None

            bib_data = exchange_doc.get("bibliographic-data", {})

            # Title
            title = ""
            inv_titles = bib_data.get("invention-title", [])
            if not isinstance(inv_titles, list):
                inv_titles = [inv_titles]
            for t in inv_titles:
                if t.get("@lang") == "en":
                    title = t.get("$", "")
                    break
            if not title and inv_titles:
                title = inv_titles[0].get("$", "")

            # Abstract
            abstract = ""
            abstracts = exchange_doc.get("abstract", [])
            if not isinstance(abstracts, list):
                abstracts = [abstracts]
            for a in abstracts:
                if a.get("@lang") == "en":
                    p = a.get("p", {})
                    if isinstance(p, dict):
                        abstract = p.get("$", "")
                    break

            # IPC codes
            ipc_codes = []
            class_data = bib_data.get("classifications-ipcr", {})
            class_list = class_data.get("classification-ipcr", [])
            if not isinstance(class_list, list):
                class_list = [class_list]
            for c in class_list:
                text = c.get("text", {})
                if isinstance(text, dict):
                    code = text.get("$", "")
                else:
                    code = str(text)
                if code:
                    ipc_codes.append(code.strip())

            return Document(
                id=f"patent:{pub_number}",
                source="patent",
                title=title,
                text=abstract,
                ipc_codes=ipc_codes,
                url=f"https://worldwide.espacenet.com/patent/search?q=pn%3D{pub_number}",
                metadata={"publication_number": pub_number},
            )

        except Exception as e:
            logger.warning(f"Failed to parse EPO biblio: {e}")
            return None

    async def health_check(self) -> bool:
        """Check if EPO API is accessible."""
        try:
            await self._get_access_token()
            return True
        except Exception:
            return False


def create_patentsview_connector(
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> PatentsViewConnector:
    """Factory function for PatentsView connector."""
    from pathlib import Path

    config = ConnectorConfig(
        api_key=api_key,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )
    return PatentsViewConnector(config=config)


def create_epo_connector(
    consumer_key: Optional[str] = None,
    consumer_secret: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> EPOConnector:
    """Factory function for EPO connector."""
    from pathlib import Path

    config = ConnectorConfig(
        cache_dir=Path(cache_dir) if cache_dir else None,
    )
    return EPOConnector(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        config=config,
    )
