"""NCBI Sequence Connectors (GenBank, Protein, Nucleotide).

Provides access to NCBI sequence databases via E-utilities:
- GenBank/GenPept for patent sequences
- Protein database for protein sequences
- Nucleotide database for DNA/RNA sequences

Uses the same API as PubMed (E-utilities).

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


class NCBISequenceConnector(BaseConnector):
    """Connector for NCBI Sequence databases (GenBank, Protein, Nucleotide).

    Supports fetching:
    - Patent sequences from GenBank/GenPept
    - Protein sequences from NCBI Protein
    - Nucleotide sequences from NCBI Nucleotide

    Example:
        ```python
        async with NCBISequenceConnector(api_key="your-key") as ncbi:
            # Search patent sequences
            seqs = await ncbi.search_patent_sequences(
                "pembrolizumab antibody",
                limit=50
            )

            # Search protein sequences
            proteins = await ncbi.search_protein("PD-1 human", limit=100)

            # Get sequence by accession
            seq = await ncbi.get_sequence("AAA123456")
        ```
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # Database names
    DB_PROTEIN = "protein"
    DB_NUCLEOTIDE = "nucleotide"
    DB_GENE = "gene"

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ConnectorConfig] = None,
    ):
        """
        Initialize NCBI sequence connector.

        Args:
            api_key: NCBI API key (increases rate limit from 3 to 10 req/sec)
            config: Connector configuration
        """
        if config is None:
            config = ConnectorConfig(api_key=api_key)
        elif api_key:
            config.api_key = api_key

        # NCBI rate limits
        if config.api_key:
            config.rate_limit = 10.0
        else:
            config.rate_limit = 3.0

        super().__init__(config)
        self._last_request_ts = 0.0
        self._rate_lock = asyncio.Lock()

    @property
    def source_name(self) -> str:
        return "ncbi_sequences"

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
        """Make rate-limited GET request."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required. Install with: pip install httpx")

        if self.config.api_key:
            params["api_key"] = self.config.api_key

        client = await self._get_client()
        last_exc: Optional[Exception] = None

        for attempt in range(retries):
            await self._throttle()

            try:
                resp = await client.get(url, params=params)

                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"NCBI rate limited. Waiting {wait}s...")
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

    async def search(
        self,
        query: str,
        limit: int = 100,
        database: str = "protein",
        **kwargs,
    ) -> List[Document]:
        """
        Search NCBI sequence databases.

        Args:
            query: Search query
            limit: Maximum results
            database: "protein", "nucleotide", or "gene"

        Returns:
            List of Document objects with sequences
        """
        if database == "protein":
            return await self.search_protein(query, limit)
        elif database == "nucleotide":
            return await self.search_nucleotide(query, limit)
        else:
            return await self.search_protein(query, limit)

    async def search_protein(
        self,
        query: str,
        limit: int = 100,
        organism: Optional[str] = None,
    ) -> List[Document]:
        """
        Search NCBI Protein database.

        Args:
            query: Search query
            limit: Maximum results
            organism: Filter by organism (e.g., "human", "Homo sapiens")

        Returns:
            List of Document objects with protein sequences
        """
        search_query = query
        if organism:
            search_query += f" AND {organism}[Organism]"

        return await self._search_database(self.DB_PROTEIN, search_query, limit)

    async def search_nucleotide(
        self,
        query: str,
        limit: int = 100,
        organism: Optional[str] = None,
    ) -> List[Document]:
        """
        Search NCBI Nucleotide database.

        Args:
            query: Search query
            limit: Maximum results
            organism: Filter by organism

        Returns:
            List of Document objects with nucleotide sequences
        """
        search_query = query
        if organism:
            search_query += f" AND {organism}[Organism]"

        return await self._search_database(self.DB_NUCLEOTIDE, search_query, limit)

    async def search_patent_sequences(
        self,
        query: str,
        limit: int = 100,
        sequence_type: str = "protein",
    ) -> List[Document]:
        """
        Search for patent sequences in GenBank/GenPept.

        Args:
            query: Search query
            limit: Maximum results
            sequence_type: "protein" or "nucleotide"

        Returns:
            List of Document objects with patent sequences
        """
        # Add patent filter to query
        patent_query = f"({query}) AND patent[filter]"

        db = self.DB_PROTEIN if sequence_type == "protein" else self.DB_NUCLEOTIDE
        return await self._search_database(db, patent_query, limit)

    async def _search_database(
        self,
        database: str,
        query: str,
        limit: int,
    ) -> List[Document]:
        """Execute search on an NCBI database."""
        try:
            # Step 1: Search for IDs
            search_params = {
                "db": database,
                "term": query,
                "retmax": limit,
                "retmode": "json",
            }

            resp = await self._get(f"{self.BASE_URL}/esearch.fcgi", search_params)
            data = resp.json()

            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            # Step 2: Fetch sequences
            return await self._fetch_sequences(database, id_list)

        except Exception as e:
            logger.error(f"NCBI sequence search failed: {e}")
            return []

    async def _fetch_sequences(
        self,
        database: str,
        ids: List[str],
    ) -> List[Document]:
        """Fetch sequences by ID."""
        # Fetch in GenPept/GenBank format
        fetch_params = {
            "db": database,
            "id": ",".join(ids),
            "rettype": "gp" if database == self.DB_PROTEIN else "gb",
            "retmode": "xml",
        }

        resp = await self._get(f"{self.BASE_URL}/efetch.fcgi", fetch_params)
        return self._parse_sequence_xml(resp.content, database)

    async def get_sequence(
        self,
        accession: str,
        database: str = "protein",
    ) -> Optional[Document]:
        """
        Get sequence by accession number.

        Args:
            accession: NCBI accession (e.g., "AAA123456", "NP_001234")
            database: "protein" or "nucleotide"

        Returns:
            Document with sequence or None
        """
        try:
            docs = await self._fetch_sequences(database, [accession])
            return docs[0] if docs else None
        except Exception as e:
            logger.error(f"Failed to get sequence {accession}: {e}")
            return None

    async def get_sequence_fasta(
        self,
        accession: str,
        database: str = "protein",
    ) -> Optional[str]:
        """
        Get sequence in FASTA format.

        Args:
            accession: NCBI accession
            database: "protein" or "nucleotide"

        Returns:
            FASTA string or None
        """
        try:
            fetch_params = {
                "db": database,
                "id": accession,
                "rettype": "fasta",
                "retmode": "text",
            }

            resp = await self._get(f"{self.BASE_URL}/efetch.fcgi", fetch_params)
            return resp.text

        except Exception as e:
            logger.error(f"Failed to get FASTA for {accession}: {e}")
            return None

    def _parse_sequence_xml(
        self,
        content: bytes,
        database: str,
    ) -> List[Document]:
        """Parse NCBI sequence XML."""
        documents = []

        try:
            root = ET.fromstring(content)

            # Find sequence entries (GBSeq for GenBank format)
            for seq_elem in root.findall(".//GBSeq"):
                try:
                    doc = self._parse_gbseq(seq_elem, database)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to parse sequence: {e}")

        except Exception as e:
            logger.error(f"Failed to parse NCBI XML: {e}")

        return documents

    def _parse_gbseq(
        self,
        seq_elem: ET.Element,
        database: str,
    ) -> Optional[Document]:
        """Parse GBSeq element."""
        # Accession
        accession = seq_elem.findtext("GBSeq_primary-accession", "")
        if not accession:
            accession = seq_elem.findtext("GBSeq_accession-version", "")

        if not accession:
            return None

        # Definition (title)
        definition = seq_elem.findtext("GBSeq_definition", "")

        # Organism
        organism = seq_elem.findtext("GBSeq_organism", "")

        # Sequence
        sequence = seq_elem.findtext("GBSeq_sequence", "")

        # Sequence type
        mol_type = seq_elem.findtext("GBSeq_moltype", "")
        if database == self.DB_PROTEIN or mol_type.lower() in ["aa", "protein"]:
            seq_type = "protein"
        else:
            seq_type = "nucleotide"

        # Source (check if patent)
        source = seq_elem.findtext("GBSeq_source", "")
        is_patent = "patent" in source.lower()

        # Keywords
        keywords = []
        for kw in seq_elem.findall(".//GBKeyword"):
            if kw.text:
                keywords.append(kw.text)

        # References (look for patent info)
        patent_id = None
        for ref in seq_elem.findall(".//GBReference"):
            ref_text = ref.findtext("GBReference_journal", "")
            if "patent" in ref_text.lower():
                # Try to extract patent number
                match = re.search(r'(US|EP|WO)\s*[\d,]+', ref_text)
                if match:
                    patent_id = match.group().replace(" ", "").replace(",", "")

        # Build description
        desc_parts = []
        if organism:
            desc_parts.append(f"Organism: {organism}")
        if mol_type:
            desc_parts.append(f"Type: {mol_type}")
        if keywords:
            desc_parts.append(f"Keywords: {', '.join(keywords[:5])}")

        return Document(
            id=f"ncbi:{database}:{accession}",
            source=f"ncbi_{database}",
            title=definition or f"{seq_type.title()} sequence {accession}",
            text=" | ".join(desc_parts) if desc_parts else f"NCBI {seq_type} sequence",
            sequence=sequence.upper() if sequence else None,
            sequence_type=seq_type,
            url=f"https://www.ncbi.nlm.nih.gov/{database}/{accession}",
            metadata={
                "accession": accession,
                "organism": organism,
                "mol_type": mol_type,
                "is_patent_sequence": is_patent,
                "patent_id": patent_id,
                "keywords": keywords,
                "length": len(sequence) if sequence else 0,
            },
        )

    async def health_check(self) -> bool:
        """Check if NCBI API is accessible."""
        try:
            params = {"db": "protein", "term": "insulin", "retmax": 1, "retmode": "json"}
            client = await self._get_client()
            resp = await client.get(f"{self.BASE_URL}/esearch.fcgi", params=params, timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False


def create_ncbi_sequence_connector(
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> NCBISequenceConnector:
    """Factory function for NCBI sequence connector.

    Args:
        api_key: NCBI API key
        cache_dir: Directory for caching

    Returns:
        Configured NCBISequenceConnector
    """
    from pathlib import Path

    config = ConnectorConfig(
        api_key=api_key,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )
    return NCBISequenceConnector(config=config)
