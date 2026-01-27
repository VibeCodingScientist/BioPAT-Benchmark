"""UniProt REST API Client Wrapper.

Phase 4.0 (Advanced): Provides access to protein sequence data and
annotations using the UniProt REST API.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Union

import httpx
from diskcache import Cache
from pathlib import Path

logger = logging.getLogger(__name__)

UNIPROT_API_BASE = "https://rest.uniprot.org/uniprotkb"


class UniProtClient:
    """Wrapper for UniProt REST API.
    
    Provides high-level methods for fetching protein sequences,
    metadata, and database cross-references.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        rate_limit: int = 5,  # UniProt is generous but prefers polite clients
    ):
        """Initialize UniProt client.
        
        Args:
            cache_dir: Optional directory for caching responses.
            rate_limit: Maximum requests per second.
        """
        self.rate_limit = rate_limit
        self.cache = Cache(str(cache_dir / "uniprot")) if cache_dir else None
        self._semaphore = asyncio.Semaphore(rate_limit)

    async def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        format: str = "json"
    ) -> Optional[Union[Dict[str, Any], str]]:
        """Make rate-limited request to UniProt."""
        url = f"{UNIPROT_API_BASE}/{endpoint}"
        if format != "json" and not endpoint.endswith(f".{format}"):
            url = f"{url}.{format}"
        
        async with self._semaphore:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(url, params=params, timeout=60.0)
                    if response.status_code == 404:
                        return None
                    response.raise_for_status()
                    
                    if format == "json":
                        return response.json()
                    return response.text
                except Exception as e:
                    logger.error(f"UniProt request failed: {url} - {e}")
                    return None

    async def get_protein_by_accession(self, accession: str) -> Optional[Dict[str, Any]]:
        """Fetch protein record by UniProt accession ID."""
        cache_key = f"acc:{accession}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        data = await self._make_request(accession)
        
        if data and self.cache:
            self.cache[cache_key] = data
        return data

    async def get_sequence_fasta(self, accession: str) -> Optional[str]:
        """Fetch protein sequence in FASTA format."""
        cache_key = f"fasta:{accession}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        fasta = await self._make_request(accession, format="fasta")
        
        if fasta and self.cache:
            self.cache[cache_key] = fasta
        return fasta

    async def search_proteins(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search UniProtKB for proteins matching a query string."""
        params = {
            "query": query,
            "size": limit,
            "fields": "accession,protein_name,gene_names,organism_name,sequence"
        }
        
        data = await self._make_request("search", params=params)
        
        if data and "results" in data:
            return data["results"]
        return []

    async def get_pubmed_ids(self, accession: str) -> List[str]:
        """Extract PubMed IDs (PMIDs) associated with a protein accession."""
        cache_key = f"pubmed:{accession}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        data = await self.get_protein_by_accession(accession)
        pmids = []
        
        if data and "references" in data:
            for ref in data["references"]:
                citation = ref.get("citation", {})
                db_refs = citation.get("dbReferences", [])
                for db_ref in db_refs:
                    if db_ref.get("database") == "PubMed":
                        pmids.append(db_ref.get("id"))
        
        if pmids and self.cache:
            self.cache[cache_key] = pmids
        return sorted(list(set(pmids)))

    async def resolve_patent_references(self, accession: str) -> List[str]:
        """Identify patent identifiers linked to this UniProt accession."""
        data = await self.get_protein_by_accession(accession)
        patents = []
        
        if data and "extraAttributes" in data:
            # Note: UniProt response structure for patent links can be deep
            # This is a simplified extraction logic
            for ref in data.get("references", []):
                citation = ref.get("citation", {})
                if citation.get("citationType") == "patent":
                    patent_id = citation.get("number")
                    if patent_id:
                        patents.append(patent_id)
        
        return sorted(list(set(patents)))
