"""SureChEMBL V3 REST API Client Wrapper.

Phase 4.0 (Advanced): Provides access to SureChEMBL chemical structures
extracted from patents using the unified V3 REST API.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Union

import httpx
from diskcache import Cache
from pathlib import Path

logger = logging.getLogger(__name__)

SURECHEMBL_API_BASE = "https://www.surechembl.org/api"


class SureChEMBLClient:
    """Wrapper for SureChEMBL V3 REST API.
    
    Provides high-level methods for mapping patents to chemical structures
    and performing structural similarity searches.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        rate_limit: int = 2,  # Careful with SureChEMBL rate limits
    ):
        """Initialize SureChEMBL client.
        
        Args:
            api_key: SureChEMBL API key (required for V3).
            cache_dir: Optional directory for caching responses.
            rate_limit: Maximum requests per second.
        """
        self.api_key = api_key
        self.cache = Cache(str(cache_dir / "surechembl")) if cache_dir else None
        self._semaphore = asyncio.Semaphore(rate_limit)
        
        if not api_key:
            logger.warning("No SureChEMBL API key provided. Many V3 endpoints may fail.")

    async def _make_request(
        self, 
        method: str,
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Make rate-limited request to SureChEMBL."""
        url = f"{SURECHEMBL_API_BASE}/{endpoint}"
        headers = {}
        if self.api_key:
            headers["apikey"] = self.api_key
            
        async with self._semaphore:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.request(
                        method, url, params=params, json=json_data, 
                        headers=headers, timeout=60.0
                    )
                    if response.status_code == 404:
                        return None
                    response.raise_for_status()
                    return response.json()
                except Exception as e:
                    logger.error(f"SureChEMBL request failed: {url} - {e}")
                    return None

    async def get_document_chemistry(self, patent_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chemical structures extracted from a given patent.
        
        Args:
            patent_id: Standard patent number (e.g., "US10123456").
            
        Returns:
            List of chemical entity records.
        """
        cache_key = f"doc_chem:{patent_id}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        # Use the contents endpoint which includes extracted chemistry
        endpoint = f"document/{patent_id}/contents"
        data = await self._make_request("GET", endpoint)
        
        chemicals = []
        if data and "data" in data:
            # SureChEMBL response structure often nests chemistry under 'entities'
            # Note: This parsing logic may need refinement based on exact V3 response format
            chemicals = data["data"].get("chemicals", [])
            
            if self.cache:
                self.cache[cache_key] = chemicals
        return chemicals

    async def structure_search(
        self, 
        structure: str, 
        search_type: str = "SIMILARITY", 
        threshold: float = 0.7,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """Perform a chemical structure search.
        
        Args:
            structure: SMILES string of the query structure.
            search_type: "SIMILARITY", "SUBSTRUCTURE", or "EXACT".
            threshold: Similarity threshold (0-1).
            max_results: Max results to return.
            
        Returns:
            List of matching chemical entities.
        """
        endpoint = "search/structure"
        payload = {
            "struct": structure,
            "structSearchType": search_type.upper(),
            "maxResults": max_results
        }
        
        data = await self._make_request("POST", endpoint, json_data=payload)
        
        if data and "data" in data:
            return data["data"].get("results", [])
        return []

    async def get_chemical_metadata(self, chemical_id: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata (SMILES, InChIKey, properties) for a SureChEMBL ID.
        
        Args:
            chemical_id: SureChEMBL ID (e.g., "SCHEMBL123").
        """
        cache_key = f"chem_meta:{chemical_id}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        endpoint = f"chemical/id/{chemical_id}"
        data = await self._make_request("GET", endpoint)
        
        if data and "data" in data:
            metadata = data["data"]
            if self.cache:
                self.cache[cache_key] = metadata
            return metadata
        return None

    async def get_documents_for_chemical(self, chemical_id: str, page: int = 1) -> List[str]:
        """Find patent documents containing a specific chemical ID.
        
        Args:
            chemical_id: SureChEMBL ID.
            page: Results page.
        """
        endpoint = "search/documents_for_structures"
        params = {
            "chemicalIds": [chemical_id],
            "page": page
        }
        
        data = await self._make_request("POST", endpoint, params=params)
        
        if data and "data" in data:
            return [doc["scbdId"] for doc in data["data"].get("results", [])]
        return []
