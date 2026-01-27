"""PubChem PUG-REST API Client Wrapper.

Phase 4.0 (Advanced): Provides access to PubChem chemical data
using the Power User Gateway (PUG-REST) interface.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Union

import httpx
from diskcache import Cache
from pathlib import Path

logger = logging.getLogger(__name__)

PUG_REST_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


class PubChemClient:
    """Wrapper for PubChem PUG-REST API.
    
    Provides high-level methods for resolving chemical identifiers
    and fetching associated literature (PubMed) links.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        rate_limit: int = 5,  # requests per second
    ):
        """Initialize PubChem client.
        
        Args:
            cache_dir: Optional directory for caching responses.
            rate_limit: Maximum requests per second.
        """
        self.rate_limit = rate_limit
        self.cache = Cache(str(cache_dir / "pubchem")) if cache_dir else None
        self._semaphore = asyncio.Semaphore(rate_limit)

    async def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Make rate-limited request to PubChem."""
        url = f"{PUG_REST_BASE}/{endpoint}"
        
        async with self._semaphore:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(url, params=params, timeout=30.0)
                    if response.status_code == 404:
                        return None
                    response.raise_for_status()
                    return response.json()
                except Exception as e:
                    logger.error(f"PubChem request failed: {url} - {e}")
                    return None

    async def get_cid_by_structure(
        self, 
        structure: str, 
        id_type: str = "smiles"
    ) -> Optional[int]:
        """Resolve a structure to its PubChem Compound ID (CID)."""
        cache_key = f"cid:{id_type}:{structure}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        endpoint = f"compound/{id_type}/{structure}/cids/JSON"
        data = await self._make_request(endpoint)
        
        if data and "IdentifierList" in data:
            cid = data["IdentifierList"]["CID"][0]
            if self.cache:
                self.cache[cache_key] = cid
            return cid
        return None

    async def get_properties(
        self, 
        cid: int, 
        properties: List[str] = ["CanonicalSMILES", "InChIKey", "MolecularFormula", "MolecularWeight"]
    ) -> Optional[Dict[str, Any]]:
        """Fetch molecular properties for a given CID."""
        cache_key = f"props:{cid}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        prop_str = ",".join(properties)
        endpoint = f"compound/cid/{cid}/property/{prop_str}/JSON"
        data = await self._make_request(endpoint)
        
        if data and "PropertyTable" in data:
            props = data["PropertyTable"]["Properties"][0]
            if self.cache:
                self.cache[cache_key] = props
            return props
        return None

    async def get_pubmed_links(self, cid: int) -> List[int]:
        """Fetch PubMed IDs (PMIDs) associated with a CID."""
        cache_key = f"pubmed:{cid}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        # Use the xrefs endpoint for literature links
        endpoint = f"compound/cid/{cid}/xrefs/PubMedID/JSON"
        data = await self._make_request(endpoint)
        
        pmids = []
        if data and "InformationList" in data:
            info = data["InformationList"]["Information"][0]
            pmids = info.get("PubMedID", [])
            
            if self.cache:
                self.cache[cache_key] = pmids
        return pmids

    async def resolve_full_entity(self, structure: str, id_type: str = "smiles") -> Optional[Dict[str, Any]]:
        """Fully resolve a structure to CID, properties, and literature links."""
        cid = await self.get_cid_by_structure(structure, id_type)
        if not cid:
            return None
            
        props = await self.get_properties(cid)
        pmids = await self.get_pubmed_links(cid)
        
        return {
            "pubchem_cid": cid,
            "properties": props,
            "pubmed_ids": pmids
        }

    async def search_by_name(self, name: str) -> Optional[int]:
        """Search PubChem by common name."""
        return await self.get_cid_by_structure(name, "name")
