"""PubChem Connector for chemical data.

Provides access to PubChem for:
- Compound search by name, SMILES, InChI
- SMILES retrieval by compound ID
- Chemical property data

API Documentation: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
"""

import logging
import re
from typing import Any, Dict, List, Optional

from biopat.data.base import BaseConnector, ConnectorConfig, Document

logger = logging.getLogger(__name__)

# Try to import httpx
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class PubChemConnector(BaseConnector):
    """Connector for PubChem PUG REST API.

    Example:
        ```python
        async with PubChemConnector() as pubchem:
            # Search by name
            compounds = await pubchem.search("aspirin", limit=10)

            # Get SMILES by name
            smiles = await pubchem.get_smiles("pembrolizumab")

            # Get compound by CID
            compound = await pubchem.get_compound(2244)  # Aspirin
        ```
    """

    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def __init__(self, config: Optional[ConnectorConfig] = None):
        """Initialize PubChem connector."""
        if config is None:
            config = ConnectorConfig(rate_limit=5.0)  # PubChem allows ~5 req/sec
        super().__init__(config)

    @property
    def source_name(self) -> str:
        return "pubchem"

    async def search(
        self,
        query: str,
        limit: int = 100,
        **kwargs,
    ) -> List[Document]:
        """
        Search PubChem compounds by name.

        Args:
            query: Compound name or keyword
            limit: Maximum results

        Returns:
            List of Document objects with compound info
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required. Install with: pip install httpx")

        try:
            # Search for CIDs by name
            cids = await self._search_by_name(query, limit)
            if not cids:
                return []

            # Fetch compound data
            return await self._get_compounds_batch(cids)

        except Exception as e:
            logger.error(f"PubChem search failed: {e}")
            return []

    async def get_smiles(self, name: str) -> Optional[str]:
        """
        Get canonical SMILES for a compound by name.

        Args:
            name: Compound name

        Returns:
            SMILES string or None
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required")

        try:
            await self._rate_limiter.acquire()
            client = await self._get_client()

            # URL encode the name
            url = f"{self.BASE_URL}/compound/name/{name}/property/CanonicalSMILES/JSON"

            resp = await client.get(url)

            if resp.status_code == 404:
                return None

            resp.raise_for_status()
            data = resp.json()

            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                return props[0].get("CanonicalSMILES")

            return None

        except Exception as e:
            logger.warning(f"Failed to get SMILES for {name}: {e}")
            return None

    async def get_smiles_batch(self, names: List[str]) -> Dict[str, str]:
        """
        Get SMILES for multiple compounds.

        Args:
            names: List of compound names

        Returns:
            Dict mapping name to SMILES
        """
        results = {}
        for name in names:
            smiles = await self.get_smiles(name)
            if smiles:
                results[name] = smiles
        return results

    async def get_compound(self, cid: int) -> Optional[Document]:
        """
        Get compound by PubChem CID.

        Args:
            cid: PubChem Compound ID

        Returns:
            Document with compound info
        """
        compounds = await self._get_compounds_batch([cid])
        return compounds[0] if compounds else None

    async def get_compound_by_smiles(self, smiles: str) -> Optional[Document]:
        """
        Get compound info by SMILES.

        Args:
            smiles: SMILES string

        Returns:
            Document with compound info
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required")

        try:
            await self._rate_limiter.acquire()
            client = await self._get_client()

            # Search by SMILES
            url = f"{self.BASE_URL}/compound/smiles/{smiles}/cids/JSON"
            resp = await client.get(url)

            if resp.status_code == 404:
                return None

            resp.raise_for_status()
            data = resp.json()

            cids = data.get("IdentifierList", {}).get("CID", [])
            if cids:
                return await self.get_compound(cids[0])

            return None

        except Exception as e:
            logger.warning(f"Failed to get compound by SMILES: {e}")
            return None

    async def _search_by_name(self, name: str, limit: int) -> List[int]:
        """Search for CIDs by compound name."""
        await self._rate_limiter.acquire()
        client = await self._get_client()

        url = f"{self.BASE_URL}/compound/name/{name}/cids/JSON"
        params = {"MaxRecords": limit}

        resp = await client.get(url, params=params)

        if resp.status_code == 404:
            return []

        resp.raise_for_status()
        data = resp.json()

        return data.get("IdentifierList", {}).get("CID", [])[:limit]

    async def _get_compounds_batch(self, cids: List[int]) -> List[Document]:
        """Fetch compound data for multiple CIDs."""
        if not cids:
            return []

        await self._rate_limiter.acquire()
        client = await self._get_client()

        # Request properties
        cid_str = ",".join(str(c) for c in cids)
        properties = [
            "Title",
            "CanonicalSMILES",
            "IsomericSMILES",
            "MolecularFormula",
            "MolecularWeight",
            "IUPACName",
        ]
        prop_str = ",".join(properties)

        url = f"{self.BASE_URL}/compound/cid/{cid_str}/property/{prop_str}/JSON"

        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

        props_list = data.get("PropertyTable", {}).get("Properties", [])

        documents = []
        for props in props_list:
            cid = props.get("CID")
            if not cid:
                continue

            title = props.get("Title") or props.get("IUPACName") or f"CID:{cid}"
            smiles = props.get("CanonicalSMILES") or props.get("IsomericSMILES")

            # Build description
            desc_parts = []
            if props.get("MolecularFormula"):
                desc_parts.append(f"Formula: {props['MolecularFormula']}")
            if props.get("MolecularWeight"):
                desc_parts.append(f"MW: {props['MolecularWeight']}")
            if props.get("IUPACName"):
                desc_parts.append(f"IUPAC: {props['IUPACName']}")

            documents.append(Document(
                id=f"pubchem:{cid}",
                source="pubchem",
                title=title,
                text=" | ".join(desc_parts) if desc_parts else f"PubChem compound {cid}",
                smiles=smiles,
                url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
                metadata={
                    "cid": cid,
                    "molecular_formula": props.get("MolecularFormula"),
                    "molecular_weight": props.get("MolecularWeight"),
                    "iupac_name": props.get("IUPACName"),
                },
            ))

        return documents

    async def health_check(self) -> bool:
        """Check if PubChem API is accessible."""
        try:
            client = await self._get_client()
            resp = await client.get(
                f"{self.BASE_URL}/compound/name/aspirin/cids/JSON",
                timeout=5.0,
            )
            return resp.status_code == 200
        except Exception:
            return False


def create_pubchem_connector(
    cache_dir: Optional[str] = None,
) -> PubChemConnector:
    """Factory function for PubChem connector."""
    from pathlib import Path

    config = ConnectorConfig(
        cache_dir=Path(cache_dir) if cache_dir else None,
    )
    return PubChemConnector(config=config)
