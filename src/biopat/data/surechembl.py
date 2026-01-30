"""SureChEMBL Connector for patent-chemical mappings.

Provides access to chemical structures extracted from patents using
the SureChEMBL database. Essential for finding prior art with
similar chemical structures.

API Documentation: https://www.surechembl.org/
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from biopat.data.base import BaseConnector, ConnectorConfig, Document

logger = logging.getLogger(__name__)

# Try to import httpx
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

SURECHEMBL_API_BASE = "https://www.surechembl.org/api"


@dataclass
class SureChEMBLChemical:
    """Chemical structure from SureChEMBL."""

    surechembl_id: str
    smiles: Optional[str] = None
    inchi: Optional[str] = None
    inchikey: Optional[str] = None
    name: Optional[str] = None
    molecular_formula: Optional[str] = None
    molecular_weight: Optional[float] = None

    # Extraction metadata
    patent_id: Optional[str] = None
    mention_count: int = 0
    extraction_type: Optional[str] = None  # CHEMICAL_NAME, SMILES, etc.

    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_document(self) -> Document:
        """Convert to Document."""
        desc_parts = []
        if self.molecular_formula:
            desc_parts.append(f"Formula: {self.molecular_formula}")
        if self.molecular_weight:
            desc_parts.append(f"MW: {self.molecular_weight:.2f}")
        if self.patent_id:
            desc_parts.append(f"Patent: {self.patent_id}")

        return Document(
            id=f"surechembl:{self.surechembl_id}",
            source="surechembl",
            title=self.name or f"SureChEMBL {self.surechembl_id}",
            text=" | ".join(desc_parts) if desc_parts else f"Chemical from patent",
            smiles=self.smiles,
            url=f"https://www.surechembl.org/chemical/{self.surechembl_id}",
            metadata={
                "surechembl_id": self.surechembl_id,
                "inchi": self.inchi,
                "inchikey": self.inchikey,
                "patent_id": self.patent_id,
                "mention_count": self.mention_count,
                "extraction_type": self.extraction_type,
            },
        )


class SureChEMBLConnector(BaseConnector):
    """Connector for SureChEMBL API.

    Provides methods for:
    - Retrieving chemicals extracted from patents
    - Structure similarity search
    - Patent-chemical mappings

    Example:
        ```python
        async with SureChEMBLConnector(api_key="your-key") as surechembl:
            # Get chemicals from a patent
            chemicals = await surechembl.get_patent_chemicals("US10500001")

            # Similarity search
            similar = await surechembl.similarity_search(
                "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
                threshold=0.7
            )

            # Find patents for a chemical
            patents = await surechembl.get_patents_for_chemical("SCHEMBL123456")
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ConnectorConfig] = None,
    ):
        """
        Initialize SureChEMBL connector.

        Args:
            api_key: SureChEMBL API key (required for some endpoints)
            config: Connector configuration
        """
        if config is None:
            config = ConnectorConfig(
                api_key=api_key,
                rate_limit=2.0,  # SureChEMBL has strict rate limits
            )
        elif api_key:
            config.api_key = api_key

        super().__init__(config)

        if not api_key:
            logger.warning("SureChEMBL API key not provided. Some endpoints may not work.")

    @property
    def source_name(self) -> str:
        return "surechembl"

    def _get_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["apikey"] = self.config.api_key
        return headers

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Make rate-limited API request."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required. Install with: pip install httpx")

        await self._rate_limiter.acquire()

        client = await self._get_client()
        url = f"{SURECHEMBL_API_BASE}/{endpoint}"
        headers = self._get_headers()

        try:
            response = await client.request(
                method,
                url,
                params=params,
                json=json_body,
                headers=headers,
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.warning(f"SureChEMBL request failed: {e}")
            return None

    async def search(
        self,
        query: str,
        limit: int = 100,
        **kwargs,
    ) -> List[Document]:
        """
        Search SureChEMBL (by name or structure).

        Args:
            query: Chemical name or SMILES
            limit: Maximum results

        Returns:
            List of Document objects
        """
        # Try as SMILES first
        if self._looks_like_smiles(query):
            chemicals = await self.similarity_search(query, threshold=0.7, max_results=limit)
        else:
            # Text search not directly supported, use similarity with parsed name
            chemicals = await self._search_by_name(query, limit)

        return [chem.to_document() for chem in chemicals]

    def _looks_like_smiles(self, text: str) -> bool:
        """Check if text looks like a SMILES string."""
        smiles_chars = set("CNOSPFClBrI=#()-+[]@/\\0123456789cnops")
        return len(text) > 3 and all(c in smiles_chars for c in text)

    async def _search_by_name(
        self,
        name: str,
        limit: int,
    ) -> List[SureChEMBLChemical]:
        """Search by chemical name."""
        # SureChEMBL doesn't have direct name search, so this is limited
        # Could be enhanced with PubChem name->SMILES->SureChEMBL pipeline
        logger.warning("SureChEMBL name search is limited. Consider using SMILES.")
        return []

    async def get_patent_chemicals(
        self,
        patent_id: str,
    ) -> List[SureChEMBLChemical]:
        """
        Get chemicals extracted from a patent.

        Args:
            patent_id: Patent number (e.g., "US10500001", "EP1234567")

        Returns:
            List of SureChEMBLChemical objects
        """
        # Normalize patent ID
        normalized_id = self._normalize_patent_id(patent_id)

        data = await self._make_request(
            "GET",
            f"document/{normalized_id}/contents",
        )

        chemicals = []
        if data and "data" in data:
            for chem in data["data"].get("chemicals", []):
                chemicals.append(SureChEMBLChemical(
                    surechembl_id=chem.get("schemblId", ""),
                    smiles=chem.get("smiles"),
                    inchi=chem.get("inchi"),
                    inchikey=chem.get("inchiKey"),
                    name=chem.get("name"),
                    molecular_formula=chem.get("molecularFormula"),
                    molecular_weight=chem.get("molecularWeight"),
                    patent_id=patent_id,
                    mention_count=chem.get("mentionCount", 0),
                    extraction_type=chem.get("extractionType"),
                    raw_data=chem,
                ))

        logger.info(f"Found {len(chemicals)} chemicals in patent {patent_id}")
        return chemicals

    async def get_patent_chemicals_batch(
        self,
        patent_ids: List[str],
    ) -> Dict[str, List[SureChEMBLChemical]]:
        """
        Get chemicals for multiple patents.

        Args:
            patent_ids: List of patent IDs

        Returns:
            Dict mapping patent_id to list of chemicals
        """
        results = {}
        for patent_id in patent_ids:
            chemicals = await self.get_patent_chemicals(patent_id)
            results[patent_id] = chemicals
        return results

    async def similarity_search(
        self,
        smiles: str,
        threshold: float = 0.7,
        max_results: int = 100,
    ) -> List[SureChEMBLChemical]:
        """
        Search for similar chemicals by Tanimoto similarity.

        Args:
            smiles: Query SMILES string
            threshold: Minimum Tanimoto similarity (0-1)
            max_results: Maximum results

        Returns:
            List of similar SureChEMBLChemical objects
        """
        data = await self._make_request(
            "POST",
            "search/structure",
            json_body={
                "struct": smiles,
                "structSearchType": "SIMILARITY",
                "threshold": threshold,
                "maxResults": max_results,
            },
        )

        chemicals = []
        if data and "data" in data:
            for result in data["data"].get("results", []):
                chemicals.append(SureChEMBLChemical(
                    surechembl_id=result.get("schemblId", ""),
                    smiles=result.get("smiles"),
                    inchi=result.get("inchi"),
                    inchikey=result.get("inchiKey"),
                    name=result.get("name"),
                    raw_data=result,
                ))

        return chemicals

    async def substructure_search(
        self,
        smiles: str,
        max_results: int = 100,
    ) -> List[SureChEMBLChemical]:
        """
        Search for chemicals containing a substructure.

        Args:
            smiles: Query SMILES for substructure
            max_results: Maximum results

        Returns:
            List of matching chemicals
        """
        data = await self._make_request(
            "POST",
            "search/structure",
            json_body={
                "struct": smiles,
                "structSearchType": "SUBSTRUCTURE",
                "maxResults": max_results,
            },
        )

        chemicals = []
        if data and "data" in data:
            for result in data["data"].get("results", []):
                chemicals.append(SureChEMBLChemical(
                    surechembl_id=result.get("schemblId", ""),
                    smiles=result.get("smiles"),
                    inchi=result.get("inchi"),
                    inchikey=result.get("inchiKey"),
                    name=result.get("name"),
                    raw_data=result,
                ))

        return chemicals

    async def get_chemical(
        self,
        surechembl_id: str,
    ) -> Optional[SureChEMBLChemical]:
        """
        Get chemical metadata by SureChEMBL ID.

        Args:
            surechembl_id: SureChEMBL ID (e.g., "SCHEMBL123456")

        Returns:
            SureChEMBLChemical or None
        """
        data = await self._make_request(
            "GET",
            f"chemical/id/{surechembl_id}",
        )

        if data and "data" in data:
            chem_data = data["data"]
            return SureChEMBLChemical(
                surechembl_id=surechembl_id,
                smiles=chem_data.get("smiles"),
                inchi=chem_data.get("inchi"),
                inchikey=chem_data.get("inchiKey"),
                name=chem_data.get("name"),
                molecular_formula=chem_data.get("molecularFormula"),
                molecular_weight=chem_data.get("molecularWeight"),
                raw_data=chem_data,
            )

        return None

    async def get_patents_for_chemical(
        self,
        surechembl_id: str,
        max_results: int = 100,
    ) -> List[str]:
        """
        Find patents containing a specific chemical.

        Args:
            surechembl_id: SureChEMBL compound ID
            max_results: Maximum number of patents

        Returns:
            List of patent IDs
        """
        data = await self._make_request(
            "POST",
            "search/documents_for_structures",
            json_body={
                "chemicalIds": [surechembl_id],
                "maxResults": max_results,
            },
        )

        if data and "data" in data:
            return [doc.get("scbdId", "") for doc in data["data"].get("results", [])]

        return []

    def _normalize_patent_id(self, patent_id: str) -> str:
        """Normalize patent ID for SureChEMBL API."""
        return patent_id.replace(" ", "").replace("-", "").upper()

    async def health_check(self) -> bool:
        """Check if SureChEMBL API is accessible."""
        try:
            client = await self._get_client()
            resp = await client.get(
                f"{SURECHEMBL_API_BASE}/chemical/id/SCHEMBL1",
                headers=self._get_headers(),
                timeout=5.0,
            )
            return resp.status_code in [200, 404]
        except Exception:
            return False


def create_surechembl_connector(
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> SureChEMBLConnector:
    """Factory function for SureChEMBL connector.

    Args:
        api_key: SureChEMBL API key
        cache_dir: Directory for caching

    Returns:
        Configured SureChEMBLConnector
    """
    from pathlib import Path

    config = ConnectorConfig(
        api_key=api_key,
        cache_dir=Path(cache_dir) if cache_dir else None,
        rate_limit=2.0,
    )
    return SureChEMBLConnector(config=config)
