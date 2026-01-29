"""UniProt Connector for protein data.

Provides access to UniProtKB for:
- Protein search by name, function, or sequence
- Sequence retrieval by accession
- Cross-references to PDB, PubMed, etc.

API Documentation: https://www.uniprot.org/help/api
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


class UniProtConnector(BaseConnector):
    """Connector for UniProtKB REST API.

    Example:
        ```python
        async with UniProtConnector() as uniprot:
            # Search for proteins
            proteins = await uniprot.search("PD-1 human", limit=50)

            # Get sequence by accession
            seq = await uniprot.get_sequence("P01308")  # Insulin
            print(f"Insulin sequence: {seq[:50]}...")
        ```
    """

    BASE_URL = "https://rest.uniprot.org/uniprotkb"
    HEADERS = {
        "User-Agent": "BioPAT/1.0",
        "Accept": "application/json",
    }

    def __init__(self, config: Optional[ConnectorConfig] = None):
        """Initialize UniProt connector."""
        if config is None:
            config = ConnectorConfig(rate_limit=5.0)  # UniProt allows ~5 req/sec
        super().__init__(config)

    @property
    def source_name(self) -> str:
        return "uniprot"

    def _sanitize_query(self, query: str) -> str:
        """Normalize query to UniProt-friendly syntax."""
        if not query:
            return ""

        sanitized = query

        # Replace organism:"Homo sapiens (Human) [9606]" -> organism_id:9606
        sanitized = re.sub(
            r'organism:"[^"]*\[(\d+)\]"',
            lambda m: f"organism_id:{m.group(1)}",
            sanitized,
            flags=re.IGNORECASE,
        )

        # Replace organism:... [9606] -> organism_id:9606
        sanitized = re.sub(
            r"organism:[^\s]*\[(\d+)\]",
            lambda m: f"organism_id:{m.group(1)}",
            sanitized,
            flags=re.IGNORECASE,
        )

        return sanitized

    def _simplify_query(self, query: str) -> str:
        """Fallback to simple keyword query if UniProt rejects syntax."""
        tokens = re.findall(r"[A-Za-z0-9_-]+", query)
        tokens = [t for t in tokens if t.upper() not in {"AND", "OR", "NOT"}]
        if not tokens:
            return query
        return " ".join(tokens[:6])

    async def search(
        self,
        query: str,
        limit: int = 100,
        organism: Optional[str] = None,
        reviewed: bool = True,
        **kwargs,
    ) -> List[Document]:
        """
        Search UniProtKB for proteins.

        Args:
            query: Search query (protein name, gene, function, etc.)
            limit: Maximum number of results
            organism: Filter by organism (e.g., "human", "9606")
            reviewed: If True, only return reviewed (Swiss-Prot) entries

        Returns:
            List of Document objects with protein info
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required. Install with: pip install httpx")

        # Build query
        search_query = self._sanitize_query(query)

        # Add organism filter
        if organism:
            if organism.lower() == "human":
                search_query += " AND organism_id:9606"
            elif organism.isdigit():
                search_query += f" AND organism_id:{organism}"
            else:
                search_query += f" AND organism_name:{organism}"

        # Add reviewed filter
        if reviewed:
            search_query += " AND reviewed:true"

        params = {
            "query": search_query,
            "format": "json",
            "size": min(limit, 500),
            "fields": "accession,id,protein_name,gene_names,organism_name,length,sequence,cc_function,xref_pdb,xref_pubmed",
        }

        try:
            await self._rate_limiter.acquire()
            client = await self._get_client()

            try:
                resp = await client.get(
                    f"{self.BASE_URL}/search",
                    params=params,
                    headers=self.HEADERS,
                )
                resp.raise_for_status()
                data = resp.json()

            except Exception as e:
                # Try simplified query on error
                if "400" in str(e):
                    params["query"] = self._simplify_query(search_query)
                    resp = await client.get(
                        f"{self.BASE_URL}/search",
                        params=params,
                        headers=self.HEADERS,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                else:
                    raise

            return self._parse_results(data.get("results", []))

        except Exception as e:
            logger.error(f"UniProt search failed: {e}")
            return []

    async def get_sequence(self, accession: str) -> Optional[str]:
        """
        Get protein sequence by accession.

        Args:
            accession: UniProt accession (e.g., "P01308")

        Returns:
            Amino acid sequence string or None
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required")

        try:
            await self._rate_limiter.acquire()
            client = await self._get_client()

            resp = await client.get(
                f"{self.BASE_URL}/{accession}.fasta",
                headers=self.HEADERS,
            )
            resp.raise_for_status()

            # Parse FASTA
            lines = resp.text.strip().split("\n")
            sequence = "".join(line for line in lines if not line.startswith(">"))
            return sequence

        except Exception as e:
            logger.error(f"Failed to get sequence {accession}: {e}")
            return None

    async def get_entry(self, accession: str) -> Optional[Document]:
        """
        Get full entry by accession.

        Args:
            accession: UniProt accession

        Returns:
            Document with protein info
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required")

        try:
            await self._rate_limiter.acquire()
            client = await self._get_client()

            params = {
                "format": "json",
                "fields": "accession,id,protein_name,gene_names,organism_name,length,sequence,cc_function,xref_pdb,xref_pubmed",
            }

            resp = await client.get(
                f"{self.BASE_URL}/{accession}",
                params=params,
                headers=self.HEADERS,
            )
            resp.raise_for_status()
            data = resp.json()

            docs = self._parse_results([data])
            return docs[0] if docs else None

        except Exception as e:
            logger.error(f"Failed to get entry {accession}: {e}")
            return None

    async def get_sequences_batch(
        self,
        accessions: List[str],
    ) -> Dict[str, str]:
        """
        Get multiple sequences by accession.

        Args:
            accessions: List of UniProt accessions

        Returns:
            Dict mapping accession to sequence
        """
        results = {}
        for acc in accessions:
            seq = await self.get_sequence(acc)
            if seq:
                results[acc] = seq
        return results

    def _parse_results(self, results: List[Dict[str, Any]]) -> List[Document]:
        """Parse UniProt search results."""
        documents = []

        for item in results:
            accession = item.get("primaryAccession")
            if not accession:
                continue

            # Build title
            title = self._build_title(item, accession)

            # Get function text as abstract
            function_text = self._extract_function(item)

            # Get sequence
            sequence = item.get("sequence", {}).get("value", "")

            # Get organism
            organism = item.get("organism", {}).get("scientificName", "")

            # Get gene names
            genes = []
            for gene in item.get("genes", []):
                gene_name = gene.get("geneName", {}).get("value")
                if gene_name:
                    genes.append(gene_name)

            documents.append(Document(
                id=f"uniprot:{accession}",
                source="uniprot",
                title=title,
                text=function_text or f"Protein {accession} from {organism}",
                sequence=sequence,
                sequence_type="protein",
                url=f"https://www.uniprot.org/uniprotkb/{accession}/entry",
                metadata={
                    "accession": accession,
                    "organism": organism,
                    "genes": genes,
                    "length": item.get("sequence", {}).get("length"),
                },
            ))

        return documents

    def _build_title(self, item: Dict[str, Any], accession: str) -> str:
        """Build protein title from entry data."""
        protein_desc = item.get("proteinDescription", {})

        # Try recommended name
        recommended = protein_desc.get("recommendedName", {})
        full_name = recommended.get("fullName", {}).get("value")

        # Fall back to alternative names
        if not full_name:
            for alt in protein_desc.get("alternativeNames", []):
                alt_name = alt.get("fullName", {}).get("value")
                if alt_name:
                    full_name = alt_name
                    break

        base = full_name or item.get("uniProtkbId") or accession

        # Add gene name
        gene_name = None
        genes = item.get("genes", [])
        if genes:
            gene_name = genes[0].get("geneName", {}).get("value")

        # Add organism
        organism = item.get("organism", {}).get("scientificName")

        title = base
        if gene_name:
            title = f"{title} ({gene_name})"
        if organism:
            title = f"{title} - {organism}"

        return title

    def _extract_function(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract function annotation."""
        for comment in item.get("comments", []):
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                values = [t.get("value") for t in texts if t.get("value")]
                if values:
                    return " ".join(values)
        return None

    async def health_check(self) -> bool:
        """Check if UniProt API is accessible."""
        try:
            client = await self._get_client()
            resp = await client.get(
                f"{self.BASE_URL}/search",
                params={"query": "insulin", "format": "json", "size": 1},
                headers=self.HEADERS,
                timeout=5.0,
            )
            return resp.status_code == 200
        except Exception:
            return False


def create_uniprot_connector(
    cache_dir: Optional[str] = None,
) -> UniProtConnector:
    """Factory function for UniProt connector."""
    from pathlib import Path

    config = ConnectorConfig(
        cache_dir=Path(cache_dir) if cache_dir else None,
    )
    return UniProtConnector(config=config)
