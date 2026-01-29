"""Unified Data Acquisition for BioPAT.

Provides a single interface for fetching data from multiple sources:
- Scientific articles (PubMed, bioRxiv)
- Patents (USPTO, EPO)
- Chemical structures (PubChem)
- Protein sequences (UniProt)

This is the main entry point for building benchmark corpora.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from biopat.data.base import Document, ConnectorConfig

logger = logging.getLogger(__name__)


@dataclass
class AcquisitionConfig:
    """Configuration for data acquisition."""

    # API Keys
    pubmed_api_key: Optional[str] = None
    patentsview_api_key: Optional[str] = None
    epo_consumer_key: Optional[str] = None
    epo_consumer_secret: Optional[str] = None

    # Cache settings
    cache_dir: Optional[Path] = None
    use_cache: bool = True

    # Default limits
    default_limit: int = 100


class DataAcquisition:
    """Unified data acquisition interface.

    Provides methods for fetching data from all supported sources
    and combining them into a corpus suitable for BioPAT.

    Example:
        ```python
        from biopat.data import DataAcquisition

        async with DataAcquisition() as acq:
            # Fetch articles
            articles = await acq.fetch_pubmed(
                "CAR-T cell therapy cancer",
                limit=1000
            )

            # Fetch patents
            patents = await acq.fetch_patents(
                ipc_codes=["A61K", "C07K"],
                date_range="2020-2024",
                limit=500
            )

            # Fetch sequences
            sequences = await acq.fetch_uniprot(
                "PD-1 checkpoint inhibitor human",
                limit=100
            )

            # Build corpus
            corpus = acq.build_corpus(articles + patents)

            # Extract chemicals for molecular search
            chemicals = acq.extract_chemicals(patents)

            # Extract sequences for sequence search
            seq_data = acq.extract_sequences(sequences)
        ```
    """

    def __init__(
        self,
        config: Optional[AcquisitionConfig] = None,
        **kwargs,
    ):
        """
        Initialize data acquisition.

        Args:
            config: Acquisition configuration
            **kwargs: Override config values
        """
        self.config = config or AcquisitionConfig()

        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Connectors (initialized lazily)
        self._pubmed = None
        self._biorxiv = None
        self._medrxiv = None
        self._patentsview = None
        self._epo = None
        self._uniprot = None
        self._pubchem = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close all connectors."""
        connectors = [
            self._pubmed, self._biorxiv, self._medrxiv,
            self._patentsview, self._epo, self._uniprot, self._pubchem
        ]
        for conn in connectors:
            if conn:
                await conn.close()

    # =========================================================================
    # Article Sources
    # =========================================================================

    async def fetch_pubmed(
        self,
        query: str,
        limit: int = 100,
        date_range: Optional[str] = None,
    ) -> List[Document]:
        """
        Fetch articles from PubMed.

        Args:
            query: Search query
            limit: Maximum results
            date_range: Date filter (e.g., "2020-2024")

        Returns:
            List of Document objects
        """
        from biopat.data.pubmed import PubMedConnector

        if self._pubmed is None:
            conn_config = ConnectorConfig(
                api_key=self.config.pubmed_api_key,
                cache_dir=self.config.cache_dir,
            )
            self._pubmed = PubMedConnector(config=conn_config)

        return await self._pubmed.search(query, limit=limit, date_range=date_range)

    async def fetch_biorxiv(
        self,
        query: str,
        limit: int = 100,
        date_range: Optional[str] = None,
    ) -> List[Document]:
        """
        Fetch preprints from bioRxiv.

        Args:
            query: Search query
            limit: Maximum results
            date_range: Date filter

        Returns:
            List of Document objects
        """
        from biopat.data.biorxiv import BioRxivConnector

        if self._biorxiv is None:
            conn_config = ConnectorConfig(cache_dir=self.config.cache_dir)
            self._biorxiv = BioRxivConnector(server="biorxiv", config=conn_config)

        return await self._biorxiv.search(query, limit=limit, date_range=date_range)

    async def fetch_medrxiv(
        self,
        query: str,
        limit: int = 100,
        date_range: Optional[str] = None,
    ) -> List[Document]:
        """Fetch preprints from medRxiv."""
        from biopat.data.biorxiv import BioRxivConnector

        if self._medrxiv is None:
            conn_config = ConnectorConfig(cache_dir=self.config.cache_dir)
            self._medrxiv = BioRxivConnector(server="medrxiv", config=conn_config)

        return await self._medrxiv.search(query, limit=limit, date_range=date_range)

    async def fetch_articles(
        self,
        query: str,
        limit: int = 100,
        sources: Optional[List[str]] = None,
        date_range: Optional[str] = None,
    ) -> List[Document]:
        """
        Fetch articles from multiple sources.

        Args:
            query: Search query
            limit: Maximum results per source
            sources: List of sources ("pubmed", "biorxiv", "medrxiv")
            date_range: Date filter

        Returns:
            Combined list of Document objects
        """
        if sources is None:
            sources = ["pubmed", "biorxiv"]

        tasks = []
        if "pubmed" in sources:
            tasks.append(self.fetch_pubmed(query, limit, date_range))
        if "biorxiv" in sources:
            tasks.append(self.fetch_biorxiv(query, limit, date_range))
        if "medrxiv" in sources:
            tasks.append(self.fetch_medrxiv(query, limit, date_range))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        documents = []
        for result in results:
            if isinstance(result, list):
                documents.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Source fetch failed: {result}")

        return documents

    # =========================================================================
    # Patent Sources
    # =========================================================================

    async def fetch_patents(
        self,
        query: Optional[str] = None,
        ipc_codes: Optional[List[str]] = None,
        date_range: Optional[str] = None,
        limit: int = 100,
        source: str = "uspto",  # "uspto" or "epo"
    ) -> List[Document]:
        """
        Fetch patents from USPTO or EPO.

        Args:
            query: Text search query
            ipc_codes: IPC classification codes to filter
            date_range: Date filter (e.g., "2020-2024")
            limit: Maximum results
            source: "uspto" (PatentsView) or "epo"

        Returns:
            List of Document objects
        """
        from biopat.data.patents import PatentsViewConnector, EPOConnector

        if source == "epo":
            if self._epo is None:
                conn_config = ConnectorConfig(cache_dir=self.config.cache_dir)
                self._epo = EPOConnector(
                    consumer_key=self.config.epo_consumer_key,
                    consumer_secret=self.config.epo_consumer_secret,
                    config=conn_config,
                )

            if ipc_codes:
                # Convert date_range to EPO format
                date_from = None
                date_to = None
                if date_range:
                    import re
                    match = re.match(r'(\d{4})-(\d{4})', date_range)
                    if match:
                        date_from = f"{match.group(1)}0101"
                        date_to = f"{match.group(2)}1231"

                return await self._epo.search_by_ipc(
                    ipc_codes,
                    date_from=date_from,
                    date_to=date_to,
                    limit=limit,
                )
            else:
                return await self._epo.search(query or "", limit=limit)

        else:  # USPTO
            if self._patentsview is None:
                conn_config = ConnectorConfig(
                    api_key=self.config.patentsview_api_key,
                    cache_dir=self.config.cache_dir,
                )
                self._patentsview = PatentsViewConnector(config=conn_config)

            if ipc_codes:
                return await self._patentsview.search_biomedical(
                    ipc_codes=ipc_codes,
                    date_range=date_range,
                    limit=limit,
                )
            elif query:
                return await self._patentsview.search(query, limit=limit)
            else:
                # Default to biomedical patents
                return await self._patentsview.search_biomedical(limit=limit)

    async def fetch_patent_by_id(
        self,
        patent_id: str,
    ) -> Optional[Document]:
        """
        Fetch single patent by ID.

        Args:
            patent_id: Patent ID (e.g., "US10500001B2")

        Returns:
            Document or None
        """
        from biopat.data.patents import PatentsViewConnector

        if self._patentsview is None:
            conn_config = ConnectorConfig(
                api_key=self.config.patentsview_api_key,
                cache_dir=self.config.cache_dir,
            )
            self._patentsview = PatentsViewConnector(config=conn_config)

        return await self._patentsview.get_patent(patent_id)

    # =========================================================================
    # Chemical Sources
    # =========================================================================

    async def fetch_compounds(
        self,
        query: str,
        limit: int = 100,
    ) -> List[Document]:
        """
        Fetch compounds from PubChem.

        Args:
            query: Compound name or keyword
            limit: Maximum results

        Returns:
            List of Document objects with SMILES
        """
        from biopat.data.pubchem import PubChemConnector

        if self._pubchem is None:
            conn_config = ConnectorConfig(cache_dir=self.config.cache_dir)
            self._pubchem = PubChemConnector(config=conn_config)

        return await self._pubchem.search(query, limit=limit)

    async def get_smiles(self, compound_name: str) -> Optional[str]:
        """
        Get SMILES for a compound by name.

        Args:
            compound_name: Compound name

        Returns:
            SMILES string or None
        """
        from biopat.data.pubchem import PubChemConnector

        if self._pubchem is None:
            conn_config = ConnectorConfig(cache_dir=self.config.cache_dir)
            self._pubchem = PubChemConnector(config=conn_config)

        return await self._pubchem.get_smiles(compound_name)

    # =========================================================================
    # Protein Sources
    # =========================================================================

    async def fetch_proteins(
        self,
        query: str,
        limit: int = 100,
        organism: Optional[str] = None,
    ) -> List[Document]:
        """
        Fetch proteins from UniProt.

        Args:
            query: Search query
            limit: Maximum results
            organism: Filter by organism (e.g., "human")

        Returns:
            List of Document objects with sequences
        """
        from biopat.data.uniprot import UniProtConnector

        if self._uniprot is None:
            conn_config = ConnectorConfig(cache_dir=self.config.cache_dir)
            self._uniprot = UniProtConnector(config=conn_config)

        return await self._uniprot.search(query, limit=limit, organism=organism)

    async def get_sequence(self, accession: str) -> Optional[str]:
        """
        Get protein sequence by UniProt accession.

        Args:
            accession: UniProt accession (e.g., "P01308")

        Returns:
            Amino acid sequence or None
        """
        from biopat.data.uniprot import UniProtConnector

        if self._uniprot is None:
            conn_config = ConnectorConfig(cache_dir=self.config.cache_dir)
            self._uniprot = UniProtConnector(config=conn_config)

        return await self._uniprot.get_sequence(accession)

    # =========================================================================
    # Corpus Building
    # =========================================================================

    def build_corpus(
        self,
        documents: List[Document],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build BioPAT corpus from documents.

        Args:
            documents: List of Document objects

        Returns:
            Corpus dict mapping doc_id to document data
        """
        corpus = {}
        for doc in documents:
            corpus[doc.id] = doc.to_corpus_entry()
        return corpus

    def extract_chemicals(
        self,
        documents: List[Document],
    ) -> List[Dict[str, str]]:
        """
        Extract chemical structures from documents.

        Args:
            documents: List of Document objects

        Returns:
            List of {"doc_id": ..., "smiles": ...} dicts
        """
        chemicals = []
        for doc in documents:
            if doc.smiles:
                chemicals.append({
                    "doc_id": doc.id,
                    "smiles": doc.smiles,
                })
        return chemicals

    def extract_sequences(
        self,
        documents: List[Document],
    ) -> List[Dict[str, str]]:
        """
        Extract protein/nucleotide sequences from documents.

        Args:
            documents: List of Document objects

        Returns:
            List of {"doc_id": ..., "sequence": ..., "type": ...} dicts
        """
        sequences = []
        for doc in documents:
            if doc.sequence:
                sequences.append({
                    "doc_id": doc.id,
                    "sequence": doc.sequence,
                    "type": doc.sequence_type or "protein",
                })
        return sequences

    def save_corpus(
        self,
        documents: List[Document],
        output_dir: Union[str, Path],
        format: str = "jsonl",
    ) -> None:
        """
        Save documents to files.

        Args:
            documents: List of Document objects
            output_dir: Output directory
            format: Output format ("jsonl" or "json")
        """
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            corpus_path = output_dir / "corpus.jsonl"
            with open(corpus_path, "w") as f:
                for doc in documents:
                    f.write(json.dumps(doc.to_dict()) + "\n")
        else:
            corpus_path = output_dir / "corpus.json"
            with open(corpus_path, "w") as f:
                json.dump([doc.to_dict() for doc in documents], f, indent=2)

        # Save chemicals
        chemicals = self.extract_chemicals(documents)
        if chemicals:
            chem_path = output_dir / "chemicals.jsonl"
            with open(chem_path, "w") as f:
                for chem in chemicals:
                    f.write(json.dumps(chem) + "\n")

        # Save sequences
        sequences = self.extract_sequences(documents)
        if sequences:
            seq_path = output_dir / "sequences.jsonl"
            with open(seq_path, "w") as f:
                for seq in sequences:
                    f.write(json.dumps(seq) + "\n")

        logger.info(f"Saved {len(documents)} documents to {output_dir}")


def create_data_acquisition(
    pubmed_api_key: Optional[str] = None,
    patentsview_api_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> DataAcquisition:
    """Factory function for data acquisition.

    Args:
        pubmed_api_key: NCBI API key
        patentsview_api_key: PatentsView API key
        cache_dir: Directory for caching

    Returns:
        Configured DataAcquisition
    """
    config = AcquisitionConfig(
        pubmed_api_key=pubmed_api_key,
        patentsview_api_key=patentsview_api_key,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )
    return DataAcquisition(config=config)
