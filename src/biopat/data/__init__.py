"""Data Acquisition Module for BioPAT.

This module provides connectors to external data sources:
- Scientific literature (PubMed, bioRxiv, medRxiv)
- Patents (USPTO PatentsView, EPO)
- Chemical structures (PubChem)
- Protein sequences (UniProt)

Example:
    ```python
    from biopat.data import DataAcquisition

    async with DataAcquisition() as acq:
        # Fetch from multiple sources
        articles = await acq.fetch_pubmed("CAR-T therapy", limit=500)
        patents = await acq.fetch_patents(ipc_codes=["A61K"], limit=200)
        proteins = await acq.fetch_proteins("PD-1 human", limit=50)

        # Build corpus for BioPAT
        corpus = acq.build_corpus(articles + patents)
        chemicals = acq.extract_chemicals(patents)
        sequences = acq.extract_sequences(proteins)

        # Save to disk
        acq.save_corpus(articles + patents, "data/my_corpus")
    ```
"""

from biopat.data.base import (
    BaseConnector,
    ConnectorConfig,
    Document,
    RateLimiter,
    DiskCache,
)

from biopat.data.acquisition import (
    DataAcquisition,
    AcquisitionConfig,
    create_data_acquisition,
)

from biopat.data.pubmed import (
    PubMedConnector,
    create_pubmed_connector,
)

from biopat.data.uniprot import (
    UniProtConnector,
    create_uniprot_connector,
)

from biopat.data.patents import (
    PatentsViewConnector,
    EPOConnector,
    create_patentsview_connector,
    create_epo_connector,
)

from biopat.data.biorxiv import (
    BioRxivConnector,
    create_biorxiv_connector,
)

from biopat.data.pubchem import (
    PubChemConnector,
    create_pubchem_connector,
)

from biopat.data.ncbi_sequences import (
    NCBISequenceConnector,
    create_ncbi_sequence_connector,
)

from biopat.data.surechembl import (
    SureChEMBLConnector,
    SureChEMBLChemical,
    create_surechembl_connector,
)

__all__ = [
    # Base classes
    "BaseConnector",
    "ConnectorConfig",
    "Document",
    "RateLimiter",
    "DiskCache",

    # Unified acquisition
    "DataAcquisition",
    "AcquisitionConfig",
    "create_data_acquisition",

    # PubMed
    "PubMedConnector",
    "create_pubmed_connector",

    # UniProt
    "UniProtConnector",
    "create_uniprot_connector",

    # Patents
    "PatentsViewConnector",
    "EPOConnector",
    "create_patentsview_connector",
    "create_epo_connector",

    # bioRxiv
    "BioRxivConnector",
    "create_biorxiv_connector",

    # PubChem
    "PubChemConnector",
    "create_pubchem_connector",

    # NCBI Sequences (GenBank, Protein, Nucleotide)
    "NCBISequenceConnector",
    "create_ncbi_sequence_connector",

    # SureChEMBL (Patent-Chemical Mappings)
    "SureChEMBLConnector",
    "SureChEMBLChemical",
    "create_surechembl_connector",
]
