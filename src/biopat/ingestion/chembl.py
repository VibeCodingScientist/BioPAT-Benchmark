"""ChEMBL WebResource Client Wrapper.

Phase 4.0 (Advanced): Provides access to ChEMBL bioactivity data
using the official chembl-webresource-client library.
"""

import logging
from typing import Any, Dict, List, Optional

from chembl_webresource_client.new_client import new_client
from diskcache import Cache
from pathlib import Path

logger = logging.getLogger(__name__)


class ChEMBLClient:
    """Wrapper for official ChEMBL WebResource Client.
    
    Provides high-level methods for resolving chemical structures
    to ChEMBL IDs and fetching bioactivity/target information.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize ChEMBL client.
        
        Args:
            cache_dir: Optional directory for caching responses.
        """
        self.client = new_client
        self.cache = Cache(str(cache_dir / "chembl")) if cache_dir else None

    def get_molecule_by_id(self, chembl_id: str) -> Optional[Dict[str, Any]]:
        """Get molecule data by ChEMBL ID."""
        cache_key = f"mol:{chembl_id}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        try:
            res = self.client.molecule.filter(chembl_id=chembl_id).only([
                "molecule_chembl_id", "pref_name", "molecule_structures", 
                "molecule_properties", "molecule_synonyms"
            ])
            
            if res:
                data = res[0]
                if self.cache:
                    self.cache[cache_key] = data
                return data
            return None
        except Exception as e:
            logger.error(f"Error fetching ChEMBL molecule {chembl_id}: {e}")
            return None

    def search_molecule_by_structure(
        self, 
        structure: str, 
        search_type: str = "similarity", 
        threshold: int = 70
    ) -> List[Dict[str, Any]]:
        """Search ChEMBL by SMILES or InChIKey.
        
        Args:
            structure: SMILES or InChIKey.
            search_type: "similarity", "substructure", or "exact".
            threshold: Similarity threshold (0-100).
            
        Returns:
            List of matching molecules.
        """
        try:
            if search_type == "similarity":
                res = self.client.similarity.filter(smiles=structure, similarity=threshold)
            elif search_type == "substructure":
                res = self.client.substructure.filter(smiles=structure)
            else:
                # Detect if InChIKey
                if "-" in structure and len(structure) >= 25:
                    res = self.client.molecule.filter(molecule_structures__standard_inchi_key=structure)
                else:
                    res = self.client.molecule.filter(molecule_structures__canonical_smiles=structure)
            
            return list(res)
        except Exception as e:
            logger.error(f"Error searching ChEMBL structure {structure}: {e}")
            return []

    def get_bioactivity_for_molecule(self, chembl_id: str) -> List[Dict[str, Any]]:
        """Get all bioactivity records for a molecule."""
        cache_key = f"act:{chembl_id}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        try:
            res = self.client.activity.filter(molecule_chembl_id=chembl_id).only([
                "activity_id", "target_chembl_id", "standard_type", 
                "standard_value", "standard_units", "pchembl_value"
            ])
            
            data = list(res)
            if self.cache:
                self.cache[cache_key] = data
            return data
        except Exception as e:
            logger.error(f"Error fetching bioactivity for {chembl_id}: {e}")
            return []

    def get_targets_for_molecule(self, chembl_id: str) -> List[Dict[str, Any]]:
        """Get unique targets associated with a molecule's bioactivity."""
        activities = self.get_bioactivity_for_molecule(chembl_id)
        target_ids = list(set([a["target_chembl_id"] for a in activities if a.get("target_chembl_id")]))
        
        targets = []
        for tid in target_ids:
            t_data = self.get_target_by_id(tid)
            if t_data:
                targets.append(t_data)
        return targets

    def get_target_by_id(self, target_id: str) -> Optional[Dict[str, Any]]:
        """Get target data by ChEMBL ID."""
        cache_key = f"target:{target_id}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        try:
            res = self.client.target.filter(target_chembl_id=target_id).only([
                "target_chembl_id", "pref_name", "target_type", "organism"
            ])
            
            if res:
                data = res[0]
                if self.cache:
                    self.cache[cache_key] = data
                return data
            return None
        except Exception as e:
            logger.error(f"Error fetching ChEMBL target {target_id}: {e}")
            return None
    
    def resolve_to_inchikey(self, structure: str) -> Optional[str]:
        """Utility to resolve a SMILES to its standard InChIKey via ChEMBL."""
        mols = self.search_molecule_by_structure(structure, search_type="exact")
        if mols:
            return mols[0].get("molecule_structures", {}).get("standard_inchi_key")
        return None
