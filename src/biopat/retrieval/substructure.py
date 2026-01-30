"""Substructure and Scaffold Search for BioPAT.

Extends molecular search beyond fingerprint-based similarity to include:
- Substructure matching (find molecules containing a pattern)
- Scaffold extraction and matching
- Maximum common substructure (MCS)
- Pharmacophore matching

Uses RDKit for all cheminformatics operations.

Reference:
- RDKit: https://www.rdkit.org/docs/
- Bemis-Murcko scaffolds: J. Med. Chem. 1996, 39, 2887-2893
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, Descriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import rdFMCS
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


@dataclass
class SubstructureMatch:
    """Result of a substructure search."""

    query_smiles: str
    target_smiles: str
    target_id: str
    has_substructure: bool
    match_atoms: List[Tuple[int, ...]] = field(default_factory=list)  # Atom mappings
    num_matches: int = 0


@dataclass
class ScaffoldMatch:
    """Result of scaffold matching."""

    query_smiles: str
    target_smiles: str
    target_id: str
    query_scaffold: str
    target_scaffold: str
    scaffolds_match: bool
    tanimoto_similarity: float = 0.0  # Scaffold fingerprint similarity


@dataclass
class MCSResult:
    """Maximum Common Substructure result."""

    mol1_smiles: str
    mol2_smiles: str
    mcs_smarts: str
    mcs_num_atoms: int
    mcs_num_bonds: int
    fraction_mol1: float  # Fraction of mol1 covered
    fraction_mol2: float  # Fraction of mol2 covered


class SubstructureSearcher:
    """Substructure-based molecular search.

    Finds molecules that contain a given substructure (query) pattern.
    """

    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit required. Install with: pip install rdkit")

    def parse_smiles(self, smiles: str) -> Optional[Any]:
        """Parse SMILES to molecule object."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                mol = Chem.MolFromSmarts(smiles)  # Try as SMARTS pattern
            return mol
        except Exception as e:
            logger.warning(f"Failed to parse SMILES: {smiles}, error: {e}")
            return None

    def has_substructure(
        self,
        query_smiles: str,
        target_smiles: str,
    ) -> bool:
        """
        Check if target contains query as substructure.

        Args:
            query_smiles: SMILES of substructure to find
            target_smiles: SMILES of molecule to search

        Returns:
            True if target contains query substructure
        """
        query_mol = self.parse_smiles(query_smiles)
        target_mol = self.parse_smiles(target_smiles)

        if query_mol is None or target_mol is None:
            return False

        return target_mol.HasSubstructMatch(query_mol)

    def find_substructure_matches(
        self,
        query_smiles: str,
        target_smiles: str,
        target_id: str = "",
    ) -> SubstructureMatch:
        """
        Find all substructure matches.

        Args:
            query_smiles: SMILES of substructure pattern
            target_smiles: SMILES of molecule to search
            target_id: Optional identifier for target

        Returns:
            SubstructureMatch with all match details
        """
        query_mol = self.parse_smiles(query_smiles)
        target_mol = self.parse_smiles(target_smiles)

        result = SubstructureMatch(
            query_smiles=query_smiles,
            target_smiles=target_smiles,
            target_id=target_id,
            has_substructure=False,
        )

        if query_mol is None or target_mol is None:
            return result

        matches = target_mol.GetSubstructMatches(query_mol)

        if matches:
            result.has_substructure = True
            result.match_atoms = list(matches)
            result.num_matches = len(matches)

        return result

    def search_database(
        self,
        query_smiles: str,
        database: List[Tuple[str, str]],  # List of (id, smiles)
        max_results: int = 100,
    ) -> List[SubstructureMatch]:
        """
        Search a database for substructure matches.

        Args:
            query_smiles: SMILES of substructure pattern
            database: List of (id, smiles) tuples
            max_results: Maximum results to return

        Returns:
            List of SubstructureMatch for molecules containing the substructure
        """
        query_mol = self.parse_smiles(query_smiles)
        if query_mol is None:
            return []

        matches = []
        for doc_id, smiles in database:
            target_mol = self.parse_smiles(smiles)
            if target_mol is None:
                continue

            if target_mol.HasSubstructMatch(query_mol):
                match_atoms = target_mol.GetSubstructMatches(query_mol)
                matches.append(SubstructureMatch(
                    query_smiles=query_smiles,
                    target_smiles=smiles,
                    target_id=doc_id,
                    has_substructure=True,
                    match_atoms=list(match_atoms),
                    num_matches=len(match_atoms),
                ))

                if len(matches) >= max_results:
                    break

        return matches


class ScaffoldSearcher:
    """Scaffold-based molecular search using Bemis-Murcko decomposition.

    Extracts core scaffolds from molecules for structural comparison.
    """

    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit required. Install with: pip install rdkit")

        self._scaffold_cache: Dict[str, str] = {}

    def get_scaffold(
        self,
        smiles: str,
        generic: bool = False,
    ) -> Optional[str]:
        """
        Extract Bemis-Murcko scaffold from molecule.

        Args:
            smiles: Input SMILES
            generic: If True, return generic scaffold (all atoms -> C)

        Returns:
            Scaffold SMILES or None
        """
        cache_key = f"{smiles}_{generic}"
        if cache_key in self._scaffold_cache:
            return self._scaffold_cache[cache_key]

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            if generic:
                core = MurckoScaffold.MakeScaffoldGeneric(
                    MurckoScaffold.GetScaffoldForMol(mol)
                )
            else:
                core = MurckoScaffold.GetScaffoldForMol(mol)

            scaffold_smiles = Chem.MolToSmiles(core)
            self._scaffold_cache[cache_key] = scaffold_smiles

            return scaffold_smiles

        except Exception as e:
            logger.warning(f"Failed to get scaffold for {smiles}: {e}")
            return None

    def get_ring_systems(self, smiles: str) -> List[str]:
        """
        Extract ring systems from a molecule.

        Args:
            smiles: Input SMILES

        Returns:
            List of ring system SMILES
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []

            ring_info = mol.GetRingInfo()
            ring_systems = []

            for ring in ring_info.AtomRings():
                ring_mol = Chem.RWMol(mol)
                atoms_to_remove = [i for i in range(mol.GetNumAtoms()) if i not in ring]
                for idx in sorted(atoms_to_remove, reverse=True):
                    ring_mol.RemoveAtom(idx)

                try:
                    ring_smiles = Chem.MolToSmiles(ring_mol)
                    ring_systems.append(ring_smiles)
                except Exception:
                    pass

            return ring_systems

        except Exception as e:
            logger.warning(f"Failed to get ring systems: {e}")
            return []

    def scaffolds_match(
        self,
        smiles1: str,
        smiles2: str,
        generic: bool = False,
    ) -> bool:
        """Check if two molecules have the same scaffold."""
        scaffold1 = self.get_scaffold(smiles1, generic=generic)
        scaffold2 = self.get_scaffold(smiles2, generic=generic)

        if scaffold1 is None or scaffold2 is None:
            return False

        return scaffold1 == scaffold2

    def scaffold_similarity(
        self,
        smiles1: str,
        smiles2: str,
    ) -> float:
        """
        Compute Tanimoto similarity between scaffolds.

        Args:
            smiles1: First molecule SMILES
            smiles2: Second molecule SMILES

        Returns:
            Tanimoto similarity (0-1)
        """
        scaffold1 = self.get_scaffold(smiles1)
        scaffold2 = self.get_scaffold(smiles2)

        if scaffold1 is None or scaffold2 is None:
            return 0.0

        mol1 = Chem.MolFromSmiles(scaffold1)
        mol2 = Chem.MolFromSmiles(scaffold2)

        if mol1 is None or mol2 is None:
            return 0.0

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)

        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def search_by_scaffold(
        self,
        query_smiles: str,
        database: List[Tuple[str, str]],
        similarity_threshold: float = 0.7,
        generic: bool = False,
        max_results: int = 100,
    ) -> List[ScaffoldMatch]:
        """
        Search database for molecules with similar scaffolds.

        Args:
            query_smiles: Query molecule SMILES
            database: List of (id, smiles) tuples
            similarity_threshold: Minimum scaffold similarity
            generic: Use generic scaffolds
            max_results: Maximum results

        Returns:
            List of ScaffoldMatch results
        """
        query_scaffold = self.get_scaffold(query_smiles, generic=generic)
        if query_scaffold is None:
            return []

        matches = []
        for doc_id, smiles in database:
            target_scaffold = self.get_scaffold(smiles, generic=generic)
            if target_scaffold is None:
                continue

            # Exact match
            exact_match = query_scaffold == target_scaffold

            # Similarity
            sim = self.scaffold_similarity(query_smiles, smiles)

            if exact_match or sim >= similarity_threshold:
                matches.append(ScaffoldMatch(
                    query_smiles=query_smiles,
                    target_smiles=smiles,
                    target_id=doc_id,
                    query_scaffold=query_scaffold,
                    target_scaffold=target_scaffold,
                    scaffolds_match=exact_match,
                    tanimoto_similarity=sim,
                ))

                if len(matches) >= max_results:
                    break

        # Sort by similarity
        matches.sort(key=lambda x: x.tanimoto_similarity, reverse=True)

        return matches


class MCSSearcher:
    """Maximum Common Substructure (MCS) search.

    Finds the largest common substructure between molecules.
    """

    def __init__(
        self,
        timeout: int = 60,  # seconds
        threshold: float = 0.5,  # minimum fraction of query atoms
    ):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit required. Install with: pip install rdkit")

        self.timeout = timeout
        self.threshold = threshold

    def find_mcs(
        self,
        smiles1: str,
        smiles2: str,
    ) -> Optional[MCSResult]:
        """
        Find maximum common substructure between two molecules.

        Args:
            smiles1: First molecule SMILES
            smiles2: Second molecule SMILES

        Returns:
            MCSResult or None
        """
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return None

        try:
            mcs = rdFMCS.FindMCS(
                [mol1, mol2],
                timeout=self.timeout,
                threshold=self.threshold,
                ringMatchesRingOnly=True,
                completeRingsOnly=True,
            )

            if mcs.canceled or mcs.numAtoms == 0:
                return None

            return MCSResult(
                mol1_smiles=smiles1,
                mol2_smiles=smiles2,
                mcs_smarts=mcs.smartsString,
                mcs_num_atoms=mcs.numAtoms,
                mcs_num_bonds=mcs.numBonds,
                fraction_mol1=mcs.numAtoms / mol1.GetNumAtoms(),
                fraction_mol2=mcs.numAtoms / mol2.GetNumAtoms(),
            )

        except Exception as e:
            logger.warning(f"MCS search failed: {e}")
            return None

    def mcs_similarity(
        self,
        smiles1: str,
        smiles2: str,
    ) -> float:
        """
        Compute MCS-based similarity.

        Returns the fraction of atoms in the smaller molecule
        that are part of the MCS.
        """
        result = self.find_mcs(smiles1, smiles2)

        if result is None:
            return 0.0

        return max(result.fraction_mol1, result.fraction_mol2)

    def search_by_mcs(
        self,
        query_smiles: str,
        database: List[Tuple[str, str]],
        min_similarity: float = 0.5,
        max_results: int = 100,
    ) -> List[Tuple[str, str, MCSResult]]:
        """
        Search database using MCS similarity.

        Args:
            query_smiles: Query molecule SMILES
            database: List of (id, smiles) tuples
            min_similarity: Minimum MCS similarity
            max_results: Maximum results

        Returns:
            List of (doc_id, smiles, MCSResult) tuples
        """
        results = []

        for doc_id, smiles in database:
            mcs_result = self.find_mcs(query_smiles, smiles)

            if mcs_result is not None:
                similarity = max(mcs_result.fraction_mol1, mcs_result.fraction_mol2)

                if similarity >= min_similarity:
                    results.append((doc_id, smiles, mcs_result))

        # Sort by similarity
        results.sort(
            key=lambda x: max(x[2].fraction_mol1, x[2].fraction_mol2),
            reverse=True
        )

        return results[:max_results]


class MolecularDescriptorCalculator:
    """Calculate molecular descriptors for filtering/ranking."""

    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit required. Install with: pip install rdkit")

    def calculate_descriptors(self, smiles: str) -> Dict[str, float]:
        """
        Calculate molecular descriptors.

        Args:
            smiles: Input SMILES

        Returns:
            Dict of descriptor name -> value
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        try:
            return {
                "molecular_weight": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "hbd": Descriptors.NumHDonors(mol),
                "hba": Descriptors.NumHAcceptors(mol),
                "tpsa": Descriptors.TPSA(mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "ring_count": Descriptors.RingCount(mol),
                "aromatic_rings": Descriptors.NumAromaticRings(mol),
                "heavy_atoms": Descriptors.HeavyAtomCount(mol),
                "fraction_sp3": Descriptors.FractionCSP3(mol),
            }
        except Exception as e:
            logger.warning(f"Descriptor calculation failed: {e}")
            return {}

    def is_drug_like(
        self,
        smiles: str,
        rule: str = "lipinski",
    ) -> bool:
        """
        Check if molecule passes drug-likeness rules.

        Args:
            smiles: Input SMILES
            rule: "lipinski" (Ro5), "veber", or "ghose"

        Returns:
            True if passes rules
        """
        desc = self.calculate_descriptors(smiles)
        if not desc:
            return False

        if rule == "lipinski":
            # Lipinski's Rule of 5
            violations = 0
            if desc["molecular_weight"] > 500:
                violations += 1
            if desc["logp"] > 5:
                violations += 1
            if desc["hbd"] > 5:
                violations += 1
            if desc["hba"] > 10:
                violations += 1
            return violations <= 1

        elif rule == "veber":
            # Veber rules
            return desc["rotatable_bonds"] <= 10 and desc["tpsa"] <= 140

        elif rule == "ghose":
            # Ghose filter
            return (
                160 <= desc["molecular_weight"] <= 480 and
                -0.4 <= desc["logp"] <= 5.6 and
                40 <= desc["heavy_atoms"] <= 70
            )

        return True


def create_substructure_searcher() -> SubstructureSearcher:
    """Factory function for SubstructureSearcher."""
    return SubstructureSearcher()


def create_scaffold_searcher() -> ScaffoldSearcher:
    """Factory function for ScaffoldSearcher."""
    return ScaffoldSearcher()


def create_mcs_searcher(
    timeout: int = 60,
    threshold: float = 0.5,
) -> MCSSearcher:
    """Factory function for MCSSearcher."""
    return MCSSearcher(timeout=timeout, threshold=threshold)
