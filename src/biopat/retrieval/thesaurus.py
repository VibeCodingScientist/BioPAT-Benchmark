"""Biomedical Thesaurus Integration for BioPAT.

Provides access to controlled vocabularies and ontologies for
query expansion and vocabulary gap bridging:

- MeSH (Medical Subject Headings)
- ChEBI (Chemical Entities of Biological Interest)
- GO (Gene Ontology)
- DrugBank synonyms
- UMLS (Unified Medical Language System) concepts

Reference:
- MeSH: https://www.nlm.nih.gov/mesh/
- ChEBI: https://www.ebi.ac.uk/chebi/
- UMLS: https://www.nlm.nih.gov/research/umls/
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class ThesaurusTerm:
    """A term from a controlled vocabulary."""

    id: str  # e.g., "D001241" for MeSH, "CHEBI:15365" for ChEBI
    preferred_name: str
    synonyms: List[str] = field(default_factory=list)
    source: str = ""  # "mesh", "chebi", "drugbank", etc.

    # Hierarchical relationships
    broader_terms: List[str] = field(default_factory=list)  # Parent concepts
    narrower_terms: List[str] = field(default_factory=list)  # Child concepts
    related_terms: List[str] = field(default_factory=list)

    # Additional metadata
    definition: Optional[str] = None
    semantic_types: List[str] = field(default_factory=list)

    def all_names(self) -> List[str]:
        """Get all names including preferred and synonyms."""
        return [self.preferred_name] + self.synonyms


@dataclass
class ExpansionResult:
    """Result of query expansion."""

    original_query: str
    expanded_query: str
    added_terms: List[str]
    matched_concepts: List[ThesaurusTerm]
    expansion_method: str  # "synonym", "broader", "narrower", "related"


class MeSHThesaurus:
    """Medical Subject Headings (MeSH) thesaurus.

    MeSH is the NLM controlled vocabulary for indexing articles in PubMed.
    """

    # Common MeSH terms for biomedical/pharmaceutical domain
    MESH_TERMS: Dict[str, ThesaurusTerm] = {
        # Drug classes
        "antibiotics": ThesaurusTerm(
            id="D000900",
            preferred_name="Anti-Bacterial Agents",
            synonyms=["antibiotics", "antibacterials", "antimicrobials"],
            source="mesh",
            semantic_types=["Pharmacologic Substance"],
        ),
        "antineoplastic": ThesaurusTerm(
            id="D000970",
            preferred_name="Antineoplastic Agents",
            synonyms=["anticancer agents", "chemotherapy", "cancer drugs", "antitumor agents"],
            source="mesh",
            semantic_types=["Pharmacologic Substance"],
        ),
        "immunotherapy": ThesaurusTerm(
            id="D007167",
            preferred_name="Immunotherapy",
            synonyms=["immune therapy", "biologic therapy", "immunologic therapy"],
            source="mesh",
            semantic_types=["Therapeutic Procedure"],
        ),

        # Specific drugs
        "aspirin": ThesaurusTerm(
            id="D001241",
            preferred_name="Aspirin",
            synonyms=["acetylsalicylic acid", "ASA", "acetyl salicylic acid"],
            source="mesh",
            broader_terms=["Anti-Inflammatory Agents", "Platelet Aggregation Inhibitors"],
        ),
        "pembrolizumab": ThesaurusTerm(
            id="C582435",
            preferred_name="Pembrolizumab",
            synonyms=["Keytruda", "MK-3475", "lambrolizumab"],
            source="mesh",
            broader_terms=["Immune Checkpoint Inhibitors"],
        ),
        "nivolumab": ThesaurusTerm(
            id="C555450",
            preferred_name="Nivolumab",
            synonyms=["Opdivo", "BMS-936558", "MDX-1106"],
            source="mesh",
            broader_terms=["Immune Checkpoint Inhibitors"],
        ),
        "trastuzumab": ThesaurusTerm(
            id="C508053",
            preferred_name="Trastuzumab",
            synonyms=["Herceptin", "anti-HER2", "anti-ERBB2"],
            source="mesh",
            broader_terms=["Monoclonal Antibodies"],
        ),

        # Proteins/Targets
        "pd-1": ThesaurusTerm(
            id="D061026",
            preferred_name="Programmed Cell Death 1 Receptor",
            synonyms=["PD-1", "PDCD1", "CD279", "programmed death 1"],
            source="mesh",
            semantic_types=["Receptor"],
        ),
        "pd-l1": ThesaurusTerm(
            id="D061027",
            preferred_name="B7-H1 Antigen",
            synonyms=["PD-L1", "CD274", "programmed death-ligand 1", "B7-H1"],
            source="mesh",
            semantic_types=["Receptor"],
        ),
        "egfr": ThesaurusTerm(
            id="D011958",
            preferred_name="Receptor, Epidermal Growth Factor",
            synonyms=["EGFR", "ErbB-1", "HER1", "epidermal growth factor receptor"],
            source="mesh",
            semantic_types=["Receptor"],
        ),
        "her2": ThesaurusTerm(
            id="D018719",
            preferred_name="Receptor, erbB-2",
            synonyms=["HER2", "HER2/neu", "ERBB2", "CD340", "neu receptor"],
            source="mesh",
            semantic_types=["Receptor"],
        ),

        # Diseases
        "cancer": ThesaurusTerm(
            id="D009369",
            preferred_name="Neoplasms",
            synonyms=["cancer", "tumor", "tumour", "carcinoma", "malignancy", "neoplasm"],
            source="mesh",
            narrower_terms=["Breast Neoplasms", "Lung Neoplasms", "Melanoma"],
        ),
        "breast cancer": ThesaurusTerm(
            id="D001943",
            preferred_name="Breast Neoplasms",
            synonyms=["breast cancer", "breast carcinoma", "mammary cancer"],
            source="mesh",
            broader_terms=["Neoplasms"],
        ),
        "lung cancer": ThesaurusTerm(
            id="D008175",
            preferred_name="Lung Neoplasms",
            synonyms=["lung cancer", "pulmonary cancer", "lung carcinoma", "NSCLC", "SCLC"],
            source="mesh",
            broader_terms=["Neoplasms"],
        ),
        "melanoma": ThesaurusTerm(
            id="D008545",
            preferred_name="Melanoma",
            synonyms=["melanoma", "malignant melanoma", "skin melanoma"],
            source="mesh",
            broader_terms=["Neoplasms", "Skin Neoplasms"],
        ),
        "diabetes": ThesaurusTerm(
            id="D003920",
            preferred_name="Diabetes Mellitus",
            synonyms=["diabetes", "DM", "diabetic"],
            source="mesh",
            narrower_terms=["Diabetes Mellitus, Type 1", "Diabetes Mellitus, Type 2"],
        ),

        # Biological processes
        "apoptosis": ThesaurusTerm(
            id="D017209",
            preferred_name="Apoptosis",
            synonyms=["apoptosis", "programmed cell death", "cell death"],
            source="mesh",
            semantic_types=["Cell Function"],
        ),
        "angiogenesis": ThesaurusTerm(
            id="D018919",
            preferred_name="Neovascularization, Physiologic",
            synonyms=["angiogenesis", "blood vessel formation", "vascularization"],
            source="mesh",
            semantic_types=["Physiologic Function"],
        ),

        # Organisms
        "human": ThesaurusTerm(
            id="D006801",
            preferred_name="Humans",
            synonyms=["human", "Homo sapiens", "man", "person"],
            source="mesh",
            semantic_types=["Mammal"],
        ),
        "mouse": ThesaurusTerm(
            id="D051379",
            preferred_name="Mice",
            synonyms=["mouse", "Mus musculus", "murine"],
            source="mesh",
            semantic_types=["Mammal"],
        ),
    }

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize MeSH thesaurus.

        Args:
            data_dir: Directory containing MeSH data files
        """
        self.data_dir = data_dir
        self._term_index: Dict[str, ThesaurusTerm] = {}
        self._name_to_id: Dict[str, str] = {}

        # Build index from built-in terms
        self._build_index()

        # Load additional terms from files if available
        if data_dir and data_dir.exists():
            self._load_from_files(data_dir)

    def _build_index(self):
        """Build lookup index from terms."""
        for key, term in self.MESH_TERMS.items():
            self._term_index[term.id] = term
            self._term_index[key.lower()] = term

            # Index all names
            self._name_to_id[term.preferred_name.lower()] = term.id
            for syn in term.synonyms:
                self._name_to_id[syn.lower()] = term.id

    def _load_from_files(self, data_dir: Path):
        """Load additional terms from files."""
        mesh_file = data_dir / "mesh_terms.json"
        if mesh_file.exists():
            try:
                with open(mesh_file) as f:
                    data = json.load(f)
                for term_data in data:
                    term = ThesaurusTerm(**term_data)
                    self._term_index[term.id] = term
                    self._name_to_id[term.preferred_name.lower()] = term.id
                    for syn in term.synonyms:
                        self._name_to_id[syn.lower()] = term.id
                logger.info(f"Loaded {len(data)} additional MeSH terms")
            except Exception as e:
                logger.warning(f"Failed to load MeSH file: {e}")

    def lookup(self, term: str) -> Optional[ThesaurusTerm]:
        """
        Look up a term in the thesaurus.

        Args:
            term: Term to look up (name or ID)

        Returns:
            ThesaurusTerm or None
        """
        normalized = term.lower().strip()

        # Try direct lookup
        if normalized in self._term_index:
            return self._term_index[normalized]

        # Try name lookup
        if normalized in self._name_to_id:
            term_id = self._name_to_id[normalized]
            return self._term_index.get(term_id)

        return None

    def get_synonyms(self, term: str) -> List[str]:
        """Get synonyms for a term."""
        result = self.lookup(term)
        if result:
            return result.all_names()
        return []

    def get_broader_terms(self, term: str) -> List[str]:
        """Get broader (parent) terms."""
        result = self.lookup(term)
        if result:
            return result.broader_terms
        return []

    def get_narrower_terms(self, term: str) -> List[str]:
        """Get narrower (child) terms."""
        result = self.lookup(term)
        if result:
            return result.narrower_terms
        return []


class ChEBIThesaurus:
    """Chemical Entities of Biological Interest (ChEBI) thesaurus.

    ChEBI is a dictionary of molecular entities focused on 'small'
    chemical compounds.
    """

    # Common ChEBI terms
    CHEBI_TERMS: Dict[str, ThesaurusTerm] = {
        "aspirin": ThesaurusTerm(
            id="CHEBI:15365",
            preferred_name="aspirin",
            synonyms=["acetylsalicylic acid", "ASA", "2-acetoxybenzoic acid"],
            source="chebi",
            broader_terms=["salicylate", "carboxylic acid"],
        ),
        "ibuprofen": ThesaurusTerm(
            id="CHEBI:5855",
            preferred_name="ibuprofen",
            synonyms=["Advil", "Motrin", "α-methyl-4-(2-methylpropyl)benzeneacetic acid"],
            source="chebi",
            broader_terms=["propionic acid derivative"],
        ),
        "paracetamol": ThesaurusTerm(
            id="CHEBI:46195",
            preferred_name="paracetamol",
            synonyms=["acetaminophen", "Tylenol", "APAP", "N-acetyl-p-aminophenol"],
            source="chebi",
            broader_terms=["phenol derivative"],
        ),
        "metformin": ThesaurusTerm(
            id="CHEBI:6801",
            preferred_name="metformin",
            synonyms=["Glucophage", "dimethylbiguanide"],
            source="chebi",
            broader_terms=["biguanide"],
        ),
        "insulin": ThesaurusTerm(
            id="CHEBI:145810",
            preferred_name="insulin",
            synonyms=["insulin human", "regular insulin"],
            source="chebi",
            semantic_types=["hormone", "protein"],
        ),
    }

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize ChEBI thesaurus."""
        self.data_dir = data_dir
        self._term_index: Dict[str, ThesaurusTerm] = {}
        self._name_to_id: Dict[str, str] = {}

        self._build_index()

    def _build_index(self):
        """Build lookup index."""
        for key, term in self.CHEBI_TERMS.items():
            self._term_index[term.id] = term
            self._term_index[key.lower()] = term

            self._name_to_id[term.preferred_name.lower()] = term.id
            for syn in term.synonyms:
                self._name_to_id[syn.lower()] = term.id

    def lookup(self, term: str) -> Optional[ThesaurusTerm]:
        """Look up a term."""
        normalized = term.lower().strip()

        if normalized in self._term_index:
            return self._term_index[normalized]

        if normalized in self._name_to_id:
            term_id = self._name_to_id[normalized]
            return self._term_index.get(term_id)

        return None

    def get_synonyms(self, term: str) -> List[str]:
        """Get synonyms for a term."""
        result = self.lookup(term)
        if result:
            return result.all_names()
        return []


class DrugBankThesaurus:
    """DrugBank drug synonym lookup.

    DrugBank is a comprehensive drug database with extensive
    synonym information.
    """

    # Common drug synonyms
    DRUG_SYNONYMS: Dict[str, ThesaurusTerm] = {
        "pembrolizumab": ThesaurusTerm(
            id="DB09037",
            preferred_name="Pembrolizumab",
            synonyms=["Keytruda", "MK-3475", "lambrolizumab", "SCH 900475"],
            source="drugbank",
            semantic_types=["monoclonal antibody", "PD-1 inhibitor"],
        ),
        "nivolumab": ThesaurusTerm(
            id="DB09035",
            preferred_name="Nivolumab",
            synonyms=["Opdivo", "BMS-936558", "MDX-1106", "ONO-4538"],
            source="drugbank",
            semantic_types=["monoclonal antibody", "PD-1 inhibitor"],
        ),
        "atezolizumab": ThesaurusTerm(
            id="DB11595",
            preferred_name="Atezolizumab",
            synonyms=["Tecentriq", "MPDL3280A", "RG7446"],
            source="drugbank",
            semantic_types=["monoclonal antibody", "PD-L1 inhibitor"],
        ),
        "durvalumab": ThesaurusTerm(
            id="DB11714",
            preferred_name="Durvalumab",
            synonyms=["Imfinzi", "MEDI4736"],
            source="drugbank",
            semantic_types=["monoclonal antibody", "PD-L1 inhibitor"],
        ),
        "ipilimumab": ThesaurusTerm(
            id="DB06186",
            preferred_name="Ipilimumab",
            synonyms=["Yervoy", "MDX-010", "BMS-734016"],
            source="drugbank",
            semantic_types=["monoclonal antibody", "CTLA-4 inhibitor"],
        ),
        "trastuzumab": ThesaurusTerm(
            id="DB00072",
            preferred_name="Trastuzumab",
            synonyms=["Herceptin", "anti-HER2", "rhuMAb HER2"],
            source="drugbank",
            semantic_types=["monoclonal antibody", "HER2 inhibitor"],
        ),
        "rituximab": ThesaurusTerm(
            id="DB00073",
            preferred_name="Rituximab",
            synonyms=["Rituxan", "MabThera", "anti-CD20"],
            source="drugbank",
            semantic_types=["monoclonal antibody", "CD20 inhibitor"],
        ),
        "bevacizumab": ThesaurusTerm(
            id="DB00112",
            preferred_name="Bevacizumab",
            synonyms=["Avastin", "anti-VEGF", "rhuMAb-VEGF"],
            source="drugbank",
            semantic_types=["monoclonal antibody", "VEGF inhibitor"],
        ),
    }

    def __init__(self):
        """Initialize DrugBank thesaurus."""
        self._term_index: Dict[str, ThesaurusTerm] = {}
        self._name_to_id: Dict[str, str] = {}

        self._build_index()

    def _build_index(self):
        """Build lookup index."""
        for key, term in self.DRUG_SYNONYMS.items():
            self._term_index[term.id] = term
            self._term_index[key.lower()] = term

            self._name_to_id[term.preferred_name.lower()] = term.id
            for syn in term.synonyms:
                self._name_to_id[syn.lower()] = term.id

    def lookup(self, term: str) -> Optional[ThesaurusTerm]:
        """Look up a drug term."""
        normalized = term.lower().strip()

        if normalized in self._term_index:
            return self._term_index[normalized]

        if normalized in self._name_to_id:
            term_id = self._name_to_id[normalized]
            return self._term_index.get(term_id)

        return None

    def get_synonyms(self, term: str) -> List[str]:
        """Get drug synonyms."""
        result = self.lookup(term)
        if result:
            return result.all_names()
        return []


class UnifiedThesaurus:
    """Unified interface to all biomedical thesauri.

    Combines MeSH, ChEBI, DrugBank, and other vocabularies
    into a single query expansion interface.
    """

    def __init__(
        self,
        mesh_data_dir: Optional[Path] = None,
        chebi_data_dir: Optional[Path] = None,
    ):
        """
        Initialize unified thesaurus.

        Args:
            mesh_data_dir: Directory for MeSH data
            chebi_data_dir: Directory for ChEBI data
        """
        self.mesh = MeSHThesaurus(mesh_data_dir)
        self.chebi = ChEBIThesaurus(chebi_data_dir)
        self.drugbank = DrugBankThesaurus()

    def lookup(self, term: str) -> Optional[ThesaurusTerm]:
        """Look up a term in all thesauri."""
        # Try each thesaurus in order
        result = self.drugbank.lookup(term)
        if result:
            return result

        result = self.mesh.lookup(term)
        if result:
            return result

        result = self.chebi.lookup(term)
        if result:
            return result

        return None

    def get_all_synonyms(self, term: str) -> Set[str]:
        """Get all synonyms from all sources."""
        synonyms = set()

        # Try each thesaurus
        for thesaurus in [self.drugbank, self.mesh, self.chebi]:
            syns = thesaurus.get_synonyms(term)
            synonyms.update(syns)

        return synonyms

    def expand_query(
        self,
        query: str,
        expansion_types: Optional[List[str]] = None,
        max_expansions_per_term: int = 5,
    ) -> ExpansionResult:
        """
        Expand query using thesaurus synonyms.

        Args:
            query: Original query
            expansion_types: Types of expansion ("synonym", "broader", "narrower")
            max_expansions_per_term: Max expansions per matched term

        Returns:
            ExpansionResult with expanded query and metadata
        """
        if expansion_types is None:
            expansion_types = ["synonym"]

        # Tokenize query
        words = query.lower().split()

        added_terms = []
        matched_concepts = []

        # Try to match multi-word terms first
        for n in range(min(4, len(words)), 0, -1):  # 4-grams down to 1-grams
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n])
                concept = self.lookup(phrase)

                if concept:
                    matched_concepts.append(concept)

                    # Add synonyms
                    if "synonym" in expansion_types:
                        for syn in concept.synonyms[:max_expansions_per_term]:
                            if syn.lower() not in query.lower():
                                added_terms.append(syn)

                    # Add broader terms
                    if "broader" in expansion_types:
                        for broader in concept.broader_terms[:2]:
                            if broader.lower() not in query.lower():
                                added_terms.append(broader)

                    # Add narrower terms
                    if "narrower" in expansion_types:
                        for narrower in concept.narrower_terms[:2]:
                            if narrower.lower() not in query.lower():
                                added_terms.append(narrower)

        # Build expanded query
        if added_terms:
            # Remove duplicates while preserving order
            seen = set()
            unique_terms = []
            for term in added_terms:
                if term.lower() not in seen:
                    seen.add(term.lower())
                    unique_terms.append(term)

            expanded_query = f"{query} {' '.join(unique_terms)}"
        else:
            expanded_query = query

        return ExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            added_terms=added_terms,
            matched_concepts=matched_concepts,
            expansion_method="synonym" if "synonym" in expansion_types else expansion_types[0],
        )

    def expand_with_hierarchy(
        self,
        query: str,
        include_broader: bool = True,
        include_narrower: bool = False,
    ) -> ExpansionResult:
        """Expand query using hierarchical relationships."""
        expansion_types = ["synonym"]
        if include_broader:
            expansion_types.append("broader")
        if include_narrower:
            expansion_types.append("narrower")

        return self.expand_query(query, expansion_types=expansion_types)


def create_unified_thesaurus(
    mesh_data_dir: Optional[str] = None,
    chebi_data_dir: Optional[str] = None,
) -> UnifiedThesaurus:
    """Factory function for unified thesaurus.

    Args:
        mesh_data_dir: Path to MeSH data directory
        chebi_data_dir: Path to ChEBI data directory

    Returns:
        Configured UnifiedThesaurus
    """
    return UnifiedThesaurus(
        mesh_data_dir=Path(mesh_data_dir) if mesh_data_dir else None,
        chebi_data_dir=Path(chebi_data_dir) if chebi_data_dir else None,
    )
