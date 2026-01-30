"""Biomedical Named Entity Recognition (NER) for BioPAT.

Extracts structured entities from patent claims and scientific abstracts:
- Drugs/Chemicals (drug names, chemical compounds)
- Proteins/Genes (protein names, gene symbols)
- Diseases/Conditions (disease mentions, indications)
- Organisms (species, cell lines)
- Dosages/Concentrations

Uses state-of-the-art biomedical NER models from HuggingFace.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Try to import transformers
try:
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        pipeline,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class BioEntity:
    """A biomedical entity extracted from text."""

    text: str
    entity_type: str  # CHEMICAL, PROTEIN, DISEASE, ORGANISM, DOSAGE, etc.
    start: int
    end: int
    confidence: float = 1.0

    # Normalized forms
    canonical_name: Optional[str] = None
    identifiers: Dict[str, str] = field(default_factory=dict)  # e.g., {"mesh": "D001241", "drugbank": "DB00945"}

    def __hash__(self):
        return hash((self.text.lower(), self.entity_type))

    def __eq__(self, other):
        if not isinstance(other, BioEntity):
            return False
        return self.text.lower() == other.text.lower() and self.entity_type == other.entity_type


@dataclass
class ExtractedClaim:
    """Structured representation of a patent claim."""

    raw_text: str

    # Extracted entities by type
    chemicals: List[BioEntity] = field(default_factory=list)
    proteins: List[BioEntity] = field(default_factory=list)
    diseases: List[BioEntity] = field(default_factory=list)
    organisms: List[BioEntity] = field(default_factory=list)
    dosages: List[BioEntity] = field(default_factory=list)

    # Claim structure
    claim_type: Optional[str] = None  # "composition", "method", "use", "system"
    is_independent: bool = True
    depends_on: Optional[int] = None

    def all_entities(self) -> List[BioEntity]:
        """Get all entities."""
        return self.chemicals + self.proteins + self.diseases + self.organisms + self.dosages

    def entity_types_present(self) -> Set[str]:
        """Get set of entity types present in this claim."""
        types = set()
        if self.chemicals:
            types.add("CHEMICAL")
        if self.proteins:
            types.add("PROTEIN")
        if self.diseases:
            types.add("DISEASE")
        if self.organisms:
            types.add("ORGANISM")
        if self.dosages:
            types.add("DOSAGE")
        return types


# Model configurations for different NER tasks
NER_MODELS = {
    # General biomedical NER
    "pubmedbert_ner": "alvaroalon2/biobert_chemical_ner",

    # Chemical NER
    "chemical_ner": "alvaroalon2/biobert_chemical_ner",
    "chemberta_ner": "seyonec/ChemBERTa-zinc-base-v1",  # Can be fine-tuned for NER

    # Disease NER
    "disease_ner": "alvaroalon2/biobert_diseases_ner",

    # Gene/Protein NER
    "gene_ner": "alvaroalon2/biobert_genetic_ner",

    # Multi-type biomedical NER
    "hunflair": "hunflair/hunflair-paper",  # Recognizes genes, chemicals, diseases, species
    "biobert_bc5cdr": "dmis-lab/biobert-base-cased-v1.2",  # Trained on BC5CDR

    # SciBERT-based
    "scibert_ner": "allenai/scibert_scivocab_uncased",
}


class BiomedicalNER:
    """Biomedical Named Entity Recognition using transformer models.

    Supports multiple specialized models for different entity types
    and can combine results for comprehensive extraction.
    """

    def __init__(
        self,
        chemical_model: str = "alvaroalon2/biobert_chemical_ner",
        disease_model: str = "alvaroalon2/biobert_diseases_ner",
        gene_model: str = "alvaroalon2/biobert_genetic_ner",
        device: int = -1,  # -1 for CPU, 0+ for GPU
        batch_size: int = 8,
    ):
        """
        Initialize BiomedicalNER.

        Args:
            chemical_model: Model for chemical/drug NER
            disease_model: Model for disease NER
            gene_model: Model for gene/protein NER
            device: Device to use (-1 for CPU)
            batch_size: Batch size for inference
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers required. Install with: pip install transformers"
            )

        self.device = device
        self.batch_size = batch_size

        # Lazy loading of models
        self._chemical_model_name = chemical_model
        self._disease_model_name = disease_model
        self._gene_model_name = gene_model

        self._chemical_pipeline = None
        self._disease_pipeline = None
        self._gene_pipeline = None

        # Regex patterns for rule-based extraction
        self._init_patterns()

    def _init_patterns(self):
        """Initialize regex patterns for rule-based extraction."""
        # Dosage patterns
        self._dosage_pattern = re.compile(
            r'\b(\d+(?:\.\d+)?)\s*(mg|g|kg|μg|µg|ng|ml|mL|L|μL|µL|mM|μM|µM|nM|%|ppm|IU|U)\b',
            re.IGNORECASE
        )

        # Concentration patterns
        self._concentration_pattern = re.compile(
            r'\b(\d+(?:\.\d+)?)\s*(mg/ml|mg/L|μg/ml|g/L|mol/L|M|mM|μM|nM)\b',
            re.IGNORECASE
        )

        # Chemical formula pattern (e.g., H2O, C6H12O6)
        self._formula_pattern = re.compile(
            r'\b([A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?(?:\d+)?)+)\b'
        )

        # Claim type indicators
        self._claim_patterns = {
            "composition": re.compile(r'\b(composition|formulation|compound|mixture)\b', re.I),
            "method": re.compile(r'\b(method|process|procedure)\b', re.I),
            "use": re.compile(r'\b(use of|for use|for treating|for the treatment)\b', re.I),
            "system": re.compile(r'\b(system|device|apparatus|kit)\b', re.I),
        }

        # Dependency claim pattern
        self._dependency_pattern = re.compile(
            r'^(?:the\s+)?(?:composition|method|compound|use|system)\s+(?:of|according to)\s+claim\s+(\d+)',
            re.IGNORECASE
        )

    @property
    def chemical_pipeline(self):
        """Lazy load chemical NER pipeline."""
        if self._chemical_pipeline is None:
            logger.info(f"Loading chemical NER model: {self._chemical_model_name}")
            self._chemical_pipeline = pipeline(
                "ner",
                model=self._chemical_model_name,
                device=self.device,
                aggregation_strategy="simple",
            )
        return self._chemical_pipeline

    @property
    def disease_pipeline(self):
        """Lazy load disease NER pipeline."""
        if self._disease_pipeline is None:
            logger.info(f"Loading disease NER model: {self._disease_model_name}")
            self._disease_pipeline = pipeline(
                "ner",
                model=self._disease_model_name,
                device=self.device,
                aggregation_strategy="simple",
            )
        return self._disease_pipeline

    @property
    def gene_pipeline(self):
        """Lazy load gene/protein NER pipeline."""
        if self._gene_pipeline is None:
            logger.info(f"Loading gene NER model: {self._gene_model_name}")
            self._gene_pipeline = pipeline(
                "ner",
                model=self._gene_model_name,
                device=self.device,
                aggregation_strategy="simple",
            )
        return self._gene_pipeline

    def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
    ) -> List[BioEntity]:
        """
        Extract biomedical entities from text.

        Args:
            text: Input text
            entity_types: Types to extract (None = all)
                Options: "CHEMICAL", "DISEASE", "PROTEIN", "ORGANISM", "DOSAGE"

        Returns:
            List of extracted entities
        """
        if entity_types is None:
            entity_types = ["CHEMICAL", "DISEASE", "PROTEIN", "DOSAGE"]

        entities = []

        # Chemical extraction
        if "CHEMICAL" in entity_types:
            entities.extend(self._extract_chemicals(text))

        # Disease extraction
        if "DISEASE" in entity_types:
            entities.extend(self._extract_diseases(text))

        # Protein/Gene extraction
        if "PROTEIN" in entity_types:
            entities.extend(self._extract_genes(text))

        # Dosage extraction (rule-based)
        if "DOSAGE" in entity_types:
            entities.extend(self._extract_dosages(text))

        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity.text.lower(), entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities

    def _extract_chemicals(self, text: str) -> List[BioEntity]:
        """Extract chemical entities using NER model."""
        entities = []

        try:
            results = self.chemical_pipeline(text)

            for result in results:
                # Map model labels to our entity types
                entity_type = "CHEMICAL"
                if "drug" in result.get("entity_group", "").lower():
                    entity_type = "DRUG"

                entities.append(BioEntity(
                    text=result["word"],
                    entity_type=entity_type,
                    start=result["start"],
                    end=result["end"],
                    confidence=result["score"],
                ))
        except Exception as e:
            logger.warning(f"Chemical NER failed: {e}")

        # Also extract chemical formulas via regex
        for match in self._formula_pattern.finditer(text):
            formula = match.group(1)
            # Filter out common non-chemical matches
            if len(formula) >= 3 and not formula.isupper():
                entities.append(BioEntity(
                    text=formula,
                    entity_type="CHEMICAL",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7,
                ))

        return entities

    def _extract_diseases(self, text: str) -> List[BioEntity]:
        """Extract disease entities using NER model."""
        entities = []

        try:
            results = self.disease_pipeline(text)

            for result in results:
                entities.append(BioEntity(
                    text=result["word"],
                    entity_type="DISEASE",
                    start=result["start"],
                    end=result["end"],
                    confidence=result["score"],
                ))
        except Exception as e:
            logger.warning(f"Disease NER failed: {e}")

        return entities

    def _extract_genes(self, text: str) -> List[BioEntity]:
        """Extract gene/protein entities using NER model."""
        entities = []

        try:
            results = self.gene_pipeline(text)

            for result in results:
                # Distinguish genes from proteins based on context/label
                entity_type = "PROTEIN"
                label = result.get("entity_group", "").lower()
                if "gene" in label or "dna" in label:
                    entity_type = "GENE"

                entities.append(BioEntity(
                    text=result["word"],
                    entity_type=entity_type,
                    start=result["start"],
                    end=result["end"],
                    confidence=result["score"],
                ))
        except Exception as e:
            logger.warning(f"Gene NER failed: {e}")

        return entities

    def _extract_dosages(self, text: str) -> List[BioEntity]:
        """Extract dosage mentions using regex patterns."""
        entities = []

        # Dosage patterns
        for match in self._dosage_pattern.finditer(text):
            entities.append(BioEntity(
                text=match.group(0),
                entity_type="DOSAGE",
                start=match.start(),
                end=match.end(),
                confidence=0.9,
            ))

        # Concentration patterns
        for match in self._concentration_pattern.finditer(text):
            entities.append(BioEntity(
                text=match.group(0),
                entity_type="CONCENTRATION",
                start=match.start(),
                end=match.end(),
                confidence=0.9,
            ))

        return entities

    def extract_claim(self, claim_text: str) -> ExtractedClaim:
        """
        Extract structured information from a patent claim.

        Args:
            claim_text: Raw claim text

        Returns:
            ExtractedClaim with entities and structure
        """
        entities = self.extract_entities(claim_text)

        claim = ExtractedClaim(raw_text=claim_text)

        # Sort entities by type
        for entity in entities:
            if entity.entity_type in ("CHEMICAL", "DRUG"):
                claim.chemicals.append(entity)
            elif entity.entity_type in ("PROTEIN", "GENE"):
                claim.proteins.append(entity)
            elif entity.entity_type == "DISEASE":
                claim.diseases.append(entity)
            elif entity.entity_type == "ORGANISM":
                claim.organisms.append(entity)
            elif entity.entity_type in ("DOSAGE", "CONCENTRATION"):
                claim.dosages.append(entity)

        # Determine claim type
        claim.claim_type = self._classify_claim_type(claim_text)

        # Check for dependency
        dep_match = self._dependency_pattern.match(claim_text)
        if dep_match:
            claim.is_independent = False
            claim.depends_on = int(dep_match.group(1))

        return claim

    def _classify_claim_type(self, text: str) -> str:
        """Classify claim type based on keywords."""
        for claim_type, pattern in self._claim_patterns.items():
            if pattern.search(text):
                return claim_type
        return "other"

    def extract_batch(
        self,
        texts: List[str],
        entity_types: Optional[List[str]] = None,
    ) -> List[List[BioEntity]]:
        """
        Extract entities from multiple texts.

        Args:
            texts: List of input texts
            entity_types: Types to extract

        Returns:
            List of entity lists, one per input text
        """
        return [self.extract_entities(text, entity_types) for text in texts]


class EntityLinker:
    """Link extracted entities to knowledge bases.

    Links entities to:
    - MeSH (Medical Subject Headings)
    - DrugBank
    - UniProt
    - ChEBI (Chemical Entities of Biological Interest)
    """

    def __init__(self):
        """Initialize entity linker with lookup tables."""
        # These would typically be loaded from files or APIs
        self._mesh_cache: Dict[str, str] = {}
        self._drugbank_cache: Dict[str, str] = {}
        self._uniprot_cache: Dict[str, str] = {}
        self._chebi_cache: Dict[str, str] = {}

    def link_entity(self, entity: BioEntity) -> BioEntity:
        """
        Link entity to knowledge base identifiers.

        Args:
            entity: Entity to link

        Returns:
            Entity with identifiers populated
        """
        # Normalize entity text
        normalized = entity.text.lower().strip()

        if entity.entity_type in ("CHEMICAL", "DRUG"):
            # Try DrugBank
            if normalized in self._drugbank_cache:
                entity.identifiers["drugbank"] = self._drugbank_cache[normalized]

            # Try ChEBI
            if normalized in self._chebi_cache:
                entity.identifiers["chebi"] = self._chebi_cache[normalized]

        elif entity.entity_type in ("PROTEIN", "GENE"):
            # Try UniProt
            if normalized in self._uniprot_cache:
                entity.identifiers["uniprot"] = self._uniprot_cache[normalized]

        elif entity.entity_type == "DISEASE":
            # Try MeSH
            if normalized in self._mesh_cache:
                entity.identifiers["mesh"] = self._mesh_cache[normalized]

        return entity

    def link_entities(self, entities: List[BioEntity]) -> List[BioEntity]:
        """Link multiple entities."""
        return [self.link_entity(e) for e in entities]


class QueryEntityExpander:
    """Expand query using extracted entities and synonyms.

    Uses NER to identify key entities in query, then expands
    using synonyms from knowledge bases.
    """

    def __init__(
        self,
        ner: Optional[BiomedicalNER] = None,
        linker: Optional[EntityLinker] = None,
    ):
        self.ner = ner or BiomedicalNER()
        self.linker = linker or EntityLinker()

        # Common synonyms (would typically be loaded from knowledge bases)
        self._synonyms: Dict[str, List[str]] = {
            "aspirin": ["acetylsalicylic acid", "ASA"],
            "ibuprofen": ["advil", "motrin"],
            "pembrolizumab": ["keytruda", "MK-3475"],
            "nivolumab": ["opdivo", "BMS-936558"],
            "trastuzumab": ["herceptin"],
            "pd-1": ["programmed death 1", "PDCD1", "CD279"],
            "pd-l1": ["programmed death-ligand 1", "CD274", "B7-H1"],
            "egfr": ["epidermal growth factor receptor", "HER1", "ERBB1"],
            "her2": ["ERBB2", "HER2/neu", "CD340"],
            "cancer": ["carcinoma", "tumor", "tumour", "neoplasm", "malignancy"],
            "diabetes": ["diabetes mellitus", "DM"],
        }

    def expand_query(
        self,
        query: str,
        max_synonyms_per_entity: int = 3,
    ) -> Tuple[str, List[str]]:
        """
        Expand query with entity synonyms.

        Args:
            query: Original query
            max_synonyms_per_entity: Max synonyms to add per entity

        Returns:
            Tuple of (expanded_query, list of added terms)
        """
        # Extract entities from query
        entities = self.ner.extract_entities(query)

        added_terms = []
        for entity in entities:
            normalized = entity.text.lower()

            if normalized in self._synonyms:
                synonyms = self._synonyms[normalized][:max_synonyms_per_entity]
                added_terms.extend(synonyms)

        # Build expanded query
        if added_terms:
            expanded = f"{query} {' '.join(added_terms)}"
        else:
            expanded = query

        return expanded, added_terms

    def get_entity_variations(self, entity_text: str) -> List[str]:
        """Get all known variations of an entity."""
        normalized = entity_text.lower()

        variations = [entity_text]
        if normalized in self._synonyms:
            variations.extend(self._synonyms[normalized])

        return variations


def create_biomedical_ner(
    device: int = -1,
    load_all_models: bool = False,
) -> BiomedicalNER:
    """Factory function for BiomedicalNER.

    Args:
        device: Device to use (-1 for CPU)
        load_all_models: Whether to load all models immediately

    Returns:
        Configured BiomedicalNER
    """
    ner = BiomedicalNER(device=device)

    if load_all_models:
        # Trigger lazy loading of all pipelines
        _ = ner.chemical_pipeline
        _ = ner.disease_pipeline
        _ = ner.gene_pipeline

    return ner
