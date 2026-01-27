"""NCBI Patent Sequence Database Client.

Phase 4.0 (Advanced): Provides access to patent-associated biological sequences
via NCBI Entrez and regex-based SEQ ID NO extraction from patent text.

The NCBI Patent Sequence Database contains sequences disclosed in patents
and can be searched via Entrez. This module handles:
- Fetching sequences associated with patent numbers
- Extracting SEQ ID NO references from patent claim text
- Converting sequences to FASTA format for BLAST indexing
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from diskcache import Cache

logger = logging.getLogger(__name__)

# NCBI Entrez base URL
ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Regex patterns for SEQ ID NO extraction
SEQ_ID_PATTERNS = [
    # Standard formats
    re.compile(r"SEQ\s*ID\s*NO[:\s]*(\d+)", re.IGNORECASE),
    re.compile(r"SEQ\s*ID\s*NOS?[:\s]*([\d,\s\-and]+)", re.IGNORECASE),
    re.compile(r"sequence\s+(?:listing\s+)?(?:number|no\.?)[:\s]*(\d+)", re.IGNORECASE),
    # Range patterns: SEQ ID NOs: 1-5
    re.compile(r"SEQ\s*ID\s*NOS?[:\s]*(\d+)\s*(?:to|-)\s*(\d+)", re.IGNORECASE),
]

# NCBI nucleotide/protein sequence patterns
ACCESSION_PATTERNS = [
    re.compile(r"\b([A-Z]{1,2}_?\d{5,9}(?:\.\d+)?)\b"),  # GenBank/RefSeq
    re.compile(r"\b(AAA\d{5}(?:\.\d+)?)\b"),  # GenBank protein
    re.compile(r"\b(NP_\d{6,9}(?:\.\d+)?)\b"),  # RefSeq protein
    re.compile(r"\b(NM_\d{6,9}(?:\.\d+)?)\b"),  # RefSeq mRNA
]


@dataclass
class PatentSequence:
    """Sequence associated with a patent."""

    sequence_id: str  # Internal ID (e.g., "SEQ ID NO:1")
    sequence: str  # Actual sequence string
    sequence_type: str  # "AA" (amino acid) or "NT" (nucleotide)
    patent_number: str  # Associated patent
    description: str = ""
    length: int = 0
    organism: str = ""
    accession: Optional[str] = None  # NCBI accession if available

    def __post_init__(self):
        if not self.length:
            self.length = len(self.sequence)

    def to_fasta(self) -> str:
        """Convert to FASTA format string."""
        header = f">{self.sequence_id}|{self.patent_number}"
        if self.description:
            header += f"|{self.description}"
        if self.organism:
            header += f"|{self.organism}"

        # Wrap sequence at 80 characters
        wrapped = "\n".join(
            self.sequence[i : i + 80] for i in range(0, len(self.sequence), 80)
        )
        return f"{header}\n{wrapped}"


@dataclass
class SeqIdReference:
    """Reference to a SEQ ID NO in patent text."""

    seq_id_no: int
    context: str = ""  # Surrounding text
    claim_number: Optional[int] = None


class PatentSequenceClient:
    """Client for NCBI Patent Sequence Database.

    Provides methods for:
    - Fetching sequences associated with patent numbers
    - Searching the patent sequence database
    - Extracting SEQ ID NO references from text
    """

    def __init__(
        self,
        email: str,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        rate_limit: int = 3,  # NCBI default without API key
    ):
        """Initialize patent sequence client.

        Args:
            email: Required email for NCBI Entrez.
            api_key: Optional NCBI API key for higher rate limits.
            cache_dir: Optional directory for caching responses.
            rate_limit: Maximum requests per second.
        """
        self.email = email
        self.api_key = api_key
        self.cache = Cache(str(cache_dir / "patent_sequences")) if cache_dir else None
        self._semaphore = asyncio.Semaphore(rate_limit if not api_key else 10)

    def _get_params(self, **kwargs) -> Dict[str, Any]:
        """Build request parameters with required email/api_key."""
        params = {"email": self.email, "tool": "biopat"}
        if self.api_key:
            params["api_key"] = self.api_key
        params.update(kwargs)
        return params

    async def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        timeout: float = 60.0,
    ) -> Optional[str]:
        """Make rate-limited request to NCBI Entrez."""
        url = f"{ENTREZ_BASE}/{endpoint}"

        async with self._semaphore:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(url, params=params, timeout=timeout)
                    response.raise_for_status()
                    return response.text
                except Exception as e:
                    logger.error(f"NCBI request failed: {url} - {e}")
                    return None

    async def search_by_patent(
        self,
        patent_number: str,
        db: str = "nuccore",
    ) -> List[str]:
        """Search NCBI for sequences associated with a patent number.

        Args:
            patent_number: Patent number (e.g., "US10123456", "EP1234567").
            db: Database to search ("nuccore" for nucleotides, "protein" for proteins).

        Returns:
            List of NCBI GI/accession numbers.
        """
        # Normalize patent number for search
        normalized = patent_number.upper().replace(" ", "").replace("-", "")

        cache_key = f"search:{db}:{normalized}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        # Search query format for patent sequences
        query = f'"{normalized}"[Patent]'

        params = self._get_params(
            db=db,
            term=query,
            retmax=500,
            retmode="json",
        )

        response = await self._make_request("esearch.fcgi", params)
        if not response:
            return []

        try:
            import json

            data = json.loads(response)
            id_list = data.get("esearchresult", {}).get("idlist", [])

            if self.cache:
                self.cache[cache_key] = id_list

            return id_list

        except Exception as e:
            logger.error(f"Failed to parse search results: {e}")
            return []

    async def fetch_sequences(
        self,
        ids: List[str],
        db: str = "nuccore",
        rettype: str = "fasta",
    ) -> str:
        """Fetch sequences by ID in FASTA format.

        Args:
            ids: List of NCBI GI or accession numbers.
            db: Database ("nuccore" or "protein").
            rettype: Return type ("fasta", "gb", etc.).

        Returns:
            Raw FASTA or GenBank format string.
        """
        if not ids:
            return ""

        params = self._get_params(
            db=db,
            id=",".join(ids),
            rettype=rettype,
            retmode="text",
        )

        return await self._make_request("efetch.fcgi", params) or ""

    async def get_patent_sequences(
        self,
        patent_number: str,
        include_proteins: bool = True,
        include_nucleotides: bool = True,
    ) -> List[PatentSequence]:
        """Get all sequences associated with a patent.

        Args:
            patent_number: Patent number to search.
            include_proteins: Include protein sequences.
            include_nucleotides: Include nucleotide sequences.

        Returns:
            List of PatentSequence objects.
        """
        sequences = []

        if include_nucleotides:
            nt_ids = await self.search_by_patent(patent_number, db="nuccore")
            if nt_ids:
                fasta = await self.fetch_sequences(nt_ids, db="nuccore")
                sequences.extend(
                    self._parse_fasta(fasta, patent_number, "NT")
                )

        if include_proteins:
            aa_ids = await self.search_by_patent(patent_number, db="protein")
            if aa_ids:
                fasta = await self.fetch_sequences(aa_ids, db="protein")
                sequences.extend(
                    self._parse_fasta(fasta, patent_number, "AA")
                )

        return sequences

    def _parse_fasta(
        self,
        fasta_text: str,
        patent_number: str,
        seq_type: str,
    ) -> List[PatentSequence]:
        """Parse FASTA text into PatentSequence objects."""
        sequences = []

        if not fasta_text:
            return sequences

        current_header = None
        current_seq = []

        for line in fasta_text.split("\n"):
            line = line.strip()
            if line.startswith(">"):
                # Save previous sequence
                if current_header and current_seq:
                    seq_str = "".join(current_seq)
                    seq_id, desc, acc = self._parse_fasta_header(current_header)
                    sequences.append(
                        PatentSequence(
                            sequence_id=seq_id,
                            sequence=seq_str,
                            sequence_type=seq_type,
                            patent_number=patent_number,
                            description=desc,
                            accession=acc,
                        )
                    )

                current_header = line[1:]  # Remove '>'
                current_seq = []
            elif line:
                current_seq.append(line)

        # Don't forget the last sequence
        if current_header and current_seq:
            seq_str = "".join(current_seq)
            seq_id, desc, acc = self._parse_fasta_header(current_header)
            sequences.append(
                PatentSequence(
                    sequence_id=seq_id,
                    sequence=seq_str,
                    sequence_type=seq_type,
                    patent_number=patent_number,
                    description=desc,
                    accession=acc,
                )
            )

        return sequences

    def _parse_fasta_header(self, header: str) -> Tuple[str, str, Optional[str]]:
        """Parse FASTA header to extract ID, description, and accession."""
        parts = header.split("|")

        accession = None
        seq_id = parts[0] if parts else header
        description = ""

        # Try to extract accession
        for part in parts:
            for pattern in ACCESSION_PATTERNS:
                match = pattern.search(part)
                if match:
                    accession = match.group(1)
                    break
            if accession:
                break

        # Description is usually after the first few identifier fields
        if len(parts) > 2:
            description = " ".join(parts[2:])

        return seq_id, description, accession


class SeqIdExtractor:
    """Extracts SEQ ID NO references from patent text.

    Handles various formats:
    - SEQ ID NO: 1
    - SEQ ID NOs: 1, 2, and 3
    - SEQ ID NOs: 1-5
    - sequences of SEQ ID NO: 1 through SEQ ID NO: 10
    """

    def __init__(self):
        """Initialize extractor."""
        self._patterns = SEQ_ID_PATTERNS

    def extract_from_text(
        self,
        text: str,
        context_chars: int = 100,
    ) -> List[SeqIdReference]:
        """Extract all SEQ ID NO references from text.

        Args:
            text: Patent text (claims, description, etc.).
            context_chars: Number of characters of context to include.

        Returns:
            List of SeqIdReference objects.
        """
        references = []
        seen_ids: Set[int] = set()

        for pattern in self._patterns:
            for match in pattern.finditer(text):
                # Extract context
                start = max(0, match.start() - context_chars)
                end = min(len(text), match.end() + context_chars)
                context = text[start:end]

                # Parse the matched IDs
                ids = self._parse_seq_ids(match)

                for seq_id in ids:
                    if seq_id not in seen_ids:
                        seen_ids.add(seq_id)
                        references.append(
                            SeqIdReference(
                                seq_id_no=seq_id,
                                context=context,
                            )
                        )

        return sorted(references, key=lambda x: x.seq_id_no)

    def _parse_seq_ids(self, match: re.Match) -> List[int]:
        """Parse matched text to extract individual SEQ ID numbers."""
        ids = []
        groups = match.groups()

        if len(groups) == 2 and groups[1] is not None:
            # Range pattern: SEQ ID NOs: 1-5
            try:
                start = int(groups[0])
                end = int(groups[1])
                ids.extend(range(start, end + 1))
            except ValueError:
                pass
        elif len(groups) >= 1 and groups[0]:
            text = groups[0]
            # Handle comma-separated and "and" patterns
            text = text.replace("and", ",").replace("-", ",")
            for part in text.split(","):
                part = part.strip()
                if part.isdigit():
                    ids.append(int(part))

        return ids

    def extract_from_claims(
        self,
        claims: List[str],
    ) -> Dict[int, List[SeqIdReference]]:
        """Extract SEQ ID references from numbered claims.

        Args:
            claims: List of claim texts.

        Returns:
            Dict mapping claim number to list of references.
        """
        result: Dict[int, List[SeqIdReference]] = {}

        for i, claim_text in enumerate(claims, 1):
            refs = self.extract_from_text(claim_text)
            if refs:
                for ref in refs:
                    ref.claim_number = i
                result[i] = refs

        return result

    def get_unique_seq_ids(self, text: str) -> List[int]:
        """Get sorted list of unique SEQ ID numbers from text.

        Args:
            text: Patent text.

        Returns:
            Sorted list of unique SEQ ID numbers.
        """
        refs = self.extract_from_text(text, context_chars=0)
        return sorted(set(ref.seq_id_no for ref in refs))


def parse_ncbi_fasta(fasta_content: str) -> List[Dict[str, Any]]:
    """Parse NCBI FASTA content into structured records.

    Args:
        fasta_content: Raw FASTA text from NCBI.

    Returns:
        List of dicts with sequence data.
    """
    records = []
    current_header = None
    current_seq = []

    for line in fasta_content.split("\n"):
        line = line.strip()
        if line.startswith(">"):
            if current_header and current_seq:
                records.append({
                    "header": current_header,
                    "sequence": "".join(current_seq),
                    "length": len("".join(current_seq)),
                })
            current_header = line[1:]
            current_seq = []
        elif line:
            current_seq.append(line)

    # Last record
    if current_header and current_seq:
        records.append({
            "header": current_header,
            "sequence": "".join(current_seq),
            "length": len("".join(current_seq)),
        })

    return records


def sequences_to_fasta_file(
    sequences: List[PatentSequence],
    output_path: Path,
) -> int:
    """Write sequences to a FASTA file.

    Args:
        sequences: List of PatentSequence objects.
        output_path: Path to output FASTA file.

    Returns:
        Number of sequences written.
    """
    with open(output_path, "w") as f:
        for seq in sequences:
            f.write(seq.to_fasta())
            f.write("\n")

    return len(sequences)
