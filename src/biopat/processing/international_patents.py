"""International Patent Corpus Assembly.

Phase 6 (v3.0): Assembles a unified corpus of US, EP, and WO patents
for the international prior art retrieval benchmark.

This module handles:
- Merging patents from multiple jurisdictions into unified corpus
- Patent family deduplication to avoid redundant entries
- IPC-based filtering for biomedical domains
- Temporal filtering for prior art validity
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import polars as pl

from biopat.processing.patent_ids import (
    Jurisdiction,
    ParsedPatentId,
    PatentIdNormalizer,
    normalize_patent_id,
    group_by_jurisdiction,
)
from biopat.ingestion.epo import EPOClient
from biopat.ingestion.wipo import WIPOClient
from biopat.ingestion.patentsview import PatentsViewClient
from biopat.reproducibility import AuditLogger

logger = logging.getLogger(__name__)

# Biomedical IPC prefixes
DEFAULT_IPC_PREFIXES = ["A61", "C07", "C12"]


@dataclass
class InternationalPatent:
    """Unified patent record across jurisdictions."""

    patent_id: str
    jurisdiction: Jurisdiction
    title: str
    abstract: str
    publication_date: str
    priority_date: str
    ipc_codes: List[str] = field(default_factory=list)
    claims_text: str = ""
    family_id: Optional[str] = None
    source: str = ""  # "us", "ep", "wo"

    @property
    def text_for_corpus(self) -> str:
        """Generate text representation for corpus entry."""
        parts = []
        if self.title:
            parts.append(self.title)
        if self.abstract:
            parts.append(self.abstract)
        if self.claims_text:
            parts.append(self.claims_text)
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "patent_id": self.patent_id,
            "jurisdiction": self.jurisdiction.value,
            "title": self.title,
            "abstract": self.abstract,
            "publication_date": self.publication_date,
            "priority_date": self.priority_date,
            "ipc_codes": self.ipc_codes,
            "claims_text": self.claims_text,
            "family_id": self.family_id,
            "source": self.source,
        }


@dataclass
class InternationalCorpusConfig:
    """Configuration for international corpus assembly."""

    include_us: bool = True
    include_ep: bool = True
    include_wo: bool = True
    ipc_prefixes: List[str] = field(default_factory=lambda: DEFAULT_IPC_PREFIXES.copy())
    publication_date_from: Optional[str] = None
    publication_date_to: Optional[str] = None
    max_patents_per_jurisdiction: Optional[int] = None
    deduplicate_families: bool = True
    include_claims: bool = True


class InternationalCorpusBuilder:
    """Builds international patent corpus from multiple sources.

    Integrates US (PatentsView), EP (EPO OPS), and WO (WIPO) patent data
    into a unified corpus for the v3.0 benchmark.
    """

    def __init__(
        self,
        us_client: Optional[PatentsViewClient] = None,
        epo_client: Optional[EPOClient] = None,
        wipo_client: Optional[WIPOClient] = None,
        audit_logger: Optional[AuditLogger] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize corpus builder.

        Args:
            us_client: PatentsView client for US patents.
            epo_client: EPO OPS client for EP patents.
            wipo_client: WIPO client for PCT/WO patents.
            audit_logger: Optional audit logger.
            cache_dir: Directory for caching.
        """
        self.us_client = us_client
        self.epo_client = epo_client
        self.wipo_client = wipo_client
        self.audit_logger = audit_logger
        self.cache_dir = cache_dir

        self._id_normalizer = PatentIdNormalizer()
        self._family_cache: Dict[str, str] = {}

    async def build_corpus(
        self,
        config: InternationalCorpusConfig,
        show_progress: bool = True,
    ) -> pl.DataFrame:
        """Build unified international patent corpus.

        Args:
            config: Corpus assembly configuration.
            show_progress: Show progress bars.

        Returns:
            DataFrame with unified patent records.
        """
        all_patents: List[InternationalPatent] = []

        # Fetch from each jurisdiction
        if config.include_us and self.us_client:
            us_patents = await self._fetch_us_patents(config, show_progress)
            all_patents.extend(us_patents)
            logger.info(f"Fetched {len(us_patents)} US patents")

        if config.include_ep and self.epo_client:
            ep_patents = await self._fetch_ep_patents(config, show_progress)
            all_patents.extend(ep_patents)
            logger.info(f"Fetched {len(ep_patents)} EP patents")

        if config.include_wo and self.wipo_client:
            wo_patents = await self._fetch_wo_patents(config, show_progress)
            all_patents.extend(wo_patents)
            logger.info(f"Fetched {len(wo_patents)} WO patents")

        # Deduplicate by family if enabled
        if config.deduplicate_families:
            all_patents = self._deduplicate_by_family(all_patents)
            logger.info(f"After family deduplication: {len(all_patents)} patents")

        # Convert to DataFrame
        records = [p.to_dict() for p in all_patents]
        df = pl.DataFrame(records)

        logger.info(f"Built international corpus with {len(df)} patents")
        return df

    async def _fetch_us_patents(
        self,
        config: InternationalCorpusConfig,
        show_progress: bool,
    ) -> List[InternationalPatent]:
        """Fetch US patents from PatentsView."""
        if not self.us_client:
            return []

        patents = []

        # Build IPC query
        ipc_filter = [{"cpc_subgroup_id": f"~{ipc}"} for ipc in config.ipc_prefixes]

        try:
            # Use existing PatentsView search
            results = await self.us_client.search_patents_by_ipc(
                ipc_prefixes=config.ipc_prefixes,
                limit=config.max_patents_per_jurisdiction,
            )

            for r in results:
                patents.append(InternationalPatent(
                    patent_id=f"US{r.get('patent_number', '')}",
                    jurisdiction=Jurisdiction.US,
                    title=r.get("patent_title", ""),
                    abstract=r.get("patent_abstract", ""),
                    publication_date=r.get("patent_date", ""),
                    priority_date=r.get("application_date", ""),
                    ipc_codes=r.get("cpc_codes", []),
                    claims_text=r.get("claim_text", "") if config.include_claims else "",
                    source="us",
                ))

        except Exception as e:
            logger.error(f"Failed to fetch US patents: {e}")

        return patents

    async def _fetch_ep_patents(
        self,
        config: InternationalCorpusConfig,
        show_progress: bool,
    ) -> List[InternationalPatent]:
        """Fetch EP patents from EPO OPS."""
        if not self.epo_client:
            return []

        patents = []

        try:
            results = await self.epo_client.get_biomedical_patents(
                ipc_classes=config.ipc_prefixes,
                limit=config.max_patents_per_jurisdiction,
                publication_date_from=config.publication_date_from,
                publication_date_to=config.publication_date_to,
            )

            # Convert to DataFrame for easier processing
            df = self.epo_client.patents_to_dataframe(results)

            for row in df.iter_rows(named=True):
                patents.append(InternationalPatent(
                    patent_id=row["patent_id"],
                    jurisdiction=Jurisdiction.EP,
                    title=row["title"],
                    abstract=row["abstract"],
                    publication_date=row["publication_date"],
                    priority_date=row["priority_date"],
                    ipc_codes=row["ipc_codes"],
                    source="ep",
                ))

        except Exception as e:
            logger.error(f"Failed to fetch EP patents: {e}")

        return patents

    async def _fetch_wo_patents(
        self,
        config: InternationalCorpusConfig,
        show_progress: bool,
    ) -> List[InternationalPatent]:
        """Fetch WO patents from WIPO PATENTSCOPE."""
        if not self.wipo_client:
            return []

        patents = []

        try:
            results = await self.wipo_client.get_biomedical_pct(
                ipc_classes=config.ipc_prefixes,
                limit=config.max_patents_per_jurisdiction,
                publication_date_from=config.publication_date_from,
                publication_date_to=config.publication_date_to,
            )

            # Convert to DataFrame
            df = self.wipo_client.patents_to_dataframe(results)

            for row in df.iter_rows(named=True):
                patents.append(InternationalPatent(
                    patent_id=row["patent_id"],
                    jurisdiction=Jurisdiction.WO,
                    title=row["title"],
                    abstract=row["abstract"],
                    publication_date=row["publication_date"],
                    priority_date=row["priority_date"],
                    ipc_codes=row["ipc_codes"],
                    source="wo",
                ))

        except Exception as e:
            logger.error(f"Failed to fetch WO patents: {e}")

        return patents

    def _deduplicate_by_family(
        self,
        patents: List[InternationalPatent],
    ) -> List[InternationalPatent]:
        """Remove duplicate patents from the same family.

        Priority: US > EP > WO (keeps most authoritative version).

        Args:
            patents: List of patents from multiple jurisdictions.

        Returns:
            Deduplicated list.
        """
        # Group by family ID where available
        family_groups: Dict[str, List[InternationalPatent]] = {}
        no_family: List[InternationalPatent] = []

        for p in patents:
            if p.family_id:
                if p.family_id not in family_groups:
                    family_groups[p.family_id] = []
                family_groups[p.family_id].append(p)
            else:
                no_family.append(p)

        # Select one representative per family
        result = []
        priority_order = [Jurisdiction.US, Jurisdiction.EP, Jurisdiction.WO]

        for family_id, family_patents in family_groups.items():
            # Sort by priority
            family_patents.sort(
                key=lambda p: priority_order.index(p.jurisdiction)
                if p.jurisdiction in priority_order else 999
            )
            result.append(family_patents[0])

        # Add patents without family info
        result.extend(no_family)

        return result

    async def fetch_patents_by_ids(
        self,
        patent_ids: List[str],
        show_progress: bool = True,
    ) -> List[InternationalPatent]:
        """Fetch specific patents by ID from appropriate sources.

        Automatically routes requests to correct client based on jurisdiction.

        Args:
            patent_ids: List of patent IDs (any jurisdiction).
            show_progress: Show progress bars.

        Returns:
            List of fetched patents.
        """
        # Group by jurisdiction
        grouped = group_by_jurisdiction(patent_ids)
        results: List[InternationalPatent] = []

        # Fetch from each jurisdiction
        for jurisdiction, parsed_ids in grouped.items():
            id_strings = [p.full for p in parsed_ids]

            if jurisdiction == Jurisdiction.US and self.us_client:
                us_results = await self._fetch_us_by_ids(id_strings)
                results.extend(us_results)

            elif jurisdiction == Jurisdiction.EP and self.epo_client:
                ep_results = await self._fetch_ep_by_ids(id_strings)
                results.extend(ep_results)

            elif jurisdiction == Jurisdiction.WO and self.wipo_client:
                wo_results = await self._fetch_wo_by_ids(id_strings)
                results.extend(wo_results)

            else:
                logger.warning(
                    f"No client available for jurisdiction {jurisdiction}, "
                    f"skipping {len(parsed_ids)} patents"
                )

        return results

    async def _fetch_us_by_ids(
        self,
        patent_ids: List[str],
    ) -> List[InternationalPatent]:
        """Fetch US patents by ID."""
        if not self.us_client:
            return []

        patents = []
        try:
            results = await self.us_client.search_patents_by_ids(patent_ids)

            for r in results:
                patents.append(InternationalPatent(
                    patent_id=f"US{r.get('patent_number', '')}",
                    jurisdiction=Jurisdiction.US,
                    title=r.get("patent_title", ""),
                    abstract=r.get("patent_abstract", ""),
                    publication_date=r.get("patent_date", ""),
                    priority_date=r.get("application_date", ""),
                    ipc_codes=r.get("cpc_codes", []),
                    source="us",
                ))

        except Exception as e:
            logger.error(f"Failed to fetch US patents by ID: {e}")

        return patents

    async def _fetch_ep_by_ids(
        self,
        patent_ids: List[str],
    ) -> List[InternationalPatent]:
        """Fetch EP patents by ID."""
        if not self.epo_client:
            return []

        patents = []
        try:
            results = await self.epo_client.get_patents_batch(patent_ids)
            df = self.epo_client.patents_to_dataframe(results)

            for row in df.iter_rows(named=True):
                patents.append(InternationalPatent(
                    patent_id=row["patent_id"],
                    jurisdiction=Jurisdiction.EP,
                    title=row["title"],
                    abstract=row["abstract"],
                    publication_date=row["publication_date"],
                    priority_date=row["priority_date"],
                    ipc_codes=row["ipc_codes"],
                    source="ep",
                ))

        except Exception as e:
            logger.error(f"Failed to fetch EP patents by ID: {e}")

        return patents

    async def _fetch_wo_by_ids(
        self,
        patent_ids: List[str],
    ) -> List[InternationalPatent]:
        """Fetch WO patents by ID."""
        if not self.wipo_client:
            return []

        patents = []
        try:
            results = await self.wipo_client.get_patents_batch(patent_ids)
            df = self.wipo_client.patents_to_dataframe(results)

            for row in df.iter_rows(named=True):
                patents.append(InternationalPatent(
                    patent_id=row["patent_id"],
                    jurisdiction=Jurisdiction.WO,
                    title=row["title"],
                    abstract=row["abstract"],
                    publication_date=row["publication_date"],
                    priority_date=row["priority_date"],
                    ipc_codes=row["ipc_codes"],
                    source="wo",
                ))

        except Exception as e:
            logger.error(f"Failed to fetch WO patents by ID: {e}")

        return patents


def create_international_corpus_entry(
    patent: InternationalPatent,
    doc_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a BEIR-format corpus entry from an international patent.

    Args:
        patent: International patent record.
        doc_id: Optional custom document ID.

    Returns:
        Dict with _id, text, title, metadata fields.
    """
    return {
        "_id": doc_id or patent.patent_id,
        "text": patent.text_for_corpus,
        "title": patent.title,
        "metadata": {
            "doc_type": "patent",
            "jurisdiction": patent.jurisdiction.value,
            "publication_date": patent.publication_date,
            "priority_date": patent.priority_date,
            "ipc_codes": patent.ipc_codes,
            "source": patent.source,
        },
    }


def merge_corpus_dataframes(
    us_df: Optional[pl.DataFrame],
    ep_df: Optional[pl.DataFrame],
    wo_df: Optional[pl.DataFrame],
) -> pl.DataFrame:
    """Merge corpus DataFrames from multiple jurisdictions.

    Args:
        us_df: US patent DataFrame.
        ep_df: EP patent DataFrame.
        wo_df: WO patent DataFrame.

    Returns:
        Merged DataFrame with jurisdiction column.
    """
    dfs = []

    if us_df is not None and len(us_df) > 0:
        us_df = us_df.with_columns(pl.lit("US").alias("jurisdiction"))
        dfs.append(us_df)

    if ep_df is not None and len(ep_df) > 0:
        ep_df = ep_df.with_columns(pl.lit("EP").alias("jurisdiction"))
        dfs.append(ep_df)

    if wo_df is not None and len(wo_df) > 0:
        wo_df = wo_df.with_columns(pl.lit("WO").alias("jurisdiction"))
        dfs.append(wo_df)

    if not dfs:
        return pl.DataFrame()

    # Concatenate all
    return pl.concat(dfs, how="diagonal")


def get_corpus_statistics(df: pl.DataFrame) -> Dict[str, Any]:
    """Compute statistics for an international corpus.

    Args:
        df: Corpus DataFrame with jurisdiction column.

    Returns:
        Dict with corpus statistics.
    """
    stats = {
        "total_documents": len(df),
        "by_jurisdiction": {},
    }

    if "jurisdiction" in df.columns:
        for jur in ["US", "EP", "WO"]:
            count = len(df.filter(pl.col("jurisdiction") == jur))
            stats["by_jurisdiction"][jur] = count

    if "doc_type" in df.columns:
        stats["by_doc_type"] = {}
        for dtype in df["doc_type"].unique().to_list():
            count = len(df.filter(pl.col("doc_type") == dtype))
            stats["by_doc_type"][dtype] = count

    return stats
