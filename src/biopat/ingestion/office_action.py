"""USPTO Office Action Research Dataset loader.

Downloads and parses the Office Action dataset which contains
examiner citations under §102 and §103 rejections.
"""

import gzip
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import httpx
import polars as pl
from tqdm import tqdm

from biopat.reproducibility import ChecksumEngine, AuditLogger

logger = logging.getLogger(__name__)

# USPTO Office Action dataset URLs
OA_DATASET_BASE = "https://bulkdata.uspto.gov/data/patent/office/actions/bigdata"
OA_FILES = {
    "office_actions": "office_actions.csv.zip",
    "rejections": "rejections.csv.zip",
    "citations": "citations.csv.zip",
}


class OfficeActionLoader:
    """Loader for USPTO Office Action Research Dataset."""

    def __init__(
        self,
        raw_dir: Path,
        cache_dir: Path,
        checksum_engine: Optional[ChecksumEngine] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.cache_dir = Path(cache_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.oa_dir = self.raw_dir / "office_actions"
        self.oa_dir.mkdir(parents=True, exist_ok=True)
        self.checksum_engine = checksum_engine
        self.audit_logger = audit_logger

    def download_file(self, filename: str, force: bool = False) -> Path:
        """Download a single Office Action file.

        Args:
            filename: Name of file to download.
            force: If True, re-download even if exists.

        Returns:
            Path to downloaded file.
        """
        local_path = self.oa_dir / filename
        if local_path.exists() and not force:
            logger.info(f"File already exists: {local_path}")
            return local_path

        url = f"{OA_DATASET_BASE}/{filename}"
        logger.info(f"Downloading {url}")

        # Log API call
        if self.audit_logger:
            self.audit_logger.log_api_call(
                service="uspto",
                endpoint=f"office/actions/bigdata/{filename}",
                method="GET",
                params={"file": filename},
            )

        with httpx.stream("GET", url, follow_redirects=True, timeout=600.0) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            with open(local_path, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

        logger.info(f"Downloaded to {local_path}")

        # Log download with checksum
        if self.checksum_engine:
            self.checksum_engine.log_download(
                file_path=local_path,
                source_url=url,
                compute_hash=True,
            )

        return local_path

    def download_all(self, force: bool = False) -> Dict[str, Path]:
        """Download all Office Action files.

        Args:
            force: If True, re-download even if exists.

        Returns:
            Dict mapping table names to file paths.
        """
        paths = {}
        for table_name, filename in OA_FILES.items():
            paths[table_name] = self.download_file(filename, force)
        return paths

    def _extract_csv_from_zip(self, zip_path: Path) -> Path:
        """Extract CSV from zip file.

        Args:
            zip_path: Path to zip file.

        Returns:
            Path to extracted CSV.
        """
        csv_path = zip_path.with_suffix("")  # Remove .zip
        if csv_path.exists():
            return csv_path

        logger.info(f"Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Get the CSV file name inside the zip
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if csv_names:
                zf.extract(csv_names[0], self.oa_dir)
                extracted = self.oa_dir / csv_names[0]
                if extracted != csv_path:
                    extracted.rename(csv_path)

        return csv_path

    def load_office_actions(self, force_reload: bool = False) -> pl.DataFrame:
        """Load office actions metadata.

        Args:
            force_reload: If True, reload from CSV.

        Returns:
            DataFrame with office action metadata.
        """
        cache_path = self.cache_dir / "office_actions.parquet"
        if cache_path.exists() and not force_reload:
            logger.info(f"Loading cached office actions from {cache_path}")
            return pl.read_parquet(cache_path)

        zip_path = self.oa_dir / OA_FILES["office_actions"]
        if not zip_path.exists():
            self.download_file(OA_FILES["office_actions"])

        csv_path = self._extract_csv_from_zip(zip_path)
        logger.info(f"Parsing office actions from {csv_path}")

        df = pl.read_csv(
            csv_path,
            schema_overrides={
                "app_id": pl.Utf8,
                "ifw_number": pl.Utf8,
                "mail_dt": pl.Utf8,
            },
            ignore_errors=True,
        )

        # Standardize column names
        df = df.rename({col: col.lower().replace(" ", "_") for col in df.columns})

        df.write_parquet(cache_path)
        logger.info(f"Cached office actions to {cache_path}")
        return df

    def load_rejections(self, force_reload: bool = False) -> pl.DataFrame:
        """Load rejection data from Office Actions.

        Args:
            force_reload: If True, reload from CSV.

        Returns:
            DataFrame with rejection details.
        """
        cache_path = self.cache_dir / "rejections.parquet"
        if cache_path.exists() and not force_reload:
            logger.info(f"Loading cached rejections from {cache_path}")
            return pl.read_parquet(cache_path)

        zip_path = self.oa_dir / OA_FILES["rejections"]
        if not zip_path.exists():
            self.download_file(OA_FILES["rejections"])

        csv_path = self._extract_csv_from_zip(zip_path)
        logger.info(f"Parsing rejections from {csv_path}")

        df = pl.read_csv(
            csv_path,
            schema_overrides={
                "app_id": pl.Utf8,
                "ifw_number": pl.Utf8,
                "rejection_fp": pl.Utf8,
                "alice_ind": pl.Utf8,
            },
            ignore_errors=True,
        )

        df = df.rename({col: col.lower().replace(" ", "_") for col in df.columns})

        df.write_parquet(cache_path)
        logger.info(f"Cached rejections to {cache_path}")
        return df

    def load_citations(self, force_reload: bool = False) -> pl.DataFrame:
        """Load citation data from Office Actions.

        Args:
            force_reload: If True, reload from CSV.

        Returns:
            DataFrame with citation details.
        """
        cache_path = self.cache_dir / "oa_citations.parquet"
        if cache_path.exists() and not force_reload:
            logger.info(f"Loading cached citations from {cache_path}")
            return pl.read_parquet(cache_path)

        zip_path = self.oa_dir / OA_FILES["citations"]
        if not zip_path.exists():
            self.download_file(OA_FILES["citations"])

        csv_path = self._extract_csv_from_zip(zip_path)
        logger.info(f"Parsing citations from {csv_path}")

        df = pl.read_csv(
            csv_path,
            schema_overrides={
                "app_id": pl.Utf8,
                "ifw_number": pl.Utf8,
                "cite_pat_pgpub_id": pl.Utf8,
                "cite_npl_str": pl.Utf8,
            },
            ignore_errors=True,
        )

        df = df.rename({col: col.lower().replace(" ", "_") for col in df.columns})

        df.write_parquet(cache_path)
        logger.info(f"Cached citations to {cache_path}")
        return df

    def get_npl_citations(
        self,
        citations_df: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """Get non-patent literature citations.

        Args:
            citations_df: Pre-loaded citations DataFrame.

        Returns:
            DataFrame of NPL citations only.
        """
        if citations_df is None:
            citations_df = self.load_citations()

        # Filter to NPL citations (cite_npl_str is not null/empty)
        npl = citations_df.filter(
            pl.col("cite_npl_str").is_not_null() &
            (pl.col("cite_npl_str").str.len_chars() > 0)
        )

        logger.info(f"Found {len(npl)} NPL citations")
        return npl

    def get_102_103_rejections(
        self,
        rejections_df: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """Get rejections under 35 USC 102 and 103.

        Args:
            rejections_df: Pre-loaded rejections DataFrame.

        Returns:
            DataFrame of 102/103 rejections only.
        """
        if rejections_df is None:
            rejections_df = self.load_rejections()

        # Filter to 102 and 103 rejections
        # The rejection_fp field contains fingerprint including rejection type
        filtered = rejections_df.filter(
            pl.col("rejection_fp").str.contains("102") |
            pl.col("rejection_fp").str.contains("103")
        )

        # Add rejection type column
        filtered = filtered.with_columns(
            pl.when(pl.col("rejection_fp").str.contains("102"))
            .then(pl.lit("102"))
            .when(pl.col("rejection_fp").str.contains("103"))
            .then(pl.lit("103"))
            .otherwise(pl.lit("other"))
            .alias("rejection_type")
        )

        logger.info(
            f"Found {len(filtered)} 102/103 rejections "
            f"(102: {len(filtered.filter(pl.col('rejection_type') == '102'))}, "
            f"103: {len(filtered.filter(pl.col('rejection_type') == '103'))})"
        )
        return filtered

    def join_rejections_with_citations(
        self,
        rejections_df: Optional[pl.DataFrame] = None,
        citations_df: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """Join rejections with their associated citations.

        Args:
            rejections_df: Rejections DataFrame.
            citations_df: Citations DataFrame.

        Returns:
            DataFrame joining rejections to NPL citations.
        """
        if rejections_df is None:
            rejections_df = self.get_102_103_rejections()
        if citations_df is None:
            citations_df = self.get_npl_citations()

        # Join on app_id and ifw_number
        joined = rejections_df.join(
            citations_df,
            on=["app_id", "ifw_number"],
            how="inner"
        )

        logger.info(f"Joined {len(joined)} rejection-citation pairs")
        return joined

    def get_stats(self) -> Dict:
        """Get statistics about the Office Action data.

        Returns:
            Dict of statistics.
        """
        stats = {}

        try:
            oa_df = self.load_office_actions()
            stats["total_office_actions"] = len(oa_df)
            stats["unique_applications"] = oa_df["app_id"].n_unique()
        except Exception as e:
            logger.warning(f"Could not load office actions: {e}")

        try:
            rej_df = self.load_rejections()
            stats["total_rejections"] = len(rej_df)
            filtered = self.get_102_103_rejections(rej_df)
            stats["102_rejections"] = len(filtered.filter(pl.col("rejection_type") == "102"))
            stats["103_rejections"] = len(filtered.filter(pl.col("rejection_type") == "103"))
        except Exception as e:
            logger.warning(f"Could not load rejections: {e}")

        try:
            cit_df = self.load_citations()
            stats["total_citations"] = len(cit_df)
            npl = self.get_npl_citations(cit_df)
            stats["npl_citations"] = len(npl)
        except Exception as e:
            logger.warning(f"Could not load citations: {e}")

        return stats
