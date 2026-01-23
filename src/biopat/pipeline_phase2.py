"""Phase 2 Pipeline for BioPAT.

Extends Phase 1 with Office Action data for examiner-validated
graded relevance (0-3 scale) and claim-level mapping.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import polars as pl

from . import compat
from .config import BioPatConfig
from .ingestion import (
    RelianceOnScienceLoader,
    PatentsViewClient,
    OpenAlexClient,
    OfficeActionLoader,
)
from .processing import (
    PatentProcessor,
    PaperProcessor,
    CitationLinker,
    NPLParser,
    NPLLinker,
    ClaimCitationMapper,
)
from .groundtruth import (
    RelevanceAssigner,
    TemporalValidator,
    DomainStratifier,
    create_stratified_evaluation,
)
from .benchmark import BenchmarkSampler, DatasetSplitter, BEIRFormatter

logger = logging.getLogger(__name__)


class Phase2Pipeline:
    """Pipeline for Phase 2 enhanced ground truth benchmark."""

    def __init__(self, config: BioPatConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.cache_dir = self.data_dir / "cache"
        self.processed_dir = self.data_dir / "processed"
        self.benchmark_dir = self.data_dir / "benchmark"

        # Create directories
        for d in [self.raw_dir, self.cache_dir, self.processed_dir, self.benchmark_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.ros_loader = RelianceOnScienceLoader(self.raw_dir, self.cache_dir)
        self.oa_loader = OfficeActionLoader(self.raw_dir, self.cache_dir)
        self.patentsview = PatentsViewClient(
            api_key=config.patentsview_api_key,
            cache_dir=self.cache_dir,
        )
        self.openalex = OpenAlexClient(
            mailto=config.openalex_mailto,
            cache_dir=self.cache_dir,
        )
        self.patent_processor = PatentProcessor(self.processed_dir)
        self.paper_processor = PaperProcessor(self.processed_dir)
        self.citation_linker = CitationLinker(self.processed_dir)
        self.relevance_assigner = RelevanceAssigner(self.processed_dir)
        self.temporal_validator = TemporalValidator(self.processed_dir)
        self.domain_stratifier = DomainStratifier()
        self.sampler = BenchmarkSampler(seed=config.seed)
        self.splitter = DatasetSplitter(seed=config.seed)
        self.formatter = BEIRFormatter(self.benchmark_dir)

    def step1_load_ros_data(self) -> pl.DataFrame:
        """Load Reliance on Science citations.

        Returns:
            RoS DataFrame.
        """
        logger.info("Phase 2 Step 1: Loading RoS data")
        return self.ros_loader.load(
            confidence_threshold=self.config.ros_confidence_threshold
        )

    def step2_load_office_action_data(self) -> pl.DataFrame:
        """Load and process Office Action data.

        Returns:
            Office Action citations with rejection types.
        """
        logger.info("Phase 2 Step 2: Loading Office Action data")

        # Load OA data
        rejections = self.oa_loader.get_102_103_rejections()
        citations = self.oa_loader.get_npl_citations()

        # Join rejections with citations
        oa_with_citations = self.oa_loader.join_rejections_with_citations(
            rejections, citations
        )

        logger.info(f"Loaded {len(oa_with_citations)} OA rejection-citation pairs")
        return oa_with_citations

    def step3_parse_npl_citations(
        self,
        oa_data: pl.DataFrame,
    ) -> pl.DataFrame:
        """Parse NPL citations and extract identifiers.

        Args:
            oa_data: Office Action data with NPL citation strings.

        Returns:
            Parsed citations with PMID, DOI, title.
        """
        logger.info("Phase 2 Step 3: Parsing NPL citations")

        parser = NPLParser()
        parsed = parser.parse_citations_df(oa_data, text_col="cite_npl_str")

        return parsed

    def step4_link_citations_to_papers(
        self,
        parsed_citations: pl.DataFrame,
        papers_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Link parsed NPL citations to papers in corpus.

        Args:
            parsed_citations: Parsed citation DataFrame.
            papers_df: Papers corpus DataFrame.

        Returns:
            Citations linked to paper IDs.
        """
        logger.info("Phase 2 Step 4: Linking citations to papers")

        linker = NPLLinker(papers_df)
        linked = linker.link_citations_df(parsed_citations)

        return linked

    def step5_map_claims_to_citations(
        self,
        linked_citations: pl.DataFrame,
        patents_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Map citations to specific claims.

        Args:
            linked_citations: Linked citation DataFrame.
            patents_df: Patents DataFrame with claims.

        Returns:
            Claim-paper mapping with rejection types.
        """
        logger.info("Phase 2 Step 5: Mapping claims to citations")

        mapper = ClaimCitationMapper()

        # We need rejections data with claim info
        rejections = linked_citations

        # Create mapping
        mapping = mapper.create_claim_paper_mapping(
            rejections_df=rejections,
            linked_citations_df=linked_citations,
            patents_df=patents_df,
        )

        return mapping

    def step6_create_graded_qrels(
        self,
        ros_links: pl.DataFrame,
        oa_claim_mapping: pl.DataFrame,
    ) -> pl.DataFrame:
        """Create graded relevance judgments.

        Args:
            ros_links: RoS citation links.
            oa_claim_mapping: Office Action claim-paper mapping.

        Returns:
            Graded qrels DataFrame.
        """
        logger.info("Phase 2 Step 6: Creating graded qrels")

        # Merge RoS and OA data
        merged = self.relevance_assigner.merge_ros_and_oa_citations(
            ros_links, oa_claim_mapping
        )

        # Create graded qrels
        qrels = self.relevance_assigner.create_graded_qrels_from_merged(merged)

        return qrels

    def step7_stratify_evaluation(
        self,
        qrels: pl.DataFrame,
        queries_df: pl.DataFrame,
        papers_df: pl.DataFrame,
    ) -> dict:
        """Create domain-stratified evaluation splits.

        Args:
            qrels: Graded qrels DataFrame.
            queries_df: Queries with IPC codes.
            papers_df: Papers with concepts.

        Returns:
            Dict with all, in_domain, out_domain qrels.
        """
        logger.info("Phase 2 Step 7: Creating stratified evaluation")

        result = create_stratified_evaluation(
            qrels_df=qrels,
            queries_df=queries_df,
            papers_df=papers_df,
            output_dir=str(self.processed_dir / "stratified"),
        )

        return result

    def step8_format_benchmark(
        self,
        queries_df: pl.DataFrame,
        papers_df: pl.DataFrame,
        qrels: pl.DataFrame,
    ):
        """Format and save BEIR-compatible benchmark.

        Args:
            queries_df: Queries DataFrame.
            papers_df: Papers DataFrame.
            qrels: Qrels DataFrame.
        """
        logger.info("Phase 2 Step 8: Formatting BEIR benchmark")

        # Create train/dev/test splits
        splits = self.splitter.create_splits(
            queries_df=queries_df,
            qrels_df=qrels,
            stratify_col="domain",
            split_by_patent=True,
        )

        # Format corpus
        self.formatter.format_corpus(papers_df)

        # Format queries
        self.formatter.format_queries(queries_df)

        # Format qrels for each split
        for split_name, (split_queries, split_qrels) in splits.items():
            self.formatter.format_qrels(split_qrels, split_name)

        # Validate output
        validation = self.formatter.validate_output()
        logger.info(f"Benchmark validation: {validation}")

    async def run(self) -> dict:
        """Run the complete Phase 2 pipeline.

        Returns:
            Dict with pipeline results and statistics.
        """
        results = {}

        try:
            # Step 1: Load RoS data
            ros_df = self.step1_load_ros_data()
            results["ros_citations"] = len(ros_df)

            # Step 2: Load Office Action data
            oa_df = self.step2_load_office_action_data()
            results["oa_citations"] = len(oa_df)

            # Load existing Phase 1 data (patents and papers)
            patents_path = self.processed_dir / "patents.parquet"
            papers_path = self.processed_dir / "papers.parquet"

            if patents_path.exists():
                patents_df = pl.read_parquet(patents_path)
            else:
                logger.warning("Patents not found, running Phase 1 first")
                # Would need to run Phase 1 pipeline
                return {"error": "Run Phase 1 first"}

            if papers_path.exists():
                papers_df = pl.read_parquet(papers_path)
            else:
                logger.warning("Papers not found, running Phase 1 first")
                return {"error": "Run Phase 1 first"}

            results["patents"] = len(patents_df)
            results["papers"] = len(papers_df)

            # Step 3: Parse NPL citations
            parsed_citations = self.step3_parse_npl_citations(oa_df)
            results["parsed_citations"] = len(parsed_citations)

            # Step 4: Link to papers
            linked_citations = self.step4_link_citations_to_papers(
                parsed_citations, papers_df
            )
            linked_count = linked_citations.filter(
                pl.col("linked_paper_id").is_not_null()
            ).height
            results["linked_citations"] = linked_count

            # Step 5: Map claims
            claim_mapping = self.step5_map_claims_to_citations(
                linked_citations, patents_df
            )
            results["claim_mappings"] = len(claim_mapping)

            # Load existing Phase 1 citation links
            ros_links_path = self.processed_dir / "citation_links.parquet"
            if ros_links_path.exists():
                ros_links = pl.read_parquet(ros_links_path)
            else:
                ros_links = pl.DataFrame()

            # Step 6: Create graded qrels
            if len(claim_mapping) > 0:
                qrels = self.step6_create_graded_qrels(ros_links, claim_mapping)
            else:
                # Fall back to Phase 1 style if no OA data linked
                qrels = self.relevance_assigner.assign_graded_relevance(ros_links)

            results["qrels"] = len(qrels)

            # Get claims as queries
            claims_path = self.processed_dir / "claims.parquet"
            if claims_path.exists():
                queries_df = pl.read_parquet(claims_path)
            else:
                queries_df = self.patent_processor.extract_independent_claims(patents_df)

            # Step 7: Stratify
            stratified = self.step7_stratify_evaluation(qrels, queries_df, papers_df)
            results["in_domain_qrels"] = len(stratified.get("in_domain", []))
            results["out_domain_qrels"] = len(stratified.get("out_domain", []))

            # Step 8: Format benchmark
            self.step8_format_benchmark(queries_df, papers_df, qrels)

            # Summary statistics
            results["status"] = "success"
            logger.info(f"Phase 2 pipeline complete: {results}")

        except Exception as e:
            logger.exception(f"Phase 2 pipeline failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)

        return results


async def run_phase2(config_path: Optional[str] = None) -> dict:
    """Convenience function to run Phase 2 pipeline.

    Args:
        config_path: Optional path to config file.

    Returns:
        Pipeline results.
    """
    if config_path:
        config = BioPatConfig.from_yaml(config_path)
    else:
        config = BioPatConfig()

    pipeline = Phase2Pipeline(config)
    return await pipeline.run()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    results = asyncio.run(run_phase2(config_path))
    print(results)
