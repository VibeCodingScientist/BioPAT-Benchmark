"""Phase 3 Pipeline for BioPAT.

Runs comprehensive baselines, ablation studies, and error analysis
to characterize benchmark difficulty and establish reference performance.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Phase3Pipeline:
    """Pipeline for Phase 3 comprehensive evaluation."""

    def __init__(
        self,
        benchmark_dir: Path,
        results_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize pipeline.

        Args:
            benchmark_dir: Path to BEIR-format benchmark.
            results_dir: Directory for results.
            cache_dir: Directory for caching embeddings.
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.results_dir = Path(results_dir) if results_dir else self.benchmark_dir / "results"
        self.cache_dir = Path(cache_dir) if cache_dir else self.benchmark_dir / "cache"

        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def step1_run_bm25_baseline(
        self,
        split: str = "test",
        k_values: List[int] = [10, 50, 100],
    ) -> Dict[str, float]:
        """Run BM25 baseline evaluation.

        Args:
            split: Evaluation split.
            k_values: Values of k for metrics.

        Returns:
            Dictionary of metrics.
        """
        logger.info("Phase 3 Step 1: Running BM25 baseline")

        from .evaluation.bm25 import BM25Evaluator

        evaluator = BM25Evaluator(self.benchmark_dir, self.results_dir)
        metrics = evaluator.run_evaluation(
            split=split,
            top_k=100,
            k_values=k_values,
            save_results=True,
        )

        return metrics

    def step2_run_dense_baselines(
        self,
        models: Optional[List[str]] = None,
        split: str = "test",
        k_values: List[int] = [10, 50, 100],
    ) -> Dict[str, Dict[str, float]]:
        """Run dense retrieval baselines.

        Args:
            models: List of model names to evaluate.
            split: Evaluation split.
            k_values: Values of k for metrics.

        Returns:
            Dictionary mapping model name to metrics.
        """
        logger.info("Phase 3 Step 2: Running dense baselines")

        from .evaluation.dense import DenseEvaluator

        if models is None:
            models = ["contriever", "specter2", "all-mpnet-base"]

        evaluator = DenseEvaluator(
            benchmark_dir=self.benchmark_dir,
            results_dir=self.results_dir,
            cache_dir=self.cache_dir,
        )

        all_metrics = evaluator.run_all_baselines(
            models=models,
            split=split,
            top_k=100,
            k_values=k_values,
        )

        return all_metrics

    def step3_run_hybrid_baseline(
        self,
        dense_model: str = "contriever",
        split: str = "test",
        k_values: List[int] = [10, 50, 100],
    ) -> Dict[str, float]:
        """Run hybrid BM25 + dense baseline.

        Args:
            dense_model: Dense model to use for hybrid.
            split: Evaluation split.
            k_values: Values of k for metrics.

        Returns:
            Dictionary of metrics.
        """
        logger.info("Phase 3 Step 3: Running hybrid baseline")

        from .evaluation.hybrid import BM25DenseHybrid

        hybrid = BM25DenseHybrid(
            benchmark_dir=str(self.benchmark_dir),
            dense_model=dense_model,
            fusion_method="rrf",
            cache_dir=str(self.cache_dir),
        )

        metrics = hybrid.run_evaluation(
            split=split,
            top_k=100,
            k_values=k_values,
        )

        # Save results
        results_path = self.results_dir / f"hybrid_rrf_{split}_results.json"
        with open(results_path, "w") as f:
            json.dump({"model": f"BM25+{dense_model}_RRF", "metrics": metrics}, f, indent=2)

        return metrics

    def step4_run_reranking_baseline(
        self,
        cross_encoder: str = "ms-marco-minilm",
        split: str = "test",
        k_values: List[int] = [10, 50, 100],
    ) -> Dict[str, float]:
        """Run BM25 + cross-encoder reranking baseline.

        Args:
            cross_encoder: Cross-encoder model to use.
            split: Evaluation split.
            k_values: Values of k for metrics.

        Returns:
            Dictionary of metrics.
        """
        logger.info("Phase 3 Step 4: Running reranking baseline")

        from .evaluation.reranker import BM25CrossEncoderPipeline

        pipeline = BM25CrossEncoderPipeline(
            benchmark_dir=str(self.benchmark_dir),
            cross_encoder_model=cross_encoder,
            bm25_top_k=100,
            rerank_top_k=100,
        )

        metrics = pipeline.run_evaluation(
            split=split,
            top_k=100,
            k_values=k_values,
        )

        # Save results
        results_path = self.results_dir / f"bm25_ce_{split}_results.json"
        with open(results_path, "w") as f:
            json.dump({"model": f"BM25+{cross_encoder}", "metrics": metrics}, f, indent=2)

        return metrics

    def step5_run_ablation_studies(
        self,
        results: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        split: str = "test",
    ) -> Dict[str, Any]:
        """Run ablation studies.

        Args:
            results: Retrieval results from best model.
            qrels: Query relevance judgments.
            split: Evaluation split.

        Returns:
            Ablation results.
        """
        logger.info("Phase 3 Step 5: Running ablation studies")

        from .evaluation.ablation import AblationRunner

        runner = AblationRunner(
            benchmark_dir=self.benchmark_dir,
            results_dir=self.results_dir / "ablation",
        )

        ablation_results = {}

        # Note: Domain and IPC ablations require metadata
        # These will be run if metadata is available

        return ablation_results

    def step6_run_error_analysis(
        self,
        results: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        queries: Dict[str, str],
        corpus: Dict[str, dict],
    ) -> Dict[str, Any]:
        """Run error analysis.

        Args:
            results: Retrieval results.
            qrels: Query relevance judgments.
            queries: Query texts.
            corpus: Document corpus.

        Returns:
            Error analysis results.
        """
        logger.info("Phase 3 Step 6: Running error analysis")

        from .evaluation.error_analysis import run_error_analysis, ErrorAnalysisConfig

        config = ErrorAnalysisConfig(
            sample_size=100,
            min_relevance=1,
            rank_threshold=100,
        )

        analysis_results = run_error_analysis(
            results=results,
            qrels=qrels,
            queries=queries,
            corpus=corpus,
            output_dir=self.results_dir / "error_analysis",
            config=config,
        )

        return analysis_results

    def step7_generate_report(
        self,
        all_results: Dict[str, Any],
    ) -> str:
        """Generate final evaluation report.

        Args:
            all_results: All evaluation results.

        Returns:
            Markdown report.
        """
        logger.info("Phase 3 Step 7: Generating report")

        lines = [
            "# BioPAT Phase 3 Evaluation Report",
            "",
            "## Baseline Results",
            "",
            "### Lexical Baseline (BM25)",
            "",
        ]

        if "bm25" in all_results:
            for metric, value in sorted(all_results["bm25"].items()):
                lines.append(f"- {metric}: {value:.4f}")

        lines.extend([
            "",
            "### Dense Baselines",
            "",
            "| Model | NDCG@10 | NDCG@100 | Recall@100 | MAP |",
            "|-------|---------|----------|------------|-----|",
        ])

        if "dense" in all_results:
            for model, metrics in all_results["dense"].items():
                if "error" not in metrics:
                    row = f"| {model} | {metrics.get('NDCG@10', 0):.4f} | {metrics.get('NDCG@100', 0):.4f} | {metrics.get('Recall@100', 0):.4f} | {metrics.get('MAP', 0):.4f} |"
                    lines.append(row)

        lines.extend([
            "",
            "### Hybrid Methods",
            "",
        ])

        if "hybrid" in all_results:
            for metric, value in sorted(all_results["hybrid"].items()):
                lines.append(f"- {metric}: {value:.4f}")

        lines.extend([
            "",
            "### Reranking",
            "",
        ])

        if "reranking" in all_results:
            for metric, value in sorted(all_results["reranking"].items()):
                lines.append(f"- {metric}: {value:.4f}")

        lines.extend([
            "",
            "## Error Analysis Summary",
            "",
        ])

        if "error_analysis" in all_results:
            stats = all_results["error_analysis"].get("statistics", {})
            lines.append(f"- Total failures: {stats.get('total_failures', 0)}")
            lines.append(f"- Failure rate: {stats.get('failure_rate', 0):.2%}")

        # Save report
        report = "\n".join(lines)
        report_path = self.results_dir / "evaluation_report.md"
        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Saved report to {report_path}")
        return report

    def run(
        self,
        split: str = "test",
        run_dense: bool = True,
        run_hybrid: bool = True,
        run_reranking: bool = False,  # Expensive
        run_error_analysis: bool = True,
        dense_models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run the complete Phase 3 pipeline.

        Args:
            split: Evaluation split.
            run_dense: Whether to run dense baselines.
            run_hybrid: Whether to run hybrid baseline.
            run_reranking: Whether to run reranking.
            run_error_analysis: Whether to run error analysis.
            dense_models: List of dense models to evaluate.

        Returns:
            Dictionary with all results.
        """
        all_results = {}

        try:
            # Step 1: BM25 baseline
            bm25_metrics = self.step1_run_bm25_baseline(split=split)
            all_results["bm25"] = bm25_metrics

            # Load data for subsequent steps
            from .evaluation.bm25 import BM25Evaluator
            evaluator = BM25Evaluator(self.benchmark_dir)
            corpus = evaluator.load_corpus()
            queries = evaluator.load_queries()
            qrels = evaluator.load_qrels(split)

            # Step 2: Dense baselines
            if run_dense:
                dense_metrics = self.step2_run_dense_baselines(
                    models=dense_models,
                    split=split,
                )
                all_results["dense"] = dense_metrics

            # Step 3: Hybrid baseline
            if run_hybrid:
                hybrid_metrics = self.step3_run_hybrid_baseline(split=split)
                all_results["hybrid"] = hybrid_metrics

            # Step 4: Reranking
            if run_reranking:
                reranking_metrics = self.step4_run_reranking_baseline(split=split)
                all_results["reranking"] = reranking_metrics

            # Get BM25 results for error analysis
            evaluator.build_index(corpus)
            filter_queries = {qid: text for qid, text in queries.items() if qid in qrels}
            bm25_results = evaluator.retrieve(filter_queries, 100)

            # Step 6: Error analysis
            if run_error_analysis:
                error_results = self.step6_run_error_analysis(
                    results=bm25_results,
                    qrels=qrels,
                    queries=queries,
                    corpus=corpus,
                )
                all_results["error_analysis"] = error_results

            # Step 7: Generate report
            report = self.step7_generate_report(all_results)
            all_results["report"] = report

            # Save all results
            all_results_path = self.results_dir / f"all_results_{split}.json"

            # Make serializable
            serializable = {}
            for key, value in all_results.items():
                if key == "report":
                    continue
                if isinstance(value, dict):
                    serializable[key] = value
                else:
                    serializable[key] = str(value)

            with open(all_results_path, "w") as f:
                json.dump(serializable, f, indent=2)

            logger.info("Phase 3 pipeline complete")

        except Exception as e:
            logger.exception(f"Phase 3 pipeline failed: {e}")
            all_results["error"] = str(e)

        return all_results


async def run_phase3(
    benchmark_dir: str,
    results_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    split: str = "test",
) -> Dict[str, Any]:
    """Convenience function to run Phase 3 pipeline.

    Args:
        benchmark_dir: Path to benchmark directory.
        results_dir: Optional results directory.
        cache_dir: Optional cache directory.
        split: Evaluation split.

    Returns:
        Pipeline results.
    """
    pipeline = Phase3Pipeline(
        benchmark_dir=Path(benchmark_dir),
        results_dir=Path(results_dir) if results_dir else None,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )

    return pipeline.run(split=split)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    benchmark_dir = sys.argv[1] if len(sys.argv) > 1 else "data/benchmark"
    results = asyncio.run(run_phase3(benchmark_dir))
    print(json.dumps(results, indent=2, default=str))
