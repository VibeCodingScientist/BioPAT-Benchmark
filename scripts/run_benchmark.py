#!/usr/bin/env python3
"""BioPAT Benchmark Runner - VPS-friendly script for dataset generation.

This script provides a simple interface to run the BioPAT benchmark pipeline
with progress reporting and error handling suitable for VPS environments.

Usage:
    python scripts/run_benchmark.py --phase phase1
    python scripts/run_benchmark.py --phase phase1 --skip-baseline
    python scripts/run_benchmark.py --phase phase1 --config configs/custom.yaml

Environment variables:
    PATENTSVIEW_API_KEYS  - Comma-separated API keys for rotation
    OPENALEX_MAILTO       - Email for OpenAlex polite pool
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def setup_logging(log_dir: Path, phase: str) -> logging.Logger:
    """Set up logging with both file and console handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{phase}_{timestamp}.log"

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")

    return logger


def check_environment() -> dict:
    """Check environment setup and API keys."""
    status = {
        "patentsview_keys": 0,
        "openalex_email": False,
        "epo_credentials": False,
    }

    # Check PatentsView keys
    keys_str = os.environ.get("PATENTSVIEW_API_KEYS", "")
    if keys_str:
        status["patentsview_keys"] = len([k for k in keys_str.split(",") if k.strip()])
    elif os.environ.get("PATENTSVIEW_API_KEY"):
        status["patentsview_keys"] = 1

    # Check OpenAlex
    status["openalex_email"] = bool(os.environ.get("OPENALEX_MAILTO"))

    # Check EPO
    status["epo_credentials"] = bool(
        os.environ.get("EPO_CONSUMER_KEY") and
        os.environ.get("EPO_CONSUMER_SECRET")
    )

    return status


def print_banner():
    """Print startup banner."""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ██████╗ ██╗ ██████╗ ██████╗  █████╗ ████████╗               ║
    ║   ██╔══██╗██║██╔═══██╗██╔══██╗██╔══██╗╚══██╔══╝               ║
    ║   ██████╔╝██║██║   ██║██████╔╝███████║   ██║                  ║
    ║   ██╔══██╗██║██║   ██║██╔═══╝ ██╔══██║   ██║                  ║
    ║   ██████╔╝██║╚██████╔╝██║     ██║  ██║   ██║                  ║
    ║   ╚═════╝ ╚═╝ ╚═════╝ ╚═╝     ╚═╝  ╚═╝   ╚═╝                  ║
    ║                                                               ║
    ║   Biomedical Patent-to-Article Retrieval Benchmark            ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


def run_phase1(config_path: str, skip_download: bool, skip_baseline: bool) -> dict:
    """Run Phase 1 pipeline."""
    from biopat.pipeline import Phase1Pipeline
    from biopat.config import BioPatConfig

    config = BioPatConfig.load(config_path)
    pipeline = Phase1Pipeline(config)

    return asyncio.run(
        pipeline.run(
            skip_download=skip_download,
            skip_baseline=skip_baseline,
        )
    )


def run_phase2(config_path: str, skip_baseline: bool) -> dict:
    """Run Phase 2 pipeline."""
    from biopat.pipeline_phase2 import Phase2Pipeline
    from biopat.config import BioPatConfig

    config = BioPatConfig.load(config_path)
    pipeline = Phase2Pipeline(config)

    return asyncio.run(pipeline.run(skip_baseline=skip_baseline))


def run_phase3(config_path: str) -> dict:
    """Run Phase 3 pipeline."""
    from biopat.pipeline_phase3 import Phase3Pipeline
    from biopat.config import BioPatConfig

    config = BioPatConfig.load(config_path)
    pipeline = Phase3Pipeline(config)

    return asyncio.run(pipeline.run())


def save_results(results: dict, output_dir: Path, phase: str):
    """Save pipeline results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{phase}_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="BioPAT Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Phase 1 with default config
  python scripts/run_benchmark.py --phase phase1

  # Run without baseline evaluation (faster)
  python scripts/run_benchmark.py --phase phase1 --skip-baseline

  # Use custom config
  python scripts/run_benchmark.py --phase phase1 --config configs/production.yaml

Environment Variables:
  PATENTSVIEW_API_KEYS  - Comma-separated API keys for key rotation
  OPENALEX_MAILTO       - Email for OpenAlex polite pool (faster rate limits)
  EPO_CONSUMER_KEY      - EPO OPS consumer key (for Phase 6)
  EPO_CONSUMER_SECRET   - EPO OPS consumer secret (for Phase 6)
        """
    )

    parser.add_argument(
        "--phase",
        choices=["phase1", "phase2", "phase3"],
        default="phase1",
        help="Pipeline phase to run (default: phase1)"
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration file (default: configs/default.yaml)"
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data download if already present"
    )

    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip BM25 baseline evaluation"
    )

    parser.add_argument(
        "--output-dir",
        default="logs",
        help="Directory for logs and results (default: logs)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress banner and verbose output"
    )

    args = parser.parse_args()

    # Print banner
    if not args.quiet:
        print_banner()

    # Setup paths
    project_dir = Path(__file__).parent.parent
    output_dir = project_dir / args.output_dir
    config_path = project_dir / args.config if not Path(args.config).is_absolute() else Path(args.config)

    # Setup logging
    logger = setup_logging(output_dir, args.phase)

    # Check environment
    env_status = check_environment()
    logger.info("Environment check:")
    logger.info(f"  PatentsView API keys: {env_status['patentsview_keys']}")
    logger.info(f"  OpenAlex email configured: {env_status['openalex_email']}")
    logger.info(f"  EPO credentials configured: {env_status['epo_credentials']}")

    if env_status["patentsview_keys"] == 0:
        logger.warning(
            "No PatentsView API keys found. Rate limits will be restricted. "
            "Get a key at: https://patentsview.org/apis/keyrequest"
        )

    if not env_status["openalex_email"]:
        logger.warning(
            "No OpenAlex email configured. Consider setting OPENALEX_MAILTO "
            "for faster rate limits."
        )

    # Check config exists
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        logger.info("Creating default configuration...")
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Create minimal config
        default_config = """# BioPAT Configuration
phase: "phase1"

paths:
  data_dir: "data"
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  cache_dir: "data/cache"
  benchmark_dir: "data/benchmark"

phase1:
  target_patent_count: 2000
  target_query_count: 5000
  min_citations: 3
  ros_confidence_threshold: 8
  ipc_prefixes:
    - "A61"
    - "C07"
    - "C12"
  seed: 42
"""
        with open(config_path, "w") as f:
            f.write(default_config)
        logger.info(f"Created: {config_path}")

    # Run pipeline
    logger.info(f"Starting {args.phase} pipeline...")
    logger.info(f"Config: {config_path}")

    start_time = time.time()

    try:
        if args.phase == "phase1":
            results = run_phase1(
                str(config_path),
                skip_download=args.skip_download,
                skip_baseline=args.skip_baseline,
            )
        elif args.phase == "phase2":
            results = run_phase2(str(config_path), skip_baseline=args.skip_baseline)
        elif args.phase == "phase3":
            results = run_phase3(str(config_path))
        else:
            logger.error(f"Unknown phase: {args.phase}")
            sys.exit(1)

        elapsed = time.time() - start_time

        # Add timing to results
        results["elapsed_seconds"] = elapsed
        results["elapsed_formatted"] = f"{elapsed/60:.1f} minutes"

        # Save results
        save_results(results, output_dir, args.phase)

        # Print summary
        print("\n" + "=" * 60)
        print(f"{args.phase.upper()} COMPLETE")
        print("=" * 60)
        print(f"Elapsed time: {results['elapsed_formatted']}")
        print()

        for key, value in results.items():
            if key not in ["elapsed_seconds", "elapsed_formatted"]:
                print(f"  {key}: {value}")

        print("\n" + "=" * 60)
        print("Output location: data/benchmark/")
        print("=" * 60)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
