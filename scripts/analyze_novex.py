#!/usr/bin/env python3
"""NovEx Analysis CLI — generate paper figures and tables.

Usage:
    python scripts/analyze_novex.py --config configs/novex.yaml
    python scripts/analyze_novex.py --analysis vocab_gap
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from biopat.novex._util import setup_logging, load_yaml_config


def main():
    p = argparse.ArgumentParser(description="NovEx paper analysis")
    p.add_argument("--config", default="configs/novex.yaml")
    p.add_argument("--data-dir", default="data/novex")
    p.add_argument("--results-dir", default="data/novex/results")
    p.add_argument("--analysis", choices=["vocab_gap", "per_domain", "cross_domain", "tables", "summary", "all"], default="all")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    setup_logging(args.verbose)

    config = load_yaml_config(args.config)

    from biopat.novex.benchmark import NovExBenchmark
    from biopat.novex.evaluator import TierResult
    from biopat.novex.analysis import NovExAnalyzer

    b = NovExBenchmark(data_dir=args.data_dir)
    b.load()

    results = []
    rp = Path(args.results_dir) / "all_results.json"
    if rp.exists():
        with open(rp) as f:
            results = [TierResult(**r) for r in json.load(f)]

    out_dir = config.get("analysis", {}).get("output_dir", "data/novex/analysis")
    a = NovExAnalyzer(b, results, output_dir=out_dir)

    if args.analysis == "all":
        a.run_all()
    elif args.analysis == "vocab_gap":
        a.vocabulary_gap()
    elif args.analysis == "per_domain":
        a.per_domain()
    elif args.analysis == "cross_domain":
        a.cross_domain()
    elif args.analysis == "tables":
        a.tier1_table(); a.tier2_table(); a.tier3_table()
    elif args.analysis == "summary":
        a.summary()

    print(f"Done. Outputs in {out_dir}/")


if __name__ == "__main__":
    sys.exit(main())
