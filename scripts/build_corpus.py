#!/usr/bin/env python3
"""Build a Real BioPAT Corpus from External Data Sources.

This script demonstrates how to use the data acquisition module
to fetch real data and build a benchmark corpus.

Requirements:
    pip install httpx

Optional API Keys (for higher rate limits):
    - NCBI_API_KEY: PubMed API key (https://www.ncbi.nlm.nih.gov/account/)
    - PATENTSVIEW_API_KEY: PatentsView API key (https://patentsview.org/)

Usage:
    # Basic usage (uses public APIs)
    python scripts/build_corpus.py --topic "CAR-T cell therapy" --output data/car_t_corpus

    # With API keys
    export NCBI_API_KEY=your_key
    python scripts/build_corpus.py --topic "CRISPR gene editing" --output data/crispr_corpus

    # Custom limits
    python scripts/build_corpus.py \\
        --topic "checkpoint inhibitor cancer" \\
        --pubmed-limit 500 \\
        --patent-limit 200 \\
        --output data/checkpoint_corpus
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biopat.data import DataAcquisition, AcquisitionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def build_corpus(
    topic: str,
    output_dir: Path,
    pubmed_limit: int = 500,
    biorxiv_limit: int = 100,
    patent_limit: int = 200,
    protein_limit: int = 50,
    patent_sequence_limit: int = 50,
    date_range: str = "2020-2024",
):
    """Build a corpus on a specific biomedical topic.

    Args:
        topic: Research topic (e.g., "CAR-T cell therapy")
        output_dir: Directory to save corpus
        pubmed_limit: Max PubMed articles
        biorxiv_limit: Max bioRxiv preprints
        patent_limit: Max patents
        protein_limit: Max proteins
        patent_sequence_limit: Max patent sequences from NCBI
        date_range: Date filter
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get API keys from environment
    config = AcquisitionConfig(
        pubmed_api_key=os.environ.get("NCBI_API_KEY"),  # Also used for NCBI sequences
        patentsview_api_key=os.environ.get("PATENTSVIEW_API_KEY"),
        surechembl_api_key=os.environ.get("SURECHEMBL_API_KEY"),
        cache_dir=output_dir / ".cache",
    )

    logger.info(f"Building corpus for: {topic}")
    logger.info(f"Date range: {date_range}")
    logger.info(f"Output directory: {output_dir}")

    all_documents = []

    async with DataAcquisition(config=config) as acq:
        # 1. Fetch PubMed articles
        logger.info(f"\n[1/4] Fetching PubMed articles (limit={pubmed_limit})...")
        try:
            pubmed_docs = await acq.fetch_pubmed(
                topic,
                limit=pubmed_limit,
                date_range=date_range,
            )
            logger.info(f"  Retrieved {len(pubmed_docs)} articles from PubMed")
            all_documents.extend(pubmed_docs)
        except Exception as e:
            logger.error(f"  PubMed fetch failed: {e}")

        # 2. Fetch bioRxiv preprints
        logger.info(f"\n[2/4] Fetching bioRxiv preprints (limit={biorxiv_limit})...")
        try:
            biorxiv_docs = await acq.fetch_biorxiv(
                topic,
                limit=biorxiv_limit,
                date_range=date_range,
            )
            logger.info(f"  Retrieved {len(biorxiv_docs)} preprints from bioRxiv")
            all_documents.extend(biorxiv_docs)
        except Exception as e:
            logger.error(f"  bioRxiv fetch failed: {e}")

        # 3. Fetch patents
        logger.info(f"\n[3/4] Fetching USPTO patents (limit={patent_limit})...")
        try:
            patent_docs = await acq.fetch_patents(
                query=topic,
                date_range=date_range,
                limit=patent_limit,
            )
            logger.info(f"  Retrieved {len(patent_docs)} patents from USPTO")
            all_documents.extend(patent_docs)
        except Exception as e:
            logger.error(f"  Patent fetch failed: {e}")

        # 4. Fetch protein sequences from UniProt
        logger.info(f"\n[4/5] Fetching UniProt proteins (limit={protein_limit})...")
        try:
            protein_docs = await acq.fetch_proteins(
                topic,
                limit=protein_limit,
                organism="human",
            )
            logger.info(f"  Retrieved {len(protein_docs)} proteins from UniProt")
            all_documents.extend(protein_docs)
        except Exception as e:
            logger.error(f"  UniProt fetch failed: {e}")

        # 5. Fetch patent sequences from NCBI
        logger.info(f"\n[5/5] Fetching NCBI patent sequences (limit={patent_sequence_limit})...")
        try:
            patent_seq_docs = await acq.fetch_patent_sequences(
                topic,
                limit=patent_sequence_limit,
                sequence_type="protein",
            )
            logger.info(f"  Retrieved {len(patent_seq_docs)} patent sequences from NCBI")
            all_documents.extend(patent_seq_docs)
        except Exception as e:
            logger.error(f"  NCBI patent sequences fetch failed: {e}")

        # Save corpus
        logger.info(f"\nSaving corpus ({len(all_documents)} total documents)...")

        # Save as JSONL
        corpus_path = output_dir / "corpus.jsonl"
        with open(corpus_path, "w") as f:
            for doc in all_documents:
                f.write(json.dumps(doc.to_dict()) + "\n")
        logger.info(f"  Saved corpus to {corpus_path}")

        # Save chemicals
        chemicals = acq.extract_chemicals(all_documents)
        if chemicals:
            chem_path = output_dir / "chemicals.jsonl"
            with open(chem_path, "w") as f:
                for chem in chemicals:
                    f.write(json.dumps(chem) + "\n")
            logger.info(f"  Saved {len(chemicals)} chemicals to {chem_path}")

        # Save sequences
        sequences = acq.extract_sequences(all_documents)
        if sequences:
            seq_path = output_dir / "sequences.jsonl"
            with open(seq_path, "w") as f:
                for seq in sequences:
                    f.write(json.dumps(seq) + "\n")
            logger.info(f"  Saved {len(sequences)} sequences to {seq_path}")

        # Build BioPAT corpus dict
        corpus = acq.build_corpus(all_documents)
        corpus_dict_path = output_dir / "corpus_dict.json"
        with open(corpus_dict_path, "w") as f:
            json.dump(corpus, f, indent=2)
        logger.info(f"  Saved corpus dict to {corpus_dict_path}")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("CORPUS SUMMARY")
        logger.info("=" * 60)

        source_counts = {}
        for doc in all_documents:
            source = doc.source
            source_counts[source] = source_counts.get(source, 0) + 1

        for source, count in sorted(source_counts.items()):
            logger.info(f"  {source}: {count} documents")

        logger.info(f"\n  Total documents: {len(all_documents)}")
        logger.info(f"  Documents with chemicals: {len(chemicals)}")
        logger.info(f"  Documents with sequences: {len(sequences)}")
        logger.info(f"\nOutput saved to: {output_dir}")

        return all_documents


def main():
    parser = argparse.ArgumentParser(
        description="Build a BioPAT corpus from external data sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/build_corpus.py --topic "CAR-T cell therapy" --output data/car_t
    python scripts/build_corpus.py --topic "CRISPR" --pubmed-limit 1000 --output data/crispr
    python scripts/build_corpus.py --topic "PD-1 antibody" --date-range "2022-2024" --output data/pd1
        """,
    )

    parser.add_argument(
        "--topic",
        required=True,
        help="Research topic to search for",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for corpus",
    )
    parser.add_argument(
        "--pubmed-limit",
        type=int,
        default=500,
        help="Maximum PubMed articles (default: 500)",
    )
    parser.add_argument(
        "--biorxiv-limit",
        type=int,
        default=100,
        help="Maximum bioRxiv preprints (default: 100)",
    )
    parser.add_argument(
        "--patent-limit",
        type=int,
        default=200,
        help="Maximum patents (default: 200)",
    )
    parser.add_argument(
        "--protein-limit",
        type=int,
        default=50,
        help="Maximum proteins (default: 50)",
    )
    parser.add_argument(
        "--patent-sequence-limit",
        type=int,
        default=50,
        help="Maximum patent sequences from NCBI (default: 50)",
    )
    parser.add_argument(
        "--date-range",
        default="2020-2024",
        help="Date range filter (default: 2020-2024)",
    )

    args = parser.parse_args()

    asyncio.run(build_corpus(
        topic=args.topic,
        output_dir=Path(args.output),
        pubmed_limit=args.pubmed_limit,
        biorxiv_limit=args.biorxiv_limit,
        patent_limit=args.patent_limit,
        protein_limit=args.protein_limit,
        patent_sequence_limit=args.patent_sequence_limit,
        date_range=args.date_range,
    ))


if __name__ == "__main__":
    main()
