"""CLI entry point for the skin permeation benchmark pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PUBLICATION_DIR = Path(__file__).resolve().parent
SRC_DIR = PUBLICATION_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from skin_benchmark.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the reproducible skin permeation benchmark dataset.")
    parser.add_argument("--config", type=Path, default=PUBLICATION_DIR / "configs" / "merge_config.yaml")
    parser.add_argument("--raw-dir", type=Path, default=PUBLICATION_DIR / "data_raw")
    parser.add_argument("--output-dir", type=Path, default=PUBLICATION_DIR / "outputs")
    parser.add_argument("--report-dir", type=Path, default=PUBLICATION_DIR / "reports")
    parser.add_argument("--source-registry", type=Path, default=PUBLICATION_DIR / "configs" / "source_registry.yaml")
    parser.add_argument("--pubchem-cache", type=Path, default=PUBLICATION_DIR / "configs" / "pubchem_cas_cache.csv")
    parser.add_argument("--sources", nargs="*", default=None, help="Optional subset of sources to run.")
    parser.add_argument("--sheet-override", nargs="*", default=None, help="Overrides like source=SheetName.")
    parser.add_argument("--refresh-pubchem-cache", action="store_true", help="Re-query PubChem even if CAS values exist in cache.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(
        publication_dir=PUBLICATION_DIR,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        source_registry_path=args.source_registry,
        merge_config_path=args.config,
        pubchem_cache_path=args.pubchem_cache,
        selected_sources=args.sources,
        sheet_overrides=args.sheet_override,
        refresh_pubchem_cache=args.refresh_pubchem_cache,
    )


if __name__ == "__main__":
    main()
