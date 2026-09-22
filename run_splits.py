"""CLI entry point for benchmark split generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PUBLICATION_DIR = Path(__file__).resolve().parent
SRC_DIR = PUBLICATION_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from skin_benchmark.splits import generate_split_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate benchmark 10-fold cross-validation split artifacts (random_10fold, target_stratified_10fold)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PUBLICATION_DIR / "outputs" / "final" / "benchmark_conflict_filtered.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PUBLICATION_DIR / "outputs" / "final" / "splits",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        choices=[10],
        default=10,
        help="Fold count for the 10-fold cross-validation methods.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    generate_split_outputs(
        input_path=args.input.resolve(),
        output_dir=args.output_dir.resolve(),
        n_splits=args.n_splits,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
