"""Single CLI entry point for benchmark training and evaluation.

This thin wrapper is the only command-line entry point for benchmark experiments.
Its job is only to:
1. define the stable CLI contract for reproducible runs
2. resolve project-local default paths
3. forward parsed arguments into `skin_benchmark.benchmarking.pipeline.run_benchmark`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PUBLICATION_DIR = Path(__file__).resolve().parent
SRC_DIR = PUBLICATION_DIR / "src"

# Allow the repository-local package to be imported when the CLI is executed directly from
# the project root without requiring a separate editable install step.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from skin_benchmark.benchmarking.pipeline import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface used to launch benchmark runs.

    The defaults intentionally point at the shared curated benchmark and shared split directory
    so a plain `python run_benchmark.py` executes the repository-standard comparison setup.
    The standard way to run only a subset of models is `--models ...`; no secondary wrapper
    script is needed for baseline-plus-external-model comparisons.
    """

    parser = argparse.ArgumentParser(
        description="Run any selected fixed benchmark models on the curated skin permeability benchmark."
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=PUBLICATION_DIR / "outputs" / "final" / "benchmark_conflict_filtered.csv",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=PUBLICATION_DIR / "outputs" / "final" / "splits",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=PUBLICATION_DIR / "outputs" / "features",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=PUBLICATION_DIR / "runs",
    )
    parser.add_argument(
        "--split-methods",
        nargs="*",
        default=["random_10fold", "target_stratified_10fold"],
        help=(
            "Split methods to evaluate. Defaults to the shared 10-fold cross-validation splits "
            "(`random_10fold`, `target_stratified_10fold`)."
        ),
    )
    parser.add_argument("--folds", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument(
        "--models",
        nargs="*",
        default=["all"],
        # The model list mixes fixed registry models with the optional `external_model` adapter.
        # `all` expands only to the fixed model ids defined in the registry.
        # This is the standard public interface for selecting exactly which models to run.
        help=(
            "Model ids to execute. Use `all` for every fixed model, or pass an explicit subset. "
            "Valid ids: all, zeng_svr_proxy, abdallah_lgbm, "
            "waters_fragment_linear, fpadmet_rf, gate_lgbm, external_model"
        ),
    )
    parser.add_argument(
        "--external-model",
        type=str,
        default=None,
        help="Python adapter factory ref for a user-supplied model: package.module:factory",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paper-sanity", choices=["auto", "off"], default="auto")
    return parser


def main() -> None:
    """Parse CLI arguments and hand off execution to the benchmark pipeline."""

    parser = build_parser()
    args = parser.parse_args()

    # Path resolution happens here so the runner receives absolute paths regardless of the
    # current shell working directory used to launch the command.
    run_benchmark(
        publication_dir=PUBLICATION_DIR,
        benchmark_path=args.benchmark_path.resolve(),
        splits_dir=args.splits_dir.resolve(),
        features_dir=args.features_dir.resolve(),
        runs_dir=args.runs_dir.resolve(),
        run_id=args.run_id,
        split_methods=args.split_methods,
        folds=args.folds,
        models=args.models,
        external_model_factory_ref=args.external_model,
        seed=args.seed,
        paper_sanity=args.paper_sanity,
    )


if __name__ == "__main__":
    # Keeping the side effect inside `main()` makes the module import-safe for tests and reuse.
    main()
