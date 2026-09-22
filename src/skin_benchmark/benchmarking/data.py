"""Benchmark data loading and fold materialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from skin_benchmark.splits import (
    RANDOM_10FOLD_SPLIT_METHOD,
    TARGET_STRATIFIED_10FOLD_SPLIT_METHOD,
    is_supported_split_method,
    load_benchmark_for_splitting,
)

ASSIGNMENT_COLUMNS = ["smiles_std", "split_method", "test_fold", "random_seed", "target_bin", "target_logkp"]


@dataclass
class FoldDataset:
    """Materialized train/test rows for one outer fold."""

    split_method: str
    fold_index: int
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def default_benchmark_path(publication_dir: Path) -> Path:
    return publication_dir / "outputs" / "final" / "benchmark_conflict_filtered.csv"


def default_splits_dir(publication_dir: Path) -> Path:
    return publication_dir / "outputs" / "final" / "splits"


def default_runs_dir(publication_dir: Path) -> Path:
    return publication_dir / "runs"


def assignment_path(splits_dir: Path, split_method: str) -> Path:
    if not is_supported_split_method(split_method):
        raise ValueError(f"Unsupported split method: {split_method}")
    return splits_dir / split_method / f"{split_method}_fold_assignments.csv"


def load_assignment_table(path: Path, split_method: str) -> pd.DataFrame:
    """Load one shared split assignment table from disk.

    Input:
    - path: canonical CSV path for the requested split assignment table
    - split_method: split identifier expected inside the CSV contents

    Output:
    - assignment dataframe with one row per `smiles_std`, validated and stably sorted
    """

    if not path.exists():
        raise FileNotFoundError(
            "Required split assignment file is missing: "
            f"{path}. Generate shared split artifacts first with "
            "`conda run -n hnh_publish python run_splits.py`, or pass the correct "
            "`--splits-dir` if the files were written elsewhere."
        )
    assignment = pd.read_csv(path)
    missing_columns = [column for column in ASSIGNMENT_COLUMNS if column not in assignment.columns]
    if missing_columns:
        raise ValueError(
            f"Split assignment at {path} is missing required columns: {', '.join(missing_columns)}"
        )
    if set(assignment["split_method"].astype(str)) != {split_method}:
        raise ValueError(f"Split assignment at {path} contains inconsistent split_method values.")
    if assignment["smiles_std"].duplicated().any():
        raise ValueError(f"Split assignment at {path} contains duplicate smiles_std rows.")
    return assignment.sort_values("smiles_std", kind="stable").reset_index(drop=True)


def load_benchmark_with_assignments(
    benchmark_path: Path,
    splits_dir: Path,
    split_method: str,
) -> pd.DataFrame:
    benchmark = load_benchmark_for_splitting(benchmark_path)
    assignment = load_assignment_table(assignment_path(splits_dir, split_method), split_method)
    benchmark_smiles = set(benchmark["smiles_std"])
    assignment_smiles = set(assignment["smiles_std"])
    if benchmark_smiles != assignment_smiles:
        raise ValueError(
            f"Benchmark rows and split assignment rows do not match for split '{split_method}'."
        )

    merged = benchmark.merge(
        assignment[ASSIGNMENT_COLUMNS].rename(columns={"target_logkp": "assignment_target_logkp"}),
        on="smiles_std",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(benchmark):
        raise ValueError(f"Failed to align benchmark rows and split assignments for split '{split_method}'.")
    mismatched_targets = (merged["target_logkp"] - merged["assignment_target_logkp"]).abs() > 1e-9
    if mismatched_targets.any():
        raise ValueError(
            f"Benchmark target_logkp values do not match split assignment values for split '{split_method}'."
        )
    return (
        merged.drop(columns=["assignment_target_logkp"])
        .sort_values("smiles_std", kind="stable")
        .reset_index(drop=True)
    )


def available_folds(assignment_df: pd.DataFrame) -> list[int]:
    return sorted(int(value) for value in assignment_df["test_fold"].dropna().astype(int).unique().tolist())


def materialize_fold_dataset(
    assignment_df: pd.DataFrame,
    split_method: str,
    fold_index: int,
) -> FoldDataset:
    folds = available_folds(assignment_df)
    if fold_index not in folds:
        raise ValueError(f"Fold {fold_index} is not available for split '{split_method}'. Available folds: {folds}")
    test_folds = pd.to_numeric(assignment_df["test_fold"], errors="coerce").astype("Int64")
    # Holdout-style split assignments keep train rows as `NA`, so train membership is
    # defined as either "assigned to a different fold" or "not assigned to any test fold".
    train_mask = (test_folds.isna() | test_folds.ne(fold_index)).to_numpy(dtype=bool)
    test_mask = test_folds.eq(fold_index).fillna(False).to_numpy(dtype=bool)
    train_df = assignment_df.loc[train_mask].copy().reset_index(drop=True)
    test_df = assignment_df.loc[test_mask].copy().reset_index(drop=True)
    return FoldDataset(split_method=split_method, fold_index=fold_index, train_df=train_df, test_df=test_df)


def leakage_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, object]:
    train_smiles = set(train_df["smiles_std"].astype(str))
    test_smiles = set(test_df["smiles_std"].astype(str))
    overlap = sorted(train_smiles & test_smiles)
    return {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_unique_train_smiles": len(train_smiles),
        "n_unique_test_smiles": len(test_smiles),
        "n_overlap_smiles": len(overlap),
        "overlap_example": overlap[0] if overlap else None,
        "passed": len(overlap) == 0,
    }


def default_split_methods() -> list[str]:
    return [RANDOM_10FOLD_SPLIT_METHOD, TARGET_STRATIFIED_10FOLD_SPLIT_METHOD]
