"""Benchmark split generation utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from skin_benchmark.utils.io import ensure_directory, save_csv

RANDOM_10FOLD_SPLIT_METHOD = "random_10fold"
TARGET_STRATIFIED_10FOLD_SPLIT_METHOD = "target_stratified_10fold"
BASE_SUPPORTED_SPLIT_METHODS = (
    RANDOM_10FOLD_SPLIT_METHOD,
    TARGET_STRATIFIED_10FOLD_SPLIT_METHOD,
)
SUPPORTED_SPLIT_METHODS = BASE_SUPPORTED_SPLIT_METHODS
REQUIRED_COLUMNS = ("smiles_std", "target_logkp")
BASE_SPLIT_METHOD_TO_N_SPLITS = {
    RANDOM_10FOLD_SPLIT_METHOD: 10,
    TARGET_STRATIFIED_10FOLD_SPLIT_METHOD: 10,
}


def is_supported_split_method(split_method: str) -> bool:
    """Return whether the split id is supported by the benchmark repository."""

    return split_method in BASE_SUPPORTED_SPLIT_METHODS


def prepare_benchmark_for_splitting(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and sort the benchmark dataframe used for fold generation."""

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Benchmark split input is missing required columns: " + ", ".join(missing_columns)
        )

    out = df.copy()
    smiles_text = out["smiles_std"].astype(str).str.strip()
    missing_smiles = out["smiles_std"].isna() | smiles_text.eq("") | smiles_text.eq("nan")
    if missing_smiles.any():
        raise ValueError("Benchmark split input contains missing smiles_std values.")
    if out["target_logkp"].isna().any():
        raise ValueError("Benchmark split input contains missing target_logkp values.")
    if out["smiles_std"].duplicated().any():
        duplicate_count = int(out["smiles_std"].duplicated().sum())
        raise ValueError(
            f"Benchmark split input contains duplicate smiles_std values ({duplicate_count} duplicate rows)."
        )

    return out.sort_values("smiles_std", kind="stable").reset_index(drop=True)


def load_benchmark_for_splitting(path: Path) -> pd.DataFrame:
    """Load and validate the benchmark input CSV."""

    return prepare_benchmark_for_splitting(pd.read_csv(path))


def _empty_target_bins(length: int) -> pd.Series:
    return pd.Series(pd.array([pd.NA] * length, dtype="Int64"))


def _build_target_bins(df: pd.DataFrame, n_splits: int) -> pd.Series:
    bins = pd.qcut(df["target_logkp"], q=n_splits, labels=False, duplicates="drop")
    bin_series = pd.Series(bins, index=df.index, dtype="Int64")
    n_bins = int(bin_series.nunique(dropna=True))
    if n_bins != n_splits:
        raise ValueError(
            f"Expected {n_splits} target quantile bins, but qcut produced {n_bins}. "
            "The target distribution is too degenerate for the requested target-stratified split."
        )
    return bin_series


def _expected_n_splits(split_method: str) -> int:
    if split_method in BASE_SPLIT_METHOD_TO_N_SPLITS:
        return BASE_SPLIT_METHOD_TO_N_SPLITS[split_method]
    raise ValueError(f"Unsupported split method: {split_method}")


def _assign_test_folds(
    df: pd.DataFrame,
    split_method: str,
    n_splits: int,
    random_seed: int,
    target_bins: pd.Series | None = None,
) -> pd.Series:
    test_folds = pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"), index=df.index)
    if split_method == RANDOM_10FOLD_SPLIT_METHOD:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        split_iter = splitter.split(df)
    elif split_method == TARGET_STRATIFIED_10FOLD_SPLIT_METHOD:
        if target_bins is None:
            raise ValueError("target_bins are required for target-stratified split generation.")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        split_iter = splitter.split(df, target_bins)
    else:
        raise ValueError(f"Unsupported split method: {split_method}")

    for fold_index, (_, test_index) in enumerate(split_iter, start=1):
        test_folds.iloc[test_index] = fold_index

    if test_folds.isna().any():
        raise RuntimeError(f"Split assignment for '{split_method}' did not populate every row.")
    return test_folds


def build_assignment_table(
    df: pd.DataFrame,
    split_method: str,
    n_splits: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Build a one-row-per-compound assignment table for a split method."""

    working = prepare_benchmark_for_splitting(df)
    if not is_supported_split_method(split_method):
        raise ValueError(f"Unsupported split method: {split_method}")
    expected_n_splits = _expected_n_splits(split_method)
    if n_splits is None:
        n_splits = expected_n_splits
    if n_splits != expected_n_splits:
        raise ValueError(
            f"Split method '{split_method}' requires n_splits={expected_n_splits}, received {n_splits}."
        )
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1.")
    if len(working) < n_splits:
        raise ValueError("n_splits cannot exceed the number of benchmark rows.")

    target_bins = _empty_target_bins(len(working))
    if split_method == TARGET_STRATIFIED_10FOLD_SPLIT_METHOD:
        target_bins = _build_target_bins(working, n_splits)

    assignment = working.copy()
    assignment["split_method"] = split_method
    assignment["test_fold"] = _assign_test_folds(
        assignment,
        split_method=split_method,
        n_splits=n_splits,
        random_seed=random_seed,
        target_bins=target_bins if split_method == TARGET_STRATIFIED_10FOLD_SPLIT_METHOD else None,
    )
    assignment["random_seed"] = random_seed
    assignment["target_bin"] = target_bins
    return assignment


def assignment_output_path(output_dir: Path, split_method: str) -> Path:
    """Return the canonical assignment output path for a split method."""

    return output_dir / split_method / f"{split_method}_fold_assignments.csv"


def _fold_output_path(output_dir: Path, split_method: str, fold_index: int, split_role: str) -> Path:
    return output_dir / split_method / f"fold_{fold_index}_{split_role}.csv"


def _build_fold_frame(
    assignment: pd.DataFrame,
    base_columns: list[str],
    fold_index: int,
    split_role: str,
) -> pd.DataFrame:
    test_folds = pd.to_numeric(assignment["test_fold"], errors="coerce").astype("Int64")
    if split_role == "train":
        membership_mask = test_folds.isna() | test_folds.ne(fold_index)
        subset = assignment.loc[membership_mask].copy()
    elif split_role == "test":
        membership_mask = test_folds.eq(fold_index).fillna(False)
        subset = assignment.loc[membership_mask].copy()
    else:
        raise ValueError(f"Unsupported split role: {split_role}")

    out = subset[base_columns + ["split_method", "random_seed", "target_bin"]].copy()
    out["fold_index"] = fold_index
    out["split_role"] = split_role
    ordered_columns = base_columns + [
        "split_method",
        "fold_index",
        "split_role",
        "random_seed",
        "target_bin",
    ]
    return out[ordered_columns]


def _persist_assignment_outputs(
    *,
    root_output_dir: Path,
    split_method: str,
    assignment: pd.DataFrame,
    base_columns: list[str],
) -> dict[str, object]:
    """Write one assignment CSV plus its per-fold train/test materializations.

    Input:
    - root_output_dir: shared root split artifact directory
    - split_method: split identifier used for directory and file naming
    - assignment: one-row-per-compound assignment dataframe
    - base_columns: original benchmark columns preserved in fold CSV files

    Output:
    - dictionary describing the persisted assignment path, output directory, and fold paths
    """

    method_dir = ensure_directory(root_output_dir / split_method)
    assignment_path = assignment_output_path(root_output_dir, split_method)
    save_csv(assignment, assignment_path)

    fold_paths: list[Path] = []
    fold_indices = sorted(int(value) for value in assignment["test_fold"].dropna().astype(int).unique().tolist())
    for fold_index in fold_indices:
        for split_role in ("train", "test"):
            fold_path = _fold_output_path(root_output_dir, split_method, fold_index, split_role)
            fold_frame = _build_fold_frame(
                assignment,
                base_columns=base_columns,
                fold_index=fold_index,
                split_role=split_role,
            )
            save_csv(fold_frame, fold_path)
            fold_paths.append(fold_path)

    return {
        "assignment": assignment,
        "assignment_path": assignment_path,
        "output_dir": method_dir,
        "fold_paths": fold_paths,
    }


def generate_split_outputs(
    input_path: Path,
    output_dir: Path,
    n_splits: int = 10,
    random_seed: int = 42,
) -> dict[str, dict[str, object]]:
    """Generate split assignment and per-fold train/test CSV files.

    Input:
    - input_path: benchmark CSV used to derive the split assignments
    - output_dir: root directory where split artifacts are written
    - n_splits: fold count to materialize for the 10-fold cross-validation methods
    - random_seed: seed forwarded to the randomized split builders

    Output:
    - mapping from split method name to generated artifact paths and assignment frame
    """

    benchmark = load_benchmark_for_splitting(input_path)
    base_columns = benchmark.columns.tolist()
    root_output_dir = ensure_directory(output_dir)
    results: dict[str, dict[str, object]] = {}

    for split_method in BASE_SUPPORTED_SPLIT_METHODS:
        method_n_splits = _expected_n_splits(split_method)
        if method_n_splits > n_splits:
            continue
        assignment = build_assignment_table(
            benchmark,
            split_method=split_method,
            n_splits=method_n_splits,
            random_seed=random_seed,
        )
        results[split_method] = _persist_assignment_outputs(
            root_output_dir=root_output_dir,
            split_method=split_method,
            assignment=assignment,
            base_columns=base_columns,
        )

    return results
