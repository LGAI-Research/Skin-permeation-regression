"""Duplicate-audit helpers for notebook-based literature evidence review."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "source",
    "source_record_id",
    "smiles_std",
    "target_logkp_round3",
    "author_year_key",
    "reference_raw",
    "author_raw",
    "publication_year_raw",
    "doi_raw",
    "is_benchmark_eligible",
]


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Duplicate audit requires the following columns in merged_all_records.csv: "
            + ", ".join(missing)
        )


def _eligible_rows(df: pd.DataFrame) -> pd.DataFrame:
    _validate_required_columns(df)
    eligible = df.loc[df["is_benchmark_eligible"].fillna(False)].copy()
    if "target_logkp_round3" in eligible.columns:
        eligible["target_logkp_round3"] = pd.to_numeric(eligible["target_logkp_round3"], errors="coerce")
    return eligible


def _sorted_pipe_join(values: pd.Series) -> str:
    unique = sorted({str(value) for value in values if pd.notna(value) and str(value).strip() not in {"", "nan"}})
    return "|".join(unique)


def _is_informative_author_year(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return False
    if text.lower().startswith("unknown_"):
        return False
    return True


def build_duplicate_audit_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build notebook-friendly duplicate audit tables from merged_all_records rows."""

    eligible = _eligible_rows(df)

    same_smiles_logkp = (
        eligible.groupby(["smiles_std", "target_logkp_round3"], dropna=False)
        .agg(
            n_rows=("source", "size"),
            n_sources=("source", "nunique"),
            source_list=("source", _sorted_pipe_join),
        )
        .reset_index()
    )
    same_smiles_logkp = same_smiles_logkp.loc[same_smiles_logkp["n_sources"] > 1].reset_index(drop=True)

    shared_rows: list[dict[str, object]] = []
    for (smiles_std, target_logkp_round3), group in eligible.groupby(
        ["smiles_std", "target_logkp_round3"],
        dropna=False,
    ):
        if group["source"].nunique() <= 1:
            continue
        key_to_sources: dict[str, set[str]] = {}
        for _, row in group.iterrows():
            key = row.get("author_year_key")
            if not _is_informative_author_year(key):
                continue
            key_to_sources.setdefault(str(key), set()).add(str(row["source"]))
        shared_keys = sorted(key for key, sources in key_to_sources.items() if len(sources) >= 2)
        if not shared_keys:
            continue
        shared_rows.append(
            {
                "smiles_std": smiles_std,
                "target_logkp_round3": target_logkp_round3,
                "shared_author_year_keys": "|".join(shared_keys),
                "n_shared_author_year_keys": len(shared_keys),
                "source_list": _sorted_pipe_join(group["source"]),
            }
        )

    same_smiles_logkp_author_year = pd.DataFrame(
        shared_rows,
        columns=[
            "smiles_std",
            "target_logkp_round3",
            "shared_author_year_keys",
            "n_shared_author_year_keys",
            "source_list",
        ],
    )

    mismatch_groups = same_smiles_logkp.merge(
        same_smiles_logkp_author_year[["smiles_std", "target_logkp_round3"]],
        on=["smiles_std", "target_logkp_round3"],
        how="left",
        indicator=True,
    )
    mismatch_groups = (
        mismatch_groups.loc[mismatch_groups["_merge"] == "left_only", ["smiles_std", "target_logkp_round3", "n_rows", "n_sources", "source_list"]]
        .reset_index(drop=True)
    )

    mismatch_rows = eligible.merge(
        mismatch_groups[["smiles_std", "target_logkp_round3"]],
        on=["smiles_std", "target_logkp_round3"],
        how="inner",
    )[
        [
            "smiles_std",
            "target_logkp_round3",
            "source",
            "source_record_id",
            "author_year_key",
            "reference_raw",
            "author_raw",
            "publication_year_raw",
            "doi_raw",
        ]
    ].copy()

    mismatch_rows = mismatch_rows.sort_values(
        by=["smiles_std", "target_logkp_round3", "source", "source_record_id"],
        kind="stable",
    ).reset_index(drop=True)

    same_smiles_logkp = same_smiles_logkp.sort_values(
        by=["n_sources", "n_rows", "smiles_std", "target_logkp_round3"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    same_smiles_logkp_author_year = same_smiles_logkp_author_year.sort_values(
        by=["n_shared_author_year_keys", "smiles_std", "target_logkp_round3"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    mismatch_groups = mismatch_groups.sort_values(
        by=["n_sources", "n_rows", "smiles_std", "target_logkp_round3"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    return {
        "same_smiles_logkp": same_smiles_logkp,
        "same_smiles_logkp_author_year": same_smiles_logkp_author_year,
        "same_smiles_logkp_author_year_mismatch_groups": mismatch_groups,
        "same_smiles_logkp_author_year_mismatch_rows": mismatch_rows,
    }


def duplicate_audit_output_paths(output_dir: Path) -> dict[str, Path]:
    """Return canonical output paths for duplicate audit CSV exports."""

    return {
        "same_smiles_logkp": output_dir / "duplicate_audit_same_smiles_logkp.csv",
        "same_smiles_logkp_author_year": output_dir / "duplicate_audit_same_smiles_logkp_author_year.csv",
        "same_smiles_logkp_author_year_mismatch_groups": output_dir / "duplicate_audit_same_smiles_logkp_author_year_mismatch_groups.csv",
        "same_smiles_logkp_author_year_mismatch_rows": output_dir / "duplicate_audit_same_smiles_logkp_author_year_mismatch_rows.csv",
    }
