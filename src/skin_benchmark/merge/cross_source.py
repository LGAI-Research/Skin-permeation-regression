"""Global pooled merge and conflict handling."""

from __future__ import annotations

import json
from math import inf

import numpy as np
import pandas as pd

from skin_benchmark.config import MergeConfig
from skin_benchmark.utils.text import json_dumps


def _compute_threshold(values: pd.Series, method: str, fixed_threshold: float | None) -> float:
    if method == "fixed":
        if fixed_threshold is None:
            raise ValueError("fixed conflict threshold requested but fixed_conflict_threshold is not configured.")
        return float(fixed_threshold)
    if values.empty:
        return inf
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    if method == "tukey_iqr_1.5":
        return q3 + 1.5 * iqr
    if method == "tukey_iqr_3.0":
        return q3 + 3.0 * iqr
    raise ValueError(f"Unsupported conflict threshold method: {method}")


def _threshold_details(values: pd.Series, method: str, fixed_threshold: float | None) -> dict[str, object]:
    values = values.astype(float)
    q1 = float(values.quantile(0.25)) if not values.empty else np.nan
    median = float(values.quantile(0.5)) if not values.empty else np.nan
    q3 = float(values.quantile(0.75)) if not values.empty else np.nan
    iqr = q3 - q1 if not values.empty else np.nan
    threshold = _compute_threshold(values, method, fixed_threshold)
    return {
        "method": method,
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "threshold": threshold,
    }


def build_global_pooled_benchmark(all_records: pd.DataFrame, merge_config: MergeConfig) -> pd.DataFrame:
    """Merge remaining eligible raw rows into a global pooled benchmark."""

    if all_records.empty:
        return all_records.copy()

    records: list[dict[str, object]] = []
    for smiles_std, group in all_records.groupby(merge_config.identity_key, dropna=False):
        eligible_group = group.loc[group["is_benchmark_eligible"].fillna(False)].copy()
        if eligible_group.empty:
            continue
        included_group = eligible_group.loc[eligible_group["publication_duplicate_keep"].ne(False)].copy()
        if included_group.empty:
            continue

        included_targets = included_group["target_logkp"].astype(float).tolist()
        source_list = sorted(included_group["source"].astype(str).unique().tolist())
        family_list = sorted(included_group["source_family"].astype(str).unique().tolist())
        source_target_map = {
            source: [float(value) for value in values.tolist()]
            for source, values in included_group.groupby("source")["target_logkp"]
        }
        source_record_map = {
            source: values.astype(str).tolist()
            for source, values in included_group.groupby("source")["source_record_id"]
        }
        source_row_count_map = {
            source: int(count) for source, count in included_group.groupby("source").size().to_dict().items()
        }
        ineligible_rows = group.loc[~group["is_benchmark_eligible"].fillna(False), ["source", "source_record_id", "exclusion_reason"]]
        ineligible_reasons = [
            {
                "source": str(row["source"]),
                "source_record_id": str(row["source_record_id"]),
                "reason": None if pd.isna(row["exclusion_reason"]) else str(row["exclusion_reason"]),
            }
            for _, row in ineligible_rows.iterrows()
        ]
        removed_rows = eligible_group.loc[eligible_group["publication_duplicate_keep"].eq(False)]

        representative_row = included_group.sort_values(["source", "source_record_id"]).iloc[0]
        records.append(
            {
                "source": "merged",
                "source_family": "merged",
                "source_record_id": f"merged_{len(records) + 1}",
                "compound_name_raw": representative_row.get("compound_name_raw"),
                "cas_number_raw": representative_row.get("cas_number_raw"),
                "smiles_raw": representative_row.get("smiles_raw"),
                "smiles_std": smiles_std,
                "target_logkp_raw": None,
                "target_logkp_native": pd.NA,
                "target_logkp": float(np.median(included_targets)),
                "target_unit_native": None,
                "target_unit": representative_row.get("target_unit"),
                "target_type": representative_row.get("target_type"),
                "notes": representative_row.get("notes"),
                "is_valid_structure": True,
                "standardization_status": "merged",
                "within_source_duplicate": bool(group["within_source_duplicate"].fillna(False).any()),
                "cross_source_duplicate": len(source_list) > 1,
                "conflict_flag": False,
                "n_sources": len(source_list),
                "source_list": "|".join(source_list),
                "sheet_name": None,
                "raw_row_index": pd.NA,
                "target_conversion_applied": "pooled_median_all_rows",
                "n_source_families": len(family_list),
                "source_family_list": "|".join(family_list),
                "source_record_ids_json": json_dumps(source_record_map),
                "source_targets_json": json_dumps(source_target_map),
                "all_targets_json": json_dumps([float(value) for value in included_targets]),
                "source_row_counts_json": json_dumps(source_row_count_map),
                "ineligible_reasons_json": json_dumps(ineligible_reasons),
                "conflict_range": float(max(included_targets) - min(included_targets)) if len(included_targets) > 1 else 0.0,
                "conflict_std": float(np.std(included_targets, ddof=0)) if len(included_targets) > 1 else 0.0,
                "conflict_threshold_used": pd.NA,
                "is_benchmark_eligible": True,
                "exclusion_reason": None,
                "doi_raw": representative_row.get("doi_raw"),
                "reference_raw": representative_row.get("reference_raw"),
                "author_raw": representative_row.get("author_raw"),
                "publication_year_raw": representative_row.get("publication_year_raw"),
                "publication_id_raw": representative_row.get("publication_id_raw"),
                "author_year_key": representative_row.get("author_year_key"),
                "target_logkp_round3": round(float(np.median(included_targets)), merge_config.round_target_decimals),
                "duplicate_sources": representative_row.get("duplicate_sources"),
                "publication_duplicate_group_id": None,
                "publication_duplicate_keep": True,
                "publication_duplicate_reason": None,
                "n_rows_total": len(group),
                "n_rows_eligible": len(eligible_group),
                "n_rows_after_duplicate_removal": len(included_group),
                "n_rows_removed_as_exact_duplicates": len(removed_rows),
            }
        )

    benchmark = pd.DataFrame(records)
    if benchmark.empty:
        return benchmark
    return benchmark.sort_values(["smiles_std", "source_record_id"]).reset_index(drop=True)


def derive_conflict_threshold(
    benchmark_with_provenance: pd.DataFrame,
    merge_config: MergeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive conflict thresholds from overlap compounds and annotate the benchmark."""

    benchmark = benchmark_with_provenance.copy()
    overlap = benchmark.loc[benchmark["n_sources"].fillna(0).astype(int) > 1].copy()
    values = overlap["conflict_range"].astype(float) if not overlap.empty else pd.Series(dtype=float)

    if len(values) < merge_config.threshold_min_overlap_compounds and merge_config.conflict_threshold_method != "fixed":
        methods = [merge_config.conflict_threshold_method, *merge_config.threshold_sensitivity_methods]
        methods = list(dict.fromkeys(methods))
        summary_rows = []
        for method in methods:
            summary_rows.append(
                {
                    "method": method,
                    "n_overlap_compounds": len(values),
                    "q1": np.nan,
                    "median": np.nan,
                    "q3": np.nan,
                    "iqr": np.nan,
                    "threshold": np.inf,
                    "n_flagged": 0,
                    "fraction_flagged": 0.0,
                }
            )
        benchmark["conflict_flag"] = False
        benchmark["conflict_threshold_used"] = np.nan
        return benchmark, pd.DataFrame(summary_rows)

    methods = [merge_config.conflict_threshold_method, *merge_config.threshold_sensitivity_methods]
    methods = list(dict.fromkeys(methods))
    summary_rows = []
    for method in methods:
        details = _threshold_details(values, method, merge_config.fixed_conflict_threshold)
        threshold = float(details["threshold"])
        n_flagged = int((values > threshold).sum()) if np.isfinite(threshold) else 0
        summary_rows.append(
            {
                "method": method,
                "n_overlap_compounds": len(values),
                "q1": details["q1"],
                "median": details["median"],
                "q3": details["q3"],
                "iqr": details["iqr"],
                "threshold": threshold,
                "n_flagged": n_flagged,
                "fraction_flagged": 0.0 if len(values) == 0 else n_flagged / len(values),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    main_threshold = float(
        summary_df.loc[summary_df["method"] == merge_config.conflict_threshold_method, "threshold"].iloc[0]
    )
    benchmark["conflict_flag"] = (
        benchmark["n_sources"].fillna(0).astype(int).gt(1)
        & benchmark["conflict_range"].astype(float).gt(main_threshold)
    )
    benchmark["conflict_threshold_used"] = main_threshold
    return benchmark, summary_df


def build_high_conflict_examples(benchmark_with_provenance: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    """Return the highest-conflict pooled benchmark compounds."""

    if benchmark_with_provenance.empty:
        return benchmark_with_provenance.copy()
    working = benchmark_with_provenance.loc[
        benchmark_with_provenance["n_sources"].fillna(0).astype(int) > 1
    ].copy()
    if working.empty:
        return working
    columns = [
        "smiles_std",
        "target_logkp",
        "n_sources",
        "source_list",
        "conflict_flag",
        "conflict_range",
        "conflict_std",
        "source_targets_json",
        "source_record_ids_json",
        "n_rows_total",
        "n_rows_after_duplicate_removal",
        "n_rows_removed_as_exact_duplicates",
    ]
    return working[columns].sort_values(
        ["conflict_flag", "conflict_range", "n_sources"],
        ascending=[False, False, False],
    ).head(limit).reset_index(drop=True)


def build_benchmark_row_evidence(all_records: pd.DataFrame, benchmark_with_provenance: pd.DataFrame) -> pd.DataFrame:
    """Return row-level evidence that contributed to the pooled benchmark."""

    if all_records.empty:
        return all_records.copy()
    included_mask = (
        all_records["is_benchmark_eligible"].fillna(False)
        & all_records["publication_duplicate_keep"].ne(False)
        & all_records["smiles_std"].notna()
        & all_records["target_logkp"].notna()
    )
    evidence = all_records.loc[included_mask].copy()
    return evidence


def annotate_row_level_records(all_records: pd.DataFrame, benchmark_with_provenance: pd.DataFrame) -> pd.DataFrame:
    """Propagate benchmark annotations back to the full row-level record table."""

    annotated = all_records.copy()
    if benchmark_with_provenance.empty:
        return annotated

    subset = benchmark_with_provenance[
        [
            "smiles_std",
            "cross_source_duplicate",
            "conflict_flag",
            "n_sources",
            "source_list",
            "n_source_families",
            "source_family_list",
            "conflict_range",
            "conflict_std",
            "conflict_threshold_used",
            "n_rows_total",
            "n_rows_eligible",
            "n_rows_after_duplicate_removal",
            "n_rows_removed_as_exact_duplicates",
        ]
    ].drop_duplicates(subset=["smiles_std"])
    annotated = annotated.merge(subset, on="smiles_std", how="left", suffixes=("", "_merged"))
    for column in [
        "cross_source_duplicate",
        "conflict_flag",
        "n_sources",
        "source_list",
        "n_source_families",
        "source_family_list",
        "conflict_range",
        "conflict_std",
        "conflict_threshold_used",
        "n_rows_total",
        "n_rows_eligible",
        "n_rows_after_duplicate_removal",
        "n_rows_removed_as_exact_duplicates",
    ]:
        merged_column = f"{column}_merged"
        if merged_column in annotated.columns:
            annotated[column] = annotated[merged_column].combine_first(annotated[column])
            annotated = annotated.drop(columns=[merged_column])
    annotated["cross_source_duplicate"] = annotated["cross_source_duplicate"].eq(True)
    annotated["conflict_flag"] = annotated["conflict_flag"].eq(True)
    annotated["n_sources"] = annotated["n_sources"].fillna(0).astype(int)
    annotated["n_source_families"] = annotated["n_source_families"].fillna(0).astype(int)
    return annotated
