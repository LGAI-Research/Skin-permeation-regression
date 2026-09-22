"""Within-source deduplication logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skin_benchmark.config import MergeConfig
from skin_benchmark.utils.text import append_note, json_dumps


def deduplicate_source(df: pd.DataFrame, merge_config: MergeConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate raw rows and create a deduplicated source-level table."""

    annotated = df.copy()
    eligible_mask = annotated["is_benchmark_eligible"].fillna(False) & annotated["smiles_std"].notna() & annotated["target_logkp"].notna()
    eligible = annotated.loc[eligible_mask].copy()
    if eligible.empty:
        empty = annotated.iloc[0:0].copy()
        return annotated, empty

    group_sizes = eligible.groupby(merge_config.identity_key).size()
    duplicate_keys = set(group_sizes[group_sizes > 1].index.tolist())
    annotated.loc[eligible_mask & annotated[merge_config.identity_key].isin(duplicate_keys), "within_source_duplicate"] = True

    records: list[dict[str, object]] = []
    for _, group in eligible.groupby(merge_config.identity_key, dropna=False):
        first = group.iloc[0].copy()
        source_targets = [float(value) for value in group["target_logkp"].astype(float).tolist()]
        source_targets_native = [float(value) for value in group["target_logkp_native"].astype(float).tolist()]
        source_record_ids = group["source_record_id"].astype(str).tolist()
        representative = float(np.median(source_targets))
        representative_native = float(np.median(source_targets_native))
        rounded_unique_count = len({round(value, merge_config.round_target_decimals) for value in source_targets})
        dedup_note = "within_source_duplicate_concordant" if rounded_unique_count == 1 else "within_source_duplicate_median"

        row = first.to_dict()
        row["target_logkp"] = representative
        row["target_logkp_native"] = representative_native
        row["within_source_duplicate"] = len(group.index) > 1
        row["source_record_ids_json"] = json_dumps(source_record_ids)
        row["source_targets_json"] = json_dumps(source_targets)
        row["source_list"] = first["source"]
        row["source_family_list"] = first["source_family"]
        row["n_sources"] = 1
        row["n_source_families"] = 1
        if len(group.index) > 1:
            row["notes"] = append_note(row.get("notes"), dedup_note)
        records.append(row)

    dedup_df = pd.DataFrame(records)
    return annotated, dedup_df
