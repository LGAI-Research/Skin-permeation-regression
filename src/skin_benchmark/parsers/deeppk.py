"""Parser for DeepPK train/val/test CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from skin_benchmark.config import SourceConfig
from skin_benchmark.parsers.base import finalize_parsed_frame, find_column, load_csv_table


def parse_source(file_paths: list[Path], source_config: SourceConfig):
    """Parse DeepPK split CSV files into the common raw schema."""

    frames: list[pd.DataFrame] = []
    for file_path in file_paths:
        table, _, header_row = load_csv_table(file_path)
        table["__file_name"] = file_path.name
        table["__header_row"] = header_row
        frames.append(table)
    combined = pd.concat(frames, ignore_index=True)
    columns = list(combined.columns)
    smiles_column = find_column(columns, source_config.column_aliases["smiles_raw"], source_config.name, "smiles_raw")
    target_column = find_column(columns, source_config.column_aliases["target_logkp_raw"], source_config.name, "target_logkp_raw")
    split_column = find_column(columns, source_config.column_aliases["notes"], source_config.name, "group", required=False)
    out = finalize_parsed_frame(
        combined,
        source_config,
        sheet_name=None,
        header_row=0,
        source_record_column=None,
        compound_column=None,
        cas_column=None,
        smiles_column=smiles_column,
        target_column=target_column,
        note_columns=[split_column] if split_column else [],
    )
    source_prefix = combined["__file_name"].str.replace(".csv", "", regex=False)
    row_suffix = combined.groupby("__file_name").cumcount() + 2
    out["source_record_id"] = source_prefix + "_row_" + row_suffix.astype(str)
    if split_column:
        out["source_split"] = combined[split_column].astype(str)
    else:
        out["source_split"] = combined["__file_name"].str.extract(r"_(train|val|test)\.csv$", expand=False)
    out["raw_file_name"] = combined["__file_name"]
    return out
