"""Parser for HuskinDB."""

from __future__ import annotations

from pathlib import Path

from skin_benchmark.config import SourceConfig
from skin_benchmark.parsers.base import finalize_parsed_frame, find_column, load_excel_table


def parse_source(file_path: Path, source_config: SourceConfig, sheet_override: str | None = None):
    """Parse HuskinDB into the common raw schema."""

    table, sheet_name, header_row = load_excel_table(file_path, source_config, sheet_override=sheet_override)
    columns = list(table.columns)
    compound_column = find_column(columns, source_config.column_aliases["compound_name_raw"], source_config.name, "compound_name_raw")
    smiles_column = find_column(columns, source_config.column_aliases["smiles_raw"], source_config.name, "smiles_raw")
    target_column = find_column(columns, source_config.column_aliases["target_logkp_raw"], source_config.name, "target_logkp_raw")
    cas_column = find_column(columns, source_config.column_aliases.get("cas_number_raw", []), source_config.name, "cas_number_raw", required=False)
    note_columns = [column for column in columns if column not in {compound_column, smiles_column, target_column, cas_column}]
    return finalize_parsed_frame(
        table,
        source_config,
        sheet_name,
        header_row,
        source_record_column=None,
        compound_column=compound_column,
        cas_column=cas_column,
        smiles_column=smiles_column,
        target_column=target_column,
        note_columns=note_columns,
    )
