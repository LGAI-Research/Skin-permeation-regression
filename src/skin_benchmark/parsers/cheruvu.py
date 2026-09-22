"""Parser for Cheruvu et al."""

from __future__ import annotations

from pathlib import Path

from skin_benchmark.config import SourceConfig
from skin_benchmark.parsers.base import finalize_parsed_frame, find_column, load_excel_table


def parse_source(file_path: Path, source_config: SourceConfig, sheet_override: str | None = None):
    """Parse Cheruvu et al. into the common raw schema."""

    table, sheet_name, header_row = load_excel_table(file_path, source_config, sheet_override=sheet_override)
    columns = list(table.columns)
    source_record_column = find_column(columns, source_config.column_aliases["source_record_id"], source_config.name, "source_record_id")
    compound_column = find_column(columns, source_config.column_aliases["compound_name_raw"], source_config.name, "compound_name_raw")
    cas_column = find_column(columns, source_config.column_aliases["cas_number_raw"], source_config.name, "cas_number_raw", required=False)
    target_column = find_column(columns, source_config.column_aliases["target_logkp_raw"], source_config.name, "target_logkp_raw")
    smiles_column = find_column(columns, source_config.column_aliases.get("smiles_raw", []), source_config.name, "smiles_raw", required=False)
    note_columns = [column for column in columns if column not in {source_record_column, compound_column, cas_column, smiles_column, target_column}]
    return finalize_parsed_frame(
        table,
        source_config,
        sheet_name,
        header_row,
        source_record_column=source_record_column,
        compound_column=compound_column,
        cas_column=cas_column,
        smiles_column=smiles_column,
        target_column=target_column,
        note_columns=note_columns,
    )
