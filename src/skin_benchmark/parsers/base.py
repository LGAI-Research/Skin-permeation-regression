"""Shared parser utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from skin_benchmark.config import SourceConfig
from skin_benchmark.utils.text import clean_text, is_missing, normalize_column_name

COMMON_COLUMNS = [
    "source",
    "source_family",
    "source_record_id",
    "compound_name_raw",
    "cas_number_raw",
    "smiles_raw",
    "smiles_std",
    "target_logkp_raw",
    "target_logkp_native",
    "target_logkp",
    "target_unit_native",
    "target_unit",
    "target_type",
    "notes",
    "is_valid_structure",
    "standardization_status",
    "within_source_duplicate",
    "cross_source_duplicate",
    "conflict_flag",
    "n_sources",
    "source_list",
    "sheet_name",
    "raw_row_index",
    "target_conversion_applied",
    "n_source_families",
    "source_family_list",
    "source_record_ids_json",
    "source_targets_json",
    "all_targets_json",
    "source_row_counts_json",
    "ineligible_reasons_json",
    "conflict_range",
    "conflict_std",
    "conflict_threshold_used",
    "is_benchmark_eligible",
    "exclusion_reason",
    "doi_raw",
    "reference_raw",
    "author_raw",
    "publication_year_raw",
    "publication_id_raw",
    "author_year_key",
    "target_logkp_round3",
    "duplicate_sources",
    "publication_duplicate_group_id",
    "publication_duplicate_keep",
    "publication_duplicate_reason",
    "n_rows_total",
    "n_rows_eligible",
    "n_rows_after_duplicate_removal",
    "n_rows_removed_as_exact_duplicates",
]


class ParserError(RuntimeError):
    """Raised when a source cannot be parsed reproducibly."""


def _collect_alias_terms(source_config: SourceConfig) -> set[str]:
    terms: set[str] = set()
    for aliases in source_config.column_aliases.values():
        for alias in aliases:
            normalized = normalize_column_name(alias)
            if normalized:
                terms.add(normalized)
    for keyword in source_config.header_keywords:
        normalized = normalize_column_name(keyword)
        if normalized:
            terms.add(normalized)
    return terms


def select_excel_sheet(file_path: Path, source_config: SourceConfig, sheet_override: str | None = None) -> str:
    """Choose the most likely sheet for an Excel source."""

    workbook = pd.ExcelFile(file_path)
    if sheet_override is not None:
        if sheet_override not in workbook.sheet_names:
            raise ParserError(
                f"{source_config.name}: requested sheet '{sheet_override}' not found in {file_path.name}. "
                f"Available sheets: {workbook.sheet_names}"
            )
        return sheet_override
    if source_config.preferred_sheet and source_config.preferred_sheet in workbook.sheet_names:
        return source_config.preferred_sheet

    alias_terms = _collect_alias_terms(source_config)
    best_sheet = workbook.sheet_names[0]
    best_score = -1
    for sheet_name in workbook.sheet_names:
        preview = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=None,
            nrows=source_config.header_search_rows,
            dtype=object,
        )
        sheet_score = 0
        for row in preview.itertuples(index=False):
            normalized_row = {normalize_column_name(value) for value in row if not is_missing(value)}
            sheet_score = max(sheet_score, len(normalized_row & alias_terms))
        if sheet_score > best_score:
            best_score = sheet_score
            best_sheet = sheet_name
    return best_sheet


def detect_header_row(preview_df: pd.DataFrame, source_config: SourceConfig) -> int:
    """Detect the header row in a preview dataframe."""

    alias_terms = _collect_alias_terms(source_config)
    best_row = 0
    best_score = -1
    for row_index in range(len(preview_df.index)):
        row = preview_df.iloc[row_index].tolist()
        normalized_row = {normalize_column_name(value) for value in row if not is_missing(value)}
        score = len(normalized_row & alias_terms)
        if score > best_score:
            best_row = row_index
            best_score = score
    if best_score <= 0:
        raise ParserError(
            f"{source_config.name}: failed to detect a header row. "
            f"Observed preview rows: {preview_df.fillna('').astype(str).values.tolist()}"
        )
    return best_row


def load_excel_table(file_path: Path, source_config: SourceConfig, sheet_override: str | None = None) -> tuple[pd.DataFrame, str, int]:
    """Load an Excel table using heuristic sheet and header-row detection."""

    sheet_name = select_excel_sheet(file_path, source_config, sheet_override=sheet_override)
    preview = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=None,
        nrows=source_config.header_search_rows,
        dtype=object,
    )
    header_row = detect_header_row(preview, source_config)
    table = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row, dtype=object)
    table = table.dropna(how="all").reset_index(drop=True)
    table.columns = [clean_text(column) or f"unnamed_{idx}" for idx, column in enumerate(table.columns)]
    return table, sheet_name, header_row


def load_csv_table(file_path: Path) -> tuple[pd.DataFrame, str | None, int]:
    """Load a CSV table."""

    table = pd.read_csv(file_path, dtype=object)
    table = table.dropna(how="all").reset_index(drop=True)
    table.columns = [clean_text(column) or f"unnamed_{idx}" for idx, column in enumerate(table.columns)]
    return table, None, 0


def find_column(columns: list[str], aliases: list[str], source_name: str, label: str, required: bool = True) -> str | None:
    """Resolve a column using normalized aliases."""

    normalized_map = {normalize_column_name(column): column for column in columns}
    for alias in aliases:
        normalized = normalize_column_name(alias)
        if normalized in normalized_map:
            return normalized_map[normalized]
    if required:
        raise ParserError(
            f"{source_name}: required field '{label}' could not be mapped. "
            f"Aliases tried: {aliases}. Available columns: {columns}"
        )
    return None


def build_notes(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Serialize selected metadata columns into a concise notes string."""

    if not columns:
        return pd.Series([None] * len(df), index=df.index, dtype=object)

    def _join(row: pd.Series) -> str | None:
        parts: list[str] = []
        for column in columns:
            value = clean_text(row.get(column))
            if value:
                parts.append(f"{column}={value}")
        if not parts:
            return None
        return "; ".join(parts)

    return df.apply(_join, axis=1)


def finalize_parsed_frame(
    df: pd.DataFrame,
    source_config: SourceConfig,
    sheet_name: str | None,
    header_row: int,
    source_record_column: str | None,
    compound_column: str | None,
    cas_column: str | None,
    smiles_column: str | None,
    target_column: str | None,
    note_columns: list[str] | None = None,
    extra_columns: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Create a common-schema parsed dataframe from a source table."""

    out = pd.DataFrame(index=df.index)
    out["source"] = source_config.name
    out["source_family"] = source_config.family
    raw_row_index = pd.Series(df.index + header_row + 2, index=df.index, dtype="Int64")
    out["raw_row_index"] = raw_row_index
    if source_record_column:
        out["source_record_id"] = df[source_record_column].map(clean_text)
        fallback_ids = raw_row_index.map(lambda value: f"{source_config.name}_row_{value}")
        out["source_record_id"] = out["source_record_id"].fillna(fallback_ids)
    else:
        out["source_record_id"] = raw_row_index.map(lambda value: f"{source_config.name}_row_{value}")
    out["compound_name_raw"] = df[compound_column].map(clean_text) if compound_column else None
    out["cas_number_raw"] = df[cas_column].map(clean_text) if cas_column else None
    out["smiles_raw"] = df[smiles_column].map(clean_text) if smiles_column else None
    out["target_logkp_raw"] = df[target_column].map(clean_text) if target_column else None
    out["target_unit_native"] = source_config.native_target_unit
    out["target_unit"] = None
    out["target_type"] = "logKp"
    out["sheet_name"] = sheet_name
    out["notes"] = build_notes(df, note_columns or [])
    out["smiles_std"] = None
    out["target_logkp_native"] = pd.NA
    out["target_logkp"] = pd.NA
    out["is_valid_structure"] = False
    out["standardization_status"] = None
    out["within_source_duplicate"] = False
    out["cross_source_duplicate"] = False
    out["conflict_flag"] = False
    out["n_sources"] = 0
    out["source_list"] = None
    out["target_conversion_applied"] = None
    out["n_source_families"] = 0
    out["source_family_list"] = None
    out["source_record_ids_json"] = None
    out["source_targets_json"] = None
    out["all_targets_json"] = None
    out["source_row_counts_json"] = None
    out["ineligible_reasons_json"] = None
    out["conflict_range"] = pd.NA
    out["conflict_std"] = pd.NA
    out["conflict_threshold_used"] = pd.NA
    out["is_benchmark_eligible"] = False
    out["exclusion_reason"] = None
    out["doi_raw"] = None
    out["reference_raw"] = None
    out["author_raw"] = None
    out["publication_year_raw"] = None
    out["publication_id_raw"] = None
    out["author_year_key"] = None
    out["target_logkp_round3"] = pd.NA
    out["duplicate_sources"] = None
    out["publication_duplicate_group_id"] = None
    out["publication_duplicate_keep"] = pd.NA
    out["publication_duplicate_reason"] = None
    out["n_rows_total"] = pd.NA
    out["n_rows_eligible"] = pd.NA
    out["n_rows_after_duplicate_removal"] = pd.NA
    out["n_rows_removed_as_exact_duplicates"] = pd.NA

    if extra_columns:
        for column, value in extra_columns.items():
            out[column] = value

    for column in COMMON_COLUMNS:
        if column not in out.columns:
            out[column] = None
    return out
