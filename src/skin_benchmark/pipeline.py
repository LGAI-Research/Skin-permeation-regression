"""Top-level pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from skin_benchmark.config import MergeConfig, SourceConfig, load_merge_config, load_source_registry
from skin_benchmark.merge.cross_source import (
    annotate_row_level_records,
    build_benchmark_row_evidence,
    build_global_pooled_benchmark,
    build_high_conflict_examples,
    derive_conflict_threshold,
)
from skin_benchmark.merge.publication_duplicates import (
    enrich_publication_metadata,
    remove_cross_source_exact_duplicates,
)
from skin_benchmark.parsers import cheruvu, deeppk, huskin, skinpix, zeng
from skin_benchmark.reporting.figures import generate_figures
from skin_benchmark.reporting.summary import (
    build_cheruvu_resolution_details,
    build_cheruvu_resolution_summary,
    build_conflict_summary,
    build_pairwise_overlap_preview,
    build_source_summary,
    render_markdown_report,
)
from skin_benchmark.resolve.pubchem_cas import resolve_cheruvu_smiles
from skin_benchmark.standardize.structure import standardize_dataframe
from skin_benchmark.utils.io import ensure_directory, round_numeric_columns, save_csv
from skin_benchmark.utils.logging import configure_logger

PARSER_MODULES = {
    "huskin": huskin,
    "skinpix": skinpix,
    "zeng": zeng,
    "cheruvu": cheruvu,
    "deeppk": deeppk,
}


def _load_source_data(
    raw_dir: Path,
    source_config: SourceConfig,
    sheet_override: str | None,
) -> pd.DataFrame:
    if isinstance(source_config.file_name, list):
        file_paths = [raw_dir / name for name in source_config.file_name]
        missing = [path.name for path in file_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{source_config.name}: missing raw files: {missing}")
        return PARSER_MODULES[source_config.parser].parse_source(file_paths, source_config)
    file_path = raw_dir / source_config.file_name
    if not file_path.exists():
        raise FileNotFoundError(f"{source_config.name}: missing raw file '{file_path.name}' in {raw_dir}")
    parser = PARSER_MODULES[source_config.parser]
    return parser.parse_source(file_path, source_config, sheet_override=sheet_override)


def _parse_sheet_overrides(values: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Sheet override '{value}' must use the form source=sheet_name.")
        source, sheet = value.split("=", 1)
        overrides[source.strip()] = sheet.strip()
    return overrides


def _round_for_export(df: pd.DataFrame, merge_config: MergeConfig) -> pd.DataFrame:
    numeric_columns = [
        "target_logkp_native",
        "target_logkp",
        "target_logkp_round3",
        "conflict_range",
        "conflict_std",
        "conflict_threshold_used",
        "threshold",
        "q1",
        "median",
        "q3",
        "iqr",
        "fraction_flagged",
        "pearson_r",
        "spearman_r",
        "mae",
        "rmse",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
    ]
    return round_numeric_columns(df, numeric_columns, merge_config.round_target_decimals)


def _print_summary(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    if df.empty:
        print("(empty)")
    else:
        print(df.to_string(index=False))


def run_pipeline(
    publication_dir: Path,
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
    report_dir: Path | None = None,
    source_registry_path: Path | None = None,
    merge_config_path: Path | None = None,
    pubchem_cache_path: Path | None = None,
    selected_sources: list[str] | None = None,
    sheet_overrides: list[str] | None = None,
    refresh_pubchem_cache: bool = False,
) -> dict[str, Any]:
    """Run the full benchmark construction pipeline."""

    publication_dir = publication_dir.resolve()
    raw_dir = (raw_dir or publication_dir / "data_raw").resolve()
    output_dir = (output_dir or publication_dir / "outputs").resolve()
    report_dir = (report_dir or publication_dir / "reports").resolve()
    source_registry_path = (source_registry_path or publication_dir / "configs" / "source_registry.yaml").resolve()
    merge_config_path = (merge_config_path or publication_dir / "configs" / "merge_config.yaml").resolve()
    pubchem_cache_path = (pubchem_cache_path or publication_dir / "configs" / "pubchem_cas_cache.csv").resolve()

    merge_config: MergeConfig = load_merge_config(merge_config_path)
    registry = load_source_registry(source_registry_path)
    requested_sources = selected_sources or list(registry.keys())
    for source_name in requested_sources:
        if source_name not in registry:
            raise ValueError(f"Unknown source '{source_name}'. Available sources: {sorted(registry)}")

    intermediate_dir = ensure_directory(output_dir / "intermediate")
    final_dir = ensure_directory(output_dir / "final")
    logs_dir = ensure_directory(output_dir / "logs")
    reports_dir = ensure_directory(report_dir)
    logger = configure_logger(logs_dir / "pipeline.log")
    sheet_override_map = _parse_sheet_overrides(sheet_overrides)

    parsed_by_source: dict[str, pd.DataFrame] = {}
    standardized_by_source: dict[str, pd.DataFrame] = {}
    cheruvu_cache = pd.DataFrame()

    for source_name in requested_sources:
        source_config = registry[source_name]
        logger.info("Parsing source %s", source_name)
        parsed = _load_source_data(raw_dir, source_config, sheet_override_map.get(source_name))
        parsed_by_source[source_name] = parsed
        save_csv(parsed, intermediate_dir / f"{source_name}_parsed.csv")

    if "cheruvu" in parsed_by_source:
        logger.info("Resolving Cheruvu CAS numbers through PubChem")
        resolved_cheruvu, cheruvu_cache = resolve_cheruvu_smiles(
            parsed_by_source["cheruvu"],
            pubchem_cache_path,
            logger,
            refresh_cache=refresh_pubchem_cache,
        )
        parsed_by_source["cheruvu"] = resolved_cheruvu

    standardized_frames: list[pd.DataFrame] = []
    for source_name in requested_sources:
        logger.info("Standardizing source %s", source_name)
        standardized = standardize_dataframe(parsed_by_source[source_name], merge_config)
        standardized = enrich_publication_metadata(standardized, merge_config)
        standardized_by_source[source_name] = standardized
        save_csv(_round_for_export(standardized, merge_config), intermediate_dir / f"{source_name}_standardized.csv")
        standardized_frames.append(standardized)

    merged_all_records = pd.concat(standardized_frames, ignore_index=True) if standardized_frames else pd.DataFrame()

    logger.info("Removing exact cross-source duplicates")
    merged_all_records, pooled_row_evidence, removed_rows, publication_duplicate_summary = remove_cross_source_exact_duplicates(
        merged_all_records,
        merge_config,
    )

    logger.info("Building global pooled benchmark")
    benchmark_with_provenance = build_global_pooled_benchmark(merged_all_records, merge_config)
    benchmark_with_provenance, conflict_threshold_summary = derive_conflict_threshold(benchmark_with_provenance, merge_config)
    merged_all_records = annotate_row_level_records(merged_all_records, benchmark_with_provenance)
    benchmark_row_evidence = build_benchmark_row_evidence(merged_all_records, benchmark_with_provenance)
    benchmark_conflict_included = benchmark_with_provenance.copy()
    benchmark_conflict_filtered = benchmark_with_provenance.loc[~benchmark_with_provenance["conflict_flag"].fillna(False)].copy()

    source_summary = build_source_summary(parsed_by_source, standardized_by_source, registry)
    conflict_summary = build_conflict_summary(benchmark_with_provenance)
    high_conflict_examples = build_high_conflict_examples(benchmark_with_provenance)
    resolution_summary = (
        build_cheruvu_resolution_summary(standardized_by_source["cheruvu"], cheruvu_cache)
        if "cheruvu" in standardized_by_source
        else pd.DataFrame(columns=["pubchem_status", "rows", "unique_cas"])
    )
    resolution_details = (
        build_cheruvu_resolution_details(standardized_by_source["cheruvu"])
        if "cheruvu" in standardized_by_source
        else pd.DataFrame(
            columns=[
                "source_record_id",
                "compound_name_raw",
                "cas_number_raw",
                "cas_number_normalized",
                "pubchem_status",
                "pubchem_cid",
                "pubchem_resolver_note",
                "smiles_raw",
                "is_valid_structure",
                "is_benchmark_eligible",
                "exclusion_reason",
            ]
        )
    )

    removed_rows_with_keep = removed_rows.copy()
    if not removed_rows_with_keep.empty:
        keep_rows = merged_all_records.loc[
            merged_all_records["publication_duplicate_reason"].eq("kept_exact_duplicate_representative"),
            [
                "publication_duplicate_group_id",
                "source",
                "source_record_id",
                "author_year_key",
                "doi_raw",
                "reference_raw",
            ],
        ].rename(
            columns={
                "source": "kept_source",
                "source_record_id": "kept_source_record_id",
                "author_year_key": "kept_author_year_key",
                "doi_raw": "kept_doi_raw",
                "reference_raw": "kept_reference_raw",
            }
        )
        removed_rows_with_keep = removed_rows_with_keep.merge(keep_rows, on="publication_duplicate_group_id", how="left")

    output_files = [
        final_dir / "merged_all_records.csv",
        final_dir / "publication_duplicate_removed_rows.csv",
        final_dir / "publication_duplicate_summary.csv",
        final_dir / "benchmark_row_evidence.csv",
        final_dir / "benchmark_with_provenance.csv",
        final_dir / "benchmark_conflict_included.csv",
        final_dir / "benchmark_conflict_filtered.csv",
        final_dir / "source_summary.csv",
        final_dir / "pairwise_source_overlap_summary.csv",
        final_dir / "conflict_summary.csv",
        final_dir / "conflict_threshold_summary.csv",
        final_dir / "conflict_threshold_sensitivity.csv",
        final_dir / "high_conflict_examples.csv",
        final_dir / "cheruvu_pubchem_resolution_summary.csv",
        final_dir / "cheruvu_pubchem_resolution_details.csv",
    ]
    save_csv(_round_for_export(merged_all_records, merge_config), output_files[0])
    save_csv(_round_for_export(removed_rows_with_keep, merge_config), output_files[1])
    save_csv(_round_for_export(publication_duplicate_summary, merge_config), output_files[2])
    save_csv(_round_for_export(benchmark_row_evidence, merge_config), output_files[3])
    save_csv(_round_for_export(benchmark_with_provenance, merge_config), output_files[4])
    save_csv(_round_for_export(benchmark_conflict_included, merge_config), output_files[5])
    save_csv(_round_for_export(benchmark_conflict_filtered, merge_config), output_files[6])
    save_csv(source_summary, output_files[7])

    logger.info("Generating figures")
    figure_paths, pairwise_source_overlap_summary = generate_figures(
        standardized_by_source=standardized_by_source,
        benchmark_conflict_included=benchmark_conflict_included,
        benchmark_conflict_filtered=benchmark_conflict_filtered,
        threshold_summary=conflict_threshold_summary,
        output_dir=final_dir,
        merge_config=merge_config,
    )
    pairwise_overlap_preview = build_pairwise_overlap_preview(pairwise_source_overlap_summary)
    save_csv(_round_for_export(pairwise_source_overlap_summary, merge_config), output_files[8])
    save_csv(_round_for_export(conflict_summary, merge_config), output_files[9])
    save_csv(_round_for_export(conflict_threshold_summary, merge_config), output_files[10])
    save_csv(_round_for_export(conflict_threshold_summary, merge_config), output_files[11])
    save_csv(_round_for_export(high_conflict_examples, merge_config), output_files[12])
    save_csv(resolution_summary, output_files[13])
    save_csv(resolution_details, output_files[14])

    audit_output_files = sorted(final_dir.glob("duplicate_audit_*.csv"))
    report_output_files = output_files + audit_output_files

    report_path = reports_dir / "benchmark_build_report.md"
    render_markdown_report(
        report_path=report_path,
        merge_config=merge_config,
        source_summary=source_summary,
        publication_duplicate_summary=publication_duplicate_summary,
        removed_rows=removed_rows_with_keep,
        conflict_summary=conflict_summary,
        threshold_summary=conflict_threshold_summary,
        high_conflict_examples=high_conflict_examples,
        pairwise_overlap_summary=pairwise_source_overlap_summary,
        pairwise_overlap_preview=pairwise_overlap_preview,
        resolution_summary=resolution_summary,
        resolution_details=resolution_details,
        registry={name: registry[name] for name in requested_sources},
        output_files=report_output_files,
        figure_paths=figure_paths,
    )

    _print_summary("Source Summary", source_summary)
    _print_summary("Publication Duplicate Summary", publication_duplicate_summary)
    _print_summary("Conflict Threshold Summary", _round_for_export(conflict_threshold_summary, merge_config))
    _print_summary("Conflict Summary", _round_for_export(conflict_summary.head(20), merge_config))
    if not resolution_summary.empty:
        _print_summary("Cheruvu Resolution Summary", resolution_summary)

    return {
        "parsed_by_source": parsed_by_source,
        "standardized_by_source": standardized_by_source,
        "merged_all_records": merged_all_records,
        "publication_duplicate_removed_rows": removed_rows_with_keep,
        "publication_duplicate_summary": publication_duplicate_summary,
        "benchmark_row_evidence": benchmark_row_evidence,
        "benchmark_with_provenance": benchmark_with_provenance,
        "benchmark_conflict_included": benchmark_conflict_included,
        "benchmark_conflict_filtered": benchmark_conflict_filtered,
        "source_summary": source_summary,
        "pairwise_source_overlap_summary": pairwise_source_overlap_summary,
        "conflict_summary": conflict_summary,
        "conflict_threshold_summary": conflict_threshold_summary,
        "high_conflict_examples": high_conflict_examples,
        "resolution_summary": resolution_summary,
        "resolution_details": resolution_details,
        "figure_paths": figure_paths,
        "report_path": report_path,
    }
