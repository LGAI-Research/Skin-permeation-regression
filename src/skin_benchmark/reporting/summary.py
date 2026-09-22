"""Summary dataframe builders and markdown report generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from skin_benchmark.config import MergeConfig, SourceConfig


def build_source_summary(
    parsed_by_source: dict[str, pd.DataFrame],
    standardized_by_source: dict[str, pd.DataFrame],
    registry: dict[str, SourceConfig],
) -> pd.DataFrame:
    """Build source-level preprocessing QC statistics."""

    rows: list[dict[str, object]] = []
    for source_name, parsed_df in parsed_by_source.items():
        standardized = standardized_by_source[source_name]
        missing_smiles = standardized["smiles_raw"].isna() | standardized["smiles_raw"].astype(str).str.strip().eq("")
        rows.append(
            {
                "source": source_name,
                "source_family": registry[source_name].family,
                "reference": registry[source_name].reference,
                "raw_rows": len(parsed_df),
                "raw_unique_smiles": int(
                    standardized["smiles_raw"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().nunique()
                ),
                "valid_rows": int(standardized["is_valid_structure"].fillna(False).sum()),
                "valid_unique_smiles": int(standardized.loc[standardized["is_valid_structure"], "smiles_std"].nunique()),
                "missing_smiles": int(missing_smiles.sum()),
                "eligible_rows": int(standardized["is_benchmark_eligible"].fillna(False).sum()),
                "eligible_unique_smiles": int(
                    standardized.loc[standardized["is_benchmark_eligible"].fillna(False), "smiles_std"].nunique()
                ),
                "within_source_dup_rows": int(standardized["within_source_duplicate"].fillna(False).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("source").reset_index(drop=True)


def build_conflict_summary(benchmark_with_provenance: pd.DataFrame) -> pd.DataFrame:
    """Build a conflict-focused benchmark summary table."""

    if benchmark_with_provenance.empty:
        return benchmark_with_provenance.copy()
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
        "n_rows_after_duplicate_removal",
        "n_rows_removed_as_exact_duplicates",
    ]
    return benchmark_with_provenance[columns].sort_values(
        by=["conflict_flag", "conflict_range"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_cheruvu_resolution_summary(standardized_cheruvu: pd.DataFrame, cache_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize CAS resolution outcomes for Cheruvu."""

    status_counts = (
        standardized_cheruvu["pubchem_status"].fillna("missing").value_counts().rename_axis("pubchem_status").reset_index(name="rows")
    )
    cache_counts = (
        cache_df["pubchem_status"].fillna("missing").value_counts().rename_axis("pubchem_status").reset_index(name="unique_cas")
    )
    summary = status_counts.merge(cache_counts, on="pubchem_status", how="outer").fillna(0)
    return summary.sort_values("pubchem_status").reset_index(drop=True)


def build_cheruvu_resolution_details(standardized_cheruvu: pd.DataFrame) -> pd.DataFrame:
    """Export row-level Cheruvu records that did not resolve to a usable PubChem-backed structure."""

    if standardized_cheruvu.empty:
        return pd.DataFrame(
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
    non_resolved = standardized_cheruvu.loc[
        standardized_cheruvu["pubchem_status"].fillna("missing") != "resolved",
        [
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
        ],
    ].copy()
    return non_resolved.sort_values(["pubchem_status", "source_record_id"], kind="stable").reset_index(drop=True)


def _build_cheruvu_status_definition_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pubchem_status": "resolved",
                "meaning": "Exact CAS synonym match found in PubChem and a structure was recovered.",
            },
            {
                "pubchem_status": "unresolved",
                "meaning": "CAS was queryable, but PubChem did not yield a uniquely resolvable exact-CAS structure.",
            },
            {
                "pubchem_status": "invalid_cas",
                "meaning": "CAS normalization or format validation failed before a valid PubChem resolution could happen.",
            },
            {
                "pubchem_status": "missing",
                "meaning": "CAS was absent in the Cheruvu source row, so PubChem resolution was not attempted.",
            },
        ]
    )


def build_pairwise_overlap_preview(pairwise_summary: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    """Build a compact preview of source-overlap agreement statistics."""

    if pairwise_summary.empty:
        return pairwise_summary.copy()
    preview = pairwise_summary[
        ["source_a", "source_b", "n_overlap", "pearson_r", "spearman_r", "mae", "rmse"]
    ].copy()
    return preview.head(limit).reset_index(drop=True)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"

    working = df.fillna("")

    def _safe(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [_safe(column) for column in working.columns]
    rows = [[_safe(value) for value in row] for row in working.itertuples(index=False, name=None)]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def _format_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(widths))) + " |"
    lines = [_format_row(headers), separator]
    lines.extend(_format_row(row) for row in rows)
    return "\n".join(lines)


def _truncate(value: object, limit: int = 56) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _display_path(path: Path, anchor: Path) -> str:
    try:
        return str(path.relative_to(anchor))
    except ValueError:
        return str(path)


def _build_benchmark_flow_mermaid() -> str:
    return """```mermaid
flowchart TD
    A[Raw source files] --> B[Source-specific parsing]
    B --> C[Cheruvu CAS to PubChem SMILES resolution]
    B --> D[Target unit normalization to canonical cm/s]
    C --> E[RDKit standardization and eligibility checks]
    D --> E
    E --> F[Cross-source exact duplicate removal same smiles_std and same logKp 3dp]
    F --> G[Global pooled merge by smiles_std]
    G --> H[Conflict range calculation on pooled raw rows]
    H --> I[Tukey IQR threshold derivation from overlap compounds]
    I --> J[Conflict-included and conflict-filtered benchmark export]
    J --> K[QC summaries report and figures]
```"""


def _build_flow_steps_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step": "1. Parse raw sources",
                "description": "Source-specific parsers detect headers, map columns, and preserve row-level provenance.",
                "major_output": "source-level parsed tables",
            },
            {
                "step": "2. Resolve Cheruvu CAS",
                "description": "Cheruvu rows use CAS -> PubChem resolution before structure eligibility is finalized.",
                "major_output": "resolved or explicitly unresolved Cheruvu rows",
            },
            {
                "step": "3. Standardize structures",
                "description": "RDKit generates canonical isomeric SMILES and computes structure-valid / benchmark-eligible flags.",
                "major_output": "standardized source tables",
            },
            {
                "step": "4. Remove exact cross-source duplicates",
                "description": "Rows with the same smiles_std and logKp rounded to 3 decimals across multiple sources are treated as duplicate literature evidence.",
                "major_output": "publication duplicate annotations and removed-row audit table",
            },
            {
                "step": "5. Global pooled merge",
                "description": "Remaining eligible raw rows are merged by smiles_std and pooled median logKp is computed.",
                "major_output": "global pooled benchmark table",
            },
            {
                "step": "6. Derive conflict cutoff",
                "description": "Overlap compounds define the pooled conflict-range distribution, and Tukey 1.5xIQR sets the main conflict threshold.",
                "major_output": "derived conflict threshold and conflict flags",
            },
            {
                "step": "7. Export report and figures",
                "description": "Final datasets, QC summaries, manuscript-oriented figures, and markdown report are written to disk.",
                "major_output": "outputs/final and reports/benchmark_build_report.md",
            },
        ]
    )


def _build_terminology_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "term": "raw_rows",
                "definition": "Total rows read by the parser before RDKit-based structure checks.",
            },
            {
                "term": "valid_rows",
                "definition": "Rows where RDKit successfully parsed smiles_raw and generated smiles_std.",
            },
            {
                "term": "eligible_rows",
                "definition": "Rows that satisfy benchmark inclusion rules: structure-valid, single-fragment, no metal, target present; Cheruvu also requires successful CAS resolution.",
            },
            {
                "term": "exact cross-source duplicate",
                "definition": "Rows from different sources with the same smiles_std and the same target_logkp rounded to 3 decimals.",
            },
            {
                "term": "pooled benchmark row",
                "definition": "A compound-level representative built from all remaining eligible raw rows after exact duplicate removal.",
            },
            {
                "term": "conflict_range",
                "definition": "Range of pooled raw-row logKp values for the same smiles_std after duplicate removal.",
            },
        ]
    )


def _build_source_preprocessing_table(source_summary: pd.DataFrame) -> pd.DataFrame:
    return source_summary[
        [
            "source",
            "raw_rows",
            "raw_unique_smiles",
            "valid_rows",
            "valid_unique_smiles",
            "eligible_rows",
            "eligible_unique_smiles",
        ]
    ].copy()


def _build_duplicate_preview(removed_rows: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    if removed_rows.empty:
        return removed_rows.copy()
    preview = removed_rows[
        [
            "publication_duplicate_group_id",
            "source",
            "source_record_id",
            "smiles_std",
            "target_logkp_round3",
            "author_year_key",
            "duplicate_sources",
            "publication_duplicate_reason",
        ]
    ].copy()
    preview["smiles_std"] = preview["smiles_std"].map(lambda value: _truncate(value, 40))
    return preview.head(limit).reset_index(drop=True)


def _build_conflict_preview(conflict_summary: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    if conflict_summary.empty:
        return conflict_summary.copy()

    preview_rows: list[dict[str, object]] = []
    for _, row in conflict_summary.head(n).iterrows():
        try:
            source_targets = json.loads(row["source_targets_json"]) if isinstance(row["source_targets_json"], str) else {}
        except Exception:
            source_targets = {}
        compact_values = []
        for source, values in source_targets.items():
            shown = ", ".join(f"{float(value):.3f}" for value in values[:3])
            if len(values) > 3:
                shown += ", ..."
            compact_values.append(f"{source}: [{shown}]")
        preview_rows.append(
            {
                "smiles_std": _truncate(row["smiles_std"], 36),
                "merged_logkp": f"{float(row['target_logkp']):.3f}",
                "n_sources": int(row["n_sources"]),
                "sources": _truncate(str(row["source_list"]).replace("|", ", "), 42),
                "range": f"{float(row['conflict_range']):.3f}",
                "std": f"{float(row['conflict_std']):.3f}",
                "source_values": _truncate("; ".join(compact_values), 120),
            }
        )
    return pd.DataFrame(preview_rows)


def _build_output_file_stage_table(output_files: list[Path], anchor: Path) -> pd.DataFrame:
    """Describe each final CSV by the pipeline stage that produced it."""

    descriptions = {
        "merged_all_records.csv": (
            "Row-level integrated records",
            "All parsed rows after standardization, eligibility annotation, exact-duplicate annotation, and pooled benchmark metadata backfill.",
        ),
        "publication_duplicate_removed_rows.csv": (
            "Cross-source exact duplicate removal",
            "Rows dropped before pooling because another source had the same smiles_std and the same logKp rounded to 3 decimals.",
        ),
        "publication_duplicate_summary.csv": (
            "Cross-source exact duplicate removal",
            "Audit summary of duplicate groups, rows involved, rows removed, and source-pair counts.",
        ),
        "duplicate_audit_same_smiles_logkp.csv": (
            "Notebook-based duplicate audit",
            "Cross-source groups defined by the same smiles_std and the same logKp rounded to 3 decimals.",
        ),
        "duplicate_audit_same_smiles_logkp_author_year.csv": (
            "Notebook-based duplicate audit",
            "Subset of exact-value duplicate groups that also show a shared author_year_key across sources.",
        ),
        "duplicate_audit_same_smiles_logkp_author_year_mismatch_groups.csv": (
            "Notebook-based duplicate audit",
            "Exact-value duplicate groups without a shared author_year_key, useful for manual metadata review.",
        ),
        "duplicate_audit_same_smiles_logkp_author_year_mismatch_rows.csv": (
            "Notebook-based duplicate audit",
            "Row-level provenance for exact-value duplicate groups that do not share an author_year_key.",
        ),
        "benchmark_row_evidence.csv": (
            "Post-duplicate-removal pooled evidence",
            "Eligible raw rows that remain after exact duplicate removal and are actually used to build the pooled benchmark.",
        ),
        "benchmark_with_provenance.csv": (
            "Global pooled benchmark with provenance",
            "Compound-level pooled benchmark table with row counts, source lists, source-specific target lists, and duplicate-removal provenance.",
        ),
        "benchmark_conflict_included.csv": (
            "Final benchmark before conflict filtering",
            "Compound-level pooled benchmark including compounds later flagged by the Tukey conflict rule.",
        ),
        "benchmark_conflict_filtered.csv": (
            "Final benchmark after conflict filtering",
            "Main benchmark after removing compounds whose pooled conflict_range exceeds the derived Tukey cutoff.",
        ),
        "source_summary.csv": (
            "Source preprocessing QC",
            "Per-source raw, valid, and eligible counts plus unique SMILES and within-source duplicate burden.",
        ),
        "pairwise_source_overlap_summary.csv": (
            "Source-overlap agreement diagnostics",
            "Pairwise source-level median logKp agreement statistics computed before exact duplicate removal.",
        ),
        "conflict_summary.csv": (
            "Conflict inspection",
            "Compound-level table sorted by conflict severity, useful for manual review of high-disagreement compounds.",
        ),
        "conflict_threshold_summary.csv": (
            "Conflict threshold derivation",
            "Tukey-statistics table with Q1, Q3, IQR, derived threshold, and flagged fraction over overlap compounds.",
        ),
        "conflict_threshold_sensitivity.csv": (
            "Conflict threshold sensitivity",
            "Alternative threshold-method outputs used to compare how benchmark size changes under different conflict cutoffs.",
        ),
        "high_conflict_examples.csv": (
            "Conflict inspection",
            "Compact preview of the highest-conflict overlap compounds for manuscript figures or supplementary review.",
        ),
        "cheruvu_pubchem_resolution_summary.csv": (
            "Cheruvu CAS resolution QC",
            "Summary of CAS-to-PubChem outcomes for Cheruvu rows: resolved, unresolved, invalid, or missing.",
        ),
        "cheruvu_pubchem_resolution_details.csv": (
            "Cheruvu CAS resolution QC",
            "Row-level Cheruvu records that did not resolve to a usable structure, with CAS status and exclusion metadata.",
        ),
    }

    rows: list[dict[str, str]] = []
    for path in output_files:
        stage, description = descriptions.get(
            path.name,
            ("Other", "Output file generated by the pipeline."),
        )
        rows.append(
            {
                "file": _display_path(path, anchor),
                "stage": stage,
                "what_it_contains": description,
            }
        )
    return pd.DataFrame(rows)


def render_markdown_report(
    report_path: Path,
    merge_config: MergeConfig,
    source_summary: pd.DataFrame,
    publication_duplicate_summary: pd.DataFrame,
    removed_rows: pd.DataFrame,
    conflict_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    high_conflict_examples: pd.DataFrame,
    pairwise_overlap_summary: pd.DataFrame,
    pairwise_overlap_preview: pd.DataFrame,
    resolution_summary: pd.DataFrame,
    resolution_details: pd.DataFrame,
    registry: dict[str, SourceConfig],
    output_files: list[Path],
    figure_paths: list[Path],
) -> None:
    """Write a markdown execution report."""

    pooled_total = 0 if conflict_summary.empty else len(conflict_summary)
    overlap_count = 0 if conflict_summary.empty else int((conflict_summary["n_sources"] > 1).sum())
    conflict_count = 0 if conflict_summary.empty else int(conflict_summary["conflict_flag"].sum())
    duplicate_overall = publication_duplicate_summary.loc[
        publication_duplicate_summary["summary_scope"] == "overall"
    ].iloc[0]
    flow_mermaid = _build_benchmark_flow_mermaid()
    flow_steps_table = _build_flow_steps_table()
    terminology_table = _build_terminology_table()
    source_preprocessing_table = _build_source_preprocessing_table(source_summary)
    duplicate_preview = _build_duplicate_preview(removed_rows)
    conflict_preview = _build_conflict_preview(high_conflict_examples)
    output_stage_table = _build_output_file_stage_table(output_files, report_path.parent.parent)
    cheruvu_status_definitions = _build_cheruvu_status_definition_table()
    resolution_details_path = next(
        (path for path in output_files if path.name == "cheruvu_pubchem_resolution_details.csv"),
        None,
    )

    references_lines = "\n".join(f"- `{name}`: {config.reference}" for name, config in registry.items())
    output_lines = "\n".join(f"- `{_display_path(path, report_path.parent.parent)}`" for path in output_files)
    figure_lines = "\n".join(f"- `{_display_path(path, report_path.parent.parent)}`" for path in figure_paths)

    main_threshold = threshold_summary.loc[threshold_summary["method"] == merge_config.conflict_threshold_method]
    if main_threshold.empty and not threshold_summary.empty:
        main_threshold = threshold_summary.iloc[[0]]
    threshold_text = "threshold unavailable"
    if not main_threshold.empty:
        row = main_threshold.iloc[0]
        threshold_text = (
            f"Q1={row['q1']:.3f}, Q3={row['q3']:.3f}, IQR={row['iqr']:.3f}, "
            f"derived threshold={row['threshold']:.3f}, flagged={int(row['n_flagged'])}/{int(row['n_overlap_compounds'])}"
        )

    report_text = f"""# Benchmark Build Report

## Task overview
- `run_merge.py` performed raw source parsing, structure standardization, exact duplicate removal, global pooled merge, conflict handling, and final export.
- The final identity is `smiles_std` (canonical isomeric SMILES).
- The final target unit is `log10(Kp in {merge_config.canonical_target_unit})`.

## Input raw files
- `data_raw/huskinDB.xlsx`
- `data_raw/SkinPix_20230620_cleanedDB.xlsx`
- `data_raw/Zeng et al.xlsx`
- `data_raw/Cheruvu et al.xlsx`
- `data_raw/deeppk_skin_permeability_train.csv`
- `data_raw/deeppk_skin_permeability_val.csv`
- `data_raw/deeppk_skin_permeability_test.csv`

## Source references
{references_lines}

## Applied config
- `identity_key`: `{merge_config.identity_key}`
- `round_target_decimals`: `{merge_config.round_target_decimals}`
- `canonical_target_unit`: `{merge_config.canonical_target_unit}`
- `conflict_threshold_method`: `{merge_config.conflict_threshold_method}`
- `exclude_mixtures_and_salts`: `{merge_config.exclude_mixtures_and_salts}`
- `exclude_metal_containing_structures`: `{merge_config.exclude_metal_containing_structures}`

## Benchmark Construction Flow
{flow_mermaid}

{_markdown_table(flow_steps_table)}

## Terminology
{_markdown_table(terminology_table)}

## Source preprocessing summary: raw -> valid -> eligible
{_markdown_table(source_preprocessing_table)}

## Cheruvu CAS resolution summary
{_markdown_table(resolution_summary)}

## Cheruvu CAS resolution status definitions
{_markdown_table(cheruvu_status_definitions)}

## Cheruvu non-resolved row details
- Full CSV export: `{_display_path(resolution_details_path, report_path.parent.parent) if resolution_details_path is not None else 'not_generated'}`
- These rows are the Cheruvu entries whose CAS-to-PubChem resolution did not produce a usable structure for benchmark inclusion.

{_markdown_table(resolution_details)}

## Source-overlap agreement diagnostics
- Diagnostic basis: `eligible` rows before exact duplicate removal.
- Scatter points use `source-level median logKp per smiles_std`, not raw row pairs.
- This figure is for source-overlap agreement inspection and is not itself the duplicate-removal rule.
- The scatter figure is organized as a fixed `5x5` source matrix with only the lower triangle populated.

{_markdown_table(pairwise_overlap_preview)}

## Cross-source exact duplicate removal
- Empirical rationale: rows sharing the same `smiles_std` and the same `logKp` rounded to 3 decimals frequently appear to represent the same literature-derived measurement curated into multiple secondary sources.
- Operational rule: same `smiles_std` and same `logKp` rounded to 3 decimals across multiple sources.
- Reporting stance: this step should be described as collapsing likely duplicate literature-derived evidence, not as proving that every removed row came from exactly the same paper.
- `author-year` and related citation fields are retained as provenance/audit metadata, but they are not used as the hard removal rule because citation metadata are incomplete and inconsistently normalized across sources.
- Duplicate groups removed: `{int(duplicate_overall['duplicate_groups'])}`
- Rows involved: `{int(duplicate_overall['rows_involved'])}`
- Rows removed: `{int(duplicate_overall['rows_removed'])}`

{_markdown_table(publication_duplicate_summary)}

### Removed duplicate preview
{_markdown_table(duplicate_preview)}

### Suggested manuscript wording
> We observed that rows sharing the same `smiles_std` and identical `logKp` values to three decimal places often reflected the same literature-derived measurement curated into multiple secondary sources. To avoid counting the same evidence multiple times, we removed such cross-source exact duplicates before global pooling. Publication metadata such as `author-year` were retained for provenance review, but were not used as the hard deduplication key because citation fields were incomplete and inconsistently normalized across sources.

## Duplicate audit notebook
- Notebook path: `notebooks/duplicate_audit.ipynb`
- Purpose: inspect three audit questions directly from `outputs/final/merged_all_records.csv`
  - same `smiles_std` + same `logKp(3dp)` group count
  - same `smiles_std` + same `logKp(3dp)` + shared `author_year_key` group count
  - same `smiles_std` + same `logKp(3dp)` but no shared `author_year_key` row-level provenance
- Interpretation: `author_year_key` is a supporting provenance field for manual review, not the hard deduplication key.

## Global pooled benchmark construction
- Pooled benchmark compounds: `{pooled_total}`
- Overlap compounds (`n_sources > 1`): `{overlap_count}`
- Conflict-flagged compounds: `{conflict_count}`

## Conflict threshold derivation
- Main conflict metric: pooled raw-row `conflict_range`
- Threshold derivation: Tukey `Q3 + 1.5 * IQR`
- {threshold_text}

{_markdown_table(threshold_summary)}

## High-conflict examples
{_markdown_table(conflict_preview)}

## Figure summary
{figure_lines}

`figure_logkp_source_vs_merged.png` shows source and final-benchmark horizontal raincloud distributions with row-local mean markers and mean/std annotations.
`figure_source_overlap_scatter_grid.png` shows a 5x5 lower-triangle source agreement matrix before exact duplicate removal.

## Final output files by stage
{_markdown_table(output_stage_table)}

## Generated output files
{output_lines}

## How to reproduce
```bash
python run_merge.py
```
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
