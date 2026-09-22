"""Publication metadata extraction and cross-source exact duplicate handling."""

from __future__ import annotations

from itertools import combinations
import re

import pandas as pd

from skin_benchmark.config import MergeConfig
from skin_benchmark.utils.text import clean_text

_FIELD_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}
_YEAR_PATTERN = re.compile(r"(19|20)\d{2}")
_SURNAME_PARTICLES = {"da", "de", "del", "der", "di", "du", "la", "le", "van", "von"}
_SOURCE_PRIORITY = {
    "skinpix": 5,
    "huskin": 4,
    "cheruvu": 3,
    "zeng": 2,
    "deeppk": 1,
}


def _note_field(note: object, field_name: str) -> str | None:
    text = clean_text(note)
    if text is None:
        return None
    pattern = _FIELD_PATTERN_CACHE.get(field_name)
    if pattern is None:
        pattern = re.compile(rf"(?:^|;\s*){re.escape(field_name)}=([^;]+)", re.IGNORECASE)
        _FIELD_PATTERN_CACHE[field_name] = pattern
    match = pattern.search(text)
    if not match:
        return None
    return clean_text(match.group(1))


def _normalize_author_token(text: object) -> str | None:
    raw = clean_text(text)
    if raw is None:
        return None
    normalized = raw.lower().replace("et al.", "").replace("et al", "")
    normalized = re.sub(r"[()]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None

    tokens = re.split(r",|\band\b", normalized, maxsplit=1)
    lead = tokens[0].strip()
    parts = [part for part in re.split(r"\s+", lead) if part]
    if not parts:
        return None
    if len(parts) >= 2 and parts[0] in _SURNAME_PARTICLES:
        surname = "".join(parts[:2])
    else:
        surname = parts[0]
    surname = re.sub(r"[^a-z0-9]+", "", surname)
    return surname or None


def _extract_publication_year(reference_raw: object, date_raw: object) -> str | None:
    for value in [reference_raw, date_raw]:
        text = clean_text(value)
        if text is None:
            continue
        match = _YEAR_PATTERN.search(text)
        if match:
            return match.group(0)
    return None


def _author_year_key(reference_raw: object, author_raw: object, date_raw: object) -> str | None:
    author_from_reference = _normalize_author_token(reference_raw)
    author_from_author = _normalize_author_token(author_raw)
    year = _extract_publication_year(reference_raw, date_raw)
    author = author_from_reference or author_from_author
    if author or year:
        return f"{author or 'unknown'}_{year or 'unknown'}"
    return None


def enrich_publication_metadata(df: pd.DataFrame, merge_config: MergeConfig) -> pd.DataFrame:
    """Extract publication metadata from notes and annotate source-level duplicates."""

    out = df.copy()
    out["doi_raw"] = out["notes"].map(lambda value: _note_field(value, "doi") or _note_field(value, "DOI"))
    out["reference_raw"] = out["notes"].map(
        lambda value: _note_field(value, "reference") or _note_field(value, "Reference")
    )
    out["author_raw"] = out["notes"].map(lambda value: _note_field(value, "author") or _note_field(value, "Author"))
    out["publication_year_raw"] = [
        _extract_publication_year(reference_raw, _note_field(note, "date") or _note_field(note, "Date"))
        for reference_raw, note in zip(out["reference_raw"], out["notes"], strict=False)
    ]
    out["publication_id_raw"] = out["notes"].map(
        lambda value: _note_field(value, "publication ID") or _note_field(value, "publication id")
    )
    out["author_year_key"] = [
        _author_year_key(reference_raw, author_raw, _note_field(note, "date") or _note_field(note, "Date"))
        for reference_raw, author_raw, note in zip(out["reference_raw"], out["author_raw"], out["notes"], strict=False)
    ]
    out["target_logkp_round3"] = out["target_logkp"].round(merge_config.round_target_decimals)
    out = annotate_within_source_duplicates(out, merge_config)
    return out


def annotate_within_source_duplicates(df: pd.DataFrame, merge_config: MergeConfig) -> pd.DataFrame:
    """Mark row-level within-source duplicates without collapsing them."""

    out = df.copy()
    out["within_source_duplicate"] = False
    eligible_mask = out["is_benchmark_eligible"].fillna(False) & out[merge_config.identity_key].notna()
    eligible = out.loc[eligible_mask, ["source", merge_config.identity_key]].copy()
    if eligible.empty:
        return out

    duplicate_groups = (
        eligible.groupby(["source", merge_config.identity_key]).size().reset_index(name="group_size")
    )
    duplicate_groups = duplicate_groups.loc[duplicate_groups["group_size"] > 1, ["source", merge_config.identity_key]]
    if duplicate_groups.empty:
        return out

    out = out.merge(
        duplicate_groups.assign(_duplicate_marker=True),
        on=["source", merge_config.identity_key],
        how="left",
    )
    out["within_source_duplicate"] = out["_duplicate_marker"].eq(True)
    out = out.drop(columns=["_duplicate_marker"])
    return out


def _metadata_completeness(row: pd.Series) -> int:
    fields = ["author_year_key", "doi_raw", "reference_raw", "author_raw", "publication_year_raw"]
    return sum(clean_text(row.get(field)) is not None for field in fields)


def _source_priority(source: object) -> int:
    return _SOURCE_PRIORITY.get(clean_text(source) or "", 0)


def _select_keep_index(group: pd.DataFrame) -> int:
    ranked = group.assign(
        _metadata_score=group.apply(_metadata_completeness, axis=1),
        _source_priority=group["source"].map(_source_priority),
        _record_id=group["source_record_id"].astype(str),
    ).sort_values(
        by=["_metadata_score", "_source_priority", "_record_id"],
        ascending=[False, False, True],
    )
    return int(ranked.index[0])


def remove_cross_source_exact_duplicates(
    df: pd.DataFrame,
    merge_config: MergeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove cross-source exact duplicates based on smiles and rounded target."""

    annotated = df.copy()
    annotated["publication_duplicate_group_id"] = None
    annotated["duplicate_sources"] = None
    annotated["publication_duplicate_keep"] = pd.NA
    annotated["publication_duplicate_reason"] = None

    eligible_mask = (
        annotated["is_benchmark_eligible"].fillna(False)
        & annotated[merge_config.identity_key].notna()
        & annotated["target_logkp"].notna()
    )
    candidate = annotated.loc[eligible_mask].copy()
    if candidate.empty:
        empty = annotated.iloc[0:0].copy()
        return annotated, annotated.loc[eligible_mask].copy(), empty, pd.DataFrame(
            columns=[
                "summary_scope",
                "source_pair",
                "duplicate_groups",
                "rows_involved",
                "rows_removed",
            ]
        )

    records: list[dict[str, object]] = []
    removed_index: set[int] = set()
    kept_index: set[int] = set()
    pair_counts: dict[str, int] = {}
    group_id = 0

    group_cols = [merge_config.identity_key, "target_logkp_round3"]
    for _, group in candidate.groupby(group_cols, dropna=False):
        unique_sources = sorted(group["source"].astype(str).unique().tolist())
        if len(unique_sources) <= 1:
            continue
        group_id += 1
        duplicate_group_id = f"pubdup_{group_id:05d}"
        keep_index = _select_keep_index(group)
        duplicate_sources = "|".join(unique_sources)

        annotated.loc[group.index, "publication_duplicate_group_id"] = duplicate_group_id
        annotated.loc[group.index, "duplicate_sources"] = duplicate_sources
        annotated.loc[group.index, "publication_duplicate_keep"] = False
        annotated.loc[group.index, "publication_duplicate_reason"] = "removed_cross_source_exact_duplicate"
        annotated.loc[keep_index, "publication_duplicate_keep"] = True
        annotated.loc[keep_index, "publication_duplicate_reason"] = "kept_exact_duplicate_representative"

        kept_index.add(keep_index)
        removed_index.update(idx for idx in group.index if idx != keep_index)

        keep_row = annotated.loc[keep_index]
        records.append(
            {
                "publication_duplicate_group_id": duplicate_group_id,
                "smiles_std": keep_row.get(merge_config.identity_key),
                "target_logkp_round3": keep_row.get("target_logkp_round3"),
                "duplicate_sources": duplicate_sources,
                "n_sources": len(unique_sources),
                "rows_involved": len(group),
                "rows_removed": len(group) - 1,
                "kept_source": keep_row.get("source"),
                "kept_source_record_id": keep_row.get("source_record_id"),
                "kept_author_year_key": keep_row.get("author_year_key"),
                "kept_reference_raw": keep_row.get("reference_raw"),
                "kept_doi_raw": keep_row.get("doi_raw"),
            }
        )
        for source_a, source_b in combinations(unique_sources, 2):
            pair_key = f"{source_a}|{source_b}"
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

    kept_mask = eligible_mask & ~annotated.index.isin(removed_index)
    kept_rows = annotated.loc[kept_mask].copy()
    removed_rows = annotated.loc[sorted(removed_index)].copy() if removed_index else annotated.iloc[0:0].copy()
    group_summary = pd.DataFrame(records)

    overall_rows_involved = 0 if group_summary.empty else int(group_summary["rows_involved"].sum())
    overall_rows_removed = 0 if group_summary.empty else int(group_summary["rows_removed"].sum())
    summary_rows: list[dict[str, object]] = [
        {
            "summary_scope": "overall",
            "source_pair": "ALL",
            "duplicate_groups": len(group_summary),
            "rows_involved": overall_rows_involved,
            "rows_removed": overall_rows_removed,
        }
    ]
    for pair_key, duplicate_groups in sorted(pair_counts.items()):
        summary_rows.append(
            {
                "summary_scope": "source_pair",
                "source_pair": pair_key,
                "duplicate_groups": duplicate_groups,
                "rows_involved": pd.NA,
                "rows_removed": pd.NA,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    return annotated, kept_rows, removed_rows, summary_df
