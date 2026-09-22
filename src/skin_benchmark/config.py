"""Configuration loaders for the benchmark pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class MergeConfig:
    """Pipeline-level merge and curation settings."""

    identity_key: str = "smiles_std"
    round_target_decimals: int = 3
    canonical_target_unit: str = "cm/s"
    conflict_threshold_method: str = "tukey_iqr_1.5"
    conflict_metric: str = "pooled_range"
    threshold_population: str = "overlap_compounds"
    threshold_min_overlap_compounds: int = 10
    threshold_sensitivity_methods: list[str] = field(
        default_factory=lambda: ["tukey_iqr_1.5", "tukey_iqr_3.0"]
    )
    fixed_conflict_threshold: float | None = None
    drop_severe_conflicts: bool = False
    exclude_invalid_structures: bool = True
    exclude_mixtures_and_salts: bool = True
    exclude_metal_containing_structures: bool = True
    keep_raw_records: bool = True
    pubchem_query_enabled: bool = True


@dataclass(slots=True)
class SourceConfig:
    """Metadata needed to parse a specific raw source."""

    name: str
    file_name: str | list[str]
    parser: str
    file_type: str
    family: str
    preferred_sheet: str | None = None
    native_target_unit: str = "cm/s"
    header_search_rows: int = 6
    header_keywords: list[str] = field(default_factory=list)
    column_aliases: dict[str, list[str]] = field(default_factory=dict)
    reference: str = ""


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config at {path} must contain a mapping at the top level.")
    return data


def load_merge_config(path: Path) -> MergeConfig:
    """Load merge configuration from YAML."""

    raw = _load_yaml(path)
    if "conflict_threshold" in raw and "fixed_conflict_threshold" not in raw:
        raw["fixed_conflict_threshold"] = raw["conflict_threshold"]
        if "conflict_threshold_method" not in raw:
            raw["conflict_threshold_method"] = "fixed"
    raw.pop("conflict_threshold", None)

    valid_fields = {item.name for item in fields(MergeConfig)}
    unknown_fields = sorted(set(raw) - valid_fields)
    if unknown_fields:
        raise ValueError(f"Unknown merge config fields in {path}: {unknown_fields}")
    return MergeConfig(**raw)


def load_source_registry(path: Path) -> dict[str, SourceConfig]:
    """Load source registry entries keyed by source name."""

    raw = _load_yaml(path)
    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, dict):
        raise ValueError(f"Source registry at {path} must define a 'sources' mapping.")
    registry: dict[str, SourceConfig] = {}
    for name, source_data in sources_raw.items():
        if not isinstance(source_data, dict):
            raise ValueError(f"Registry entry for source '{name}' must be a mapping.")
        registry[name] = SourceConfig(name=name, **source_data)
    return registry
