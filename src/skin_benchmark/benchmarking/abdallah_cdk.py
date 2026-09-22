"""Curated CDK 2.8 descriptor contract for the Abdallah benchmark family.

The Abdallah descriptor table ships frozen under `outputs/features/`, so this
package loads it as provided. The CDK 2.8 extraction code that produced it —
the Maven bootstrap, the JVM bridge, and the descriptor computation — is not
part of this package.

What remains here is the tracked descriptor contract: the curated 141-descriptor
subset that passed the reference audit. `load_feature_store` validates the frozen
table against it on every run, so this module is on the reproduction path.
"""

from __future__ import annotations

from pathlib import Path

import yaml


ABDALLAH_DESCRIPTOR_CONFIG_PATH = Path("configs") / "abdallah_descriptor_subset.yaml"


def load_abdallah_descriptor_config(publication_dir: Path) -> dict[str, object]:
    """Load and validate the tracked curated-descriptor contract for Abdallah."""

    config_path = (publication_dir / ABDALLAH_DESCRIPTOR_CONFIG_PATH).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Abdallah descriptor config must be a mapping: {config_path}")

    selected = config.get("selected_descriptors", [])
    excluded = config.get("excluded_descriptors", [])
    if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
        raise ValueError(f"selected_descriptors must be a list of strings: {config_path}")
    if not isinstance(excluded, list) or not all(isinstance(item, str) for item in excluded):
        raise ValueError(f"excluded_descriptors must be a list of strings: {config_path}")
    if len(set(selected)) != len(selected):
        raise ValueError(f"selected_descriptors contains duplicates: {config_path}")
    if len(set(excluded)) != len(excluded):
        raise ValueError(f"excluded_descriptors contains duplicates: {config_path}")
    overlap = sorted(set(selected) & set(excluded))
    if overlap:
        raise ValueError(f"Selected/excluded Abdallah descriptors overlap: {overlap}")

    reference_count = int(config.get("reference_descriptor_count", len(selected) + len(excluded)))
    if len(selected) + len(excluded) != reference_count:
        raise ValueError(
            "Abdallah descriptor config counts are inconsistent: "
            f"selected={len(selected)}, excluded={len(excluded)}, reference={reference_count}"
        )

    return {
        "config_path": str(config_path),
        "reference_source": str(config.get("reference_source", "abdallah_ref/clean_trial4.csv")),
        "reference_descriptor_count": reference_count,
        "selection_provenance": str(
            config.get("selection_provenance", "audit against abdallah_ref/clean_trial4.csv")
        ),
        "pass_criteria": str(
            config.get("pass_criteria", "pearson_r_ge_0.85_or_constant_exact_match")
        ),
        "selected_descriptors": selected,
        "excluded_descriptors": excluded,
    }
