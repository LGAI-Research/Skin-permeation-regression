"""Structure standardization and benchmark eligibility logic."""

from __future__ import annotations

import math

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

from skin_benchmark.config import MergeConfig
from skin_benchmark.utils.text import append_note, clean_text

RDLogger.DisableLog("rdApp.*")
LOG10_3600 = math.log10(3600.0)

# Metals are excluded, while common metalloids and non-metals remain allowed.
ALLOWED_NON_METALS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 52, 53}
METAL_ATOMIC_NUMBERS = set(range(1, 119)) - ALLOWED_NON_METALS


def _safe_float(value: object) -> float | None:
    try:
        if value is None or (isinstance(value, float) and value != value):
            return None
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None


def _convert_target(native_value: float | None, native_unit: str | None) -> tuple[float | None, str]:
    if native_value is None:
        return None, "none"
    if native_unit == "cm/h":
        return native_value - LOG10_3600, "cm/h_to_cm/s"
    return native_value, "none"


def _contains_metal(mol: Chem.Mol) -> bool:
    return any(atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS for atom in mol.GetAtoms())


def standardize_dataframe(df: pd.DataFrame, merge_config: MergeConfig) -> pd.DataFrame:
    """Standardize structures and compute benchmark eligibility."""

    out = df.copy()
    if "pubchem_status" not in out.columns:
        out["pubchem_status"] = pd.NA
    if "pubchem_cid" not in out.columns:
        out["pubchem_cid"] = pd.NA
    if "pubchem_resolver_note" not in out.columns:
        out["pubchem_resolver_note"] = pd.NA
    smiles_std: list[str | None] = []
    target_native: list[float | None] = []
    target_canonical: list[float | None] = []
    conversion_applied: list[str] = []
    valid_flags: list[bool] = []
    status_values: list[str] = []
    eligible_flags: list[bool] = []
    exclusion_reasons: list[str | None] = []
    notes_values: list[str | None] = []

    for _, row in out.iterrows():
        current_notes = row.get("notes")
        smiles_raw = clean_text(row.get("smiles_raw"))
        native_value = _safe_float(row.get("target_logkp_raw"))
        canonical_value, conversion = _convert_target(native_value, clean_text(row.get("target_unit_native")))

        target_native.append(native_value)
        target_canonical.append(canonical_value)
        conversion_applied.append(conversion)
        notes_value = current_notes

        if smiles_raw is None:
            smiles_std.append(None)
            valid_flags.append(False)
            status_values.append("missing_smiles")
            eligible_flags.append(False)
            exclusion_reasons.append("missing_smiles")
            notes_values.append(notes_value)
            continue

        mol = Chem.MolFromSmiles(smiles_raw)
        if mol is None:
            smiles_std.append(None)
            valid_flags.append(False)
            status_values.append("invalid_smiles")
            eligible_flags.append(False)
            exclusion_reasons.append("invalid_smiles")
            notes_values.append(notes_value)
            continue

        standardized_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        smiles_std.append(standardized_smiles)
        valid_flags.append(True)

        fragment_count = len(Chem.GetMolFrags(mol))
        if fragment_count != 1 and merge_config.exclude_mixtures_and_salts:
            status_values.append("mixture_or_salt")
            eligible_flags.append(False)
            exclusion_reasons.append("mixture_or_salt")
            notes_values.append(append_note(notes_value, "Excluded because the structure contains multiple fragments."))
            continue

        if _contains_metal(mol) and merge_config.exclude_metal_containing_structures:
            status_values.append("contains_metal")
            eligible_flags.append(False)
            exclusion_reasons.append("contains_metal")
            notes_values.append(append_note(notes_value, "Excluded because the structure contains at least one metal atom."))
            continue

        if canonical_value is None:
            status_values.append("valid_structure")
            eligible_flags.append(False)
            exclusion_reasons.append("missing_target")
            notes_values.append(notes_value)
            continue

        status_values.append("valid_structure")
        eligible_flags.append(True)
        exclusion_reasons.append(None)
        notes_values.append(notes_value)

    out["smiles_std"] = smiles_std
    out["target_logkp_native"] = target_native
    out["target_logkp"] = target_canonical
    out["target_conversion_applied"] = conversion_applied
    out["target_unit"] = merge_config.canonical_target_unit
    out["is_valid_structure"] = valid_flags
    out["standardization_status"] = status_values
    out["is_benchmark_eligible"] = eligible_flags
    out["exclusion_reason"] = exclusion_reasons
    out["notes"] = notes_values

    unresolved_mask = out["source"].eq("cheruvu") & out["pubchem_status"].isin(["unresolved", "invalid_cas", "query_error"])
    ambiguous_mask = out["source"].eq("cheruvu") & out["pubchem_status"].eq("ambiguous")
    out.loc[unresolved_mask, "is_benchmark_eligible"] = False
    out.loc[unresolved_mask, "exclusion_reason"] = out.loc[unresolved_mask, "exclusion_reason"].fillna("pubchem_unresolved")
    out.loc[ambiguous_mask, "is_benchmark_eligible"] = False
    out.loc[ambiguous_mask, "exclusion_reason"] = out.loc[ambiguous_mask, "exclusion_reason"].fillna("pubchem_ambiguous")
    return out
