"""Feature builders and train-only preprocessing utilities.

This module does two jobs for the benchmark runner:
1. build raw feature tables for each literature-style baseline family
2. apply train-only preprocessing so no test-fold information leaks into fit-time transforms

The feature families intentionally mix several open-source backends.
For `fpadmet_rf`, the PubChem-style fingerprint table is a deterministic local
representation table rather than a REST lookup cache. This package ships that
table frozen under `outputs/features/` and loads it as provided; the generator
that produced it is not part of this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, GraphDescriptors
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import yaml

from skin_benchmark.benchmarking.abdallah_cdk import load_abdallah_descriptor_config
from skin_benchmark.splits import load_benchmark_for_splitting
from skin_benchmark.utils.io import ensure_directory, save_csv

ZENG_FEATURE_FAMILY = "zeng_proxy_descriptors"
ABDALLAH_FEATURE_FAMILY = "abdallah_descriptor_table"
WATERS_FEATURE_FAMILY = "waters_functional_groups"
FPADMET_FEATURE_FAMILY = "fpadmet_pubchem"
GATE_FEATURE_FAMILY = "gate_predicted_properties"
# GATE feature 는 배포되지 않는 foundation-model weight 의 출력이라 재생성이 불가능하고,
# frozen table 로만 제공된다. audit 산출물 두 곳(registry 의 감사 표, feature metadata)이
# 같은 값을 쓰도록 여기서 한 번만 정의한다.
GATE_FIDELITY_TAG = "frozen_pretrained_features"

# `skfp.PubChemFingerprint` returns the PubChem CACTVS-style fingerprint as an 881-bit vector.
# We validate the width explicitly so downstream models never see a silently malformed feature table.
PUBCHEM_FINGERPRINT_LENGTH = 881
PUBCHEM_FINGERPRINT_BIT_COLUMNS = [f"bit_{index}" for index in range(PUBCHEM_FINGERPRINT_LENGTH)]
FEATURE_STORE_FILENAMES = {
    ZENG_FEATURE_FAMILY: "zeng_proxy_descriptors.csv",
    ABDALLAH_FEATURE_FAMILY: "abdallah_descriptor_table.csv",
    WATERS_FEATURE_FAMILY: "waters_functional_groups.csv",
    FPADMET_FEATURE_FAMILY: "fpadmet_pubchem.csv",
    GATE_FEATURE_FAMILY: "gate_predicted_properties.csv",
}

FRAGMENT_SMARTS = {
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "bromine": "[Br]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "chlorine": "[Cl]",
    "ether": "[OD2]([#6])[#6]",
    "ester": "[CX3](=O)[OX2][#6]",
    "hydroxyl": "[OX2H]",
    "ketone": "[#6][CX3](=O)[#6]",
}
FRAGMENT_PATTERNS = {name: Chem.MolFromSmarts(pattern) for name, pattern in FRAGMENT_SMARTS.items()}
WATERS_FEATURE_COLUMNS = [
    "amide",
    "amine",
    "aromatic",
    "bromine",
    "carboxylic_acid",
    "chlorine",
    "ether",
    "ester",
    "hydroxyl",
    "ketone",
]
ZENG_FEATURE_COLUMNS = [
    "cos2_mollogp_proxy",
    "neoplastic_80_rule_proxy",
    "x3v_chi3v_proxy",
]
ZENG_BENZENE_RING_PATTERN = Chem.MolFromSmarts("c1ccccc1")
ZENG_ALIPHATIC_AMINE_PATTERN = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O);!$([N]a)]")
ZENG_CARBOXAMIDE_PATTERN = Chem.MolFromSmarts("[NX3][CX3](=[OX1])[#6]")
ZENG_ALCOHOLIC_HYDROXYL_PATTERN = Chem.MolFromSmarts("[OX2H][CX4]")
ZENG_ESTER_PATTERN = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
ZENG_KETONE_PATTERN = Chem.MolFromSmarts("[#6][CX3](=O)[#6]")
GATE_EXCLUDED_COLUMNS = [
    "f_unnormalized",
    "s_unnormalized",
    "aa_unnormalized",
    "psa_unnormalized",
    "rb_unnormalized",
    "hbd_unnormalized",
    "hba_unnormalized",
    "mw_unnormalized",
]


@dataclass
class FeatureBuildResult:
    """Raw feature table and provenance metadata."""

    feature_family: str
    frame: pd.DataFrame
    metadata: dict[str, object]
    store_frame: pd.DataFrame | None = None


class FeaturePreprocessor:
    """Train-only feature transformation protocol."""

    def fit(self, train_frame: pd.DataFrame) -> "FeaturePreprocessor":
        raise NotImplementedError

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def summary(self) -> dict[str, object]:
        raise NotImplementedError


class IdentityPreprocessor(FeaturePreprocessor):
    """Return raw features without train-dependent filtering."""

    def __init__(self) -> None:
        self.columns: list[str] = []

    def fit(self, train_frame: pd.DataFrame) -> "IdentityPreprocessor":
        self.columns = train_frame.columns.tolist()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[:, self.columns].copy()

    def summary(self) -> dict[str, object]:
        return {"kept_columns": self.columns, "n_output_features": len(self.columns)}


class ScaledNumericPreprocessor(FeaturePreprocessor):
    """Scale a dense numeric feature table without data leakage."""

    def __init__(self) -> None:
        self.columns: list[str] = []
        self.scaler = StandardScaler()

    def fit(self, train_frame: pd.DataFrame) -> "ScaledNumericPreprocessor":
        self.columns = train_frame.columns.tolist()
        self.scaler.fit(train_frame[self.columns].astype(float))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = self.scaler.transform(frame.loc[:, self.columns].astype(float))
        return pd.DataFrame(transformed, index=frame.index, columns=self.columns)

    def summary(self) -> dict[str, object]:
        return {"kept_columns": self.columns, "n_output_features": len(self.columns)}


class DescriptorTablePreprocessor(FeaturePreprocessor):
    """Train-only imputation, filtering, correlation pruning, and scaling."""

    def __init__(self, correlation_threshold: float = 0.95) -> None:
        self.correlation_threshold = correlation_threshold
        self.input_columns: list[str] = []
        self.columns_after_missing: list[str] = []
        self.columns_after_variance: list[str] = []
        self.final_columns: list[str] = []
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.dropped_missing: list[str] = []
        self.dropped_zero_variance: list[str] = []
        self.dropped_correlation: list[str] = []

    def fit(self, train_frame: pd.DataFrame) -> "DescriptorTablePreprocessor":
        # Every decision below is fit on the training fold only. The retained columns,
        # imputation statistics, and scaler parameters are then reused unchanged on test rows.
        working = train_frame.astype(float).replace([np.inf, -np.inf], np.nan)
        self.input_columns = working.columns.tolist()

        # Drop descriptor columns that are entirely missing in the training fold.
        # Keeping them would create unstable schema drift with no usable information.
        keep_missing = working.columns[working.notna().any(axis=0)].tolist()
        self.dropped_missing = [column for column in self.input_columns if column not in keep_missing]
        working = working.loc[:, keep_missing]
        self.columns_after_missing = working.columns.tolist()

        # Median imputation is also fit on training data only.
        imputed = pd.DataFrame(
            self.imputer.fit_transform(working),
            index=working.index,
            columns=self.columns_after_missing,
        )

        # Remove zero-variance columns after imputation because they cannot help any downstream model.
        variance = imputed.var(axis=0, ddof=0)
        keep_variance = variance[variance > 0.0].index.tolist()
        self.dropped_zero_variance = [column for column in self.columns_after_missing if column not in keep_variance]
        imputed = imputed.loc[:, keep_variance]
        self.columns_after_variance = imputed.columns.tolist()

        if imputed.empty:
            raise ValueError("Descriptor preprocessing removed all columns before correlation filtering.")

        # Correlation pruning is intentionally ordered and deterministic: later columns are dropped
        # when they are too correlated with an earlier retained column.
        corr = imputed.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        keep_columns: list[str] = []
        dropped_correlation: list[str] = []
        for column in upper.columns:
            correlated = upper[column].dropna()
            if not correlated.empty and (correlated >= self.correlation_threshold).any():
                dropped_correlation.append(column)
                continue
            keep_columns.append(column)
        self.dropped_correlation = dropped_correlation
        self.final_columns = keep_columns
        if not self.final_columns:
            raise ValueError("Descriptor preprocessing removed all columns after correlation filtering.")

        # Scaling is the final train-only step and is applied only on the surviving schema.
        final_imputed = imputed.loc[:, self.final_columns]
        self.scaler.fit(final_imputed)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        # Test-time transform reuses the training-derived schema exactly:
        # same kept columns, same imputation statistics, same scaler parameters.
        working = frame.astype(float).replace([np.inf, -np.inf], np.nan)
        working = working.reindex(columns=self.columns_after_missing)
        imputed = pd.DataFrame(
            self.imputer.transform(working),
            index=working.index,
            columns=self.columns_after_missing,
        )
        final_frame = imputed.loc[:, self.final_columns]
        transformed = self.scaler.transform(final_frame)
        return pd.DataFrame(transformed, index=frame.index, columns=self.final_columns)

    def summary(self) -> dict[str, object]:
        return {
            "input_columns": self.input_columns,
            "dropped_missing": self.dropped_missing,
            "dropped_zero_variance": self.dropped_zero_variance,
            "dropped_correlation": self.dropped_correlation,
            "kept_columns": self.final_columns,
            "n_output_features": len(self.final_columns),
            "correlation_threshold": self.correlation_threshold,
        }


def signed_log1p(value: float) -> float:
    return float(np.sign(value) * np.log1p(abs(value)))


def build_molecule_cache(smiles_values: pd.Series) -> dict[str, Chem.Mol]:
    """Parse canonical benchmark SMILES once and share the molecule objects across RDKit feature builders."""

    cache: dict[str, Chem.Mol] = {}
    for smiles in smiles_values.astype(str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Failed to parse canonical smiles_std value: {smiles}")
        cache[smiles] = mol
    return cache


def count_aromatic_rings(mol: Chem.Mol) -> int:
    """Count aromatic rings rather than aromatic atoms for the Waters feature set.

    RDKit SMARTS `"a"` counts aromatic atoms, which overstates this feature for benzene-like
    systems. The Waters representation needs ring counts, so each aromatic ring returned by the
    ring topology is counted once if all atoms in that ring are aromatic.
    """

    aromatic_rings = 0
    for atom_ring in mol.GetRingInfo().AtomRings():
        if all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in atom_ring):
            aromatic_rings += 1
    return aromatic_rings


def zeng_has_heterocyclic_ring(mol: Chem.Mol) -> bool:
    """Return True if the molecule contains any ring with at least one heteroatom."""

    for atom_ring in mol.GetRingInfo().AtomRings():
        if any(mol.GetAtomWithIdx(atom_idx).GetAtomicNum() not in {1, 6} for atom_idx in atom_ring):
            return True
    return False


def zeng_neoplastic_80_rule_components(mol: Chem.Mol) -> dict[str, int | float | bool]:
    """Approximate the paper-described Neoplastic-80 rule with RDKit-accessible properties.

    The paper describes Neoplastic-80 as a binary indicator that becomes 1 when:
    1. at least one listed structural motif is present
    2. AlogP, molar refractivity, molecular weight, and total atom count all fall in range

    Here we map those terms to open-source RDKit quantities:
    - AlogP -> MolLogP
    - molar refractivity -> MolMR
    - molecular weight -> MolWt
    - total number of atoms -> atom count after explicit hydrogens are added
    """

    mol_with_h = Chem.AddHs(mol)
    mollogp = float(Crippen.MolLogP(mol))
    molmr = float(Crippen.MolMR(mol))
    molwt = float(Descriptors.MolWt(mol))
    total_atoms = int(mol_with_h.GetNumAtoms())

    structural_alert_any = any(
        [
            mol.HasSubstructMatch(ZENG_BENZENE_RING_PATTERN),
            zeng_has_heterocyclic_ring(mol),
            mol.HasSubstructMatch(ZENG_ALIPHATIC_AMINE_PATTERN),
            mol.HasSubstructMatch(ZENG_CARBOXAMIDE_PATTERN),
            mol.HasSubstructMatch(ZENG_ALCOHOLIC_HYDROXYL_PATTERN),
            mol.HasSubstructMatch(ZENG_ESTER_PATTERN),
            mol.HasSubstructMatch(ZENG_KETONE_PATTERN),
        ]
    )
    in_logp_range = -1.5 <= mollogp <= 4.7
    in_mr_range = 43.0 <= molmr <= 128.0
    in_mw_range = 180.0 <= molwt <= 470.0
    in_atom_range = 21 <= total_atoms <= 63
    rule_proxy = int(structural_alert_any and in_logp_range and in_mr_range and in_mw_range and in_atom_range)

    return {
        "rdkit_mollogp": mollogp,
        "rdkit_molmr": molmr,
        "rdkit_molwt": molwt,
        "rdkit_total_atoms_with_h": total_atoms,
        "neoplastic_80_rule_proxy": rule_proxy,
        "neoplastic_80_structural_alert_any": structural_alert_any,
        "neoplastic_80_in_logp_range": in_logp_range,
        "neoplastic_80_in_mr_range": in_mr_range,
        "neoplastic_80_in_mw_range": in_mw_range,
        "neoplastic_80_in_atom_range": in_atom_range,
    }


def default_features_dir(publication_dir: Path | None) -> Path | None:
    if publication_dir is None:
        return None
    return publication_dir / "outputs" / "features"


def default_frozen_feature_source_path(publication_dir: Path, feature_family: str) -> Path:
    """Return the checked-in frozen feature table shipped with this package."""

    return publication_dir / "outputs" / "features" / FEATURE_STORE_FILENAMES[feature_family]


def feature_store_path(features_dir: Path, feature_family: str) -> Path:
    if feature_family not in FEATURE_STORE_FILENAMES:
        raise ValueError(f"Unsupported feature family: {feature_family}")
    return features_dir / FEATURE_STORE_FILENAMES[feature_family]


def feature_store_manifest_path(features_dir: Path) -> Path:
    return features_dir / "manifest.yaml"


def _validate_selected_families(selected_families: set[str] | None) -> set[str]:
    supported = set(FEATURE_STORE_FILENAMES)
    if selected_families is None:
        return supported
    unknown = sorted(set(selected_families) - supported)
    if unknown:
        raise ValueError(f"Unsupported feature families requested: {unknown}. Supported: {sorted(supported)}")
    return set(selected_families)


def _feature_store_columns(feature_family: str, df: pd.DataFrame) -> list[str]:
    if feature_family == FPADMET_FEATURE_FAMILY:
        return PUBCHEM_FINGERPRINT_BIT_COLUMNS
    if feature_family == GATE_FEATURE_FAMILY:
        return [column for column in df.columns if column != "SMILES"]
    return [column for column in df.columns if column != "smiles_std"]


def _frozen_prediction_metadata(
    *,
    descriptor_backend: str,
    source_path: Path,
    source_key_column: str,
    feature_columns: list[str],
    excluded_std_columns: list[str] | None = None,
    excluded_columns: list[str] | None = None,
) -> dict[str, object]:
    """Build a consistent metadata block for the frozen GATE predicted-property feature table.

    Input:
    - descriptor_backend: human-readable backend label recorded in the manifest/report
    - source_path: original checked-in CSV used as the canonical source artifact
    - source_key_column: key column name in that source artifact
    - feature_columns: final numeric feature columns exposed to the runner
    - excluded_std_columns: columns removed because their names contain `std`
    - excluded_columns: columns removed by a user-specified fixed exclusion list

    Output:
    - manifest/report-ready metadata dictionary with schema and missingness counts
    """

    metadata = {
        "descriptor_backend": descriptor_backend,
        "fidelity_tag": GATE_FIDELITY_TAG,
        "feature_columns": feature_columns,
        "source_file": str(source_path),
        "source_key_column": source_key_column,
        "n_feature_columns": len(feature_columns),
        "n_missing_cells": 0,
        "n_rows_with_missing_values": 0,
        "n_feature_columns_with_missing": 0,
        "n_all_missing_feature_columns": 0,
        "columns_with_missing_values": [],
        "missing_cells_by_column": {},
    }
    if excluded_std_columns is not None:
        metadata["excluded_std_columns"] = excluded_std_columns
    if excluded_columns is not None:
        metadata["excluded_columns"] = excluded_columns
    return metadata


def _validate_frozen_prediction_store_frame(
    store_df: pd.DataFrame,
    benchmark_smiles_set: set[str],
    path: Path,
    *,
    key_column: str,
    expected_feature_count: int,
    family_label: str,
) -> list[str]:
    """Validate one checked-in frozen feature table before it is used as a model input.

    Unlike the literature baselines, these representations are shipped as curated
    frozen artifacts. The loader therefore focuses on strict schema and benchmark
    coverage checks instead of descriptor generation.
    """

    if key_column not in store_df.columns:
        raise ValueError(f"{family_label} feature table is missing the {key_column} key column: {path}")
    if store_df[key_column].duplicated().any():
        raise ValueError(f"{family_label} feature table contains duplicate {key_column} rows: {path}")

    feature_columns = [column for column in store_df.columns if column != key_column]
    if len(feature_columns) != expected_feature_count:
        raise ValueError(
            f"{family_label} feature table must contain exactly {expected_feature_count} numeric feature columns, "
            f"found {len(feature_columns)}: {path}"
        )

    non_numeric = [column for column in feature_columns if not is_numeric_dtype(store_df[column])]
    if non_numeric:
        preview = ", ".join(non_numeric[:5])
        raise ValueError(
            f"{family_label} feature table contains non-numeric feature columns ({preview}). Path: {path}"
        )

    store_smiles = set(store_df[key_column].astype(str))
    if store_smiles != benchmark_smiles_set:
        missing = sorted(benchmark_smiles_set - store_smiles)
        extra = sorted(store_smiles - benchmark_smiles_set)
        raise ValueError(
            f"{family_label} feature table does not match benchmark coverage. "
            f"missing={len(missing)}, extra={len(extra)}, path={path}"
        )

    numeric_frame = store_df.loc[:, feature_columns].astype(float)
    missing_mask = numeric_frame.isna()
    if missing_mask.any().any():
        preview = ", ".join(missing_mask.any(axis=0).loc[lambda series: series].index[:5].tolist())
        raise ValueError(
            f"{family_label} feature table contains missing feature values ({preview}). Path: {path}"
        )
    if not np.isfinite(numeric_frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{family_label} feature table contains non-finite feature values: {path}")
    return feature_columns


def build_zeng_proxy_features(benchmark_df: pd.DataFrame, mol_cache: dict[str, Chem.Mol]) -> FeatureBuildResult:
    """Build the paper-faithful RDKit proxy set used for the Zeng-style SVR baseline."""

    rows = []
    for smiles in benchmark_df["smiles_std"].astype(str):
        mol = mol_cache[smiles]
        neoplastic = zeng_neoplastic_80_rule_components(mol)
        rows.append(
            {
                "smiles_std": smiles,
                "cos2_mollogp_proxy": float(np.cos((4.31 + neoplastic["rdkit_mollogp"]) / 8.66) ** 2),
                "neoplastic_80_rule_proxy": int(neoplastic["neoplastic_80_rule_proxy"]),
                "x3v_chi3v_proxy": float(GraphDescriptors.Chi3v(mol)),
            }
        )
    frame = pd.DataFrame(rows).set_index("smiles_std")
    return FeatureBuildResult(
        feature_family=ZENG_FEATURE_FAMILY,
        frame=frame,
        metadata={
            "descriptor_backend": "rdkit_paper_proxy",
            "fidelity_tag": "approx_open_source_proxy",
            "feature_columns": ZENG_FEATURE_COLUMNS,
        },
        store_frame=frame.reset_index(),
    )


def build_waters_fragment_features(
    benchmark_df: pd.DataFrame,
    mol_cache: dict[str, Chem.Mol],
) -> FeatureBuildResult:
    """Count fixed functional groups for the Waters-style linear baseline."""

    rows = []
    for smiles in benchmark_df["smiles_std"].astype(str):
        mol = mol_cache[smiles]
        row = {"smiles_std": smiles}
        for name in WATERS_FEATURE_COLUMNS:
            if name == "aromatic":
                # Waters uses aromatic ring counts, not aromatic atom counts.
                row[name] = count_aromatic_rings(mol)
                continue
            pattern = FRAGMENT_PATTERNS[name]
            row[name] = len(mol.GetSubstructMatches(pattern))
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("smiles_std")
    return FeatureBuildResult(
        feature_family=WATERS_FEATURE_FAMILY,
        frame=frame,
        metadata={
            "descriptor_backend": "rdkit_smarts",
            "fidelity_tag": "exact_open_source",
            "feature_columns": frame.columns.tolist(),
        },
        store_frame=frame.reset_index(),
    )


def precompute_feature_families(
    benchmark_df: pd.DataFrame,
    publication_dir: Path | None = None,
    features_dir: Path | None = None,
    selected_families: set[str] | None = None,
) -> dict[str, FeatureBuildResult]:
    """Build only the feature families needed by the selected models.

    This avoids unnecessary work for unselected baselines, which matters most for the
    denser descriptor families and for local fingerprint generation on the full benchmark.
    """

    requested = _validate_selected_families(selected_families)
    results: dict[str, FeatureBuildResult] = {}
    rdkit_families = {ZENG_FEATURE_FAMILY, WATERS_FEATURE_FAMILY}
    mol_cache: dict[str, Chem.Mol] | None = None
    if requested & rdkit_families:
        mol_cache = build_molecule_cache(benchmark_df["smiles_std"])
    if ZENG_FEATURE_FAMILY in requested:
        assert mol_cache is not None
        results[ZENG_FEATURE_FAMILY] = build_zeng_proxy_features(benchmark_df, mol_cache)
    if ABDALLAH_FEATURE_FAMILY in requested:
        raise ValueError(
            "The Abdallah CDK 2.8 descriptor table cannot be rebuilt from this package. "
            "It is shipped frozen at outputs/features/abdallah_descriptor_table.csv and is "
            "loaded, not regenerated, by the reproduction pipeline."
        )
    if WATERS_FEATURE_FAMILY in requested:
        assert mol_cache is not None
        results[WATERS_FEATURE_FAMILY] = build_waters_fragment_features(benchmark_df, mol_cache)
    if FPADMET_FEATURE_FAMILY in requested:
        raise ValueError(
            "The FP-ADMET PubChem fingerprint table cannot be rebuilt from this package. "
            "It is shipped frozen at outputs/features/fpadmet_pubchem.csv and is loaded, "
            "not regenerated, by the reproduction pipeline."
        )
    return results


def _store_frame_for_result(result: FeatureBuildResult) -> pd.DataFrame:
    if result.store_frame is not None:
        return result.store_frame.copy()
    return result.frame.reset_index()


def _manifest_entry_for_result(family: str, result: FeatureBuildResult) -> dict[str, object]:
    """Convert one freshly built feature result into the persisted manifest schema."""

    entry = {
        "file_name": FEATURE_STORE_FILENAMES[family],
        "n_rows": int(len(_store_frame_for_result(result))),
        "feature_columns": result.metadata.get("feature_columns", []),
        "n_feature_columns": int(len(result.metadata.get("feature_columns", []))),
        "descriptor_backend": result.metadata.get("descriptor_backend"),
        "fidelity_tag": result.metadata.get("fidelity_tag"),
    }
    for field in [
        "reference_descriptor_count",
        "selected_descriptor_count",
        "excluded_descriptors",
        "selection_provenance",
        "pass_criteria",
        "reference_source",
        "n_generated_descriptors",
        "n_rows_with_descriptor_failures",
        "n_missing_cells",
        "n_rows_with_missing_values",
        "n_feature_columns_with_missing",
        "n_all_missing_feature_columns",
        "columns_with_missing_values",
        "missing_cells_by_column",
        "java_runtime",
        "module_jar_count",
        "third_party_jar_count",
        "source_file",
        "source_key_column",
        "excluded_std_columns",
        "excluded_columns",
    ]:
        if field in result.metadata:
            entry[field] = result.metadata[field]
    return entry


def _load_existing_feature_store_manifest(manifest_path: Path) -> dict[str, object]:
    """Load the existing feature-store manifest if present.

    Incremental builds are allowed to materialize only a subset of families. In that case,
    the manifest should behave like an audit index for the directory rather than a log of
    just the last command, so previously recorded families are preserved unless rebuilt.
    """

    if not manifest_path.exists():
        return {"features": {}}
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if not isinstance(manifest, dict):
        raise ValueError(f"Feature-store manifest must be a mapping: {manifest_path}")
    features = manifest.get("features", {})
    if features is None:
        features = {}
    if not isinstance(features, dict):
        raise ValueError(f"Feature-store manifest 'features' must be a mapping: {manifest_path}")
    return {
        "benchmark_path": manifest.get("benchmark_path"),
        "features": dict(features),
    }


def _save_feature_store_manifest(
    manifest_path: Path,
    benchmark_path: Path,
    results: dict[str, FeatureBuildResult],
) -> None:
    existing_manifest = _load_existing_feature_store_manifest(manifest_path)
    merged_features = dict(existing_manifest["features"])

    # Incremental `run_features.py --families ...` calls should add or refresh only the
    # families built in the current command, while preserving audit metadata for the
    # previously materialized representation tables already present in the store.
    for family, result in results.items():
        merged_features[family] = _manifest_entry_for_result(family, result)

    manifest = {
        "benchmark_path": str(benchmark_path),
        "features": merged_features,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=False)


def build_feature_store(
    benchmark_path: Path,
    publication_dir: Path,
    features_dir: Path | None = None,
    selected_families: set[str] | None = None,
) -> dict[str, FeatureBuildResult]:
    """Build and persist shared representation tables for the final benchmark rows."""

    benchmark_df = load_benchmark_for_splitting(benchmark_path)
    root_features_dir = ensure_directory((features_dir or default_features_dir(publication_dir)).resolve())
    results = precompute_feature_families(
        benchmark_df,
        publication_dir=publication_dir,
        features_dir=root_features_dir,
        selected_families=selected_families,
    )

    for family, result in results.items():
        target_path = feature_store_path(root_features_dir, family)
        source_file = result.metadata.get("source_file")
        # The GATE feature table is already a curated checked-in artifact. When the feature-store root is
        # the same directory as that source file, we refresh the manifest metadata without
        # needlessly rewriting the large CSV.
        if source_file is not None and Path(str(source_file)).resolve() == target_path.resolve():
            continue
        save_csv(_store_frame_for_result(result), target_path)

    _save_feature_store_manifest(feature_store_manifest_path(root_features_dir), benchmark_path, results)

    failed_results = {
        family: result
        for family, result in results.items()
        if int(result.metadata.get("n_failed_rows", 0)) > 0
    }
    if failed_results:
        failure_messages = []
        for family, result in failed_results.items():
            examples = result.metadata.get("failure_examples", [])
            failure_messages.append(
                f"{family}: {int(result.metadata['n_failed_rows'])} unresolved rows"
                + (f" (examples: {', '.join(examples)})" if examples else "")
            )
        raise ValueError("Feature store build failed due to unresolved representations: " + "; ".join(failure_messages))
    return results


def load_feature_store(
    benchmark_df: pd.DataFrame,
    publication_dir: Path,
    features_dir: Path,
    selected_families: set[str],
) -> dict[str, FeatureBuildResult]:
    """Load persisted representation tables and validate benchmark coverage."""

    benchmark_smiles = benchmark_df["smiles_std"].astype(str).tolist()
    benchmark_smiles_set = set(benchmark_smiles)
    results: dict[str, FeatureBuildResult] = {}

    for family in _validate_selected_families(selected_families):
        path = feature_store_path(features_dir, family)
        if not path.exists():
            raise ValueError(
                f"Feature table for '{family}' was not found at {path}. Run run_features.py first."
            )
        store_df = pd.read_csv(path)
        if family == GATE_FEATURE_FAMILY:
            key_column = "SMILES"
        else:
            key_column = "smiles_std"
        if key_column not in store_df.columns:
            raise ValueError(f"Feature table for '{family}' is missing {key_column}: {path}")
        store_df = store_df.sort_values(key_column, kind="stable").reset_index(drop=True)
        if store_df[key_column].duplicated().any():
            raise ValueError(f"Feature table for '{family}' contains duplicate {key_column} rows: {path}")

        store_smiles = set(store_df[key_column].astype(str))
        if store_smiles != benchmark_smiles_set:
            missing = sorted(benchmark_smiles_set - store_smiles)
            extra = sorted(store_smiles - benchmark_smiles_set)
            raise ValueError(
                f"Feature table for '{family}' does not match benchmark coverage. "
                f"missing={len(missing)}, extra={len(extra)}, path={path}"
            )

        abdallah_config: dict[str, object] | None = None
        if family == ABDALLAH_FEATURE_FAMILY:
            abdallah_config = load_abdallah_descriptor_config(publication_dir)
            expected_columns = ["smiles_std", *abdallah_config["selected_descriptors"]]
            if store_df.columns.tolist() != expected_columns:
                raise ValueError(
                    f"Feature table for '{family}' does not match the curated 141-descriptor schema: {path}"
                )
            feature_columns = list(abdallah_config["selected_descriptors"])
        else:
            feature_columns = _feature_store_columns(family, store_df)
        if not feature_columns:
            raise ValueError(f"Feature table for '{family}' does not contain any feature columns: {path}")
        missing_feature_columns = [column for column in feature_columns if column not in store_df.columns]
        if missing_feature_columns:
            preview = ", ".join(missing_feature_columns[:5])
            raise ValueError(
                f"Feature table for '{family}' is missing expected feature columns ({preview}). Path: {path}"
            )
        metadata: dict[str, object]
        if family == ZENG_FEATURE_FAMILY:
            metadata = {
                # The persisted Zeng table now stores the paper-faithful RDKit proxy trio,
                # so the loader must report the same backend label as the builder.
                "descriptor_backend": "rdkit_paper_proxy",
                "fidelity_tag": "approx_open_source_proxy",
                "feature_columns": feature_columns,
            }
        elif family == ABDALLAH_FEATURE_FAMILY:
            assert abdallah_config is not None
            metadata = {
                "descriptor_backend": "cdk_2_8_curated_subset",
                "fidelity_tag": "validated_open_source_subset",
                "feature_columns": feature_columns,
                "reference_descriptor_count": int(abdallah_config["reference_descriptor_count"]),
                "selected_descriptor_count": len(feature_columns),
                "excluded_descriptors": list(abdallah_config["excluded_descriptors"]),
                "selection_provenance": str(abdallah_config["selection_provenance"]),
                "pass_criteria": str(abdallah_config["pass_criteria"]),
                "reference_source": str(abdallah_config["reference_source"]),
            }
        elif family == WATERS_FEATURE_FAMILY:
            metadata = {
                "descriptor_backend": "rdkit_smarts",
                "fidelity_tag": "exact_open_source",
                "feature_columns": feature_columns,
            }
        elif family == FPADMET_FEATURE_FAMILY:
            missing_bits = store_df[feature_columns].isna().any(axis=1)
            if missing_bits.any():
                preview = ", ".join(store_df.loc[missing_bits, "smiles_std"].head(5).astype(str).tolist())
                raise ValueError(
                    f"Feature table for '{family}' contains missing fingerprint bits: {preview}. Path: {path}"
                )
            metadata = {
                "descriptor_backend": "skfp_pubchem_local",
                "fidelity_tag": "approx_open_source_proxy",
                "feature_columns": feature_columns,
                "n_bits": PUBCHEM_FINGERPRINT_LENGTH,
            }
        elif family == GATE_FEATURE_FAMILY:
            feature_columns = _validate_frozen_prediction_store_frame(
                store_df,
                benchmark_smiles_set,
                path,
                key_column="SMILES",
                expected_feature_count=30,
                family_label="GATE predicted-property",
            )
            present_excluded = [column for column in GATE_EXCLUDED_COLUMNS if column in store_df.columns]
            if present_excluded:
                preview = ", ".join(present_excluded[:5])
                raise ValueError(
                    f"Feature table for '{family}' should not contain excluded core columns ({preview}): {path}"
                )
            metadata = _frozen_prediction_metadata(
                descriptor_backend="gate_predicted_property_table",
                source_path=default_frozen_feature_source_path(publication_dir, GATE_FEATURE_FAMILY),
                source_key_column="SMILES",
                feature_columns=feature_columns,
                excluded_columns=GATE_EXCLUDED_COLUMNS,
            )
        else:
            raise ValueError(f"Unsupported feature family: {family}")

        frame = store_df.loc[:, [key_column, *feature_columns]].set_index(key_column).reindex(benchmark_smiles)
        results[family] = FeatureBuildResult(feature_family=family, frame=frame, metadata=metadata, store_frame=store_df)
    return results
