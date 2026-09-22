"""Baseline registry and external-model adapter handling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from skin_benchmark.benchmarking.features import (
    ABDALLAH_FEATURE_FAMILY,
    FPADMET_FEATURE_FAMILY,
    GATE_FEATURE_FAMILY,
    GATE_FIDELITY_TAG,
    WATERS_FEATURE_FAMILY,
    ZENG_FEATURE_FAMILY,
    DescriptorTablePreprocessor,
    IdentityPreprocessor,
    ScaledNumericPreprocessor,
)


@dataclass
class ModelAuditSummary:
    """Static report metadata for one baseline family.

    This is reporting-oriented metadata: what the paper family looked like, how our benchmark
    approximates it, and whether the implementation still needs user review for fidelity reasons.
    """

    baseline_id: str
    paper_feature_stack: str
    benchmark_feature_stack: str
    open_source_status: str
    proprietary_gap: str
    fidelity_tag: str
    user_review_needed: bool
    source_url: str


@dataclass
class BaselineSpec:
    """Executable spec for one baseline family.

    Unlike `ModelAuditSummary`, this object contains the pieces the runner needs to actually train
    the model: which feature family to build, which preprocessor to fit, and which estimator to use.
    """

    model_id: str
    feature_family: str
    preprocessor_factory: Callable[[], object]
    estimator_factory: Callable[[int], object]
    audit_summary: ModelAuditSummary


def build_gate_audit_summary(
    baseline_id: str,
    paper_model_label: str,
    benchmark_model_label: str,
    *,
    feature_label: str,
    n_features: int,
) -> ModelAuditSummary:
    """Build audit metadata for the frozen GATE predicted-property fixed model row."""

    return ModelAuditSummary(
        baseline_id=baseline_id,
        paper_feature_stack=f"{feature_label} with {paper_model_label}",
        benchmark_feature_stack=f"{feature_label} ({n_features} features) + {benchmark_model_label}",
        open_source_status="not_open_source_frozen_features",
        proprietary_gap="GATE foundation-model weights are not distributed; the predicted-property features it produced are shipped as a frozen table and evaluated under the shared benchmark protocol.",
        fidelity_tag=GATE_FIDELITY_TAG,
        user_review_needed=False,
        source_url="",
    )


BASELINE_SPECS = {
    # Zeng-style baseline: a very small low-dimensional proxy descriptor set with an RBF SVR.
    "zeng_svr_proxy": BaselineSpec(
        model_id="zeng_svr_proxy",
        feature_family=ZENG_FEATURE_FAMILY,
        preprocessor_factory=ScaledNumericPreprocessor,
        estimator_factory=lambda seed: SVR(kernel="rbf", C=7.2906, gamma=1.7200, epsilon=0.001),
        audit_summary=ModelAuditSummary(
            baseline_id="zeng_svr_proxy",
            paper_feature_stack="ChemDraw/Chem3D/MOPAC + Dragon 6.0 descriptors feeding RBF-SVR",
            benchmark_feature_stack="Paper-faithful RDKit proxies for cos2[(4.31+AlogP)/8.66], Neoplastic-80, and X3v",
            open_source_status="open_source_proxy",
            proprietary_gap="Dragon 6.0 and ChemOffice-derived descriptors are approximated with RDKit MolLogP, a rule-based Neoplastic-80 reconstruction, and Chi3v.",
            fidelity_tag="approx_open_source_proxy",
            user_review_needed=True,
            source_url="https://www.nature.com/articles/s41598-021-89587-5",
        ),
    ),
    # Abdallah-style baseline: curated CDK descriptor table plus the paper's LightGBM family.
    "abdallah_lgbm": BaselineSpec(
        model_id="abdallah_lgbm",
        feature_family=ABDALLAH_FEATURE_FAMILY,
        preprocessor_factory=DescriptorTablePreprocessor,
        estimator_factory=lambda seed: LGBMRegressor(random_state=seed, n_jobs=-1),
        audit_summary=ModelAuditSummary(
            baseline_id="abdallah_lgbm",
            paper_feature_stack="CDK 2.8 1D/2D descriptor table with LGBM regression family",
            benchmark_feature_stack="Validated CDK 2.8 curated 141-descriptor subset + LGBMRegressor",
            open_source_status="validated_open_source_subset",
            proprietary_gap=(
                "No paid dependency; four reference descriptors (JPLogP, nAtomLC, topoShape, WPATH) "
                "were excluded after audit against clean_trial4.csv, and the estimator is aligned to "
                "the paper's LightGBM family."
            ),
            fidelity_tag="validated_open_source_subset",
            user_review_needed=False,
            source_url="https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000483",
        ),
    ),
    # Waters-style baseline: functional-group count features with a linear model.
    "waters_fragment_linear": BaselineSpec(
        model_id="waters_fragment_linear",
        feature_family=WATERS_FEATURE_FAMILY,
        preprocessor_factory=IdentityPreprocessor,
        estimator_factory=lambda seed: LinearRegression(),
        audit_summary=ModelAuditSummary(
            baseline_id="waters_fragment_linear",
            paper_feature_stack="Functional-group contribution model with fragment prevalence counts",
            benchmark_feature_stack="RDKit SMARTS count for 10 fixed functional groups",
            open_source_status="exact_open_source",
            proprietary_gap="none",
            fidelity_tag="exact_open_source",
            user_review_needed=False,
            source_url="https://www.nature.com/articles/s41597-023-02711-0",
        ),
    ),
    # FP-ADMET-style baseline: PubChem fingerprint features with a random forest regressor.
    "fpadmet_rf": BaselineSpec(
        model_id="fpadmet_rf",
        feature_family=FPADMET_FEATURE_FAMILY,
        preprocessor_factory=IdentityPreprocessor,
        estimator_factory=lambda seed: RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ),
        audit_summary=ModelAuditSummary(
            baseline_id="fpadmet_rf",
            paper_feature_stack="Fingerprint-based ADMET modeling with random forest regression",
            benchmark_feature_stack="PubChem CACTVS-style 881-bit fingerprint via scikit-fingerprints PubChemFingerprint + RandomForestRegressor",
            open_source_status="open_source_proxy",
            proprietary_gap=(
                "No paid-software gap; the fingerprint is a local scikit-fingerprints reimplementation "
                "that may differ slightly from PubChem API output, and the estimator backend remains scikit-learn."
            ),
            fidelity_tag="approx_open_source_proxy",
            user_review_needed=True,
            source_url="https://link.springer.com/article/10.1186/s13321-021-00557-5",
        ),
    ),
    "gate_lgbm": BaselineSpec(
        model_id="gate_lgbm",
        feature_family=GATE_FEATURE_FAMILY,
        preprocessor_factory=IdentityPreprocessor,
        estimator_factory=lambda seed: LGBMRegressor(random_state=seed, n_jobs=-1),
        audit_summary=build_gate_audit_summary(
            baseline_id="gate_lgbm",
            paper_model_label="LightGBM regression",
            benchmark_model_label="LGBMRegressor",
            feature_label="GATE predicted-property 30-feature table",
            n_features=30,
        ),
    ),
}


def list_baseline_ids() -> list[str]:
    return list(BASELINE_SPECS)


def baseline_audit_table() -> pd.DataFrame:
    return pd.DataFrame([asdict(spec.audit_summary) for spec in BASELINE_SPECS.values()])


def resolve_baseline_specs(selected_models: list[str]) -> list[BaselineSpec]:
    # Baseline selection is kept separate from external-model adapter loading so literature baselines
    # and user-provided models can share the runner without sharing the same configuration path.
    baseline_ids = list_baseline_ids()
    if "all" in selected_models:
        return [BASELINE_SPECS[model_id] for model_id in baseline_ids]
    filtered_models = [model_id for model_id in selected_models if model_id != "external_model"]
    unknown = [model_id for model_id in filtered_models if model_id not in baseline_ids]
    if unknown:
        raise ValueError(f"Unknown baseline model ids: {unknown}. Available: {baseline_ids}")
    return [BASELINE_SPECS[model_id] for model_id in filtered_models]


def load_adapter_factory(factory_ref: str) -> Callable[[], object]:
    # External-model adapters are loaded dynamically from a Python factory string instead of being baked
    # into BASELINE_SPECS because they are user/project specific rather than fixed benchmark families.
    if ":" not in factory_ref:
        raise ValueError("external-model adapter must use the form package.module:factory")
    module_name, factory_name = factory_ref.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if factory is None or not callable(factory):
        raise ValueError(f"Could not resolve callable adapter factory '{factory_ref}'.")
    return factory


def validate_model_adapter(adapter: object) -> None:
    # The adapter contract is intentionally narrow so any external model can plug into the same
    # split/metric/report pipeline as long as it obeys the shared fit/predict interface.
    missing = [name for name in ["model_id", "requires_smiles", "fit", "predict"] if not hasattr(adapter, name)]
    if missing:
        raise ValueError(
            "external-model adapter is missing required attributes/methods: " + ", ".join(missing)
        )
    if not callable(getattr(adapter, "fit")) or not callable(getattr(adapter, "predict")):
        raise ValueError("external-model adapter fit/predict attributes must be callable.")


def run_model_adapter(
    factory: Callable[[], object],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    artifact_dir: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    adapter = factory()
    validate_model_adapter(adapter)
    adapter.fit(train_df.copy(), seed=seed, artifact_dir=artifact_dir)
    predictions = np.asarray(adapter.predict(test_df.copy()), dtype=float)
    if predictions.shape[0] != len(test_df):
        raise ValueError(
            f"external-model adapter '{adapter.model_id}' returned {predictions.shape[0]} predictions for {len(test_df)} test rows."
        )
    return predictions, {
        "model_id": str(adapter.model_id),
        "requires_smiles": bool(adapter.requires_smiles),
    }
