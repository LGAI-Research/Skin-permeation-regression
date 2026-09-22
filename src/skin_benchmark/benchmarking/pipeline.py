"""Reproducible benchmark training pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml

from skin_benchmark.benchmarking.data import (
    default_benchmark_path,
    default_runs_dir,
    default_split_methods,
    default_splits_dir,
    leakage_summary,
    load_benchmark_with_assignments,
    materialize_fold_dataset,
)
from skin_benchmark.benchmarking.external_validation import run_external_validation_audit
from skin_benchmark.benchmarking.features import default_features_dir, load_feature_store
from skin_benchmark.benchmarking.metrics import compute_regression_metrics
from skin_benchmark.benchmarking.registry import (
    baseline_audit_table,
    load_adapter_factory,
    resolve_baseline_specs,
    run_model_adapter,
)
from skin_benchmark.benchmarking.reporting import write_benchmark_outputs
from skin_benchmark.utils.io import ensure_directory, save_csv
from skin_benchmark.utils.logging import configure_logger
from skin_benchmark.splits import load_benchmark_for_splitting


def _timestamp_run_id() -> str:
    """Return a compact human-facing run timestamp in local time.

    This timestamp is used only in the run directory name / run_id, so we prefer
    a short label-friendly format over an ISO-style UTC suffix.
    Example: `2603231707` means 2026-03-23 17:07 in the local system timezone.
    """

    return datetime.now().astimezone().strftime("%y%m%d%H%M")


def _resolve_run_id(requested_run_id: str | None) -> str:
    """Return the persisted run id used for the run directory and output files.

    Input
    - `requested_run_id`: optional user-provided label from the CLI or API

    Output
    - a unique run id string
      - timestamp only when no label was provided
      - `<label>_<timestamp>` when the user provided a label

    We always append the local minute-resolution timestamp to user-provided labels so repeated benchmark
    launches do not overwrite earlier run directories with the same human-readable name.
    """

    timestamp = _timestamp_run_id()
    if requested_run_id is None or not requested_run_id.strip():
        return timestamp
    return f"{requested_run_id}_{timestamp}"


def _save_yaml(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def _baseline_prediction_rows(
    fold_dataset,
    model_spec,
    feature_cache,
    seed: int,
    artifact_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    raw_feature_result = feature_cache[model_spec.feature_family]
    feature_columns = raw_feature_result.metadata.get("feature_columns", raw_feature_result.frame.columns.tolist())
    # Raw features are precomputed once on the full benchmark, but train/test rows are aligned
    # per fold here. The important leakage boundary is below: every train-dependent transform is
    # fit only on `train_features` and then applied unchanged to `test_features`.
    train_features = raw_feature_result.frame.reindex(
        fold_dataset.train_df["smiles_std"].astype(str)
    ).loc[:, feature_columns].copy()
    test_features = raw_feature_result.frame.reindex(
        fold_dataset.test_df["smiles_std"].astype(str)
    ).loc[:, feature_columns].copy()
    preprocessor = model_spec.preprocessor_factory()
    preprocessor.fit(train_features)
    train_matrix = preprocessor.transform(train_features)
    test_matrix = preprocessor.transform(test_features)
    # Estimators are trained on plain numeric matrices so model families with stricter feature-name
    # rules (for example LightGBM/XGBoost) can consume the shared feature tables without mutating
    # the original audited column names stored in the feature manifests and schema summaries.
    train_values = train_matrix.to_numpy(dtype=float)
    test_values = test_matrix.to_numpy(dtype=float)
    estimator = model_spec.estimator_factory(seed)
    estimator.fit(train_values, fold_dataset.train_df["target_logkp"].astype(float).to_numpy())
    # Persist fitted weights and the train-only preprocessor so downstream inference (e.g. HnH
    # external validation) can reuse the exact fold-level estimators without refitting on the full
    # benchmark. Saving lives next to the existing feature-schema YAML so each fold's artifacts
    # stay in one directory.
    joblib.dump(estimator, artifact_dir / f"fold_{fold_dataset.fold_index}_estimator.joblib")
    joblib.dump(preprocessor, artifact_dir / f"fold_{fold_dataset.fold_index}_preprocessor.joblib")
    predictions = estimator.predict(test_values)

    prediction_rows = pd.DataFrame(
        {
            "smiles_std": fold_dataset.test_df["smiles_std"].astype(str).tolist(),
            "y_true": fold_dataset.test_df["target_logkp"].astype(float).tolist(),
            "y_pred": [float(value) for value in predictions],
            "split_method": fold_dataset.split_method,
            "fold_index": fold_dataset.fold_index,
            "model_id": model_spec.model_id,
        }
    )
    schema_summary = preprocessor.summary()
    schema_summary.update(raw_feature_result.metadata)
    schema_summary["feature_family"] = model_spec.feature_family
    schema_summary["n_train_rows"] = len(fold_dataset.train_df)
    schema_summary["n_test_rows"] = len(fold_dataset.test_df)
    # Persist the per-fold feature schema so later audits can explain which columns survived
    # train-only preprocessing for a given baseline and split.
    _save_yaml(schema_summary, artifact_dir / f"fold_{fold_dataset.fold_index}_feature_schema.yaml")
    metadata = {
        "model_id": model_spec.model_id,
        "descriptor_backend": raw_feature_result.metadata.get("descriptor_backend"),
        "feature_family": model_spec.feature_family,
        "feature_dim": schema_summary.get("n_output_features"),
        "fidelity_tag": raw_feature_result.metadata.get("fidelity_tag", model_spec.audit_summary.fidelity_tag),
    }
    return prediction_rows, metadata, schema_summary


def _adapter_prediction_rows(
    fold_dataset,
    adapter_factory,
    seed: int,
    artifact_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    predictions, adapter_summary = run_model_adapter(
        adapter_factory,
        train_df=fold_dataset.train_df,
        test_df=fold_dataset.test_df,
        seed=seed,
        artifact_dir=artifact_dir,
    )
    prediction_rows = pd.DataFrame(
        {
            "smiles_std": fold_dataset.test_df["smiles_std"].astype(str).tolist(),
            "y_true": fold_dataset.test_df["target_logkp"].astype(float).tolist(),
            "y_pred": [float(value) for value in predictions],
            "split_method": fold_dataset.split_method,
            "fold_index": fold_dataset.fold_index,
            "model_id": adapter_summary["model_id"],
        }
    )
    _save_yaml(adapter_summary, artifact_dir / f"fold_{fold_dataset.fold_index}_adapter_summary.yaml")
    return prediction_rows, adapter_summary


def _update_audit_summary(audit_summary: pd.DataFrame, model_metadata_rows: list[dict[str, object]]) -> pd.DataFrame:
    if not model_metadata_rows:
        return audit_summary
    metadata_df = pd.DataFrame(model_metadata_rows)
    out = audit_summary.copy()
    for model_id, group in metadata_df.groupby("model_id"):
        backend = sorted({str(value) for value in group["descriptor_backend"].dropna().tolist()})
        fidelity_tags = sorted({str(value) for value in group["fidelity_tag"].dropna().tolist()})
        if model_id == "abdallah_lgbm" and backend:
            backend_text = ", ".join(backend)
            if "rdkit_2d_fallback" in backend_text:
                out.loc[out["baseline_id"] == model_id, "benchmark_feature_stack"] = "RDKit 2D descriptor fallback"
            elif "cdk_2_8_curated_subset" in backend_text:
                out.loc[out["baseline_id"] == model_id, "benchmark_feature_stack"] = (
                    "Validated CDK 2.8 curated 141-descriptor subset + LGBMRegressor"
                )
            out.loc[out["baseline_id"] == model_id, "fidelity_tag"] = (
                fidelity_tags[0] if len(fidelity_tags) == 1 else "|".join(fidelity_tags)
            )
    return out


def run_benchmark(
    publication_dir: Path,
    benchmark_path: Path | None = None,
    splits_dir: Path | None = None,
    features_dir: Path | None = None,
    runs_dir: Path | None = None,
    run_id: str | None = None,
    split_methods: list[str] | None = None,
    folds: list[int] | None = None,
    models: list[str] | None = None,
    external_model_factory_ref: str | None = None,
    seed: int = 42,
    paper_sanity: str = "auto",
) -> dict[str, object]:
    """Run the baseline benchmark pipeline."""

    publication_dir = publication_dir.resolve()
    benchmark_path = (benchmark_path or default_benchmark_path(publication_dir)).resolve()
    splits_dir = (splits_dir or default_splits_dir(publication_dir)).resolve()
    features_dir = (features_dir or default_features_dir(publication_dir)).resolve()
    runs_dir = (runs_dir or default_runs_dir(publication_dir)).resolve()
    requested_run_id = run_id
    run_id = _resolve_run_id(run_id)
    split_methods = split_methods or default_split_methods()
    folds = folds or [1, 2, 3, 4, 5]
    models = models or ["all"]

    run_dir = ensure_directory(runs_dir / run_id)
    predictions_root = ensure_directory(run_dir / "predictions")
    artifacts_root = ensure_directory(run_dir / "artifacts")
    logger = configure_logger(run_dir / "benchmark.log")

    baseline_specs = resolve_baseline_specs(models)
    adapter_factory = load_adapter_factory(external_model_factory_ref) if external_model_factory_ref else None
    external_model_status = "provided" if adapter_factory is not None else "not_provided"
    run_external_model = adapter_factory is not None and ("all" in models or "external_model" in models)
    if not baseline_specs and not run_external_model:
        raise ValueError(
            "No executable models were selected. Provide baseline model ids or pass --external-model for external_model runs."
        )

    # Stage 1: load the already-curated benchmark rows and shared representation tables.
    benchmark = load_benchmark_for_splitting(benchmark_path)
    selected_feature_families = {spec.feature_family for spec in baseline_specs}
    feature_cache = load_feature_store(
        benchmark,
        publication_dir=publication_dir,
        features_dir=features_dir,
        selected_families=selected_feature_families,
    )
    fold_metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    leakage_rows: list[dict[str, object]] = []
    model_metadata_rows: list[dict[str, object]] = []

    # Stage 2: iterate over the shared split assignments and materialize train/test rows per fold.
    for split_method in split_methods:
        logger.info("Evaluating split method %s", split_method)
        split_assignment = load_benchmark_with_assignments(benchmark_path, splits_dir, split_method)
        for fold_index in folds:
            fold_dataset = materialize_fold_dataset(split_assignment, split_method, fold_index)
            # Leakage is checked before any model fit so every baseline is forced to use the same
            # molecule-disjoint outer split definition.
            leakage = leakage_summary(fold_dataset.train_df, fold_dataset.test_df)
            leakage_rows.append(
                {
                    "split_method": split_method,
                    "fold_index": fold_index,
                    **leakage,
                }
            )
            if not leakage["passed"]:
                raise ValueError(
                    f"Leakage detected for split '{split_method}' fold {fold_index}: overlap={leakage['n_overlap_smiles']}"
                )

            # Stage 3: fit the selected baseline families on the current fold and export fold-level
            # predictions immediately so every run leaves behind auditable artifacts.
            for model_spec in baseline_specs:
                logger.info("Training %s on %s fold %s", model_spec.model_id, split_method, fold_index)
                artifact_dir = ensure_directory(artifacts_root / split_method / model_spec.model_id)
                prediction_df, model_metadata, schema_summary = _baseline_prediction_rows(
                    fold_dataset,
                    model_spec,
                    feature_cache,
                    seed=seed,
                    artifact_dir=artifact_dir,
                )
                prediction_rows.append(prediction_df)
                save_csv(
                    prediction_df.assign(run_id=run_id),
                    ensure_directory(predictions_root / split_method / model_spec.model_id) / f"fold_{fold_index}.csv",
                )
                metrics = compute_regression_metrics(
                    prediction_df["y_true"].to_numpy(dtype=float),
                    prediction_df["y_pred"].to_numpy(dtype=float),
                )
                fold_metric_rows.append(
                    {
                        "run_id": run_id,
                        "model_id": model_spec.model_id,
                        "split_method": split_method,
                        "fold_index": fold_index,
                        "seed": seed,
                        "n_train": len(fold_dataset.train_df),
                        "n_test": len(fold_dataset.test_df),
                        "feature_family": model_spec.feature_family,
                        "descriptor_backend": model_metadata["descriptor_backend"],
                        "feature_dim": schema_summary.get("n_output_features"),
                        **metrics,
                    }
                )
                model_metadata_rows.append(model_metadata)

            if run_external_model:
                # The adapter path reuses the same fold materialization, metrics, and output layout
                # so the external model is compared under the exact same evaluation contract as the baselines.
                logger.info("Training external_model on %s fold %s", split_method, fold_index)
                artifact_dir = ensure_directory(artifacts_root / split_method / "external_model")
                prediction_df, adapter_summary = _adapter_prediction_rows(
                    fold_dataset,
                    adapter_factory=adapter_factory,
                    seed=seed,
                    artifact_dir=artifact_dir,
                )
                prediction_rows.append(prediction_df)
                save_csv(
                    prediction_df.assign(run_id=run_id),
                    ensure_directory(predictions_root / split_method / "external_model") / f"fold_{fold_index}.csv",
                )
                metrics = compute_regression_metrics(
                    prediction_df["y_true"].to_numpy(dtype=float),
                    prediction_df["y_pred"].to_numpy(dtype=float),
                )
                fold_metric_rows.append(
                    {
                        "run_id": run_id,
                        "model_id": adapter_summary["model_id"],
                        "split_method": split_method,
                        "fold_index": fold_index,
                        "seed": seed,
                        "n_train": len(fold_dataset.train_df),
                        "n_test": len(fold_dataset.test_df),
                        "feature_family": "adapter",
                        "descriptor_backend": "adapter",
                        "feature_dim": pd.NA,
                        **metrics,
                    }
                )

    # Stage 4: aggregate fold outputs into benchmark-wide reports and stable run manifests.
    fold_metrics_df = pd.DataFrame(fold_metric_rows).sort_values(
        ["split_method", "model_id", "fold_index"],
        kind="stable",
    ).reset_index(drop=True)
    prediction_rows_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    leakage_df = pd.DataFrame(leakage_rows)
    audit_summary_df = _update_audit_summary(baseline_audit_table(), model_metadata_rows)
    external_validation_df = run_external_validation_audit(publication_dir, paper_sanity)

    output_paths = write_benchmark_outputs(
        run_dir=run_dir,
        fold_metrics=fold_metrics_df,
        prediction_rows=prediction_rows_df,
        leakage_checks=leakage_df,
        audit_summary=audit_summary_df,
        external_validation=external_validation_df,
        run_id=run_id,
        external_model_status=external_model_status,
    )

    _save_yaml(
        {
            "run_id": run_id,
            "requested_run_id": requested_run_id,
            "benchmark_path": str(benchmark_path),
            "splits_dir": str(splits_dir),
            "features_dir": str(features_dir),
            "runs_dir": str(runs_dir),
            "split_methods": split_methods,
            "folds": folds,
            "models": models,
            "seed": seed,
            "paper_sanity": paper_sanity,
            "external_model_factory_ref": external_model_factory_ref,
            "proprietary_descriptor_audit": audit_summary_df.to_dict(orient="records"),
        },
        run_dir / "config_snapshot.yaml",
    )
    _save_yaml(
        {
            "run_id": run_id,
            "requested_run_id": requested_run_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "executed_baselines": [spec.model_id for spec in baseline_specs],
            "executed_external_model": run_external_model,
            "external_model_status": external_model_status,
            "n_prediction_rows": int(len(prediction_rows_df)),
            "n_metric_rows": int(len(fold_metrics_df)),
            "output_files": {name: str(path) for name, path in output_paths.items()},
        },
        run_dir / "manifest.yaml",
    )

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "fold_metrics": fold_metrics_df,
        "predictions": prediction_rows_df,
        "leakage_checks": leakage_df,
        "audit_summary": audit_summary_df,
        "external_validation": external_validation_df,
        "output_paths": output_paths,
    }
