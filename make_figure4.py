# Fig 4 / S3_Table reproduction script — generates the trio case-study predictions and comparison panel

"""Fig 4 + S3_Table (cosmetic active ingredient trio case study) reproduction script.

Reuses the persisted fold models produced by the Table 3 reproduction (verify_tables.py
Part B) to run inference on the trio of three compounds (Quercetin/Phloretin/Serine) and
generate the Fig 4 panel PNG and the S3 metric CSV. No additional training is performed.

Usage:
    python make_figure4.py --benchmark-run-dir <run_benchmark output directory>

Input:
- --benchmark-run-dir: the run directory produced by run_benchmark.py.
  It must contain artifacts/target_stratified_10fold/<model_id>/fold_<n>_{estimator,preprocessor}.joblib
  and fold_<n>_feature_schema.yaml.
- outputs/features/hnh/trio_*.csv: frozen trio feature CSVs with 3 rows each (in the same
  directory as this file)
- outputs/features/hnh/trio_labels.csv: hnh_logkp measured values for the trio of 3 compounds

Output:
- _regen/figure4/figure4_panel.png: Fig 4 panel (5 models x trio scatter)
- _regen/figure4/table_s3_metrics.csv: S3 metric (model_id, pearson_r, spearman_r, rmse)
Both outputs are gitignore targets and are not committed.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# --- Path anchor: resolved relative to this script's location (supports isolated testing) ---
BASE = Path(__file__).resolve().parent

# Add the submission package to PYTHONPATH so skin_benchmark can be imported.
sys.path.insert(0, str(BASE / "src"))

from skin_benchmark.benchmarking.metrics import compute_regression_metrics  # noqa: E402


# ---------------------------------------------------------------------------
# Constant definitions
# ---------------------------------------------------------------------------

# Mapping of (feature_family, key_column, trio_csv_filename) for each of the 5 models.
# key_column: the column name that identifies compound identity in the feature CSV.
# - the 4 baselines: smiles_std
# - gate_lgbm (GATE-LGBM): SMILES
MODEL_FEATURE_MAP: dict[str, tuple[str, str, str]] = {
    "zeng_svr_proxy": (
        "zeng_proxy_descriptors",
        "smiles_std",
        "trio_zeng_proxy_descriptors.csv",
    ),
    "abdallah_lgbm": (
        "abdallah_descriptor_table",
        "smiles_std",
        "trio_abdallah_descriptor_table.csv",
    ),
    "waters_fragment_linear": (
        "waters_functional_groups",
        "smiles_std",
        "trio_waters_functional_groups.csv",
    ),
    "fpadmet_rf": (
        "fpadmet_pubchem",
        "smiles_std",
        "trio_fpadmet_pubchem.csv",
    ),
    "gate_lgbm": (
        "gate_predicted_properties",
        "SMILES",
        "trio_gate_predicted_properties.csv",
    ),
}

# Display order of the 5 models
MODEL_ORDER = [
    "zeng_svr_proxy",
    "abdallah_lgbm",
    "waters_fragment_linear",
    "fpadmet_rf",
    "gate_lgbm",
]

# Manuscript display names (used in figure titles)
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "zeng_svr_proxy": "Zeng SVR",
    "abdallah_lgbm": "Abdallah LGBM",
    "waters_fragment_linear": "Waters Linear",
    "fpadmet_rf": "FP-ADMET RF",
    "gate_lgbm": "GATE-LGBM",
}

SPLIT_METHOD = "target_stratified_10fold"
HNH_FEATURE_DIR = BASE / "outputs" / "features" / "hnh"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _load_feature_schema_columns(run_dir: Path, model_id: str) -> list[str]:
    """Read feature_columns from fold_1_feature_schema.yaml.

    Used to guarantee the column order expected by the persisted preprocessor.

    Input:
    - run_dir: persisted benchmark run directory
    - model_id: model ID string

    Output:
    - list of feature_columns (matches the preprocessor's input order)

    Raises:
    - FileNotFoundError: when the schema yaml does not exist
    """
    schema_path = (
        run_dir / "artifacts" / SPLIT_METHOD / model_id / "fold_1_feature_schema.yaml"
    )
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found: {schema_path}")
    with schema_path.open(encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    return list(schema["feature_columns"])


def _load_trio_features(
    model_id: str,
    feature_columns: list[str],
    trio_smiles: list[str],
) -> pd.DataFrame:
    """Load the frozen trio feature CSV and reorder its columns to match feature_columns.

    Input:
    - model_id: model ID string
    - feature_columns: list of feature_columns read from fold_1_feature_schema.yaml
    - trio_smiles: expected trio SMILES list (smiles_std-based strings)

    Output:
    - DataFrame (3 rows) indexed by SMILES string, with columns ordered by feature_columns

    Raises:
    - FileNotFoundError: when the trio feature CSV does not exist
    - ValueError: when a trio SMILES is missing from the CSV, or feature_columns mismatch
    """
    _feature_family, key_col, trio_csv_name = MODEL_FEATURE_MAP[model_id]
    csv_path = HNH_FEATURE_DIR / trio_csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Trio feature CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if key_col not in df.columns:
        raise ValueError(f"Key column '{key_col}' not found in {csv_path}")

    df = df.set_index(key_col)

    # Confirm that all trio SMILES are present (prevent silent drop)
    missing = [s for s in trio_smiles if s not in df.index]
    if missing:
        raise ValueError(
            f"Trio SMILES missing from {trio_csv_name}: {missing}"
        )

    # Filter to the trio order (preserving the order of trio_smiles)
    trio_df = df.reindex(trio_smiles)

    # Reindex to the same column order as the feature schema (guarantees the preprocessor's expected order)
    trio_df = trio_df.reindex(columns=feature_columns)

    # If NaN appears, it means a feature mismatch -> raise an explicit error
    if trio_df.isnull().any().any():
        nan_cols = trio_df.columns[trio_df.isnull().any()].tolist()
        raise ValueError(
            f"{model_id}: columns in feature_columns missing from the trio CSV -> NaN produced: {nan_cols[:10]}"
        )

    return trio_df


def _predict_ensemble(
    run_dir: Path,
    model_id: str,
    trio_features: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Run ensemble inference using the 10 persisted fold models.

    Implements the same logic as predict_ensemble_from_persisted_folds in the original
    inference.py, but inline. Performs only transform/predict, with no additional training.

    Input:
    - run_dir: persisted benchmark run directory
    - model_id: model ID string
    - trio_features: 3-row DataFrame indexed by SMILES, with columns=feature_columns

    Output:
    - dict: 'pred_mean' (ndarray len=3), 'pred_fold_<n>' (prediction ndarray per fold)
    """
    artifact_dir = run_dir / "artifacts" / SPLIT_METHOD / model_id
    if not artifact_dir.exists():
        raise FileNotFoundError(
            f"Persisted artifact directory not found: {artifact_dir}"
        )

    # Determine fold indices from the list of fold_*_estimator.joblib files
    estimator_paths = sorted(artifact_dir.glob("fold_*_estimator.joblib"))
    if not estimator_paths:
        raise FileNotFoundError(
            f"No fold_*_estimator.joblib in {artifact_dir}"
        )
    fold_indices = sorted(
        int(p.stem.split("_")[1]) for p in estimator_paths
    )

    n_rows = len(trio_features)
    per_fold = np.empty((n_rows, len(fold_indices)), dtype=float)

    for col_idx, fold_idx in enumerate(fold_indices):
        est_path = artifact_dir / f"fold_{fold_idx}_estimator.joblib"
        pre_path = artifact_dir / f"fold_{fold_idx}_preprocessor.joblib"
        if not est_path.exists() or not pre_path.exists():
            raise FileNotFoundError(
                f"Missing fold {fold_idx} artifacts: "
                f"estimator={est_path.exists()}, preprocessor={pre_path.exists()}"
            )
        estimator = joblib.load(est_path)
        preprocessor = joblib.load(pre_path)
        # preprocessor.transform may expect a DataFrame input, so pass it as-is
        transformed = preprocessor.transform(trio_features)
        # Convert to numpy array to match the training contract
        if isinstance(transformed, pd.DataFrame):
            transformed = transformed.to_numpy(dtype=float)
        else:
            transformed = np.asarray(transformed, dtype=float)
        per_fold[:, col_idx] = estimator.predict(transformed)

    result: dict[str, np.ndarray] = {}
    for col_idx, fold_idx in enumerate(fold_indices):
        result[f"pred_fold_{fold_idx}"] = per_fold[:, col_idx]
    result["pred_mean"] = per_fold.mean(axis=1)
    return result


def _render_panel(
    model_order: list[str],
    model_pred_map: dict[str, np.ndarray],
    compound_display_names: list[str],
    y_true: np.ndarray,
    output_path: Path,
) -> None:
    """Render the 5-model trio scatter as a 2x3 grid panel and save it.

    Input:
    - model_order: list of model IDs (display order)
    - model_pred_map: dict of model_id -> pred_mean array
    - compound_display_names: list of trio compound display names (3 items)
    - y_true: array of HnH experimental values (n=3)
    - output_path: PNG path to save to

    Output: none (writes a file)
    """
    n_cols = 3
    n_rows = 2  # ceil(5/3) = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.4, 9.6))
    axes_flat = axes.flatten()

    # Combine all model predictions and y_true to compute the global axis range once
    # -> applies the same scale to every subplot so models can be compared directly
    all_vals = np.concatenate([y_true, *[model_pred_map[m] for m in model_order]])
    min_v, max_v = float(all_vals.min()), float(all_vals.max())
    padding = max((max_v - min_v) * 0.1, 0.1)
    lo, hi = min_v - padding, max_v + padding

    for idx, model_id in enumerate(model_order):
        ax = axes_flat[idx]
        y_pred = model_pred_map[model_id]
        display_name = MODEL_DISPLAY_NAMES[model_id]

        ax.scatter(y_true, y_pred, s=42, alpha=0.85, color="#2f6f9f")
        ax.plot([lo, hi], [lo, hi], color="#444444", linewidth=1.0, linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        # Compound annotation
        for yt, yp, name in zip(y_true, y_pred, compound_display_names):
            ax.annotate(
                str(name)[:14],
                xy=(yt, yp),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color="#333333",
            )

        # Stats box (bottom right)
        pearson = float(pd.Series(y_true).corr(pd.Series(y_pred), method="pearson"))
        rmse_val = float(math.sqrt(np.mean((y_true - y_pred) ** 2)))
        ax.text(
            0.96,
            0.04,
            f"n={len(y_true)}\nPearson r={pearson:.3f}\nRMSE={rmse_val:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
        )

        ax.set_title(display_name, fontsize=9)
        ax.set_xlabel(r"Experimental $\log K_\mathrm{p}$ (cm/s)", fontsize=9)
        ax.set_ylabel(r"Predicted $\log K_\mathrm{p}$ (cm/s)", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

    # Leave the remaining 6th subplot empty (no summary panel)
    for ax in axes_flat[len(model_order):]:
        ax.set_axis_off()

    fig.suptitle(
        f"Trio ({', '.join(compound_display_names)}) — baseline vs GATE-LGBM",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Fig 4 panel saved: {output_path}")


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def run(benchmark_run_dir: Path) -> Path:
    """Run trio inference and generate the Fig 4 panel and the S3 metric CSV.

    Input:
    - benchmark_run_dir: run directory produced by run_benchmark.py (an absolute path is recommended)

    Output:
    - path to the generated metric CSV (Path)
    """
    run_dir = Path(benchmark_run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"benchmark-run-dir does not exist: {run_dir}")

    # Load trio labels
    labels_path = HNH_FEATURE_DIR / "trio_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"trio_labels.csv not found: {labels_path}")
    labels_df = pd.read_csv(labels_path)
    # Trio order: preserve the order of the compound column as-is
    trio_smiles = labels_df["smiles_std"].tolist()       # 3 SMILES strings
    trio_compounds = labels_df["compound"].tolist()      # 3 compound names
    y_true = labels_df["hnh_logkp"].to_numpy(dtype=float)  # 3 experimental values

    # Compound display names (already English in trio_labels.csv, used as-is)
    compound_display = list(trio_compounds)

    print(f"[INFO] benchmark run dir: {run_dir.name}")
    print(f"[INFO] trio SMILES: {trio_smiles}")

    # Run ensemble inference for each of the 5 models
    model_pred_map: dict[str, np.ndarray] = {}  # model_id -> pred_mean (len=3)
    metric_rows: list[dict] = []

    for model_id in MODEL_ORDER:
        print(f"[INFO] Inference: {model_id} ...")

        # 1. feature schema (the preprocessor's expected column order)
        feature_columns = _load_feature_schema_columns(run_dir, model_id)

        # 2. load trio features (columns ordered by feature_columns)
        trio_features = _load_trio_features(model_id, feature_columns, trio_smiles)

        # 3. fold ensemble inference
        preds = _predict_ensemble(run_dir, model_id, trio_features)

        # 4. map pred_mean to the smiles key and order by trio_smiles
        # trio_features.index is already in trio_smiles order -> the arrays in preds are in the same order
        pred_mean = preds["pred_mean"]
        model_pred_map[model_id] = pred_mean

        # 5. compute metrics
        metrics = compute_regression_metrics(y_true, pred_mean)
        metric_rows.append({
            "model_id": model_id,
            "pearson_r": metrics["pearson_r"],
            "spearman_r": metrics["spearman_r"],
            "rmse": metrics["rmse"],
        })
        print(
            f"       pearson_r={metrics['pearson_r']:.4f} "
            f"spearman_r={metrics['spearman_r']:.4f} "
            f"rmse={metrics['rmse']:.4f}"
        )

    # Output directory: BASE/_regen/figure4/
    out_dir = BASE / "_regen" / "figure4"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the Fig 4 panel PNG
    panel_path = out_dir / "figure4_panel.png"
    _render_panel(MODEL_ORDER, model_pred_map, compound_display, y_true, panel_path)

    # Save the S3 metric CSV
    metrics_df = pd.DataFrame(metric_rows)
    metrics_csv = out_dir / "table_s3_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"[INFO] S3 metric CSV saved: {metrics_csv}")

    return metrics_csv


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fig 4 / S3_Table reproduction: trio (3 compounds) inference + panel generation"
    )
    parser.add_argument(
        "--benchmark-run-dir",
        required=True,
        metavar="DIR",
        help=(
            "Run directory produced by run_benchmark.py. "
            "Must contain the fold joblib files for all 5 models under artifacts/target_stratified_10fold/."
        ),
    )
    args = parser.parse_args()

    try:
        run(Path(args.benchmark_run_dir))
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
