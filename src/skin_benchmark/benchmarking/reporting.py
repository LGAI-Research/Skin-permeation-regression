"""Benchmark report and figure generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from skin_benchmark.utils.io import ensure_directory, save_csv


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"

    working = df.fillna("")
    headers = [str(column).replace("|", "\\|") for column in working.columns]
    rows = [[str(value).replace("|", "\\|").replace("\n", " ") for value in row] for row in working.itertuples(index=False, name=None)]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([format_row(headers), separator, *[format_row(row) for row in rows]])


def create_metric_summaries(fold_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = ["rmse", "mae", "r2", "pearson_r", "spearman_r"]
    by_model_and_split = (
        fold_metrics.groupby(["model_id", "split_method"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    by_model_and_split.columns = [
        "_".join(part for part in column if part).rstrip("_") if isinstance(column, tuple) else str(column)
        for column in by_model_and_split.columns
    ]
    by_model = (
        fold_metrics.groupby(["model_id"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    by_model.columns = [
        "_".join(part for part in column if part).rstrip("_") if isinstance(column, tuple) else str(column)
        for column in by_model.columns
    ]
    return by_model, by_model_and_split


def generate_benchmark_figures(
    prediction_rows: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    ensure_directory(figures_dir)
    figure_paths: list[Path] = []

    if prediction_rows.empty:
        return figure_paths

    for split_method in sorted(prediction_rows["split_method"].unique()):
        subset = prediction_rows.loc[prediction_rows["split_method"] == split_method].copy()
        model_ids = sorted(subset["model_id"].unique())
        n_cols = 2
        n_rows = (len(model_ids) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), squeeze=False)
        axes_flat = axes.flatten()
        for axis, model_id in zip(axes_flat, model_ids):
            model_df = subset.loc[subset["model_id"] == model_id]
            axis.scatter(model_df["y_true"], model_df["y_pred"], s=12, alpha=0.7)
            lower = float(min(model_df["y_true"].min(), model_df["y_pred"].min()))
            upper = float(max(model_df["y_true"].max(), model_df["y_pred"].max()))
            axis.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1)
            axis.set_title(model_id)
            axis.set_xlabel("y_true")
            axis.set_ylabel("y_pred")
        for axis in axes_flat[len(model_ids):]:
            axis.axis("off")
        fig.suptitle(f"Parity Plot ({split_method})")
        fig.tight_layout()
        figure_path = figures_dir / f"parity_{split_method}.png"
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(figure_path)

    residual_fig, residual_axis = plt.subplots(figsize=(14, 6))
    grouped_labels: list[str] = []
    grouped_values: list[pd.Series] = []
    for (split_method, model_id), group in prediction_rows.groupby(["split_method", "model_id"]):
        grouped_labels.append(f"{split_method}\n{model_id}")
        grouped_values.append((group["y_pred"] - group["y_true"]).astype(float))
    residual_axis.boxplot(grouped_values, tick_labels=grouped_labels, orientation="vertical")
    residual_axis.set_ylabel("Residual (y_pred - y_true)")
    residual_axis.set_title("Residual Distribution by Split and Model")
    residual_axis.tick_params(axis="x", labelrotation=45)
    residual_fig.tight_layout()
    residual_path = figures_dir / "residual_boxplot.png"
    residual_fig.savefig(residual_path, dpi=300, bbox_inches="tight")
    plt.close(residual_fig)
    figure_paths.append(residual_path)
    return figure_paths


def render_benchmark_report(
    report_path: Path,
    run_id: str,
    fold_metrics: pd.DataFrame,
    summary_by_model: pd.DataFrame,
    summary_by_model_and_split: pd.DataFrame,
    leakage_checks: pd.DataFrame,
    audit_summary: pd.DataFrame,
    external_validation: pd.DataFrame,
    figure_paths: list[Path],
    external_model_status: str,
) -> None:
    considerations = [
        "Proprietary descriptor gaps can change baseline ranking, especially for Zeng-style descriptor stacks.",
        "Abdallah-style runs use the curated CDK 2.8 descriptor subset with an LGBMRegressor aligned to the paper's final model family.",
        "The Abdallah descriptor table still carries raw missing values, but all imputation and schema pruning stay inside the training fold to prevent leakage.",
        "The GATE-LGBM row reuses the shared frozen GATE predicted-property feature table and is evaluated under the same split and metric rules as the literature baselines.",
        "Random and target-stratified splits should both be inspected because rank order may change with split sensitivity.",
        "External paper sanity validation may remain unavailable when no machine-readable source is bundled or curated.",
        "Descriptor generation failure rate should remain zero because the benchmark only contains curated canonical smiles_std rows.",
    ]
    figure_lines = "\n".join(f"- `{path}`" for path in figure_paths) if figure_paths else "- `_No figures generated_`"
    report_text = f"""# Baseline Benchmark Report

## Summary
- `run_id`: `{run_id}`
- `external_model_status`: `{external_model_status}`
- executed fold rows: `{len(fold_metrics)}`
- primary metric: `rmse`

## Summary by model
{_markdown_table(summary_by_model)}

## Summary by model and split
{_markdown_table(summary_by_model_and_split)}

## Fold-level metrics
{_markdown_table(fold_metrics)}

## Leakage checks
{_markdown_table(leakage_checks)}

## Feature Provenance and Audit Summary
{_markdown_table(audit_summary)}

## Optional External Validation
{_markdown_table(external_validation)}

## Figures
{figure_lines}

## Additional Considerations
""" + "\n".join(f"- {line}" for line in considerations)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")


def write_benchmark_outputs(
    run_dir: Path,
    fold_metrics: pd.DataFrame,
    prediction_rows: pd.DataFrame,
    leakage_checks: pd.DataFrame,
    audit_summary: pd.DataFrame,
    external_validation: pd.DataFrame,
    run_id: str,
    external_model_status: str,
) -> dict[str, Path]:
    metrics_dir = ensure_directory(run_dir / "metrics")
    reports_dir = ensure_directory(run_dir / "reports")
    figures_dir = ensure_directory(reports_dir / "figures")
    external_dir = ensure_directory(run_dir / "external_validation")

    summary_by_model, summary_by_model_and_split = create_metric_summaries(fold_metrics)
    save_csv(fold_metrics, metrics_dir / "fold_metrics.csv")
    save_csv(summary_by_model, metrics_dir / "summary_by_model.csv")
    save_csv(summary_by_model_and_split, metrics_dir / "summary_by_model_and_split.csv")
    save_csv(leakage_checks, metrics_dir / "leakage_checks.csv")
    save_csv(audit_summary, metrics_dir / "proprietary_descriptor_audit.csv")
    save_csv(external_validation, external_dir / "status.csv")

    figure_paths = generate_benchmark_figures(prediction_rows, figures_dir)
    report_path = reports_dir / "benchmark_report.md"
    render_benchmark_report(
        report_path=report_path,
        run_id=run_id,
        fold_metrics=fold_metrics.round(3),
        summary_by_model=summary_by_model.round(3),
        summary_by_model_and_split=summary_by_model_and_split.round(3),
        leakage_checks=leakage_checks,
        audit_summary=audit_summary,
        external_validation=external_validation,
        figure_paths=figure_paths,
        external_model_status=external_model_status,
    )
    return {
        "fold_metrics": metrics_dir / "fold_metrics.csv",
        "summary_by_model": metrics_dir / "summary_by_model.csv",
        "summary_by_model_and_split": metrics_dir / "summary_by_model_and_split.csv",
        "leakage_checks": metrics_dir / "leakage_checks.csv",
        "audit_summary": metrics_dir / "proprietary_descriptor_audit.csv",
        "external_validation": external_dir / "status.csv",
        "report": report_path,
    }
