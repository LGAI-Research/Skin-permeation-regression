"""Figure generation for manuscript-friendly benchmark diagnostics."""

from __future__ import annotations

import argparse
import math
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.collections import PolyCollection

from skin_benchmark.config import MergeConfig, load_merge_config
from skin_benchmark.utils.io import ensure_directory

sns.set_theme(style="whitegrid", context="talk")

SOURCE_ORDER = ["huskin", "skinpix", "cheruvu", "zeng", "deeppk"]
SCATTER_MIN_OVERLAP = 3
FIGURE_STATS_DECIMALS = 2
SOURCE_PALETTE = {
    "huskin": "#9ecae1",
    "skinpix": "#6baed6",
    "cheruvu": "#4292c6",
    "zeng": "#2171b5",
    "deeppk": "#08519c",
    "pooled_included": "#f4a261",
    "pooled_filtered": "#2a9d8f",
}
DISPLAY_CATEGORY_NAMES = {
    "pooled_included": "conflict_included",
    "pooled_filtered": "final",
}
SOURCE_DISPLAY_NAMES = {
    "huskin": "HuskinDB",
    "skinpix": "SkinPiX",
    "cheruvu": "Cheruvu",
    "zeng": "Zeng",
    "deeppk": "Deep-PK",
}


def _display_source(name: str) -> str:
    return SOURCE_DISPLAY_NAMES.get(name, name)


def _default_publication_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_standardized_sources(intermediate_dir: Path) -> dict[str, pd.DataFrame]:
    standardized_paths = sorted(intermediate_dir.glob("*_standardized.csv"))
    if not standardized_paths:
        raise FileNotFoundError(f"No '*_standardized.csv' files found in {intermediate_dir}")

    standardized_by_source: dict[str, pd.DataFrame] = {}
    for path in standardized_paths:
        source_name = path.stem.removesuffix("_standardized")
        standardized_by_source[source_name] = pd.read_csv(path)
    return standardized_by_source


def _load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return pd.read_csv(path)


def build_parser() -> argparse.ArgumentParser:
    publication_dir = _default_publication_dir()
    parser = argparse.ArgumentParser(description="Regenerate manuscript figure PNGs from existing benchmark outputs.")
    parser.add_argument("--publication-dir", type=Path, default=publication_dir)
    parser.add_argument("--intermediate-dir", type=Path, default=None)
    parser.add_argument("--final-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> list[Path]:
    parser = build_parser()
    args = parser.parse_args(argv)

    publication_dir = args.publication_dir.resolve()
    intermediate_dir = (args.intermediate_dir or publication_dir / "outputs" / "intermediate").resolve()
    final_dir = (args.final_dir or publication_dir / "outputs" / "final").resolve()
    config_path = (args.config or publication_dir / "configs" / "merge_config.yaml").resolve()

    standardized_by_source = _load_standardized_sources(intermediate_dir)
    benchmark_conflict_included = _load_required_csv(final_dir / "benchmark_conflict_included.csv")
    benchmark_conflict_filtered = _load_required_csv(final_dir / "benchmark_conflict_filtered.csv")
    threshold_summary = _load_required_csv(final_dir / "conflict_threshold_summary.csv")
    merge_config = load_merge_config(config_path)

    figure_paths, _ = generate_figures(
        standardized_by_source=standardized_by_source,
        benchmark_conflict_included=benchmark_conflict_included,
        benchmark_conflict_filtered=benchmark_conflict_filtered,
        threshold_summary=threshold_summary,
        output_dir=final_dir,
        merge_config=merge_config,
    )

    for figure_path in figure_paths:
        print(figure_path)
    return figure_paths


def _save_placeholder(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.4, message, ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _ordered_sources(source_names: list[str]) -> list[str]:
    known = [source for source in SOURCE_ORDER if source in source_names]
    extras = sorted(source for source in source_names if source not in SOURCE_ORDER)
    return known + extras


def _label_with_n(name: str, count: int) -> str:
    display_name = DISPLAY_CATEGORY_NAMES.get(name) or SOURCE_DISPLAY_NAMES.get(name, name)
    return f"{display_name}\n(n={count})"


def _build_source_medians(standardized_by_source: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    source_medians: dict[str, pd.DataFrame] = {}
    for source_name, df in standardized_by_source.items():
        eligible = df.loc[
            df["is_benchmark_eligible"].fillna(False) & df["target_logkp"].notna(),
            ["smiles_std", "target_logkp"],
        ].copy()
        if eligible.empty:
            continue
        source_medians[source_name] = (
            eligible.groupby("smiles_std", dropna=False, as_index=False)["target_logkp"]
            .median()
            .rename(columns={"target_logkp": source_name})
        )
    return source_medians


def build_distribution_plot_inputs(
    standardized_by_source: dict[str, pd.DataFrame],
    benchmark_conflict_included: pd.DataFrame,
    benchmark_conflict_filtered: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str]]:
    """Build source/final distribution rows plus summary stats per category."""

    frames: list[pd.DataFrame] = []
    for source_name in SOURCE_ORDER:
        df = standardized_by_source.get(source_name)
        if df is None:
            continue
        eligible = df.loc[
            df["is_benchmark_eligible"].fillna(False) & df["target_logkp"].notna(),
            ["target_logkp"],
        ].copy()
        if eligible.empty:
            continue
        eligible["category"] = source_name
        frames.append(eligible)

    if not benchmark_conflict_included.empty:
        included = benchmark_conflict_included.loc[
            benchmark_conflict_included["target_logkp"].notna(),
            ["target_logkp"],
        ].copy()
        included["category"] = "pooled_included"
        frames.append(included)

    if not benchmark_conflict_filtered.empty:
        filtered = benchmark_conflict_filtered.loc[
            benchmark_conflict_filtered["target_logkp"].notna(),
            ["target_logkp"],
        ].copy()
        filtered["category"] = "pooled_filtered"
        frames.append(filtered)

    if not frames:
        empty = pd.DataFrame(columns=["target_logkp", "category", "category_label"])
        empty_stats = pd.DataFrame(columns=["category", "mean", "std", "n", "max_value", "category_label"])
        return empty, empty_stats, [], {}

    plot_df = pd.concat(frames, ignore_index=True)
    category_order = [
        category
        for category in SOURCE_ORDER + ["pooled_included", "pooled_filtered"]
        if category in plot_df["category"].unique()
    ]
    stats_df = (
        plot_df.groupby("category", as_index=False)["target_logkp"]
        .agg(mean="mean", std="std", n="size", max_value="max")
        .assign(std=lambda frame: frame["std"].fillna(0.0))
    )
    label_map = {
        row["category"]: _label_with_n(str(row["category"]), int(row["n"]))
        for _, row in stats_df.iterrows()
    }
    plot_df["category_label"] = plot_df["category"].map(label_map)
    stats_df["category_label"] = stats_df["category"].map(label_map)
    return plot_df, stats_df, category_order, label_map


def build_distribution_mean_markers(stats_df: pd.DataFrame, category_order: list[str]) -> pd.DataFrame:
    """Build row-local mean marker coordinates for the raincloud distribution plot."""

    rows: list[dict[str, object]] = []
    for idx, category in enumerate(category_order):
        row = stats_df.loc[stats_df["category"] == category].iloc[0]
        rows.append(
            {
                "category": category,
                "category_label": row["category_label"],
                "mean_x": float(row["mean"]),
                "ymin": idx - 0.18,
                "ymax": idx + 0.04,
            }
        )
    return pd.DataFrame(rows)


def build_pairwise_source_overlap_summary(standardized_by_source: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize pairwise source overlap using source-level median logKp values."""

    source_medians = _build_source_medians(standardized_by_source)
    rows: list[dict[str, object]] = []
    for source_a, source_b in combinations(_ordered_sources(list(source_medians)), 2):
        overlap = source_medians[source_a].merge(source_medians[source_b], on="smiles_std", how="inner")
        if overlap.empty:
            continue
        x = overlap[source_a].astype(float)
        y = overlap[source_b].astype(float)
        rows.append(
            {
                "source_a": source_a,
                "source_b": source_b,
                "n_overlap": int(len(overlap)),
                "pearson_r": float(x.corr(y, method="pearson")) if len(overlap) > 1 else pd.NA,
                "spearman_r": float(x.corr(y, method="spearman")) if len(overlap) > 1 else pd.NA,
                "mae": float((x - y).abs().mean()),
                "rmse": float(math.sqrt(((x - y) ** 2).mean())),
                "x_min": float(x.min()),
                "x_max": float(x.max()),
                "y_min": float(y.min()),
                "y_max": float(y.max()),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["source_a", "source_b", "n_overlap", "pearson_r", "spearman_r", "mae", "rmse", "x_min", "x_max", "y_min", "y_max"]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(by=["n_overlap", "source_a", "source_b"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def build_scatter_matrix_cells(
    standardized_by_source: dict[str, pd.DataFrame],
    pairwise_summary: pd.DataFrame,
    min_overlap: int = SCATTER_MIN_OVERLAP,
) -> pd.DataFrame:
    """Describe the status of each source-pair cell in the 5x5 scatter matrix."""

    source_medians = _build_source_medians(standardized_by_source)
    pair_lookup = {
        (str(row["source_a"]), str(row["source_b"])): row
        for _, row in pairwise_summary.iterrows()
    }

    rows: list[dict[str, object]] = []
    for row_idx, row_source in enumerate(SOURCE_ORDER):
        for col_idx, col_source in enumerate(SOURCE_ORDER):
            cell_type = "upper"
            n_overlap = pd.NA
            if row_idx == col_idx:
                cell_type = "diag"
            elif row_idx > col_idx:
                if row_source not in source_medians or col_source not in source_medians:
                    cell_type = "missing_source"
                else:
                    pair_row = pair_lookup.get((col_source, row_source))
                    if pair_row is None:
                        cell_type = "no_overlap"
                    else:
                        n_overlap = int(pair_row["n_overlap"])
                        cell_type = "scatter" if n_overlap >= min_overlap else "low_overlap"
            rows.append(
                {
                    "row_source": row_source,
                    "col_source": col_source,
                    "row_index": row_idx,
                    "col_index": col_idx,
                    "cell_type": cell_type,
                    "n_overlap": n_overlap,
                }
            )
    return pd.DataFrame(rows)


def generate_figures(
    standardized_by_source: dict[str, pd.DataFrame],
    benchmark_conflict_included: pd.DataFrame,
    benchmark_conflict_filtered: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    output_dir: Path,
    merge_config: MergeConfig,
) -> tuple[list[Path], pd.DataFrame]:
    """Generate manuscript-oriented diagnostic figures."""

    figures_dir = ensure_directory(output_dir / "figures")
    pairwise_summary = build_pairwise_source_overlap_summary(standardized_by_source)
    figure_paths = [
        figures_dir / "figure_logkp_source_vs_merged.png",
        figures_dir / "figure_source_overlap_scatter_grid.png",
        figures_dir / "figure_conflict_range_distribution.png",
        figures_dir / "figure_conflict_range_boxplot.png",
    ]
    _plot_source_vs_merged_distribution(
        standardized_by_source,
        benchmark_conflict_included,
        benchmark_conflict_filtered,
        figure_paths[0],
        merge_config,
    )
    _plot_source_overlap_scatter_grid(
        standardized_by_source,
        pairwise_summary,
        figure_paths[1],
        merge_config,
    )
    _plot_conflict_distribution(
        benchmark_conflict_included,
        threshold_summary,
        figure_paths[2],
    )
    _plot_conflict_boxplot(
        benchmark_conflict_included,
        threshold_summary,
        figure_paths[3],
    )
    return figure_paths, pairwise_summary


def _plot_source_vs_merged_distribution(
    standardized_by_source: dict[str, pd.DataFrame],
    benchmark_conflict_included: pd.DataFrame,
    benchmark_conflict_filtered: pd.DataFrame,
    path: Path,
    merge_config: MergeConfig,
) -> None:
    plot_df, stats_df, category_order, label_map = build_distribution_plot_inputs(
        standardized_by_source,
        benchmark_conflict_included,
        benchmark_conflict_filtered,
    )
    if plot_df.empty:
        _save_placeholder(path, "Source and benchmark logKp distributions", "No eligible or pooled rows were available.")
        return

    palette_map = {label_map[category]: SOURCE_PALETTE[category] for category in category_order}
    order = [label_map[category] for category in category_order]
    plot_df["target_logkp"] = plot_df["target_logkp"].astype(float)
    mean_markers = build_distribution_mean_markers(stats_df, category_order)

    fig, ax = plt.subplots(figsize=(15, 8))
    before_collections = [
        collection for collection in ax.collections if isinstance(collection, PolyCollection)
    ]
    sns.violinplot(
        data=plot_df,
        y="category_label",
        x="target_logkp",
        hue="category_label",
        order=order,
        palette=palette_map,
        inner=None,
        cut=0,
        density_norm="width",
        linewidth=0.9,
        saturation=0.95,
        legend=False,
        ax=ax,
    )
    after_collections = [
        collection for collection in ax.collections if isinstance(collection, PolyCollection)
    ]
    violin_bodies = after_collections[len(before_collections) :]
    for body in violin_bodies:
        for body_path in body.get_paths():
            vertices = body_path.vertices
            center_y = 0.5 * (vertices[:, 1].min() + vertices[:, 1].max())
            vertices[:, 1] = np.minimum(vertices[:, 1], center_y)
        body.set_alpha(0.75)

    rng = np.random.default_rng(42)
    for idx, category_label in enumerate(order):
        values = plot_df.loc[plot_df["category_label"] == category_label, "target_logkp"].to_numpy(dtype=float)
        if len(values) == 0:
            continue
        rain_y = idx + 0.18
        jitter = rng.uniform(-0.035, 0.035, size=len(values))
        ax.scatter(
            values,
            np.full(len(values), rain_y) + jitter,
            s=12,
            alpha=0.32,
            color="#2f2f2f",
            linewidths=0,
            zorder=3,
        )

    x_min = float(plot_df["target_logkp"].min())
    x_max = float(plot_df["target_logkp"].max())
    x_range = max(x_max - x_min, 1.0)
    annotation_x = x_max + 0.05 * x_range
    x_upper = x_max + 0.28 * x_range
    ax.set_xlim(x_min - 0.04 * x_range, x_upper)
    ax.set_ylim(len(order) - 0.35, -0.55)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)

    for _, marker in mean_markers.iterrows():
        ax.vlines(
            float(marker["mean_x"]),
            float(marker["ymin"]),
            float(marker["ymax"]),
            colors="#1f1f1f",
            linewidth=1.7,
            alpha=0.9,
            zorder=4.5,
        )

    for idx, category in enumerate(category_order):
        row = stats_df.loc[stats_df["category"] == category].iloc[0]
        ax.text(
            annotation_x,
            idx - 0.02,
            f"mean={float(row['mean']):.{FIGURE_STATS_DECIMALS}f}\nstd={float(row['std']):.{FIGURE_STATS_DECIMALS}f}",
            ha="left",
            va="center",
            fontsize=18,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )

    # ax.set_title("Source-level and final benchmark logKp distributions")
    ax.set_xlabel(f"logKp ({merge_config.canonical_target_unit})")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _clear_matrix_cell(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_source_overlap_scatter_grid(
    standardized_by_source: dict[str, pd.DataFrame],
    pairwise_summary: pd.DataFrame,
    path: Path,
    merge_config: MergeConfig,
) -> None:
    source_medians = _build_source_medians(standardized_by_source)
    if not source_medians:
        _save_placeholder(path, "Source-overlap agreement diagnostics", "No eligible source medians were available.")
        return

    all_values = pd.concat(
        [frame.iloc[:, 1].astype(float) for frame in source_medians.values()],
        ignore_index=True,
    )
    if all_values.empty:
        _save_placeholder(path, "Source-overlap agreement diagnostics", "No eligible source medians were available.")
        return

    axis_min = float(all_values.min())
    axis_max = float(all_values.max())
    if axis_min == axis_max:
        axis_min -= 0.5
        axis_max += 0.5
    pad = 0.05 * (axis_max - axis_min)
    axis_min -= pad
    axis_max += pad

    pair_lookup = {
        (str(row["source_a"]), str(row["source_b"])): row
        for _, row in pairwise_summary.iterrows()
    }
    matrix_cells = build_scatter_matrix_cells(
        standardized_by_source=standardized_by_source,
        pairwise_summary=pairwise_summary,
        min_overlap=SCATTER_MIN_OVERLAP,
    )

    fig, axes = plt.subplots(
        len(SOURCE_ORDER),
        len(SOURCE_ORDER),
        figsize=(18, 18),
        sharex=True,
        sharey=True,
    )

    for _, cell in matrix_cells.iterrows():
        row_idx = int(cell["row_index"])
        col_idx = int(cell["col_index"])
        row_source = str(cell["row_source"])
        col_source = str(cell["col_source"])
        ax = axes[row_idx, col_idx]
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)

        if row_idx == 0:
            ax.set_title(_display_source(col_source), fontsize=15, pad=10)
        if col_idx == 0:
            ax.set_ylabel(_display_source(row_source), fontsize=15)
        if row_idx == len(SOURCE_ORDER) - 1 and row_idx > col_idx:
            ax.set_xlabel(_display_source(col_source), fontsize=15)
        else:
            ax.set_xlabel("")

        cell_type = str(cell["cell_type"])
        if cell_type == "upper":
            _clear_matrix_cell(ax)
            continue

        if cell_type == "diag":
            _clear_matrix_cell(ax)
            ax.set_facecolor("#f4f4f4")
            ax.text(0.5, 0.5, _display_source(row_source), transform=ax.transAxes, ha="center", va="center", fontsize=50, fontweight="bold")
            continue

        if cell_type == "missing_source":
            _clear_matrix_cell(ax)
            ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", va="center", fontsize=15)
            continue

        if cell_type in {"no_overlap", "low_overlap"}:
            _clear_matrix_cell(ax)
            message = "no overlap"
            if cell_type == "low_overlap" and pd.notna(cell["n_overlap"]):
                message = f"n={int(cell['n_overlap'])} < {SCATTER_MIN_OVERLAP}"
            ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", fontsize=15)
            continue

        pair_row = pair_lookup[(col_source, row_source)]
        overlap = source_medians[col_source].merge(source_medians[row_source], on="smiles_std", how="inner")
        x = overlap[col_source].astype(float)
        y = overlap[row_source].astype(float)
        ax.scatter(x, y, alpha=0.6, s=28, color="#2b6f9e", edgecolors="none")
        ax.plot([axis_min, axis_max], [axis_min, axis_max], linestyle="--", linewidth=1.2, color="#888888")
        annotation = (
            f"n={int(pair_row['n_overlap'])}\n"
            f"r={float(pair_row['pearson_r']):.3f}\n"
            # f"rho={float(pair_row['spearman_r']):.3f}\n"
            f"MAE={float(pair_row['mae']):.3f}"
        )
        ax.text(
            0.03,
            0.97,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=20,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )

        if row_idx != len(SOURCE_ORDER) - 1:
            ax.tick_params(labelbottom=False)
        if col_idx != 0:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=8)

    fig.suptitle("Source-overlap agreement diagnostics", y=0.995, fontsize=30)
    fig.text(
        0.5,
        0.97,
        "Lower triangle uses eligible source-level median logKp per smiles_std before exact duplicate removal.",
        ha="center",
        va="top",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_conflict_distribution(
    benchmark_conflict_included: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    path: Path,
) -> None:
    overlap = benchmark_conflict_included.loc[benchmark_conflict_included["n_sources"].fillna(0).astype(int) > 1].copy()
    if overlap.empty:
        _save_placeholder(path, "Conflict range distribution", "No overlap compounds were available.")
        return

    main_summary = threshold_summary.loc[threshold_summary["method"] == "tukey_iqr_1.5"]
    if main_summary.empty:
        main_summary = threshold_summary.iloc[[0]]
    summary_row = main_summary.iloc[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(overlap["conflict_range"].astype(float), bins=24, kde=True, ax=ax, color="#4c78a8")
    for field, color in [("q1", "#999999"), ("median", "#222222"), ("q3", "#999999"), ("threshold", "#d62728")]:
        value = summary_row.get(field)
        if pd.notna(value):
            ax.axvline(float(value), color=color, linestyle="--", linewidth=2, label=field)

    three_iqr = threshold_summary.loc[threshold_summary["method"] == "tukey_iqr_3.0", "threshold"]
    if not three_iqr.empty and pd.notna(three_iqr.iloc[0]):
        ax.axvline(float(three_iqr.iloc[0]), color="#ff9896", linestyle=":", linewidth=2, label="tukey_3iqr")

    ax.set_title("Conflict range distribution across overlap compounds")
    ax.set_xlabel("Pooled conflict range")
    ax.set_ylabel("Count")
    subtitle = (
        f"Q1={summary_row['q1']:.3f}, Q3={summary_row['q3']:.3f}, IQR={summary_row['iqr']:.3f}, "
        f"Tukey 1.5xIQR={summary_row['threshold']:.3f}, flagged={int(summary_row['n_flagged'])}/{int(summary_row['n_overlap_compounds'])}"
    )
    ax.text(0.02, 0.98, subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_conflict_boxplot(
    benchmark_conflict_included: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    path: Path,
) -> None:
    overlap = benchmark_conflict_included.loc[benchmark_conflict_included["n_sources"].fillna(0).astype(int) > 1].copy()
    if overlap.empty:
        _save_placeholder(path, "Conflict range boxplot", "No overlap compounds were available.")
        return

    main_summary = threshold_summary.loc[threshold_summary["method"] == "tukey_iqr_1.5"]
    if main_summary.empty:
        main_summary = threshold_summary.iloc[[0]]
    threshold = main_summary.iloc[0]["threshold"]

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_df = overlap.assign(group="overlap compounds")
    sns.boxplot(data=plot_df, x="group", y="conflict_range", ax=ax, color="#d9e8f5", width=0.3)
    sns.stripplot(
        data=plot_df,
        x="group",
        y="conflict_range",
        ax=ax,
        color="#3b7ea1",
        size=4,
        alpha=0.65,
        jitter=0.25,
    )
    if pd.notna(threshold):
        ax.axhline(float(threshold), color="#d62728", linestyle="--", linewidth=2, label="Tukey 1.5xIQR")
    ax.set_title("Boxplot view of overlap conflict ranges")
    ax.set_xlabel("")
    ax.set_ylabel("Pooled conflict range")
    if pd.notna(threshold):
        ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# Run from the repository root:
# python src/skin_benchmark/reporting/figures.py
if __name__ == "__main__":
    main()
