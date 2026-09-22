# Standalone script that verifies reproducibility of manuscript Table 1, Table 3, and Fig 4 / S3_Table
"""Table 1 (benchmark pipeline counts), Table 3 (CV metrics), Fig 4 / S3_Table verification.

Part A — Table 1:
  Re-runs run_merge.py into a _regen/ subdirectory and verifies that the
  resulting counts match reference/table1_counts.yaml exactly.  Also checks
  that the regenerated benchmark (smiles_std, target_logkp) set is identical
  to the frozen outputs/final/benchmark_conflict_filtered.csv.

Part B — Table 3:
  Runs run_benchmark.py (5 models × 2 splits) against the frozen feature and
  split inputs, reads the produced summary_by_model_and_split.csv, and
  asserts 3-decimal match against reference/table3_metrics.csv.

Part C — Fig 4 / S3_Table:
  Reuses the persisted fold models from Part B to run make_figure4.py,
  which does ensemble inference on the frozen trio feature CSVs (no retraining).
  Asserts 3-decimal match of pearson_r, spearman_r, rmse against
  reference/table_s3_metrics.csv for all 5 models.

Exit code: 0 if all parts PASS, 1 otherwise.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

# --- All paths are resolved relative to this file's location (supports isolated testing) ---
BASE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Part A helpers
# ---------------------------------------------------------------------------

def _run_merge(regen_output_dir: Path, regen_report_dir: Path) -> None:
    """Run run_merge.py as a subprocess and save the regenerated artifacts under _regen/.

    Input:
    - regen_output_dir: _regen/outputs path (kept separate from the frozen outputs/)
    - regen_report_dir: _regen/reports path
    """
    regen_output_dir.mkdir(parents=True, exist_ok=True)
    regen_report_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(BASE / "run_merge.py"),
        "--output-dir", str(regen_output_dir),
        "--report-dir", str(regen_report_dir),
    ]
    result = subprocess.run(cmd, cwd=str(BASE), capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"run_merge.py failed with exit code {result.returncode}")


def _verify_table1(regen_final_dir: Path) -> list[str]:
    """Compare the regenerated artifacts against reference/table1_counts.yaml.

    Input:
    - regen_final_dir: _regen/outputs/final path

    Output:
    - list of mismatch messages (empty means PASS)
    """
    ref_path = BASE / "reference" / "table1_counts.yaml"
    with open(ref_path, encoding="utf-8") as f:
        ref = yaml.safe_load(f)

    # parse: number of rows in merged_all_records.csv
    merged_all = pd.read_csv(regen_final_dir / "merged_all_records.csv")
    count_parse = len(merged_all)

    # standardize: number of rows with is_benchmark_eligible == True
    count_standardize = int((merged_all["is_benchmark_eligible"] == True).sum())

    # dedup: number of rows in benchmark_row_evidence.csv
    evidence = pd.read_csv(regen_final_dir / "benchmark_row_evidence.csv")
    count_dedup = len(evidence)

    # pooled: number of rows in benchmark_with_provenance.csv
    provenance = pd.read_csv(regen_final_dir / "benchmark_with_provenance.csv")
    count_pooled = len(provenance)

    # final: number of rows in benchmark_conflict_filtered.csv
    conflict_filtered = pd.read_csv(regen_final_dir / "benchmark_conflict_filtered.csv")
    count_final = len(conflict_filtered)

    actual = {
        "parse": count_parse,
        "standardize": count_standardize,
        "dedup": count_dedup,
        "pooled": count_pooled,
        "final": count_final,
    }

    failures: list[str] = []
    for key in ("parse", "standardize", "dedup", "pooled", "final"):
        if actual[key] != ref[key]:
            failures.append(
                f"  Table1[{key}]: expected={ref[key]}, got={actual[key]}"
            )

    # Check identity between the regenerated benchmark and the frozen benchmark
    frozen_path = BASE / "outputs" / "final" / "benchmark_conflict_filtered.csv"
    frozen = pd.read_csv(frozen_path)

    # Compare as a set of (smiles_std, target_logkp)
    regen_set = set(
        zip(conflict_filtered["smiles_std"], conflict_filtered["target_logkp"].round(6))
    )
    frozen_set = set(
        zip(frozen["smiles_std"], frozen["target_logkp"].round(6))
    )
    if regen_set != frozen_set:
        only_regen = regen_set - frozen_set
        only_frozen = frozen_set - regen_set
        failures.append(
            f"  Table1[benchmark_identity]: regen vs frozen mismatch. "
            f"Only in regen: {len(only_regen)} rows. Only in frozen: {len(only_frozen)} rows."
        )

    return failures


# ---------------------------------------------------------------------------
# Part B helpers
# ---------------------------------------------------------------------------

MODELS_5 = [
    "zeng_svr_proxy",
    "abdallah_lgbm",
    "waters_fragment_linear",
    "fpadmet_rf",
    "gate_lgbm",
]
SPLITS_2 = ["random_10fold", "target_stratified_10fold"]
METRICS_TO_CHECK = ["rmse_mean", "rmse_std", "r2_mean", "pearson_r_mean"]
ROUND_DIGITS = 3


def _run_benchmark(runs_dir: Path, run_id_prefix: str) -> Path:
    """Run run_benchmark.py as a subprocess and generate the metrics results.

    Input:
    - runs_dir: path to pass as --runs-dir
    - run_id_prefix: string to pass as --run-id (a timestamp is appended automatically)

    Output:
    - path to the actually generated run directory (confirmed via glob)
    """
    cmd = [
        sys.executable,
        str(BASE / "run_benchmark.py"),
        "--run-id", run_id_prefix,
        "--runs-dir", str(runs_dir),
        "--split-methods", "random_10fold", "target_stratified_10fold",
        "--folds", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
        "--models",
    ] + MODELS_5 + ["--paper-sanity", "off"]

    result = subprocess.run(cmd, cwd=str(BASE), capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"run_benchmark.py failed with exit code {result.returncode}")

    # Find the actual timestamped directory (e.g. verify_table3_2606011234)
    matches = sorted(runs_dir.glob(f"{run_id_prefix}_*"))
    if not matches:
        raise FileNotFoundError(
            f"No run directory matching '{run_id_prefix}_*' found under {runs_dir}"
        )
    # Use the most recent one (last in alphabetical order)
    return matches[-1]


def _verify_table3(run_dir: Path) -> list[str]:
    """Compare the generated run's summary CSV against reference/table3_metrics.csv.

    Input:
    - run_dir: path to the run directory produced by run_benchmark.py

    Output:
    - list of mismatch messages (empty means PASS)
    """
    summary_path = run_dir / "metrics" / "summary_by_model_and_split.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary_by_model_and_split.csv not found at {summary_path}")

    new_df = pd.read_csv(summary_path)
    ref_df = pd.read_csv(BASE / "reference" / "table3_metrics.csv")

    failures: list[str] = []
    for model_id in MODELS_5:
        for split_method in SPLITS_2:
            new_row = new_df.loc[
                (new_df["model_id"] == model_id) & (new_df["split_method"] == split_method)
            ]
            ref_row = ref_df.loc[
                (ref_df["model_id"] == model_id) & (ref_df["split_method"] == split_method)
            ]

            if new_row.empty:
                failures.append(
                    f"  Table3[{model_id},{split_method}]: missing row in new run summary"
                )
                continue
            if ref_row.empty:
                failures.append(
                    f"  Table3[{model_id},{split_method}]: missing row in reference CSV"
                )
                continue

            for metric in METRICS_TO_CHECK:
                new_val = float(new_row[metric].iloc[0])
                ref_val = float(ref_row[metric].iloc[0])
                new_round = round(new_val, ROUND_DIGITS)
                ref_round = round(ref_val, ROUND_DIGITS)
                if new_round != ref_round:
                    failures.append(
                        f"  Table3[{model_id},{split_method},{metric}]: "
                        f"expected={ref_round} (ref={ref_val:.6f}), "
                        f"got={new_round} (new={new_val:.6f})"
                    )

    return failures


# ---------------------------------------------------------------------------
# Part C helpers
# ---------------------------------------------------------------------------

S3_METRICS_TO_CHECK = ["pearson_r", "spearman_r", "rmse"]


def _run_figure4(run_dir: Path) -> Path:
    """Run make_figure4.py as a subprocess and generate the trio inference metric CSV.

    Input:
    - run_dir: path to the persisted benchmark run directory produced by Part B

    Output:
    - path to the generated metric CSV (_regen/figure4/table_s3_metrics.csv)
    """
    cmd = [
        sys.executable,
        str(BASE / "make_figure4.py"),
        "--benchmark-run-dir", str(run_dir),
    ]
    result = subprocess.run(cmd, cwd=str(BASE), capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"make_figure4.py failed with exit code {result.returncode}")

    metrics_csv = BASE / "_regen" / "figure4" / "table_s3_metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"make_figure4.py did not produce a metric CSV: {metrics_csv}")
    return metrics_csv


def _verify_figure4(run_dir: Path) -> list[str]:
    """Run make_figure4.py and compare the generated metrics against the reference at 3-decimal rounding.

    Input:
    - run_dir: path to the persisted benchmark run directory produced by Part B

    Output:
    - list of mismatch messages (empty means PASS)
    """
    metrics_csv = _run_figure4(run_dir)

    new_df = pd.read_csv(metrics_csv)
    ref_df = pd.read_csv(BASE / "reference" / "table_s3_metrics.csv")

    failures: list[str] = []
    for model_id in MODELS_5:
        new_row = new_df.loc[new_df["model_id"] == model_id]
        ref_row = ref_df.loc[ref_df["model_id"] == model_id]

        if new_row.empty:
            failures.append(f"  Fig4[{model_id}]: missing row in make_figure4 output")
            continue
        if ref_row.empty:
            failures.append(f"  Fig4[{model_id}]: missing row in reference CSV")
            continue

        for metric in S3_METRICS_TO_CHECK:
            new_val = float(new_row[metric].iloc[0])
            ref_val = float(ref_row[metric].iloc[0])
            new_round = round(new_val, ROUND_DIGITS)
            ref_round = round(ref_val, ROUND_DIGITS)
            if new_round != ref_round:
                failures.append(
                    f"  Fig4[{model_id},{metric}]: "
                    f"expected={ref_round} (ref={ref_val:.6f}), "
                    f"got={new_round} (new={new_val:.6f})"
                )

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all verification parts in order and print the results."""

    all_pass = True

    # --- Part A: Table 1 ---
    print("=== Part A: Table 1 (benchmark pipeline counts) ===")
    regen_dir = BASE / "_regen"
    regen_output_dir = regen_dir / "outputs"
    regen_report_dir = regen_dir / "reports"
    regen_final_dir = regen_output_dir / "final"

    try:
        print("Running run_merge.py into _regen/ ...")
        _run_merge(regen_output_dir, regen_report_dir)
        failures_a = _verify_table1(regen_final_dir)
    except Exception as exc:
        print(f"Table 1: FAIL (exception: {exc})")
        all_pass = False
    else:
        if failures_a:
            print("Table 1: FAIL")
            for msg in failures_a:
                print(msg)
            all_pass = False
        else:
            print("Table 1: PASS")

    # --- Part B: Table 3 ---
    print("\n=== Part B: Table 3 (CV metrics) ===")
    runs_dir = BASE / "runs"
    runs_dir.mkdir(exist_ok=True)
    run_id_prefix = "verify_table3"

    actual_run_dir: Path | None = None
    try:
        print("Running run_benchmark.py (5 models × 2 splits) ...")
        actual_run_dir = _run_benchmark(runs_dir, run_id_prefix)
        print(f"Run directory: {actual_run_dir.name}")
        failures_b = _verify_table3(actual_run_dir)
    except Exception as exc:
        print(f"Table 3: FAIL (exception: {exc})")
        all_pass = False
    else:
        if failures_b:
            print("Table 3: FAIL")
            for msg in failures_b:
                print(msg)
            all_pass = False
        else:
            print("Table 3: PASS")

    # --- Part C: Fig 4 / S3_Table ---
    print("\n=== Part C: Fig 4 / S3_Table (trio case study metrics) ===")
    if actual_run_dir is None:
        print("Fig 4 / S3: SKIP (Part B did not produce a run directory)")
        all_pass = False
    else:
        try:
            failures_c = _verify_figure4(actual_run_dir)
        except Exception as exc:
            print(f"Fig 4 / S3: FAIL (exception: {exc})")
            all_pass = False
        else:
            if failures_c:
                print("Fig 4 / S3: FAIL")
                for msg in failures_c:
                    print(msg)
                all_pass = False
            else:
                print("Fig 4 / S3: PASS")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
