"""Optional external validation availability audit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def run_external_validation_audit(publication_dir: Path, mode: str) -> pd.DataFrame:
    """Record availability of machine-readable external validation data."""

    rows: list[dict[str, object]] = []
    if mode not in {"auto", "off"}:
        raise ValueError("paper_sanity must be 'auto' or 'off'.")

    registry = [
        {
            "baseline_id": "zeng_svr_proxy",
            "local_path": publication_dir / "outputs" / "intermediate" / "zeng_standardized.csv",
            "remote_url": None,
        },
        {
            "baseline_id": "waters_fragment_linear",
            "local_path": publication_dir / "outputs" / "intermediate" / "huskin_standardized.csv",
            "remote_url": None,
        },
        {
            "baseline_id": "abdallah_lgbm",
            "local_path": None,
            "remote_url": None,
        },
        {
            "baseline_id": "fpadmet_rf",
            "local_path": None,
            "remote_url": None,
            "validation_applicable": True,
        },
        {
            "baseline_id": "gate_lgbm",
            "local_path": None,
            "remote_url": None,
            "validation_applicable": False,
        },
    ]

    for entry in registry:
        local_path = entry["local_path"]
        if mode == "off":
            rows.append(
                {
                    "baseline_id": entry["baseline_id"],
                    "status": "not_run",
                    "detail": "paper_sanity=off",
                    "local_path": None if local_path is None else str(local_path),
                    "remote_url": entry["remote_url"],
                    "n_rows": pd.NA,
                }
            )
            continue

        if not entry.get("validation_applicable", True):
            rows.append(
                {
                    "baseline_id": entry["baseline_id"],
                    "status": "not_applicable",
                    "detail": "Model proposed in this study; there is no source-paper validation dataset to check against.",
                    "local_path": None if local_path is None else str(local_path),
                    "remote_url": entry["remote_url"],
                    "n_rows": pd.NA,
                }
            )
            continue

        if isinstance(local_path, Path) and local_path.exists():
            try:
                n_rows = len(pd.read_csv(local_path))
                rows.append(
                    {
                        "baseline_id": entry["baseline_id"],
                        "status": "local_source_available",
                        "detail": "Machine-readable local source file is available for optional sanity checks.",
                        "local_path": str(local_path),
                        "remote_url": entry["remote_url"],
                        "n_rows": n_rows,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "baseline_id": entry["baseline_id"],
                        "status": "local_source_error",
                        "detail": f"Local source exists but failed to parse: {type(exc).__name__}: {exc}",
                        "local_path": str(local_path),
                        "remote_url": entry["remote_url"],
                        "n_rows": pd.NA,
                    }
                )
            continue

        rows.append(
            {
                "baseline_id": entry["baseline_id"],
                "status": "unavailable",
                "detail": "No machine-readable external source is bundled or curated for automatic validation.",
                "local_path": None if local_path is None else str(local_path),
                "remote_url": entry["remote_url"],
                "n_rows": pd.NA,
            }
        )

    return pd.DataFrame(rows)
