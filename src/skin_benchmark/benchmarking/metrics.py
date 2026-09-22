"""Shared regression metrics for benchmark and external-validation reports."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def pearson_correlation(values_a: pd.Series, values_b: pd.Series) -> float:
    """Return Pearson correlation, or NaN when either input has no variation.

    Input:
    - values_a, values_b: paired numeric series with the same row order

    Output:
    - Pearson r as a float; NaN when a correlation is undefined
    """

    if values_a.nunique(dropna=False) <= 1 or values_b.nunique(dropna=False) <= 1:
        return float("nan")
    return float(values_a.corr(values_b, method="pearson"))


def spearman_correlation(values_a: pd.Series, values_b: pd.Series) -> float:
    """Return Spearman rank correlation, or NaN when either input has no variation.

    Input:
    - values_a, values_b: paired numeric series with the same row order

    Output:
    - Spearman rho as a float; NaN when a correlation is undefined
    """

    if values_a.nunique(dropna=False) <= 1 or values_b.nunique(dropna=False) <= 1:
        return float("nan")
    return float(values_a.corr(values_b, method="spearman"))


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the shared regression metric bundle used by benchmark-style evaluations.

    Input:
    - y_true: observed target values
    - y_pred: predicted or comparison target values aligned row-by-row to `y_true`

    Output:
    - metric dictionary containing RMSE, MAE, R2, Pearson r, and Spearman rho
    """

    residual = y_true - y_pred
    rmse = math.sqrt(float(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    y_true_series = pd.Series(y_true)
    y_pred_series = pd.Series(y_pred)
    target_mean = float(np.mean(y_true))
    total_ss = float(np.sum((y_true - target_mean) ** 2))
    residual_ss = float(np.sum((y_true - y_pred) ** 2))
    r2 = float("nan") if total_ss == 0.0 else 1.0 - (residual_ss / total_ss)
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": pearson_correlation(y_true_series, y_pred_series),
        "spearman_r": spearman_correlation(y_true_series, y_pred_series),
    }


def signed_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the mean signed prediction difference `y_pred - y_true`.

    Input:
    - y_true: observed target values
    - y_pred: predicted or comparison target values aligned row-by-row to `y_true`

    Output:
    - average signed bias, where positive means the comparison is numerically higher
    """

    return float(np.mean(y_pred - y_true))
