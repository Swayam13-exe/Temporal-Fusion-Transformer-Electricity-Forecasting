"""Evaluation metrics shared across all baselines and the TFT model."""
import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Symmetric MAPE. Bounded 0-200% by construction (divides by |actual| +
    |predicted| instead of just |actual|), so it doesn't explode on rows
    where the true load is at or near zero the way plain MAPE does.
    """
    denom = np.clip(np.abs(y_true) + np.abs(y_pred), eps, None)
    return float(np.mean(2 * np.abs(y_pred - y_true) / denom) * 100)


def pinball_loss(y_true: np.ndarray, y_pred_quantile: np.ndarray, quantile: float) -> float:
    diff = y_true - y_pred_quantile
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def quantile_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of actuals falling within [lower, upper] -- calibration check.
    For a well-calibrated p10/p90 band this should be close to 0.80.
    """
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def error_by_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """y_true, y_pred shape: (n_series, horizon). Returns MAE per horizon step."""
    return np.mean(np.abs(y_true - y_pred), axis=0)


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
    }