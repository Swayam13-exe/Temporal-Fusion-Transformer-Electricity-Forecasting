import numpy as np
from src.metrics import mae, rmse, mape, smape, pinball_loss, quantile_coverage, summarize


def test_mae_zero_when_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0


def test_rmse_zero_when_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_mape_reasonable():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    result = mape(y_true, y_pred)
    assert 9.9 <= result <= 10.1


def test_smape_zero_when_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert smape(y, y) == 0.0


def test_smape_bounded_even_near_zero():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([5.0, 10.0])
    result = smape(y_true, y_pred)
    assert 0.0 <= result <= 200.0  # would be inf/huge with plain MAPE


def test_pinball_loss_symmetric_at_median():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    loss = pinball_loss(y_true, y_pred, quantile=0.5)
    assert loss > 0


def test_quantile_coverage_full_when_wide_band():
    y_true = np.array([5.0, 10.0, 15.0])
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([100.0, 100.0, 100.0])
    assert quantile_coverage(y_true, lower, upper) == 1.0


def test_quantile_coverage_zero_when_band_misses():
    y_true = np.array([5.0, 10.0, 15.0])
    lower = np.array([100.0, 100.0, 100.0])
    upper = np.array([200.0, 200.0, 200.0])
    assert quantile_coverage(y_true, lower, upper) == 0.0


def test_summarize_returns_expected_keys():
    y = np.array([1.0, 2.0, 3.0])
    result = summarize(y, y)
    assert set(result.keys()) == {"MAE", "RMSE", "SMAPE"}