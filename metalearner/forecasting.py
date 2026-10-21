"""Univariate forecasters with a common fit(y) / predict(h) interface + rolling-origin CV."""
from __future__ import annotations

import warnings

import numpy as np

from .config import N_JOBS, RANDOM_STATE
from .models import available

MAX_HISTORY = 5000   # statistical models are refit per fold; cap history for speed


class LagForecaster:
    """Gradient-boosted trees on lagged values, recursive multi-step forecasting.
    Uses XGBoost when installed, otherwise sklearn's HistGradientBoosting (same knobs)."""

    def __init__(self, n_lags=5, **p):
        self.n_lags, self.p = n_lags, p

    def _model(self):
        if available(["xgboost"]):
            from xgboost import XGBRegressor
            return XGBRegressor(**self.p, random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0)
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=self.p.get("n_estimators", 100), learning_rate=self.p.get("learning_rate", 0.1),
            max_depth=self.p.get("max_depth", 6), l2_regularization=self.p.get("reg_lambda", 1.0),
            min_samples_leaf=max(5, int(self.p.get("min_child_weight", 1) * 5)), random_state=RANDOM_STATE)

    def fit(self, y):
        y = np.asarray(y, float)[-50000:]
        L = self.n_lags
        X = np.column_stack([y[L - k - 1:len(y) - k - 1] for k in range(L)])
        self.m = self._model().fit(X, y[L:])
        self.hist = list(y[-L:])
        return self

    def predict(self, h):
        hist, out = list(self.hist), []
        for _ in range(h):
            x = np.array(hist[::-1][: self.n_lags])[None, :]
            v = float(self.m.predict(x)[0])
            out.append(v)
            hist.append(v)
        return np.array(out)


class ARIMAForecaster:
    def __init__(self, p=1, d=1, q=1, P=0, D=0, Q=0, s=0):
        self.order, self.seasonal = (p, d, q), (P, D, Q, s) if s else (0, 0, 0, 0)

    def fit(self, y):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = SARIMAX(np.asarray(y, float)[-MAX_HISTORY:], order=self.order,
                               seasonal_order=self.seasonal).fit(disp=False)
        return self

    def predict(self, h):
        return np.asarray(self.res.forecast(h))


class ProphetForecaster:
    def __init__(self, **p):
        self.p = p

    def fit(self, y):
        import pandas as pd
        from prophet import Prophet
        y = np.asarray(y, float)[-MAX_HISTORY:]
        self.n = len(y)
        df = pd.DataFrame({"ds": pd.date_range("2000-01-01", periods=self.n, freq="D"), "y": y})
        self.m = Prophet(**self.p).fit(df)
        return self

    def predict(self, h):
        fut = self.m.make_future_dataframe(periods=h, include_history=False)
        return self.m.predict(fut)["yhat"].to_numpy()


def make_forecaster(model: str, params: dict):
    if model == "XGBoost":
        return LagForecaster(**params)
    if model in ("ARIMA", "SARIMA"):
        return ARIMAForecaster(**params)
    if model == "Prophet":
        return ProphetForecaster(**params)
    raise ValueError(model)


def forecast_metric(metric: str, y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if metric == "mae":
        return float(np.mean(np.abs(y_true - y_pred)))
    if metric == "smape":
        return float(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-12)) * 100)
    if metric == "r2":
        return float(1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - y_true.mean()) ** 2) + 1e-12))
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rolling_origin_splits(n: int, horizon: int, n_splits: int = 3):
    for i in range(n_splits, 0, -1):
        end = n - (i - 1) * horizon
        yield np.arange(0, end - horizon), np.arange(end - horizon, end)
