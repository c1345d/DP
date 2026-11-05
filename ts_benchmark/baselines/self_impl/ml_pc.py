"""
Traditional ML models with Predict-then-Correct framework.

Supports: LinearRegression, Ridge, XGBoost, LightGBM, RandomForest, KNN, SGD.
Trains multi-output regression: seq_len input -> seq_len+pred_len output,
then splits into reconstruction + prediction and applies correction.

Models that natively support multi-output (LinearRegression, Ridge, RF, KNN, SGD)
train a single estimator. Tree-based models (XGBoost, LightGBM) train one estimator
per feature to avoid the "Invalid shape for label" error with large output dims.

Model name for CLI: self_impl.Regression_PC  /  self_impl.XGB_PC  /  ...
"""

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from ts_benchmark.models.model_base import ModelBase
from ts_benchmark.baselines.time_series_library.models.correction import apply_correction


class ML_PC_Base(ModelBase):
    """Base class for traditional ML models with PC correction.

    Trains a multi-output regressor that outputs seq_len+pred_len steps,
    splits into reconstruction and prediction, then applies correction.

    For estimators without native multi-output support (XGBoost, LightGBM),
    trains one estimator per feature - each predicting all out_len steps for
    that feature. This avoids the MultiOutputRegressor explosion (out_len *
    n_features separate models) and the "Invalid shape for label" error.
    """

    def __init__(self, estimator_factory, estimator_name,
                 native_multioutput=True, **kwargs):
        """
        Args:
            estimator_factory: callable that returns a new sklearn-compatible estimator
            estimator_name: string name for model_name property
            native_multioutput: True if estimator.fit(X, y_2d) works natively
        """
        self._estimator_factory = estimator_factory
        self._estimator_name = estimator_name
        self._native_multioutput = native_multioutput
        self.scaler = StandardScaler()
        self.seq_len = kwargs.get('seq_len', kwargs.get('lags', 96))
        self.pred_len = kwargs.get('horizon', kwargs.get('output_chunk_length', 96))
        self.correction_type = kwargs.get('correction_type', 'fused')
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if k in kwargs:
                self.correction_kwargs[k] = kwargs[k]
        # For native multi-output: single model
        # For per-feature: list of models (one per feature)
        self._models = None
        self._n_features = None

    @property
    def model_name(self):
        return self._estimator_name

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def forecast_fit(self, train_data: pd.DataFrame, *,
                     train_ratio_in_tv: float = 1.0, **kwargs) -> "ModelBase":
        """Train multi-output regression: seq_len -> seq_len+pred_len."""
        if 'seq_len' in kwargs:
            self.seq_len = kwargs['seq_len']
        if 'horizon' in kwargs:
            self.pred_len = kwargs['horizon']

        values = train_data.values.astype(np.float64)
        n_samples, n_features = values.shape
        self._n_features = n_features

        self.scaler.fit(values)
        scaled = self.scaler.transform(values)

        seq_len = self.seq_len
        pred_len = self.pred_len
        out_len = seq_len + pred_len
        n_windows = n_samples - seq_len - out_len + 1
        if n_windows <= 0:
            n_windows = max(1, n_samples - seq_len - pred_len + 1)
            out_len = n_samples - seq_len

        # Build X: (n_windows, seq_len * n_features)
        idx = np.arange(n_windows)[:, None] + np.arange(seq_len)[None, :]
        X = scaled[idx].reshape(n_windows, seq_len * n_features)

        if self._native_multioutput:
            # Single estimator: y shape (n_windows, out_len * n_features)
            idx_y = np.arange(n_windows)[:, None] + np.arange(seq_len, seq_len + out_len)[None, :]
            y = scaled[idx_y].reshape(n_windows, out_len * n_features)
            self._models = self._estimator_factory()
            self._models.fit(X, y)
        else:
            # Per-feature: n_features estimators, each predicting (n_windows, out_len)
            # y_3d shape: (n_windows, out_len, n_features)
            idx_y = np.arange(n_windows)[:, None] + np.arange(seq_len, seq_len + out_len)[None, :]
            y_3d = scaled[idx_y]  # (n_windows, out_len, n_features)
            self._models = []
            for f in range(n_features):
                model_f = self._estimator_factory()
                model_f.fit(X, y_3d[:, :, f])  # (n_windows, out_len)
                self._models.append(model_f)

        return self

    def forecast(self, horizon: int, series: pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict, split into reconstruction+prediction, apply correction."""
        values = series.values.astype(np.float64)
        scaled = self.scaler.transform(values)

        seq_len = self.seq_len
        pred_len = self.pred_len
        out_len = seq_len + pred_len

        X = scaled[-seq_len:].flatten().reshape(1, -1)

        if self._native_multioutput:
            y_pred_scaled = self._models.predict(X).reshape(out_len, -1)
        else:
            # Per-feature: stack predictions -> (out_len, n_features)
            y_pred_scaled = np.column_stack(
                [m.predict(X).ravel() for m in self._models]
            )

        y_pred = self.scaler.inverse_transform(y_pred_scaled)

        if self.correction_type == 'none':
            return y_pred[seq_len:seq_len + pred_len]

        # Apply correction via torch
        x_recon = y_pred[:seq_len]
        y_pred_out = y_pred[seq_len:seq_len + pred_len]
        x_true = values[-seq_len:]

        x_true_t = torch.from_numpy(x_true.copy()).float().unsqueeze(0)
        x_recon_t = torch.from_numpy(x_recon.copy()).float().unsqueeze(0)
        y_pred_t = torch.from_numpy(y_pred_out.copy()).float().unsqueeze(0)

        try:
            corrected = apply_correction(
                x_true_t, x_recon_t, y_pred_t,
                self.correction_type, **self.correction_kwargs
            )
            result = corrected.squeeze(0).detach().cpu().numpy()
        except Exception as e:
            import sys
            print(f"  [WARN] correction failed ({e}), falling back to uncorrected",
                  file=sys.stderr)
            result = y_pred_out

        return result


# ---- Native multi-output models ----

class Regression_PC(ML_PC_Base):
    """LinearRegression with PC correction."""

    def __init__(self, **kwargs):
        super().__init__(LinearRegression, "Regression_PC",
                         native_multioutput=True, **kwargs)


class Ridge_PC(ML_PC_Base):
    """Ridge regression with PC correction."""

    def __init__(self, **kwargs):
        from sklearn.linear_model import Ridge
        super().__init__(lambda: Ridge(alpha=1.0), "Ridge_PC",
                         native_multioutput=True, **kwargs)


class RF_PC(ML_PC_Base):
    """RandomForest with PC correction."""

    def __init__(self, **kwargs):
        from sklearn.ensemble import RandomForestRegressor
        super().__init__(
            lambda: RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=1),
            "RF_PC", native_multioutput=True, **kwargs
        )


class KNN_PC(ML_PC_Base):
    """K-Nearest Neighbors with PC correction."""

    def __init__(self, **kwargs):
        from sklearn.neighbors import KNeighborsRegressor
        super().__init__(lambda: KNeighborsRegressor(n_neighbors=5), "KNN_PC",
                         native_multioutput=True, **kwargs)


class SGD_PC(ML_PC_Base):
    """SGD Regressor with PC correction."""

    def __init__(self, **kwargs):
        from sklearn.linear_model import SGDRegressor
        super().__init__(
            lambda: SGDRegressor(max_iter=1000, tol=1e-3, random_state=42),
            "SGD_PC", native_multioutput=True, **kwargs
        )


# ---- Per-feature models (no native multi-output) ----

class XGB_PC(ML_PC_Base):
    """XGBoost with PC correction - one model per feature."""

    def __init__(self, **kwargs):
        from xgboost import XGBRegressor
        super().__init__(
            lambda: XGBRegressor(n_estimators=100, max_depth=5, verbosity=0, n_jobs=1),
            "XGB_PC", native_multioutput=False, **kwargs
        )


class LGBM_PC(ML_PC_Base):
    """LightGBM with PC correction - one model per feature."""

    def __init__(self, **kwargs):
        from lightgbm import LGBMRegressor
        super().__init__(
            lambda: LGBMRegressor(n_estimators=100, max_depth=5, verbosity=-1, n_jobs=1),
            "LGBM_PC", native_multioutput=False, **kwargs
        )
