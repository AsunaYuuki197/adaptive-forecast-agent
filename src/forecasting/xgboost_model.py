import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from src.forecasting.base import BaseModel
from src.forecasting.metrics import calculate_metrics
from src.data.feature_engineering import create_lag_features
from src.utils.logger import get_logger
import warnings

warnings.filterwarnings('ignore')
logger = get_logger(__name__)


class XGBoostQuantileModel(BaseModel):
    def __init__(self, config: dict):
        self.window_size = config.pop('window_size', 7)
        self.forecast_horizon = config.pop('forecast_horizon', 7)
        self.config = config
        self.quantiles = [0.025, 0.1, 0.5, 0.9, 0.975]
        self.models = {}

    def train_and_predict(self, data: pd.DataFrame):
        df = create_lag_features(data, window=self.window_size)
        X = df[[c for c in df.columns if 'lag' in c]]
        y = df['Close']

        split_idx = int(len(df) * 0.8)
        X_train, X_holdout = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_holdout = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(
            f"Training on {len(X_train)} samples, "
            f"validating on Holdout set of {len(X_holdout)} samples."
        )

        for q in self.quantiles:
            model = XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=q,
                **self.config
            )

            model.fit(X_train, y_train)
            self.models[q] = model

        preds_holdout = self.models[0.5].predict(X_holdout)
        metrics = calculate_metrics(y_holdout, preds_holdout)

        # Forecast the next 7 days iteratively
        last_window = data['Close'].tail(self.window_size).values
        forecasts = {
            "point": [],
            "80_ci_lower": [],
            "80_ci_upper": [],
            "95_ci_lower": [],
            "95_ci_upper": []
        }

        for _ in range(self.forecast_horizon):
            inp = last_window.reshape(1, -1)

            p_median = float(self.models[0.5].predict(inp)[0])
            forecasts["point"].append(p_median)
            forecasts["80_ci_lower"].append(float(self.models[0.1].predict(inp)[0]))
            forecasts["80_ci_upper"].append(float(self.models[0.9].predict(inp)[0]))
            forecasts["95_ci_lower"].append(float(self.models[0.025].predict(inp)[0]))
            forecasts["95_ci_upper"].append(float(self.models[0.975].predict(inp)[0]))

            # Roll window for next day's prediction
            last_window = np.roll(last_window, -1)
            last_window[-1] = p_median

        return {
            "forecasts": forecasts,
            "metrics": metrics
        }
