"""
NBA Point Spread Prediction Model

Regression model to predict expected point differential,
then convert to spread cover probability.
"""

import os
import pickle
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception as e:
    xgb = None
    HAS_XGBOOST = False


class PointSpreadModel:
    """
    Regression model for predicting NBA point differentials.

    Predicts expected point differential (home - away), then
    calculates probability of covering any given spread.
    """

    DEFAULT_PARAMS = {
        "objective": "reg:squarederror",
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
    }

    # Historical std dev of prediction error (calibrated from backtest)
    # NBA games typically have ~12 point std dev in margin
    DEFAULT_STD = 12.0

    def __init__(
        self,
        params: dict = None,
        prediction_std: float = None,
    ):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.prediction_std = prediction_std or self.DEFAULT_STD
        self.model = None
        self.feature_columns = None
        self.feature_importance = None
        self.calibrated_std = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        feature_columns: List[str] = None,
    ):
        """
        Train the regression model.

        Args:
            X: Training features
            y: Point differential (home - away)
            X_val: Validation features
            y_val: Validation point differentials
            feature_columns: Columns to use as features
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost required for PointSpreadModel")

        # Store feature columns
        if feature_columns:
            self.feature_columns = feature_columns
            X = X[feature_columns]
            if X_val is not None:
                X_val = X_val[feature_columns]
        else:
            self.feature_columns = X.columns.tolist()

        # Handle missing values
        X = X.fillna(X.mean())
        if X_val is not None:
            X_val = X_val.fillna(X.mean())

        # Create and train model
        self.model = xgb.XGBRegressor(**self.params)

        if X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Calibrate prediction uncertainty from validation set
            val_preds = self.model.predict(X_val)
            errors = y_val.values - val_preds
            self.calibrated_std = np.std(errors)
            logger.info(f"Calibrated prediction std: {self.calibrated_std:.2f} points")
        else:
            self.model.fit(X, y)
            self.calibrated_std = self.prediction_std

        # Store feature importance
        self.feature_importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

        logger.info(f"PointSpreadModel trained")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected point differential (home - away)."""
        X = X[self.feature_columns].fillna(X[self.feature_columns].mean())
        return self.model.predict(X)

    def predict_spread_prob(
        self,
        X: pd.DataFrame,
        spread: float,
    ) -> np.ndarray:
        """
        Predict probability of home team covering a given spread.

        Args:
            X: Features
            spread: The spread line (negative = home favored)
                   e.g., spread = -5.5 means home favored by 5.5

        Returns:
            Probability of home covering (actual_diff + spread > 0)
        """
        # Get expected point differential
        expected_diff = self.predict(X)

        # Use calibrated std for uncertainty
        std = self.calibrated_std or self.prediction_std

        # P(home covers) = P(actual_diff > -spread)
        # = P(actual_diff - expected > -spread - expected)
        # Assuming normal distribution of errors
        # = 1 - CDF(-spread - expected)
        # = CDF(expected + spread)

        z_score = (expected_diff + spread) / std
        cover_prob = stats.norm.cdf(z_score)

        return cover_prob

    def predict_spread_probs_vectorized(
        self,
        X: pd.DataFrame,
        spreads: np.ndarray,
    ) -> np.ndarray:
        """
        Predict cover probability for different spreads per game.

        Args:
            X: Features (n_games rows)
            spreads: Array of spreads (n_games,)

        Returns:
            Cover probabilities (n_games,)
        """
        expected_diff = self.predict(X)
        std = self.calibrated_std or self.prediction_std
        z_scores = (expected_diff + spreads) / std
        return stats.norm.cdf(z_scores)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Evaluate model performance."""
        preds = self.predict(X)

        return {
            "mae": mean_absolute_error(y, preds),
            "rmse": np.sqrt(mean_squared_error(y, preds)),
            "r2": r2_score(y, preds),
            "mean_error": np.mean(y - preds),
            "std_error": np.std(y - preds),
        }

    def save(self, path: str):
        """Save model to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_columns": self.feature_columns,
                "feature_importance": self.feature_importance,
                "calibrated_std": self.calibrated_std,
                "params": self.params,
            }, f)
        logger.info(f"PointSpreadModel saved to {path}")

    @classmethod
    def load(cls, path: str) -> "PointSpreadModel":
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(params=data["params"])
        instance.model = data["model"]
        instance.feature_columns = data["feature_columns"]
        instance.feature_importance = data["feature_importance"]
        instance.calibrated_std = data.get("calibrated_std")

        logger.info(f"PointSpreadModel loaded from {path}")
        return instance


def train_point_spread_model(
    features_path: str = "data/features/game_features.parquet",
    model_path: str = "models/point_spread_model.pkl",
) -> PointSpreadModel:
    """Train a point spread regression model."""

    # Load features
    df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(df)} games")

    # Target: point differential (home - away)
    df["point_diff"] = df["home_score"] - df["away_score"]

    # Time-based split
    df = df.sort_values("date").reset_index(drop=True)
    train_size = int(0.8 * len(df))

    train_df = df[:train_size]
    val_df = df[train_size:]

    # Feature columns (exclude target and identifiers)
    exclude = ["game_id", "date", "home_team", "away_team", "home_score",
               "away_score", "point_diff", "home_win", "spread", "total", "season"]
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]

    # Train model
    model = PointSpreadModel()
    model.fit(
        train_df[feature_cols],
        train_df["point_diff"],
        val_df[feature_cols],
        val_df["point_diff"],
        feature_columns=feature_cols,
    )

    # Evaluate
    train_metrics = model.evaluate(train_df[feature_cols], train_df["point_diff"])
    val_metrics = model.evaluate(val_df[feature_cols], val_df["point_diff"])

    logger.info(f"Train metrics: {train_metrics}")
    logger.info(f"Val metrics: {val_metrics}")

    # Save
    model.save(model_path)

    return model


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n = 500

    X = pd.DataFrame({
        "home_net_rating": np.random.randn(n) * 5,
        "away_net_rating": np.random.randn(n) * 5,
        "home_rest": np.random.choice([1, 2, 3], n),
    })

    # True point diff with home advantage
    true_diff = (
        3.0  # Home advantage
        + (X["home_net_rating"] - X["away_net_rating"]) * 0.8
        + np.random.randn(n) * 10  # Noise
    )

    # Split
    X_train, X_val = X[:400], X[400:]
    y_train, y_val = pd.Series(true_diff[:400]), pd.Series(true_diff[400:])

    # Train
    model = PointSpreadModel()
    model.fit(X_train, y_train, X_val, y_val)

    # Evaluate
    metrics = model.evaluate(X_val, y_val)
    print("\nValidation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    # Test spread probabilities
    print("\nSpread Cover Probabilities (first 5 games):")
    test_spreads = np.array([-5.5, -3.5, 2.5, -7.0, 1.0])
    probs = model.predict_spread_probs_vectorized(X_val[:5], test_spreads)
    expected = model.predict(X_val[:5])

    for i in range(5):
        print(f"  Expected diff: {expected[i]:+.1f}, Spread: {test_spreads[i]:+.1f}, Cover prob: {probs[i]:.1%}")
