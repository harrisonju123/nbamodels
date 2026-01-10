"""
NBA Game Totals Prediction Model

Regression model to predict expected total points (home + away),
then convert to over/under probability.
"""

import os
import pickle
from typing import Optional, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception as e:
    xgb = None
    HAS_XGBOOST = False


class TotalsModel:
    """
    Regression model for predicting NBA game totals.

    Predicts expected total points (home + away), then
    calculates probability of going over/under any given line.
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

    # Historical std dev of total score prediction error
    # NBA games typically have ~18 point std dev in totals
    DEFAULT_STD = 18.0

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
            y: Total points (home + away)
            X_val: Validation features
            y_val: Validation totals
            feature_columns: Columns to use as features
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost required for TotalsModel")

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
            logger.info(f"Calibrated totals std: {self.calibrated_std:.2f} points")
        else:
            self.model.fit(X, y)
            self.calibrated_std = self.prediction_std

        # Store feature importance
        self.feature_importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

        logger.info(f"TotalsModel trained")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected total points."""
        X = X[self.feature_columns].fillna(X[self.feature_columns].mean())
        return self.model.predict(X)

    def predict_over_prob(
        self,
        X: pd.DataFrame,
        line: float,
    ) -> np.ndarray:
        """
        Predict probability of game going over a given line.

        Args:
            X: Features
            line: The total line (e.g., 220.5)

        Returns:
            Probability of actual total > line
        """
        # Get expected total
        expected_total = self.predict(X)

        # Use calibrated std for uncertainty
        std = self.calibrated_std or self.prediction_std

        # P(over) = P(actual > line)
        # = 1 - CDF((line - expected) / std)
        z_score = (expected_total - line) / std
        over_prob = stats.norm.cdf(z_score)

        return over_prob

    def predict_over_probs_vectorized(
        self,
        X: pd.DataFrame,
        lines: np.ndarray,
    ) -> np.ndarray:
        """
        Predict over probability for different lines per game.

        Args:
            X: Features (n_games rows)
            lines: Array of lines (n_games,)

        Returns:
            Over probabilities (n_games,)
        """
        expected_total = self.predict(X)
        std = self.calibrated_std or self.prediction_std
        z_scores = (expected_total - lines) / std
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
        logger.info(f"TotalsModel saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TotalsModel":
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(params=data["params"])
        instance.model = data["model"]
        instance.feature_columns = data["feature_columns"]
        instance.feature_importance = data["feature_importance"]
        instance.calibrated_std = data.get("calibrated_std")

        logger.info(f"TotalsModel loaded from {path}")
        return instance


def train_totals_model(
    features_path: str = "data/features/game_features.parquet",
    model_path: str = "models/totals_model.pkl",
) -> TotalsModel:
    """Train a totals regression model."""

    # Load features
    df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(df)} games")

    # Target: total points
    df["total_points"] = df["home_score"] + df["away_score"]

    # Time-based split
    df = df.sort_values("date").reset_index(drop=True)
    train_size = int(0.8 * len(df))

    train_df = df[:train_size]
    val_df = df[train_size:]

    # Feature columns - focus on pace and scoring features
    exclude = ["game_id", "date", "home_team", "away_team", "home_score",
               "away_score", "point_diff", "home_win", "spread", "total",
               "total_points", "season"]
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]

    # Train model
    model = TotalsModel()
    model.fit(
        train_df[feature_cols],
        train_df["total_points"],
        val_df[feature_cols],
        val_df["total_points"],
        feature_columns=feature_cols,
    )

    # Evaluate
    train_metrics = model.evaluate(train_df[feature_cols], train_df["total_points"])
    val_metrics = model.evaluate(val_df[feature_cols], val_df["total_points"])

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
        "home_pace": 100 + np.random.randn(n) * 3,
        "away_pace": 100 + np.random.randn(n) * 3,
        "home_off_rating": 110 + np.random.randn(n) * 5,
        "away_off_rating": 110 + np.random.randn(n) * 5,
        "home_def_rating": 110 + np.random.randn(n) * 5,
        "away_def_rating": 110 + np.random.randn(n) * 5,
    })

    # True total based on pace and ratings
    avg_pace = (X["home_pace"] + X["away_pace"]) / 2
    home_expected = (X["home_off_rating"] + X["away_def_rating"]) / 2 * avg_pace / 100
    away_expected = (X["away_off_rating"] + X["home_def_rating"]) / 2 * avg_pace / 100
    true_total = home_expected + away_expected + np.random.randn(n) * 15

    # Split
    X_train, X_val = X[:400], X[400:]
    y_train, y_val = pd.Series(true_total[:400]), pd.Series(true_total[400:])

    # Train
    model = TotalsModel()
    model.fit(X_train, y_train, X_val, y_val)

    # Evaluate
    metrics = model.evaluate(X_val, y_val)
    print("\nValidation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    # Test over/under probabilities
    print("\nOver/Under Probabilities (first 5 games):")
    test_lines = np.array([220.5, 225.0, 218.5, 230.0, 215.5])
    probs = model.predict_over_probs_vectorized(X_val[:5], test_lines)
    expected = model.predict(X_val[:5])

    for i in range(5):
        print(f"  Expected: {expected[i]:.1f}, Line: {test_lines[i]:.1f}, Over prob: {probs[i]:.1%}")
