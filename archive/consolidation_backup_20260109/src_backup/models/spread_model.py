"""
NBA Spread Prediction Model

XGBoost-based model for predicting NBA game spreads with calibrated probabilities.
"""

import os
import pickle
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception as e:
    xgb = None
    HAS_XGBOOST = False
    import warnings
    warnings.warn(f"XGBoost not available: {e}")

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from loguru import logger

from .calibration import CalibratedModel, evaluate_calibration


class SpreadPredictionModel:
    """
    XGBoost model for NBA spread prediction.

    Predicts probability of home team covering the spread.
    """

    DEFAULT_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
    }

    def __init__(
        self,
        params: dict = None,
        calibration_method: str = "isotonic",
    ):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.calibration_method = calibration_method
        self.model = None
        self.calibrated_model = None
        self.feature_columns = None
        self.feature_importance = None

    def prepare_target(
        self,
        df: pd.DataFrame,
        spread_col: str = "spread",
        actual_diff_col: str = "point_diff",
    ) -> pd.Series:
        """
        Create target variable: did home team cover the spread?

        Args:
            df: DataFrame with game data
            spread_col: Column with betting spread (negative = home favored)
            actual_diff_col: Column with actual point differential

        Returns:
            Binary series (1 if home covered, 0 otherwise)
        """
        # Home covers if actual_diff > spread
        # e.g., spread = -5 (home favored by 5), actual = -3 (home won by 3)
        # Home did NOT cover (-3 > -5 is True, but home needed to win by MORE than 5)
        # Correction: Home covers if (actual_diff + spread) > 0
        # If spread = -5 and actual = 7, then 7 + (-5) = 2 > 0, home covered

        if spread_col not in df.columns:
            logger.warning(f"Spread column '{spread_col}' not found. Using home_win as target.")
            return df["home_win"]

        return ((df[actual_diff_col] + df[spread_col]) > 0).astype(int)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        feature_columns: List[str] = None,
        early_stopping_rounds: int = 20,
    ):
        """
        Train the model with optional validation set.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_columns: Columns to use as features
            early_stopping_rounds: Early stopping patience
        """
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

        # Create XGBoost model
        self.model = xgb.XGBClassifier(**self.params)

        # Fit with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X, y)

        # Store feature importance
        self.feature_importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

        # Fit calibrated model
        self.calibrated_model = CalibratedModel(
            self.model,
            method=self.calibration_method
        )

        if X_val is not None:
            # Use validation set for calibration
            self.calibrated_model.fit(X, y, X_val, y_val)
        else:
            # Use training set (not ideal, but works)
            self.calibrated_model.fit(X, y)

        logger.info(f"Model trained. Best iteration: {self.model.best_iteration if hasattr(self.model, 'best_iteration') else 'N/A'}")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get calibrated probability predictions."""
        X = X[self.feature_columns].fillna(X[self.feature_columns].mean())
        return self.calibrated_model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Get class predictions."""
        return (self.predict_proba(X) > 0.5).astype(int)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        implied_prob: pd.Series = None,
    ) -> dict:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True labels
            implied_prob: Implied probability from odds (for edge calculation)

        Returns:
            Dictionary with metrics
        """
        probs = self.predict_proba(X)
        preds = (probs > 0.5).astype(int)

        # Basic metrics
        metrics = {
            "accuracy": accuracy_score(y, preds),
            "roc_auc": roc_auc_score(y, probs),
        }

        # Calibration metrics
        calib = evaluate_calibration(y.values, probs)
        metrics.update({
            "brier_score": calib["brier_score"],
            "log_loss": calib["log_loss"],
            "ece": calib["ece"],
        })

        # Edge over market (if implied prob available)
        if implied_prob is not None:
            edge = probs - implied_prob.values
            metrics["avg_edge"] = edge.mean()
            metrics["edge_accuracy"] = ((edge > 0) == y).mean()

        return metrics

    def get_betting_recommendations(
        self,
        X: pd.DataFrame,
        odds_df: pd.DataFrame,
        min_edge: float = 0.03,
        kelly_fraction: float = 0.2,
    ) -> pd.DataFrame:
        """
        Get betting recommendations based on model predictions.

        Args:
            X: Game features
            odds_df: DataFrame with odds data
            min_edge: Minimum edge to bet (e.g., 0.03 = 3%)
            kelly_fraction: Fraction of Kelly criterion to use

        Returns:
            DataFrame with betting recommendations
        """
        probs = self.predict_proba(X)

        recs = X[["game_id"]].copy() if "game_id" in X.columns else pd.DataFrame()
        recs["model_prob"] = probs

        # Merge with odds
        if "implied_prob" in odds_df.columns:
            recs = recs.merge(
                odds_df[["game_id", "implied_prob", "odds"]],
                on="game_id",
                how="left"
            )

            # Calculate edge
            recs["edge"] = recs["model_prob"] - recs["implied_prob"]

            # Kelly bet sizing
            # f* = (p * b - q) / b, where p = win prob, q = 1-p, b = decimal odds - 1
            decimal_odds = recs["odds"].apply(
                lambda x: (x / 100 + 1) if x > 0 else (100 / abs(x) + 1)
            )
            b = decimal_odds - 1
            p = recs["model_prob"]
            q = 1 - p

            kelly = (p * b - q) / b
            recs["kelly_fraction"] = kelly.clip(lower=0) * kelly_fraction

            # Filter to positive edge bets
            recs["recommended"] = recs["edge"] >= min_edge

        return recs

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> dict:
        """
        Time-series cross-validation.

        Args:
            X: Features (should be sorted by date)
            y: Labels
            n_splits: Number of CV splits

        Returns:
            Dictionary with CV metrics
        """
        X = X[self.feature_columns].fillna(X[self.feature_columns].mean())

        tscv = TimeSeriesSplit(n_splits=n_splits)
        model = xgb.XGBClassifier(**self.params)

        scores = cross_val_score(model, X, y, cv=tscv, scoring="roc_auc")

        return {
            "cv_scores": scores,
            "cv_mean": scores.mean(),
            "cv_std": scores.std(),
        }

    def save(self, path: str):
        """Save model to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "calibrated_model": self.calibrated_model,
                "feature_columns": self.feature_columns,
                "feature_importance": self.feature_importance,
                "params": self.params,
            }, f)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "SpreadPredictionModel":
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(params=data["params"])
        instance.model = data["model"]
        instance.calibrated_model = data["calibrated_model"]
        instance.feature_columns = data["feature_columns"]
        instance.feature_importance = data["feature_importance"]

        logger.info(f"Model loaded from {path}")
        return instance


def train_spread_model(
    features_path: str = "data/features/game_features.parquet",
    model_path: str = "models/spread_model.pkl",
    train_seasons: List[int] = None,
    val_seasons: List[int] = None,
) -> SpreadPredictionModel:
    """
    Train a spread prediction model.

    Args:
        features_path: Path to feature data
        model_path: Path to save model
        train_seasons: Seasons for training
        val_seasons: Seasons for validation

    Returns:
        Trained model
    """
    from src.features import GameFeatureBuilder

    # Default seasons
    train_seasons = train_seasons or list(range(2015, 2023))
    val_seasons = val_seasons or [2023]

    # Load features
    df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(df)} games")

    # Split data
    builder = GameFeatureBuilder()
    train_df, val_df, _ = builder.prepare_train_test_split(
        df, train_seasons, val_seasons, []
    )

    # Get feature columns
    feature_cols = builder.get_feature_columns(train_df)
    logger.info(f"Using {len(feature_cols)} features")

    # Create model
    model = SpreadPredictionModel()

    # Prepare target
    y_train = model.prepare_target(train_df)
    y_val = model.prepare_target(val_df)

    # Train
    model.fit(
        train_df[feature_cols],
        y_train,
        val_df[feature_cols],
        y_val,
        feature_columns=feature_cols,
    )

    # Evaluate
    train_metrics = model.evaluate(train_df[feature_cols], y_train)
    val_metrics = model.evaluate(val_df[feature_cols], y_val)

    logger.info(f"Train metrics: {train_metrics}")
    logger.info(f"Val metrics: {val_metrics}")

    # Save model
    model.save(model_path)

    # Print feature importance
    logger.info("\nTop 10 features:")
    print(model.feature_importance.head(10))

    return model


if __name__ == "__main__":
    # Example usage with synthetic data
    import numpy as np

    # Create synthetic training data
    np.random.seed(42)
    n_samples = 1000

    X = pd.DataFrame({
        "home_net_rating_10g": np.random.randn(n_samples) * 5,
        "away_net_rating_10g": np.random.randn(n_samples) * 5,
        "home_rest_days": np.random.choice([1, 2, 3], n_samples),
        "away_rest_days": np.random.choice([1, 2, 3], n_samples),
        "diff_win_rate_10g": np.random.randn(n_samples) * 0.2,
    })

    # Create target correlated with features
    prob = 1 / (1 + np.exp(-(X["diff_win_rate_10g"] * 2 + X["home_net_rating_10g"] * 0.1)))
    y = (np.random.random(n_samples) < prob).astype(int)

    # Split
    train_size = int(0.8 * n_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = pd.Series(y[:train_size]), pd.Series(y[train_size:])

    # Train
    model = SpreadPredictionModel()
    model.fit(X_train, y_train, X_val, y_val)

    # Evaluate
    metrics = model.evaluate(X_val, y_val)
    print("\nValidation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nFeature Importance:")
    print(model.feature_importance)
