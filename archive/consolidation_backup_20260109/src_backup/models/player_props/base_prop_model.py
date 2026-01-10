"""
Base Player Prop Model

Abstract base class for all player prop prediction models.
Provides common interface for training, prediction, and uncertainty estimation.
"""

import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    xgb = None
    HAS_XGBOOST = False


class BasePlayerPropModel(ABC):
    """
    Abstract base class for player prop prediction models.

    All prop models (Points, Rebounds, Assists, etc.) inherit from this
    and implement prop-specific feature selection.

    Model Architecture:
    - XGBoost regressor for expected value prediction
    - Calibrated std dev for uncertainty
    - Normal distribution CDF for over/under probabilities
    """

    prop_type: str = "UNKNOWN"  # Override in subclasses: "PTS", "REB", "AST", etc.

    DEFAULT_PARAMS = {
        "objective": "reg:squarederror",
        "max_depth": 4,  # Shallow to prevent overfitting on player variance
        "learning_rate": 0.05,
        "n_estimators": 150,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
    }

    def __init__(self, params: dict = None):
        """
        Initialize base prop model.

        Args:
            params: Optional XGBoost parameters override
        """
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model = None
        self.feature_columns = None
        self.feature_importance = None
        self.calibrated_std = None

    @abstractmethod
    def get_required_features(self) -> List[str]:
        """
        Return list of required feature columns.

        Each subclass implements this to specify which features are needed.
        Example: ["pts_roll3", "pts_roll5", "min_roll5", "opp_def_rating"]
        """
        pass

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ):
        """
        Train the prop model.

        Args:
            X: Training features
            y: Target values (player stat values)
            X_val: Validation features
            y_val: Validation target values

        Returns:
            self for chaining
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost required for player prop models")

        # Get required features
        self.feature_columns = self.get_required_features()

        # Filter to required columns and handle missing
        X = X[self.feature_columns].copy()
        X = X.fillna(X.mean())

        if X_val is not None:
            X_val = X_val[self.feature_columns].copy()
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

            logger.info(
                f"{self.prop_type} model trained - "
                f"Val MAE: {mean_absolute_error(y_val, val_preds):.2f}, "
                f"Std: {self.calibrated_std:.2f}"
            )
        else:
            self.model.fit(X, y)
            # Estimate std from training residuals
            train_preds = self.model.predict(X)
            errors = y.values - train_preds
            self.calibrated_std = np.std(errors)

        # Store feature importance
        self.feature_importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

        logger.info(f"{self.prop_type} model training complete")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict expected value for player prop.

        Args:
            X: Features DataFrame

        Returns:
            Array of predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = X[self.feature_columns].copy()
        X = X.fillna(0)  # Fill missing with 0 for prediction
        return self.model.predict(X)

    def predict_over_prob(
        self,
        X: pd.DataFrame,
        line: float,
    ) -> np.ndarray:
        """
        Predict probability of player going OVER a line.

        Uses calibrated standard deviation to model uncertainty.
        Assumes normal distribution around predicted value.

        Args:
            X: Features DataFrame
            line: The prop line (e.g., 25.5 points)

        Returns:
            Array of probabilities (0-1) of going over
        """
        expected = self.predict(X)

        # Use calibrated std for uncertainty
        std = self.calibrated_std if self.calibrated_std else 5.0

        # P(over) = P(actual > line)
        # = 1 - CDF((line - expected) / std)
        # = CDF((expected - line) / std)
        z_score = (expected - line) / std
        over_prob = stats.norm.cdf(z_score)

        return over_prob

    def predict_under_prob(
        self,
        X: pd.DataFrame,
        line: float,
    ) -> np.ndarray:
        """
        Predict probability of player going UNDER a line.

        Args:
            X: Features DataFrame
            line: The prop line

        Returns:
            Array of probabilities (0-1) of going under
        """
        return 1 - self.predict_over_prob(X, line)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True values

        Returns:
            Dict with evaluation metrics
        """
        preds = self.predict(X)

        return {
            "mae": mean_absolute_error(y, preds),
            "rmse": np.sqrt(mean_squared_error(y, preds)),
            "r2": r2_score(y, preds),
            "mean_error": np.mean(y - preds),
            "std_error": np.std(y - preds),
        }

    def save(self, path: str):
        """
        Save model to file.

        Args:
            path: File path to save model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "prop_type": self.prop_type,
                "model": self.model,
                "feature_columns": self.feature_columns,
                "feature_importance": self.feature_importance,
                "calibrated_std": self.calibrated_std,
                "params": self.params,
            }, f)

        logger.info(f"{self.prop_type} model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'BasePlayerPropModel':
        """
        Load model from file.

        Args:
            path: File path to load model from

        Returns:
            Loaded model instance
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(params=data["params"])
        instance.prop_type = data["prop_type"]
        instance.model = data["model"]
        instance.feature_columns = data["feature_columns"]
        instance.feature_importance = data.get("feature_importance")
        instance.calibrated_std = data.get("calibrated_std")

        logger.info(f"{instance.prop_type} model loaded from {path}")
        return instance
