"""
XGBoost Model Implementation (BaseModel Compatible)

XGBoost-based prediction model wrapped to conform to BaseModel interface.
This is a thin wrapper around the existing spread_model.py functionality.
"""

from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    xgb = None
    HAS_XGBOOST = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

from .base_model import BaseModel


class XGBSpreadModel(BaseModel):
    """
    XGBoost model for NBA spread prediction.

    Wraps XGBoost to conform to the BaseModel interface for use in ensembles.
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
        params: Optional[Dict] = None,
        name: str = "xgb_spread",
        early_stopping_rounds: int = 20,
    ):
        """
        Initialize XGBoost model.

        Args:
            params: Model parameters (overrides defaults)
            name: Model identifier
            early_stopping_rounds: Patience for early stopping
        """
        super().__init__(name=name)

        if not HAS_XGBOOST:
            raise ImportError("XGBoost required. Install with: pip install xgboost")

        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.early_stopping_rounds = early_stopping_rounds
        self.model: Optional[xgb.XGBClassifier] = None
        self._n_estimators: int = 0

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "XGBSpreadModel":
        """
        Train the XGBoost model.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features for early stopping
            y_val: Validation labels

        Returns:
            Self for method chaining
        """
        # Validate inputs
        if len(X) == 0:
            raise ValueError("Cannot fit model with empty training data")
        if len(y) == 0:
            raise ValueError("Cannot fit model with empty labels")

        # Store feature columns
        self.feature_columns = X.columns.tolist()

        # Prepare data
        X_train = self._prepare_features(X)
        if X_val is not None:
            X_val_prep = self._prepare_features(X_val)

        # Create model
        self.model = xgb.XGBClassifier(**self.params)

        # Fit with early stopping if validation provided
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y,
                eval_set=[(X_val_prep, y_val)],
                verbose=False
            )
            best_iter = getattr(self.model, 'best_iteration', self.params.get('n_estimators', 200))
            logger.info(f"{self.name}: Best iteration = {best_iter}")
        else:
            self.model.fit(X_train, y)

        self._n_estimators = self.model.n_estimators
        self.is_fitted = True

        # Compute feature importance
        self._feature_importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

        logger.info(f"{self.name}: Trained with {len(X)} samples, {len(self.feature_columns)} features")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions.

        Args:
            X: Features for prediction

        Returns:
            Array of probabilities for positive class
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_prep = self._prepare_features(X)
        return self.model.predict_proba(X_prep)[:, 1]

    def get_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimate prediction uncertainty using tree variance.

        Uses bootstrap-like approach by getting predictions from
        subsets of trees to estimate variance.

        Args:
            X: Features for prediction

        Returns:
            Array of uncertainty values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_prep = self._prepare_features(X)

        # Get predictions from different tree subsets
        booster = self.model.get_booster()
        n_trees = len(booster.get_dump())

        if n_trees < 10:
            return np.ones(len(X_prep)) * 0.5

        predictions = []
        for end_tree in [n_trees // 4, n_trees // 2, 3 * n_trees // 4, n_trees]:
            # XGBoost predict with iteration_range
            pred = self.model.predict_proba(
                X_prep,
                iteration_range=(0, end_tree)
            )[:, 1]
            predictions.append(pred)

        predictions = np.array(predictions)

        # Variance across tree subsets as uncertainty
        uncertainty = np.std(predictions, axis=0)

        # Normalize to [0, 1] range
        max_unc = uncertainty.max()
        if max_unc > 1e-6:
            uncertainty = uncertainty / max_unc
        else:
            # All predictions identical - zero uncertainty
            uncertainty = np.zeros_like(uncertainty)

        return uncertainty

    def feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns:
            DataFrame with feature importance
        """
        if self._feature_importance is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._feature_importance.copy()

    def _get_save_data(self) -> Dict[str, Any]:
        """Get data for saving."""
        data = super()._get_save_data()
        data.update({
            "params": self.params,
            "early_stopping_rounds": self.early_stopping_rounds,
            "model": self.model,
            "n_estimators": self._n_estimators,
        })
        return data

    @classmethod
    def _from_save_data(cls, data: Dict[str, Any]) -> "XGBSpreadModel":
        """Restore model from saved data."""
        instance = cls(
            params=data["params"],
            name=data["name"],
            early_stopping_rounds=data.get("early_stopping_rounds", 20),
        )
        instance.model = data["model"]
        instance.feature_columns = data["feature_columns"]
        instance.is_fitted = data["is_fitted"]
        instance._feature_importance = data.get("feature_importance")
        instance._n_estimators = data.get("n_estimators", 0)
        return instance

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "params": self.params,
            "n_estimators": self._n_estimators,
            "early_stopping_rounds": self.early_stopping_rounds,
        })
        return config


def create_xgb_model(
    tuned_params: Optional[Dict] = None,
    name: str = "xgb_spread",
) -> XGBSpreadModel:
    """
    Factory function to create an XGBoost model.

    Args:
        tuned_params: Optional tuned hyperparameters
        name: Model name

    Returns:
        Configured XGBSpreadModel
    """
    return XGBSpreadModel(params=tuned_params, name=name)
