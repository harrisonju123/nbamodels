"""
LightGBM Model Implementation

LightGBM-based prediction model with native feature importance
and uncertainty estimation via prediction variance across leaves.
"""

from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

from .base_model import BaseModel


class LGBMSpreadModel(BaseModel):
    """
    LightGBM model for NBA spread prediction.

    Uses LightGBM's gradient boosting with different regularization
    than XGBoost to provide model diversity in ensembles.
    """

    DEFAULT_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
        "force_row_wise": True,
    }

    def __init__(
        self,
        params: Optional[Dict] = None,
        name: str = "lgbm_spread",
        early_stopping_rounds: int = 20,
    ):
        """
        Initialize LightGBM model.

        Args:
            params: Model parameters (overrides defaults)
            name: Model identifier
            early_stopping_rounds: Patience for early stopping
        """
        super().__init__(name=name)

        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM required. Install with: pip install lightgbm")

        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.early_stopping_rounds = early_stopping_rounds
        self.model: Optional[lgb.LGBMClassifier] = None
        self._n_trees: int = 0

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LGBMSpreadModel":
        """
        Train the LightGBM model.

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
        self.model = lgb.LGBMClassifier(**self.params)

        # Fit with early stopping if validation provided
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y,
                eval_set=[(X_val_prep, y_val)],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=self.early_stopping_rounds,
                        verbose=False
                    )
                ]
            )
            best_iter = self.model.best_iteration_
            logger.info(f"{self.name}: Best iteration = {best_iter}")
        else:
            self.model.fit(X_train, y)

        self._n_trees = self.model.n_estimators_
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
        Estimate prediction uncertainty using leaf variance.

        Uses the variance of predictions across individual trees
        as an uncertainty measure.

        Args:
            X: Features for prediction

        Returns:
            Array of uncertainty values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_prep = self._prepare_features(X)

        # Get predictions from each tree
        booster = self.model.booster_
        n_iterations = booster.num_trees()

        # Handle edge case of no trees
        if n_iterations == 0:
            return np.ones(len(X_prep)) * 0.5

        # For binary classification, each iteration has 1 tree
        tree_predictions = []

        for i in range(n_iterations):
            # Get raw score from single tree
            pred = booster.predict(
                X_prep,
                start_iteration=i,
                num_iteration=1,
                raw_score=True
            )
            tree_predictions.append(pred)

        tree_predictions = np.array(tree_predictions)

        # Variance across trees as uncertainty
        uncertainty = np.std(tree_predictions, axis=0)

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

    def get_leaf_indices(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get leaf indices for each sample and tree.

        Useful for building leaf-based similarity measures.

        Args:
            X: Features

        Returns:
            Array of shape (n_samples, n_trees) with leaf indices
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_prep = self._prepare_features(X)
        return self.model.predict(X_prep, pred_leaf=True)

    def _get_save_data(self) -> Dict[str, Any]:
        """Get data for saving."""
        data = super()._get_save_data()
        data.update({
            "params": self.params,
            "early_stopping_rounds": self.early_stopping_rounds,
            "model": self.model,
            "n_trees": self._n_trees,
        })
        return data

    @classmethod
    def _from_save_data(cls, data: Dict[str, Any]) -> "LGBMSpreadModel":
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
        instance._n_trees = data.get("n_trees", 0)
        return instance

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "params": self.params,
            "n_trees": self._n_trees,
            "early_stopping_rounds": self.early_stopping_rounds,
        })
        return config


def create_lgbm_model(
    tuned_params: Optional[Dict] = None,
    name: str = "lgbm_spread",
) -> LGBMSpreadModel:
    """
    Factory function to create a LightGBM model.

    Args:
        tuned_params: Optional tuned hyperparameters
        name: Model name

    Returns:
        Configured LGBMSpreadModel
    """
    return LGBMSpreadModel(params=tuned_params, name=name)
