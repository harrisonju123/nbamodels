"""
CatBoost Model Implementation

CatBoost-based prediction model with native categorical feature handling
and uncertainty estimation via virtual ensembles.
"""

from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    from catboost import CatBoostClassifier, Pool
    HAS_CATBOOST = True
except ImportError:
    CatBoostClassifier = None
    Pool = None
    HAS_CATBOOST = False
    logger.warning("CatBoost not available. Install with: pip install catboost")

from .base_model import BaseModel


class CatBoostSpreadModel(BaseModel):
    """
    CatBoost model for NBA spread prediction.

    CatBoost provides:
    - Ordered boosting to reduce overfitting
    - Native categorical feature handling
    - Built-in uncertainty estimation via virtual ensembles
    - Different regularization than XGBoost/LightGBM for diversity
    """

    DEFAULT_PARAMS = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "depth": 6,
        "learning_rate": 0.05,
        "iterations": 200,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    }

    def __init__(
        self,
        params: Optional[Dict] = None,
        name: str = "catboost_spread",
        early_stopping_rounds: int = 20,
        cat_features: Optional[List[str]] = None,
        use_gpu: bool = False,
    ):
        """
        Initialize CatBoost model.

        Args:
            params: Model parameters (overrides defaults)
            name: Model identifier
            early_stopping_rounds: Patience for early stopping
            cat_features: List of categorical feature names
            use_gpu: Whether to use GPU training if available
        """
        super().__init__(name=name)

        if not HAS_CATBOOST:
            raise ImportError("CatBoost required. Install with: pip install catboost")

        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.early_stopping_rounds = early_stopping_rounds
        self.cat_features = cat_features or []
        self.use_gpu = use_gpu

        if use_gpu:
            self.params["task_type"] = "GPU"
            self.params["devices"] = "0"

        self.model: Optional[CatBoostClassifier] = None
        self._n_trees: int = 0

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "CatBoostSpreadModel":
        """
        Train the CatBoost model.

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

        # Identify categorical feature indices
        cat_feature_indices = [
            i for i, col in enumerate(self.feature_columns)
            if col in self.cat_features
        ]

        # Create training pool
        train_pool = Pool(
            X_train, y,
            cat_features=cat_feature_indices if cat_feature_indices else None
        )

        # Create model
        self.model = CatBoostClassifier(**self.params)

        # Fit with early stopping if validation provided
        if X_val is not None and y_val is not None:
            X_val_prep = self._prepare_features(X_val)
            val_pool = Pool(
                X_val_prep, y_val,
                cat_features=cat_feature_indices if cat_feature_indices else None
            )

            self.model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=self.early_stopping_rounds,
            )

            best_iter = self.model.get_best_iteration()
            logger.info(f"{self.name}: Best iteration = {best_iter}")
        else:
            self.model.fit(train_pool)

        self._n_trees = self.model.tree_count_
        self.is_fitted = True

        # Compute feature importance
        importances = self.model.get_feature_importance()
        self._feature_importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importances
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
        Estimate prediction uncertainty using virtual ensembles.

        CatBoost can provide uncertainty via virtual ensembles,
        which creates multiple models using different subsets of trees.

        Args:
            X: Features for prediction

        Returns:
            Array of uncertainty values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_prep = self._prepare_features(X)

        # Get predictions from different tree subsets to estimate variance
        n_trees = self.model.tree_count_
        if n_trees < 10:
            # Not enough trees for meaningful variance
            return np.ones(len(X_prep)) * 0.5

        # Sample different tree endpoints
        predictions = []
        for end_tree in [n_trees // 4, n_trees // 2, 3 * n_trees // 4, n_trees]:
            pred = self.model.predict_proba(
                X_prep,
                ntree_end=end_tree
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

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get SHAP values for feature importance analysis.

        CatBoost has native SHAP support for fast computation.

        Args:
            X: Features

        Returns:
            SHAP values array of shape (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_prep = self._prepare_features(X)
        pool = Pool(X_prep)

        return self.model.get_feature_importance(
            pool,
            type='ShapValues'
        )[:, :-1]  # Exclude bias term

    def _get_save_data(self) -> Dict[str, Any]:
        """Get data for saving."""
        data = super()._get_save_data()
        data.update({
            "params": self.params,
            "early_stopping_rounds": self.early_stopping_rounds,
            "cat_features": self.cat_features,
            "use_gpu": self.use_gpu,
            "model": self.model,
            "n_trees": self._n_trees,
        })
        return data

    @classmethod
    def _from_save_data(cls, data: Dict[str, Any]) -> "CatBoostSpreadModel":
        """Restore model from saved data."""
        instance = cls(
            params=data["params"],
            name=data["name"],
            early_stopping_rounds=data.get("early_stopping_rounds", 20),
            cat_features=data.get("cat_features", []),
            use_gpu=data.get("use_gpu", False),
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
            "cat_features": self.cat_features,
            "use_gpu": self.use_gpu,
            "early_stopping_rounds": self.early_stopping_rounds,
        })
        return config


def create_catboost_model(
    tuned_params: Optional[Dict] = None,
    cat_features: Optional[List[str]] = None,
    name: str = "catboost_spread",
    use_gpu: bool = False,
) -> CatBoostSpreadModel:
    """
    Factory function to create a CatBoost model.

    Args:
        tuned_params: Optional tuned hyperparameters
        cat_features: List of categorical feature names
        name: Model name
        use_gpu: Whether to use GPU

    Returns:
        Configured CatBoostSpreadModel
    """
    return CatBoostSpreadModel(
        params=tuned_params,
        cat_features=cat_features,
        name=name,
        use_gpu=use_gpu,
    )
