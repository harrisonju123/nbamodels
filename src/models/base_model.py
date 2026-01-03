"""
Abstract Base Model Interface

Defines the standard interface for all betting prediction models.
All ensemble members must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import os
import pickle

import numpy as np
import pandas as pd
from loguru import logger


class BaseModel(ABC):
    """
    Abstract base class for all prediction models.

    All models in the ensemble must implement this interface to ensure
    consistent behavior for training, prediction, and uncertainty estimation.
    """

    def __init__(self, name: str = "base_model"):
        """
        Initialize base model.

        Args:
            name: Model identifier for logging and tracking
        """
        self.name = name
        self.feature_columns: Optional[List[str]] = None
        self.is_fitted: bool = False
        self._feature_importance: Optional[pd.DataFrame] = None

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "BaseModel":
        """
        Train the model.

        Args:
            X: Training features
            y: Training labels (binary: 1 = home covers, 0 = away covers)
            X_val: Optional validation features for early stopping
            y_val: Optional validation labels

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions.

        Args:
            X: Features for prediction

        Returns:
            Array of probabilities (probability of class 1)
        """
        pass

    @abstractmethod
    def get_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction uncertainty estimates.

        Different models implement this differently:
        - Tree models: variance across trees or bootstrap
        - Neural networks: MC Dropout variance
        - Ensemble: disagreement between models

        Args:
            X: Features for prediction

        Returns:
            Array of uncertainty values (higher = less confident)
        """
        pass

    @abstractmethod
    def feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns:
            DataFrame with columns ['feature', 'importance']
            sorted by importance descending
        """
        pass

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Get binary class predictions.

        Args:
            X: Features for prediction
            threshold: Classification threshold

        Returns:
            Array of binary predictions (0 or 1)
        """
        return (self.predict_proba(X) > threshold).astype(int)

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions with uncertainty estimates.

        Args:
            X: Features for prediction

        Returns:
            Tuple of (probabilities, uncertainties)
        """
        return self.predict_proba(X), self.get_uncertainty(X)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        implied_prob: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True labels
            implied_prob: Market implied probabilities (for edge calculation)

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score, roc_auc_score, brier_score_loss, log_loss
        )

        probs = self.predict_proba(X)
        preds = (probs > 0.5).astype(int)

        metrics = {
            "accuracy": accuracy_score(y, preds),
            "roc_auc": roc_auc_score(y, probs),
            "brier_score": brier_score_loss(y, probs),
            "log_loss": log_loss(y, probs),
        }

        # Edge calculation if market odds available
        if implied_prob is not None:
            edge = probs - implied_prob.values
            metrics["avg_edge"] = edge.mean()
            metrics["edge_when_bet"] = edge[edge > 0].mean() if (edge > 0).any() else 0

            # Accuracy when we have positive edge
            positive_edge_mask = edge > 0
            if positive_edge_mask.any():
                metrics["edge_accuracy"] = accuracy_score(
                    y[positive_edge_mask], preds[positive_edge_mask]
                )

        # Uncertainty calibration
        uncertainties = self.get_uncertainty(X)
        if uncertainties is not None and len(uncertainties) > 0:
            # Check if high uncertainty correlates with errors
            errors = (preds != y).astype(int)
            # Need both arrays to have non-zero std for correlation
            if np.std(uncertainties) > 1e-6 and np.std(errors) > 1e-6:
                corr = np.corrcoef(uncertainties, errors)[0, 1]
                # Handle NaN from corrcoef
                metrics["uncertainty_error_corr"] = corr if not np.isnan(corr) else 0.0
            else:
                metrics["uncertainty_error_corr"] = 0.0

        return metrics

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction.

        Handles missing values and ensures correct column order.

        Args:
            X: Raw features

        Returns:
            Prepared features
        """
        if self.feature_columns is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if len(X) == 0:
            raise ValueError("Empty input DataFrame")

        # Select and order columns - copy once upfront
        missing_cols = set(self.feature_columns) - set(X.columns)
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}. Filling with 0.")
            X = X.copy()
            for col in missing_cols:
                X[col] = 0

        X = X[self.feature_columns].copy()

        # Handle missing values - fillna with mean, then 0 for all-NaN columns
        X = X.fillna(X.mean()).fillna(0)

        # Replace infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        return X

    def save(self, path: str) -> None:
        """
        Save model to file.

        Args:
            path: File path for saving
        """
        dir_path = os.path.dirname(path)
        if dir_path:  # Only create if directory path is not empty
            os.makedirs(dir_path, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._get_save_data(), f)
        logger.info(f"{self.name} saved to {path}")

    def _get_save_data(self) -> Dict[str, Any]:
        """
        Get data to save. Override in subclasses for custom saving.

        Returns:
            Dictionary of data to pickle
        """
        return {
            "name": self.name,
            "feature_columns": self.feature_columns,
            "is_fitted": self.is_fitted,
            "feature_importance": self._feature_importance,
        }

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """
        Load model from file.

        Args:
            path: File path to load from

        Returns:
            Loaded model instance
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls._from_save_data(data)
        logger.info(f"{instance.name} loaded from {path}")
        return instance

    @classmethod
    def _from_save_data(cls, data: Dict[str, Any]) -> "BaseModel":
        """
        Restore model from saved data. Override in subclasses.

        Args:
            data: Dictionary of saved data

        Returns:
            Restored model instance
        """
        raise NotImplementedError("Subclasses must implement _from_save_data")

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for reproducibility.

        Returns:
            Dictionary of configuration parameters
        """
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "is_fitted": self.is_fitted,
            "n_features": len(self.feature_columns) if self.feature_columns else 0,
        }

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        n_features = len(self.feature_columns) if self.feature_columns else 0
        return f"{self.__class__.__name__}(name='{self.name}', {status}, {n_features} features)"
