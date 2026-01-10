"""
Ensemble Model Orchestrator

Combines multiple diverse models with configurable weighting strategies.
Supports equal, optimized, and dynamic (CLV-based) weighting.
"""

from typing import Dict, List, Optional, Any, Tuple, Literal
from dataclasses import dataclass, field
from datetime import datetime
import os
import pickle

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize

from .base_model import BaseModel


@dataclass
class ModelPerformance:
    """Track individual model performance for dynamic weighting."""
    model_name: str
    predictions: List[float] = field(default_factory=list)
    actuals: List[int] = field(default_factory=list)
    clv_values: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

    def add_result(
        self,
        prediction: float,
        actual: int,
        clv: float,
        timestamp: Optional[datetime] = None,
    ):
        """Add a prediction result."""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.clv_values.append(clv)
        self.timestamps.append(timestamp or datetime.now())

    def get_rolling_clv(self, window: int = 50) -> float:
        """Get rolling CLV over last N bets."""
        if len(self.clv_values) < window:
            return np.mean(self.clv_values) if self.clv_values else 0.0
        return np.mean(self.clv_values[-window:])

    def get_rolling_accuracy(self, window: int = 50) -> float:
        """Get rolling accuracy over last N bets."""
        if len(self.predictions) == 0:
            return 0.5

        recent_preds = self.predictions[-window:]
        recent_actuals = self.actuals[-window:]

        binary_preds = [1 if p > 0.5 else 0 for p in recent_preds]
        correct = sum(1 for p, a in zip(binary_preds, recent_actuals) if p == a)

        return correct / len(recent_preds)


class EnsembleModel(BaseModel):
    """
    Ensemble model combining multiple diverse predictors.

    Supports three weighting strategies:
    - 'equal': Simple average of all model predictions
    - 'optimized': Weights optimized on validation set to minimize loss
    - 'dynamic': Weights adjusted based on recent CLV performance

    Uncertainty is estimated from model disagreement.
    """

    def __init__(
        self,
        models: List[BaseModel],
        weighting: Literal["equal", "optimized", "dynamic"] = "equal",
        name: str = "ensemble",
        dynamic_window: int = 50,
        min_weight: float = 0.05,
    ):
        """
        Initialize ensemble.

        Args:
            models: List of fitted BaseModel instances
            weighting: Weighting strategy
            name: Ensemble identifier
            dynamic_window: Window size for dynamic weight updates
            min_weight: Minimum weight for any model (prevents complete exclusion)
        """
        super().__init__(name=name)

        if not models:
            raise ValueError("At least one model required for ensemble")

        self.models = models
        self.weighting = weighting
        self.dynamic_window = dynamic_window
        self.min_weight = min_weight

        # Initialize equal weights
        n_models = len(models)
        self._weights = np.ones(n_models) / n_models

        # Track feature columns from first model
        self.feature_columns = models[0].feature_columns

        # Performance tracking for dynamic weighting
        self._model_performance: Dict[str, ModelPerformance] = {
            m.name: ModelPerformance(model_name=m.name) for m in models
        }

        # Check if all models are fitted
        self.is_fitted = all(m.is_fitted for m in models)

    @property
    def weights(self) -> Dict[str, float]:
        """Get current model weights as dictionary."""
        return {m.name: w for m, w in zip(self.models, self._weights)}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "EnsembleModel":
        """
        Fit the ensemble.

        If models are not already fitted, trains each model.
        If weighting='optimized', optimizes weights on validation data.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Self for method chaining
        """
        # Train any unfitted models
        for model in self.models:
            if not model.is_fitted:
                logger.info(f"Fitting {model.name}...")
                model.fit(X, y, X_val, y_val)

        self.feature_columns = self.models[0].feature_columns
        self.is_fitted = True

        # Optimize weights if requested and validation data available
        if self.weighting == "optimized" and X_val is not None and y_val is not None:
            self._optimize_weights(X_val, y_val)

        return self

    def _optimize_weights(self, X_val: pd.DataFrame, y_val: pd.Series):
        """
        Optimize weights to minimize log loss on validation data.

        Uses constrained optimization to find weights that minimize
        the ensemble's prediction error.
        """
        # Get predictions from each model
        model_preds = np.array([m.predict_proba(X_val) for m in self.models])
        y_true = y_val.values

        def neg_log_likelihood(weights):
            """Negative log-likelihood of ensemble predictions."""
            weights = weights / weights.sum()  # Normalize
            ensemble_pred = np.dot(weights, model_preds)
            # Clip to avoid log(0)
            ensemble_pred = np.clip(ensemble_pred, 1e-10, 1 - 1e-10)
            ll = y_true * np.log(ensemble_pred) + (1 - y_true) * np.log(1 - ensemble_pred)
            return -np.mean(ll)

        n_models = len(self.models)
        initial_weights = np.ones(n_models) / n_models

        # Constraints: weights sum to 1, each weight >= min_weight
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        ]
        bounds = [(self.min_weight, 1.0) for _ in range(n_models)]

        result = minimize(
            neg_log_likelihood,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            self._weights = result.x / result.x.sum()
            logger.info(f"Optimized weights: {self.weights}")
        else:
            logger.warning("Weight optimization failed, using equal weights")

    def update_weights_from_performance(
        self,
        performance_data: Optional[Dict[str, Dict]] = None,
    ):
        """
        Update weights based on recent model performance.

        Used for dynamic weighting strategy. Models with better
        recent CLV get higher weights.

        Args:
            performance_data: Optional external performance data
                Format: {model_name: {'clv': float, 'accuracy': float}}
        """
        if self.weighting != "dynamic":
            logger.warning("update_weights_from_performance only used with weighting='dynamic'")
            return

        if performance_data is not None:
            # Use provided performance data
            clv_scores = []
            for model in self.models:
                data = performance_data.get(model.name, {})
                clv_scores.append(data.get("clv", 0.0))
        else:
            # Use internal tracking
            clv_scores = [
                self._model_performance[m.name].get_rolling_clv(self.dynamic_window)
                for m in self.models
            ]

        clv_scores = np.array(clv_scores)

        # Convert CLV to weights using softmax with temperature
        # Higher CLV = higher weight, but don't be too aggressive
        temperature = 0.1  # Lower = more aggressive weighting
        shifted_clv = clv_scores - clv_scores.max()  # Numerical stability
        exp_clv = np.exp(shifted_clv / temperature)
        exp_sum = exp_clv.sum()
        if exp_sum > 1e-10:
            weights = exp_clv / exp_sum
        else:
            # Fall back to equal weights if softmax fails
            weights = np.ones(len(clv_scores)) / len(clv_scores)

        # Apply minimum weight constraint
        weights = np.maximum(weights, self.min_weight)
        weight_sum = weights.sum()
        if weight_sum > 1e-10:
            weights = weights / weight_sum
        else:
            # Fall back to equal weights
            weights = np.ones(len(weights)) / len(weights)

        self._weights = weights
        logger.info(f"Updated dynamic weights: {self.weights}")

    def record_bet_result(
        self,
        X: pd.DataFrame,
        actual: int,
        clv: float,
        timestamp: Optional[datetime] = None,
    ):
        """
        Record a bet result for dynamic weight updating.

        Args:
            X: Features for the bet
            actual: Actual outcome (0 or 1)
            clv: Closing line value for this bet
            timestamp: When the bet was placed
        """
        for model in self.models:
            pred = model.predict_proba(X)[0]
            self._model_performance[model.name].add_result(
                prediction=pred,
                actual=actual,
                clv=clv,
                timestamp=timestamp,
            )

        # Update weights if using dynamic weighting
        if self.weighting == "dynamic":
            total_bets = len(self._model_performance[self.models[0].name].predictions)
            if total_bets % 10 == 0:  # Update every 10 bets
                self.update_weights_from_performance()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get weighted ensemble probability predictions.

        Args:
            X: Features for prediction

        Returns:
            Array of probabilities
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Get predictions from each model
        model_preds = np.array([m.predict_proba(X) for m in self.models])

        # Weighted average
        ensemble_pred = np.dot(self._weights, model_preds)

        return ensemble_pred

    def get_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimate uncertainty from model disagreement.

        Uses the standard deviation of predictions across models,
        weighted by model weights.

        Args:
            X: Features for prediction

        Returns:
            Array of uncertainty values
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Get predictions from each model
        model_preds = np.array([m.predict_proba(X) for m in self.models])

        # Weighted mean
        weighted_mean = np.dot(self._weights, model_preds)

        # Weighted standard deviation as uncertainty
        weighted_var = np.dot(self._weights, (model_preds - weighted_mean) ** 2)
        uncertainty = np.sqrt(weighted_var)

        # Also incorporate individual model uncertainties
        model_uncertainties = np.array([m.get_uncertainty(X) for m in self.models])
        avg_model_uncertainty = np.dot(self._weights, model_uncertainties)

        # Combine disagreement and individual uncertainties
        combined_uncertainty = 0.5 * uncertainty + 0.5 * avg_model_uncertainty

        return combined_uncertainty

    def predict_with_breakdown(
        self,
        X: pd.DataFrame,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Get predictions with per-model breakdown.

        Args:
            X: Features for prediction

        Returns:
            Tuple of (ensemble_predictions, breakdown_df)
            breakdown_df has columns for each model's prediction and weight
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Get predictions from each model
        breakdown = {}
        for model, weight in zip(self.models, self._weights):
            pred = model.predict_proba(X)
            breakdown[f"{model.name}_pred"] = pred
            breakdown[f"{model.name}_weight"] = weight

        breakdown_df = pd.DataFrame(breakdown)

        # Ensemble prediction
        ensemble_pred = self.predict_proba(X)
        breakdown_df["ensemble_pred"] = ensemble_pred

        return ensemble_pred, breakdown_df

    def feature_importance(self) -> pd.DataFrame:
        """
        Get weighted average feature importance across models.

        Returns:
            DataFrame with aggregated feature importance
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Collect importances from each model
        all_importances = []
        for model, weight in zip(self.models, self._weights):
            try:
                imp = model.feature_importance()
                imp["weight"] = weight
                imp["model"] = model.name
                all_importances.append(imp)
            except Exception as e:
                logger.warning(f"Could not get importance from {model.name}: {e}")

        if not all_importances:
            raise ValueError("No feature importances available")

        combined = pd.concat(all_importances, ignore_index=True)

        # Weighted average importance
        aggregated = (
            combined.groupby("feature")
            .apply(
                lambda g: (g["importance"] * g["weight"]).sum() / g["weight"].sum()
            )
            .reset_index(name="importance")
            .sort_values("importance", ascending=False)
        )

        self._feature_importance = aggregated
        return aggregated

    def get_model_correlations(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get correlation matrix of model predictions.

        Low correlation indicates good model diversity.

        Args:
            X: Features to evaluate on

        Returns:
            Correlation matrix as DataFrame
        """
        model_preds = {}
        for model in self.models:
            model_preds[model.name] = model.predict_proba(X)

        pred_df = pd.DataFrame(model_preds)
        return pred_df.corr()

    def get_ensemble_stats(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Get comprehensive ensemble statistics.

        Args:
            X: Features
            y: True labels

        Returns:
            Dictionary with ensemble statistics
        """
        stats = {
            "n_models": len(self.models),
            "weighting": self.weighting,
            "weights": self.weights,
        }

        # Ensemble performance
        ensemble_pred = self.predict_proba(X)
        ensemble_binary = (ensemble_pred > 0.5).astype(int)
        stats["ensemble_accuracy"] = (ensemble_binary == y).mean()

        # Individual model performance
        for model in self.models:
            pred = model.predict_proba(X)
            binary_pred = (pred > 0.5).astype(int)
            stats[f"{model.name}_accuracy"] = (binary_pred == y).mean()

        # Model diversity (average pairwise correlation)
        corr_matrix = self.get_model_correlations(X)
        n = len(self.models)
        if n > 1:
            # Extract upper triangle (excluding diagonal)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            pairwise_corrs = corr_matrix.values[mask]
            stats["avg_model_correlation"] = pairwise_corrs.mean()
            stats["min_model_correlation"] = pairwise_corrs.min()

        return stats

    def _get_save_data(self) -> Dict[str, Any]:
        """Get data for saving."""
        data = super()._get_save_data()
        data.update({
            "weighting": self.weighting,
            "dynamic_window": self.dynamic_window,
            "min_weight": self.min_weight,
            "weights": self._weights,
            "model_names": [m.name for m in self.models],
            "model_performance": {
                name: {
                    "predictions": perf.predictions,
                    "actuals": perf.actuals,
                    "clv_values": perf.clv_values,
                }
                for name, perf in self._model_performance.items()
            },
            # Note: models need to be saved separately
        })
        return data

    def save(self, path: str) -> None:
        """
        Save ensemble and all constituent models.

        Args:
            path: Base path for saving (without extension)
        """
        base_dir = os.path.dirname(path)
        if base_dir:  # Only create if directory path is not empty
            os.makedirs(base_dir, exist_ok=True)
        base_name = os.path.basename(path).replace(".pkl", "")

        # Save each model
        model_paths = []
        for model in self.models:
            model_path = os.path.join(base_dir, f"{base_name}_{model.name}.pkl")
            model.save(model_path)
            model_paths.append(model_path)

        # Save ensemble metadata
        save_data = self._get_save_data()
        save_data["model_paths"] = model_paths

        ensemble_path = os.path.join(base_dir, f"{base_name}_ensemble.pkl")
        with open(ensemble_path, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"Ensemble saved to {ensemble_path}")

    @classmethod
    def _from_save_data(cls, data: Dict[str, Any]) -> "EnsembleModel":
        """Restore ensemble from saved data - requires models to be loaded separately."""
        raise NotImplementedError(
            "Use EnsembleModel.load() to load ensemble with its models"
        )

    @classmethod
    def load(cls, path: str) -> "EnsembleModel":
        """
        Load ensemble and all constituent models.

        Args:
            path: Path to ensemble metadata file

        Returns:
            Loaded EnsembleModel
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Load each model
        models = []
        for model_path in data["model_paths"]:
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}, skipping")
                continue

            # Determine model type from filename
            if "lgbm" in model_path:
                from .lgbm_model import LGBMSpreadModel
                model = LGBMSpreadModel.load(model_path)
            elif "catboost" in model_path:
                from .catboost_model import CatBoostSpreadModel
                model = CatBoostSpreadModel.load(model_path)
            elif "neural" in model_path:
                from .neural_model import NeuralSpreadModel
                model = NeuralSpreadModel.load(model_path)
            elif "xgb" in model_path:
                from .xgb_model import XGBSpreadModel
                model = XGBSpreadModel.load(model_path)
            else:
                # Generic load - may need custom handling
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            models.append(model)

        if not models:
            raise ValueError("No models could be loaded from saved paths")

        # Create ensemble
        instance = cls(
            models=models,
            weighting=data["weighting"],
            name=data["name"],
            dynamic_window=data.get("dynamic_window", 50),
            min_weight=data.get("min_weight", 0.05),
        )

        instance._weights = data["weights"]
        instance.feature_columns = data["feature_columns"]
        instance.is_fitted = data["is_fitted"]
        instance._feature_importance = data.get("feature_importance")

        # Restore performance tracking
        if "model_performance" in data:
            for name, perf_data in data["model_performance"].items():
                if name in instance._model_performance:
                    instance._model_performance[name].predictions = perf_data["predictions"]
                    instance._model_performance[name].actuals = perf_data["actuals"]
                    instance._model_performance[name].clv_values = perf_data["clv_values"]

        logger.info(f"Ensemble loaded from {path}")
        return instance

    def get_config(self) -> Dict[str, Any]:
        """Get ensemble configuration."""
        config = super().get_config()
        config.update({
            "weighting": self.weighting,
            "n_models": len(self.models),
            "model_names": [m.name for m in self.models],
            "weights": self.weights,
            "dynamic_window": self.dynamic_window,
            "min_weight": self.min_weight,
        })
        return config


def create_default_ensemble(
    weighting: str = "equal",
    include_neural: bool = True,
) -> EnsembleModel:
    """
    Create a default ensemble with standard model configuration.

    Args:
        weighting: Weighting strategy ('equal', 'optimized', 'dynamic')
        include_neural: Whether to include neural network model

    Returns:
        EnsembleModel with default models (unfitted)
    """
    from .lgbm_model import LGBMSpreadModel
    from .catboost_model import CatBoostSpreadModel

    models = [
        LGBMSpreadModel(name="lgbm_spread"),
        CatBoostSpreadModel(name="catboost_spread"),
    ]

    if include_neural:
        from .neural_model import NeuralSpreadModel
        models.append(NeuralSpreadModel(name="neural_spread"))

    return EnsembleModel(models=models, weighting=weighting)
