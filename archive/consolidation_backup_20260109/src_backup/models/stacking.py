"""
Stacking Ensemble with Meta-Learner

True stacking ensemble that trains a meta-learner on out-of-fold predictions
from base models, enabling learned model combination instead of fixed weights.
"""

from dataclasses import dataclass
from typing import List, Optional, Literal, Any, Dict, Tuple
import copy

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from .base_model import BaseModel

# Optional imports for advanced meta-learners
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


@dataclass
class StackingConfig:
    """Configuration for stacking ensemble."""
    meta_learner_type: Literal["logistic", "xgboost", "lightgbm"] = "logistic"
    n_folds: int = 5
    use_probas: bool = True  # Use probabilities vs binary predictions
    include_original_features: bool = False  # Blend mode: add original features
    calibrate_meta: bool = True  # Calibrate meta-learner predictions
    regularization: float = 1.0  # L2 regularization for logistic


class StackedEnsembleModel(BaseModel):
    """
    Stacking ensemble with learned meta-model.

    Training process:
    1. Split training data into K folds
    2. For each fold, train base models on K-1 folds
    3. Get out-of-fold predictions from each base model
    4. Train meta-learner on stacked OOF predictions
    5. Retrain base models on full data for inference

    This approach prevents data leakage while learning optimal
    non-linear combinations of base model predictions.
    """

    def __init__(
        self,
        base_models: List[BaseModel],
        config: Optional[StackingConfig] = None,
        name: str = "stacked_ensemble",
    ):
        """
        Initialize stacked ensemble.

        Args:
            base_models: List of BaseModel instances (will be cloned for training)
            config: Stacking configuration
            name: Model identifier
        """
        super().__init__(name=name)

        if not base_models:
            raise ValueError("At least one base model required")

        self.base_models = base_models
        self.config = config or StackingConfig()
        self.n_base_models = len(base_models)

        # Meta-learner (trained on OOF predictions)
        self.meta_learner: Optional[Any] = None

        # Final base models (retrained on full data)
        self.final_base_models: Optional[List[BaseModel]] = None

        # Track which features were used
        self.feature_columns = None
        self._meta_feature_importance: Optional[pd.DataFrame] = None

        # Track fallback/failure states
        self._meta_learner_fallback: bool = False
        self._all_base_models_failed: bool = False

    def _create_meta_learner(self) -> Any:
        """Create meta-learner based on config."""
        meta_type = self.config.meta_learner_type

        if meta_type == "logistic":
            return LogisticRegression(
                C=self.config.regularization,
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )

        elif meta_type == "xgboost":
            if not HAS_XGB:
                logger.warning("XGBoost not available, falling back to logistic")
                return LogisticRegression(C=1.0, max_iter=1000)

            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            )

        elif meta_type == "lightgbm":
            if not HAS_LGB:
                logger.warning("LightGBM not available, falling back to logistic")
                return LogisticRegression(C=1.0, max_iter=1000)

            return lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
            )

        else:
            raise ValueError(f"Unknown meta_learner_type: {meta_type}")

    def _generate_oof_predictions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int,
    ) -> Tuple[np.ndarray, List[List[BaseModel]]]:
        """
        Generate out-of-fold predictions for meta-learner training.

        Returns:
            Tuple of (oof_predictions array, list of trained fold models)
        """
        n_samples = len(X)
        oof_predictions = np.zeros((n_samples, self.n_base_models))

        # Store models from each fold
        fold_models: List[List[BaseModel]] = []

        # Use stratified k-fold
        kfold = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=42,
        )

        # Track per-fold failures: dict[model_idx] -> list of failed fold indices
        fold_failures: Dict[int, List[int]] = {i: [] for i in range(self.n_base_models)}

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            logger.info(f"Training fold {fold_idx + 1}/{n_folds}...")

            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            fold_model_list = []

            for model_idx, base_model in enumerate(self.base_models):
                # Deep copy the model to avoid contamination
                model_copy = copy.deepcopy(base_model)

                # Train on this fold WITHOUT validation data to prevent data leakage
                # The val_fold is only used for OOF predictions, not for training
                try:
                    try:
                        model_copy.fit(X_train_fold, y_train_fold, None, None)
                    except TypeError:
                        # Some models might not accept None for val data
                        model_copy.fit(X_train_fold, y_train_fold)

                    # Get out-of-fold predictions
                    if self.config.use_probas:
                        oof_preds = model_copy.predict_proba(X_val_fold)
                    else:
                        oof_preds = (model_copy.predict_proba(X_val_fold) > 0.5).astype(float)

                    oof_predictions[val_idx, model_idx] = oof_preds
                    fold_model_list.append(model_copy)
                except Exception as e:
                    # Log failure and use fallback predictions (0.5 for probabilities)
                    logger.warning(
                        f"Base model {base_model.name} failed on fold {fold_idx + 1}: {e}. "
                        "Using fallback predictions."
                    )
                    oof_predictions[val_idx, model_idx] = 0.5
                    fold_model_list.append(None)  # Mark as failed
                    fold_failures[model_idx].append(fold_idx + 1)

            fold_models.append(fold_model_list)

        # Report per-model fold failure summary
        for model_idx, failed_folds in fold_failures.items():
            if len(failed_folds) > 0:
                model_name = self.base_models[model_idx].name
                if len(failed_folds) == n_folds:
                    logger.error(
                        f"Model '{model_name}' failed on ALL {n_folds} folds. "
                        "OOF predictions for this model are all fallback values (0.5)."
                    )
                else:
                    logger.warning(
                        f"Model '{model_name}' failed on folds {failed_folds} "
                        f"({len(failed_folds)}/{n_folds}). "
                        "Partial OOF predictions using fallback values."
                    )

        return oof_predictions, fold_models

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "StackedEnsembleModel":
        """
        Fit the stacking ensemble.

        Args:
            X: Training features
            y: Training labels
            X_val: Optional validation features (used for meta-learner calibration)
            y_val: Optional validation labels

        Returns:
            Self for method chaining
        """
        self.feature_columns = list(X.columns)

        logger.info(f"Training stacked ensemble with {self.n_base_models} base models...")

        # Step 1: Generate out-of-fold predictions
        oof_predictions, fold_models = self._generate_oof_predictions(
            X, y, self.config.n_folds
        )

        # Step 2: Build meta-learner features
        meta_X = oof_predictions

        if self.config.include_original_features:
            # Blend mode: include original features
            # Warn about potential memory issues
            estimated_mb = (len(X) * (self.n_base_models + X.shape[1]) * 8) / (1024**2)
            if estimated_mb > 1000:  # 1GB threshold
                logger.warning(
                    f"Large memory footprint for blending mode: ~{estimated_mb:.0f}MB. "
                    "Consider using include_original_features=False for large datasets."
                )
            meta_X = np.hstack([oof_predictions, X.values])

        # Check for degenerate meta features (all models produce identical predictions)
        if meta_X.shape[1] > 0:
            feature_variance = np.var(meta_X, axis=0)
            zero_variance_count = np.sum(feature_variance < 1e-10)
            if zero_variance_count == meta_X.shape[1]:
                logger.warning(
                    "All base models produce identical predictions (zero variance). "
                    "Meta-learner may fail or produce trivial weights."
                )
            elif zero_variance_count > 0:
                logger.info(
                    f"{zero_variance_count}/{meta_X.shape[1]} meta features have "
                    "near-zero variance"
                )

        # Step 3: Train meta-learner
        logger.info(f"Training {self.config.meta_learner_type} meta-learner...")
        self.meta_learner = self._create_meta_learner()

        if self.config.calibrate_meta and self.config.meta_learner_type == "logistic":
            # Calibrated wrapper for logistic
            self.meta_learner = CalibratedClassifierCV(
                self.meta_learner,
                method="isotonic",
                cv=3,
            )

        # Initialize meta feature importance to None before fit attempt
        # This ensures clean state if fit fails and we fall back
        self._meta_feature_importance = None
        self._meta_learner_fallback = False

        try:
            self.meta_learner.fit(meta_X, y)
        except Exception as e:
            logger.error(
                f"Meta-learner training failed: {e}. "
                "This may be due to degenerate base model predictions. "
                "Falling back to prior-based classifier."
            )
            # Create a dummy meta-learner that outputs class probabilities based on prior
            from sklearn.dummy import DummyClassifier
            self.meta_learner = DummyClassifier(strategy="prior")
            self.meta_learner.fit(meta_X, y)
            self._meta_learner_fallback = True

        # Step 4: Retrain base models on full data for inference
        logger.info("Retraining base models on full data...")
        self.final_base_models = []

        # Warn about potential data leakage if X_val overlaps with X
        if X_val is not None and hasattr(X, 'index') and hasattr(X_val, 'index'):
            overlap = set(X.index).intersection(set(X_val.index))
            if overlap:
                logger.warning(
                    f"Validation data overlaps with training data ({len(overlap)} samples). "
                    "This may cause data leakage in stacking ensemble."
                )

        for i, base_model in enumerate(self.base_models):
            model_copy = copy.deepcopy(base_model)
            try:
                model_copy.fit(X, y, X_val, y_val)
                self.final_base_models.append(model_copy)
            except Exception as e:
                logger.error(
                    f"Failed to retrain base model {base_model.name} on full data: {e}. "
                    "Using fallback."
                )
                # Fallback: append None, handled in _get_base_predictions
                self.final_base_models.append(None)

        # Check if all base models failed
        successful_models = sum(1 for m in self.final_base_models if m is not None)
        if successful_models == 0:
            self._all_base_models_failed = True
            logger.error(
                "All base models failed to train on full data. "
                "Stacking ensemble will produce constant predictions (0.5)."
            )
        elif successful_models < self.n_base_models:
            logger.warning(
                f"Only {successful_models}/{self.n_base_models} base models trained successfully. "
                "Ensemble predictions may be degraded."
            )

        # Compute meta feature importance
        self._compute_meta_feature_importance()

        self.is_fitted = True
        logger.info("Stacked ensemble training complete")

        return self

    def _get_base_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get predictions from all base models."""
        if self.final_base_models is None:
            raise ValueError("Model not fitted. Call fit() first.")

        predictions = np.zeros((len(X), self.n_base_models))

        for i, model in enumerate(self.final_base_models):
            if model is None:
                # Fallback for failed models
                logger.warning(f"Base model {i} is None, using fallback prediction 0.5")
                predictions[:, i] = 0.5
                continue

            try:
                if self.config.use_probas:
                    preds = model.predict_proba(X)
                else:
                    preds = (model.predict_proba(X) > 0.5).astype(float)

                # Check for NaN/Inf in predictions
                if not np.all(np.isfinite(preds)):
                    nan_count = np.sum(~np.isfinite(preds))
                    logger.warning(
                        f"Base model {i} produced {nan_count} NaN/Inf values, replacing with 0.5"
                    )
                    preds = np.where(np.isfinite(preds), preds, 0.5)

                predictions[:, i] = preds
            except Exception as e:
                logger.error(f"Base model {i} prediction failed: {e}, using 0.5")
                predictions[:, i] = 0.5

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions from stacked ensemble.

        Args:
            X: Features for prediction

        Returns:
            Array of probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if len(X) == 0:
            logger.warning("predict_proba called with empty DataFrame")
            return np.array([])

        if self.meta_learner is None:
            raise ValueError(
                "Meta-learner is None. Model may have failed during training."
            )

        # Get base model predictions
        base_preds = self._get_base_predictions(X)

        # Build meta features
        meta_X = base_preds

        if self.config.include_original_features:
            # Validate feature columns match training
            if self.feature_columns is not None:
                missing_cols = set(self.feature_columns) - set(X.columns)
                extra_cols = set(X.columns) - set(self.feature_columns)
                if missing_cols:
                    raise ValueError(
                        f"Missing features for blending mode: {missing_cols}"
                    )
                if extra_cols:
                    logger.warning(f"Ignoring extra columns: {extra_cols}")
                # Ensure column order matches training
                X = X[self.feature_columns]
            # Validate row count alignment
            if len(X) != len(base_preds):
                raise ValueError(
                    f"Row count mismatch: X has {len(X)}, base_preds has {len(base_preds)}"
                )
            meta_X = np.hstack([base_preds, X.values])

        # Get meta-learner predictions
        if hasattr(self.meta_learner, "predict_proba"):
            proba = self.meta_learner.predict_proba(meta_X)[:, 1]
        else:
            proba = self.meta_learner.predict(meta_X)

        return proba

    def get_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction uncertainty from model disagreement.

        Uncertainty is the standard deviation of base model predictions,
        weighted by their meta-learner importance.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        base_preds = self._get_base_predictions(X)

        # Clip to valid probability range to prevent numerical issues
        base_preds_safe = np.clip(base_preds, 0, 1)

        # Get importance weights if available
        importance = self._get_meta_weights()

        if importance is not None:
            # Weighted std
            weighted_mean = np.average(base_preds_safe, axis=1, weights=importance)
            variance = np.average(
                (base_preds_safe - weighted_mean[:, np.newaxis]) ** 2,
                axis=1,
                weights=importance,
            )
            # Guard against negative variance from numerical error
            variance = np.maximum(variance, 0)
            uncertainty = np.sqrt(variance)
        else:
            # Simple std
            uncertainty = np.std(base_preds_safe, axis=1)

        return uncertainty

    def _get_meta_weights(self) -> Optional[np.ndarray]:
        """Extract weights/importance from meta-learner."""
        if self.meta_learner is None:
            return None

        # Handle calibrated wrapper
        meta = self.meta_learner
        if hasattr(meta, "calibrated_classifiers_"):
            if len(meta.calibrated_classifiers_) > 0:
                meta = meta.calibrated_classifiers_[0].estimator
            else:
                logger.warning(
                    "CalibratedClassifierCV has no calibrated classifiers, "
                    "cannot extract weights"
                )
                return None

        # Logistic regression coefficients
        if hasattr(meta, "coef_"):
            coefs = meta.coef_[0]
            # Validate we have enough coefficients
            if len(coefs) < self.n_base_models:
                logger.warning(
                    f"Expected {self.n_base_models} coefficients, got {len(coefs)}. "
                    "Returning None for weights."
                )
                return None
            coefs = coefs[:self.n_base_models]
            # Convert to weights (softmax of absolute values)
            weights = np.abs(coefs)
            weight_sum = weights.sum()
            if weight_sum > 1e-10:
                return weights / weight_sum
            else:
                # All coefficients near zero: return uniform weights
                logger.info("All meta-learner coefficients near zero, using uniform weights")
                return np.ones(self.n_base_models) / self.n_base_models

        # Tree-based feature importance
        if hasattr(meta, "feature_importances_"):
            importance = meta.feature_importances_
            # Validate we have enough importance values
            if len(importance) < self.n_base_models:
                logger.warning(
                    f"Expected {self.n_base_models} importance values, got {len(importance)}. "
                    "Returning None for weights."
                )
                return None
            importance = importance[:self.n_base_models]
            importance_sum = importance.sum()
            if importance_sum > 1e-10:
                return importance / importance_sum
            else:
                # All importance values near zero: return uniform weights
                logger.info("All feature importances near zero, using uniform weights")
                return np.ones(self.n_base_models) / self.n_base_models

        return None

    def _compute_meta_feature_importance(self):
        """Compute feature importance based on meta-learner weights."""
        weights = self._get_meta_weights()

        if weights is None:
            weights = np.ones(self.n_base_models) / self.n_base_models

        model_names = [m.name for m in self.base_models]

        self._meta_feature_importance = pd.DataFrame({
            "model": model_names,
            "importance": weights,
        }).sort_values("importance", ascending=False)

    def feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from meta-learner.

        Returns which base models the meta-learner trusts most.
        """
        if self._meta_feature_importance is not None:
            return self._meta_feature_importance.copy()

        # Aggregate from base models if meta importance not available
        all_importance = []

        for model in (self.final_base_models or self.base_models):
            if model is None:
                continue
            try:
                imp = model.feature_importance()
                imp["source_model"] = model.name
                all_importance.append(imp)
            except Exception:
                continue

        if all_importance:
            combined = pd.concat(all_importance, ignore_index=True)
            return (
                combined
                .groupby("feature")["importance"]
                .mean()
                .reset_index()
                .sort_values("importance", ascending=False)
            )

        return pd.DataFrame(columns=["feature", "importance"])

    def get_meta_feature_importance(self) -> pd.DataFrame:
        """Get importance of each base model according to meta-learner."""
        if self._meta_feature_importance is not None:
            return self._meta_feature_importance.copy()

        return pd.DataFrame({
            "model": [m.name for m in self.base_models],
            "importance": [1.0 / self.n_base_models] * self.n_base_models,
        })

    def get_base_model_predictions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get individual predictions from each base model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        base_preds = self._get_base_predictions(X)
        model_names = [m.name for m in self.base_models]

        return pd.DataFrame(base_preds, columns=model_names)


def create_stacked_ensemble(
    base_models: List[BaseModel],
    meta_type: str = "logistic",
    n_folds: int = 5,
    include_features: bool = False,
) -> StackedEnsembleModel:
    """
    Factory function to create a stacked ensemble.

    Args:
        base_models: List of BaseModel instances
        meta_type: Type of meta-learner ("logistic", "xgboost", "lightgbm")
        n_folds: Number of folds for OOF predictions
        include_features: Whether to include original features (blend mode)

    Returns:
        Configured StackedEnsembleModel
    """
    config = StackingConfig(
        meta_learner_type=meta_type,
        n_folds=n_folds,
        include_original_features=include_features,
    )

    return StackedEnsembleModel(
        base_models=base_models,
        config=config,
        name=f"stacked_{meta_type}",
    )
