"""
Online Learning and Adaptive Model Updates

Provides incremental model updates and retraining triggers based on
performance monitoring. Helps models adapt to changing market conditions.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from loguru import logger

from .base_model import BaseModel


@dataclass
class RetrainTrigger:
    """Information about why retraining was triggered."""
    reason: str
    metric_name: str
    metric_value: float
    threshold: float
    triggered_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'reason': self.reason,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'triggered_at': self.triggered_at.isoformat(),
        }


@dataclass
class UpdateResult:
    """Result of an incremental update."""
    success: bool
    n_samples_used: int
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    improvement: Dict[str, float]
    message: str


class OnlineUpdater:
    """
    Manages adaptive model updates based on performance monitoring.

    Supports:
    - Triggering retraining when performance degrades
    - Incremental updates with recent data
    - Performance tracking across updates
    - Version management of model states
    """

    def __init__(
        self,
        model: BaseModel,
        clv_threshold: float = -0.01,
        win_rate_threshold: float = 0.50,
        min_samples_for_retrain: int = 100,
        lookback_window: int = 50,
        update_frequency: int = 50,  # Check every N predictions
    ):
        """
        Initialize online updater.

        Args:
            model: BaseModel to manage
            clv_threshold: Trigger retrain if CLV drops below this
            win_rate_threshold: Trigger if win rate drops below this
            min_samples_for_retrain: Minimum new samples before allowing retrain
            lookback_window: Window for performance calculations
            update_frequency: How often to check for retrain trigger
        """
        self.model = model
        self.clv_threshold = clv_threshold
        self.win_rate_threshold = win_rate_threshold
        self.min_samples_for_retrain = min_samples_for_retrain
        self.lookback_window = lookback_window
        self.update_frequency = update_frequency

        # Performance history
        self._performance_history: List[Dict] = []
        self._update_history: List[Dict] = []
        self._predictions_since_check = 0

        # Buffer for incremental updates
        self._sample_buffer: List[Tuple[pd.DataFrame, pd.Series]] = []

    def add_prediction_result(
        self,
        prediction: float,
        actual: float,
        clv: float,
        features: Optional[pd.DataFrame] = None,
        label: Optional[float] = None,
    ) -> Optional[RetrainTrigger]:
        """
        Add a prediction result and check if retraining needed.

        Args:
            prediction: Model's prediction
            actual: Actual outcome (0 or 1)
            clv: Closing line value for this prediction
            features: Feature vector (for incremental learning)
            label: Label value (for incremental learning)

        Returns:
            RetrainTrigger if retraining should occur, None otherwise
        """
        win = 1 if (prediction > 0.5) == (actual > 0.5) else 0

        self._performance_history.append({
            'prediction': prediction,
            'actual': actual,
            'win': win,
            'clv': clv,
            'timestamp': datetime.now(),
        })

        # Store sample for potential incremental update
        if features is not None and label is not None:
            self._sample_buffer.append((features.copy(), pd.Series([label])))

        self._predictions_since_check += 1

        # Check if it's time to evaluate
        if self._predictions_since_check >= self.update_frequency:
            self._predictions_since_check = 0
            return self._check_retrain_trigger()

        return None

    def _check_retrain_trigger(self) -> Optional[RetrainTrigger]:
        """Check if retraining should be triggered."""
        if len(self._performance_history) < self.lookback_window:
            return None

        recent = self._performance_history[-self.lookback_window:]

        # Calculate metrics
        avg_clv = np.mean([p['clv'] for p in recent])
        win_rate = np.mean([p['win'] for p in recent])

        # Check CLV threshold
        if avg_clv < self.clv_threshold:
            return RetrainTrigger(
                reason="CLV below threshold",
                metric_name="rolling_clv",
                metric_value=avg_clv,
                threshold=self.clv_threshold,
            )

        # Check win rate threshold
        if win_rate < self.win_rate_threshold:
            return RetrainTrigger(
                reason="Win rate below threshold",
                metric_name="rolling_win_rate",
                metric_value=win_rate,
                threshold=self.win_rate_threshold,
            )

        return None

    def should_retrain(self, performance_metrics: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Determine if model should be retrained.

        Args:
            performance_metrics: Optional dict with 'clv', 'win_rate', etc.
                               If None, uses internal history.

        Returns:
            Tuple of (should_retrain, reason)
        """
        # Check sample buffer
        total_buffered = sum(len(labels) for _, labels in self._sample_buffer)
        if total_buffered < self.min_samples_for_retrain:
            return False, f"Insufficient samples ({total_buffered} < {self.min_samples_for_retrain})"

        # Use provided metrics or calculate from history
        if performance_metrics is not None:
            clv = performance_metrics.get('clv', 0)
            win_rate = performance_metrics.get('win_rate', 0.5)
        elif len(self._performance_history) >= self.lookback_window:
            recent = self._performance_history[-self.lookback_window:]
            clv = np.mean([p['clv'] for p in recent])
            win_rate = np.mean([p['win'] for p in recent])
        else:
            return False, "Insufficient history for evaluation"

        # Check triggers
        if clv < self.clv_threshold:
            return True, f"CLV ({clv:.3f}) below threshold ({self.clv_threshold})"

        if win_rate < self.win_rate_threshold:
            return True, f"Win rate ({win_rate:.1%}) below threshold ({self.win_rate_threshold:.1%})"

        return False, "Performance within acceptable bounds"

    def incremental_update(
        self,
        X_new: pd.DataFrame,
        y_new: pd.Series,
        learning_rate_multiplier: float = 0.1,
    ) -> UpdateResult:
        """
        Perform incremental update with new data.

        For XGBoost, this means continuing training from existing model.
        For other models, this may mean fine-tuning.

        Args:
            X_new: New feature data
            y_new: New labels
            learning_rate_multiplier: Scale down learning rate for fine-tuning

        Returns:
            UpdateResult with before/after metrics
        """
        if not self.model.is_fitted:
            return UpdateResult(
                success=False,
                n_samples_used=0,
                metrics_before={},
                metrics_after={},
                improvement={},
                message="Model not fitted, cannot do incremental update",
            )

        # Validate input
        if len(X_new) == 0 or len(y_new) == 0:
            return UpdateResult(
                success=False,
                n_samples_used=0,
                metrics_before={},
                metrics_after={},
                improvement={},
                message="No data provided for update",
            )

        # Calculate metrics before update
        try:
            preds_before = self.model.predict_proba(X_new)
            y_values = y_new.values if hasattr(y_new, 'values') else np.array(y_new)
            accuracy_before = np.mean((preds_before > 0.5) == (y_values > 0.5))
            brier_before = np.mean((preds_before - y_values) ** 2)
        except Exception as e:
            logger.warning(f"Could not calculate pre-update metrics: {e}")
            accuracy_before = 0
            brier_before = 1

        metrics_before = {
            'accuracy': accuracy_before,
            'brier_score': brier_before,
        }

        # Attempt incremental update
        try:
            if hasattr(self.model, 'incremental_fit'):
                # Model supports incremental fitting
                self.model.incremental_fit(X_new, y_new, learning_rate_multiplier)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'fit'):
                # Try XGBoost-style continuation
                self._xgb_incremental_update(X_new, y_new, learning_rate_multiplier)
            else:
                return UpdateResult(
                    success=False,
                    n_samples_used=len(X_new),
                    metrics_before=metrics_before,
                    metrics_after=metrics_before,
                    improvement={},
                    message="Model does not support incremental updates",
                )

            # Calculate metrics after update
            preds_after = self.model.predict_proba(X_new)
            accuracy_after = np.mean((preds_after > 0.5) == (y_values > 0.5))
            brier_after = np.mean((preds_after - y_values) ** 2)

            metrics_after = {
                'accuracy': accuracy_after,
                'brier_score': brier_after,
            }

            improvement = {
                'accuracy': accuracy_after - accuracy_before,
                'brier_score': brier_before - brier_after,  # Lower is better
            }

            # Log update
            self._update_history.append({
                'timestamp': datetime.now().isoformat(),
                'n_samples': len(X_new),
                'metrics_before': metrics_before,
                'metrics_after': metrics_after,
            })

            return UpdateResult(
                success=True,
                n_samples_used=len(X_new),
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                improvement=improvement,
                message=f"Updated with {len(X_new)} samples",
            )

        except Exception as e:
            logger.error(f"Incremental update failed: {e}")
            return UpdateResult(
                success=False,
                n_samples_used=0,
                metrics_before=metrics_before,
                metrics_after=metrics_before,
                improvement={},
                message=f"Update failed: {str(e)}",
            )

    def _xgb_incremental_update(
        self,
        X_new: pd.DataFrame,
        y_new: pd.Series,
        learning_rate_multiplier: float,
    ) -> None:
        """Perform XGBoost-specific incremental update."""
        try:
            import xgboost as xgb

            # Get current model
            if hasattr(self.model, 'model') and isinstance(self.model.model, xgb.XGBClassifier):
                booster = self.model.model.get_booster()

                # Re-fit with existing model as base
                # Note: Don't use training data as eval_set to avoid overfitting
                # Use fewer boosting rounds for incremental updates
                original_n_estimators = self.model.model.n_estimators
                self.model.model.n_estimators = max(10, int(original_n_estimators * 0.1))

                self.model.model.fit(
                    X_new, y_new,
                    xgb_model=booster,
                    eval_set=None,  # Don't evaluate on training data
                    verbose=False,
                )

                # Restore original n_estimators for future full retrains
                self.model.model.n_estimators = original_n_estimators

        except ImportError:
            raise RuntimeError("XGBoost not available for incremental update")
        except Exception as e:
            raise RuntimeError(f"XGBoost incremental update failed: {e}")

    def full_retrain(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> UpdateResult:
        """
        Perform full model retraining.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            UpdateResult with training metrics
        """
        # Calculate current performance on validation set (if available)
        metrics_before = {}
        if X_val is not None and y_val is not None and self.model.is_fitted:
            try:
                preds = self.model.predict_proba(X_val)
                y_val_values = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
                metrics_before = {
                    'accuracy': np.mean((preds > 0.5) == (y_val_values > 0.5)),
                    'brier_score': np.mean((preds - y_val_values) ** 2),
                }
            except Exception:
                pass

        # Retrain
        try:
            self.model.fit(X_train, y_train, X_val, y_val)

            # Calculate post-retrain metrics
            metrics_after = {}
            if X_val is not None and y_val is not None:
                preds = self.model.predict_proba(X_val)
                y_val_values = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
                metrics_after = {
                    'accuracy': np.mean((preds > 0.5) == (y_val_values > 0.5)),
                    'brier_score': np.mean((preds - y_val_values) ** 2),
                }

            # Calculate improvement
            improvement = {}
            for key in metrics_after:
                if key in metrics_before:
                    if key == 'brier_score':
                        improvement[key] = metrics_before[key] - metrics_after[key]
                    else:
                        improvement[key] = metrics_after[key] - metrics_before[key]

            # Clear sample buffer after retraining
            self._sample_buffer.clear()

            # Log retrain
            self._update_history.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'full_retrain',
                'n_samples': len(X_train),
                'metrics_before': metrics_before,
                'metrics_after': metrics_after,
            })

            return UpdateResult(
                success=True,
                n_samples_used=len(X_train),
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                improvement=improvement,
                message=f"Full retrain with {len(X_train)} samples",
            )

        except Exception as e:
            logger.error(f"Full retrain failed: {e}")
            return UpdateResult(
                success=False,
                n_samples_used=0,
                metrics_before=metrics_before,
                metrics_after={},
                improvement={},
                message=f"Retrain failed: {str(e)}",
            )

    def update_from_buffer(self) -> Optional[UpdateResult]:
        """
        Perform incremental update using buffered samples.

        Returns:
            UpdateResult if update performed, None if insufficient samples
        """
        if not self._sample_buffer:
            return None

        # Concatenate buffered samples
        X_parts = [x for x, _ in self._sample_buffer]
        y_parts = [y for _, y in self._sample_buffer]

        X_combined = pd.concat(X_parts, ignore_index=True)
        y_combined = pd.concat(y_parts, ignore_index=True)

        if len(X_combined) < self.min_samples_for_retrain:
            return None

        result = self.incremental_update(X_combined, y_combined)

        if result.success:
            self._sample_buffer.clear()

        return result

    def get_performance_trend(self) -> Dict[str, Any]:
        """Get trend analysis of model performance."""
        if len(self._performance_history) < 20:
            return {'status': 'insufficient_data'}

        recent_50 = self._performance_history[-50:] if len(self._performance_history) >= 50 else self._performance_history
        recent_20 = self._performance_history[-20:]

        # Calculate trends
        clv_50 = np.mean([p['clv'] for p in recent_50])
        clv_20 = np.mean([p['clv'] for p in recent_20])

        win_rate_50 = np.mean([p['win'] for p in recent_50])
        win_rate_20 = np.mean([p['win'] for p in recent_20])

        return {
            'clv_50': clv_50,
            'clv_20': clv_20,
            'clv_trend': 'declining' if clv_20 < clv_50 - 0.005 else 'stable' if abs(clv_20 - clv_50) < 0.005 else 'improving',
            'win_rate_50': win_rate_50,
            'win_rate_20': win_rate_20,
            'win_rate_trend': 'declining' if win_rate_20 < win_rate_50 - 0.02 else 'stable' if abs(win_rate_20 - win_rate_50) < 0.02 else 'improving',
            'n_samples': len(self._performance_history),
            'n_updates': len(self._update_history),
        }

    def get_update_history(self) -> List[Dict]:
        """Get history of all model updates."""
        return self._update_history.copy()

    def save_state(self, path: Union[str, Path]) -> None:
        """Save updater state to disk."""
        path = Path(path)
        state = {
            'performance_history': self._performance_history,
            'update_history': self._update_history,
            'config': {
                'clv_threshold': self.clv_threshold,
                'win_rate_threshold': self.win_rate_threshold,
                'min_samples_for_retrain': self.min_samples_for_retrain,
                'lookback_window': self.lookback_window,
            }
        }

        with open(path, 'w') as f:
            json.dump(state, f, default=str, indent=2)

        logger.info(f"Saved updater state to {path}")

    def load_state(self, path: Union[str, Path]) -> None:
        """Load updater state from disk."""
        path = Path(path)

        with open(path, 'r') as f:
            state = json.load(f)

        self._performance_history = state.get('performance_history', [])
        self._update_history = state.get('update_history', [])

        logger.info(f"Loaded updater state from {path}")


class AdaptiveEnsembleUpdater:
    """
    Manages updates for ensemble models with dynamic weight adjustment.

    Tracks per-model performance and adjusts ensemble weights based on
    recent CLV performance of each constituent model.
    """

    def __init__(
        self,
        model_names: List[str],
        lookback_window: int = 50,
        weight_update_frequency: int = 20,
    ):
        """
        Initialize adaptive ensemble updater.

        Args:
            model_names: Names of models in the ensemble
            lookback_window: Window for performance calculations
            weight_update_frequency: How often to update weights
        """
        self.model_names = model_names
        self.lookback_window = lookback_window
        self.weight_update_frequency = weight_update_frequency

        # Per-model performance tracking
        self._model_performance: Dict[str, List[Dict]] = {
            name: [] for name in model_names
        }
        self._current_weights: Dict[str, float] = {
            name: 1.0 / len(model_names) for name in model_names
        }
        self._predictions_since_update = 0

    def add_prediction_results(
        self,
        model_predictions: Dict[str, float],
        actual: float,
        clvs: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Add prediction results for all models.

        Args:
            model_predictions: Dict mapping model name to prediction
            actual: Actual outcome
            clvs: Dict mapping model name to CLV (optional)
        """
        for name, pred in model_predictions.items():
            if name not in self._model_performance:
                continue

            win = 1 if (pred > 0.5) == (actual > 0.5) else 0
            clv = clvs.get(name, 0) if clvs else 0

            self._model_performance[name].append({
                'prediction': pred,
                'actual': actual,
                'win': win,
                'clv': clv,
            })

        self._predictions_since_update += 1

        # Update weights periodically
        if self._predictions_since_update >= self.weight_update_frequency:
            self._predictions_since_update = 0
            self._update_weights()

    def _update_weights(self) -> None:
        """Update ensemble weights based on recent performance."""
        new_weights = {}
        total_score = 0

        for name in self.model_names:
            history = self._model_performance.get(name, [])
            if len(history) < self.lookback_window:
                # Not enough data, use uniform weight
                new_weights[name] = 1.0
            else:
                recent = history[-self.lookback_window:]
                # Score based on CLV and win rate
                avg_clv = np.mean([p['clv'] for p in recent])
                win_rate = np.mean([p['win'] for p in recent])

                # Combined score (CLV more important)
                score = 0.7 * (avg_clv + 0.02) + 0.3 * (win_rate - 0.5)
                score = max(score, 0.01)  # Minimum weight
                new_weights[name] = score

            total_score += new_weights[name]

        # Normalize
        for name in new_weights:
            new_weights[name] /= total_score

        self._current_weights = new_weights
        logger.debug(f"Updated ensemble weights: {new_weights}")

    def get_current_weights(self) -> Dict[str, float]:
        """Get current ensemble weights."""
        return self._current_weights.copy()

    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for each model."""
        results = {}

        for name in self.model_names:
            history = self._model_performance.get(name, [])
            if not history:
                continue

            recent = history[-self.lookback_window:] if len(history) >= self.lookback_window else history

            results[name] = {
                'n_predictions': len(history),
                'recent_clv': np.mean([p['clv'] for p in recent]),
                'recent_win_rate': np.mean([p['win'] for p in recent]),
                'current_weight': self._current_weights.get(name, 0),
            }

        return results
