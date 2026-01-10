"""
Conformal Prediction for Uncertainty Quantification

Provides coverage-guaranteed prediction intervals using conformal prediction.
This wraps any point predictor to give calibrated confidence bounds.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from .base_model import BaseModel


@dataclass
class PredictionInterval:
    """Prediction with confidence interval."""
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float

    @property
    def interval_width(self) -> float:
        """Width of the prediction interval."""
        return self.upper_bound - self.lower_bound

    @property
    def is_confident(self) -> bool:
        """Check if prediction is confident (narrow interval)."""
        # For probability predictions, interval < 0.2 is considered confident
        return self.interval_width < 0.2


class ConformalPredictor:
    """
    Conformal prediction wrapper for any BaseModel.

    Uses split conformal prediction to provide coverage-guaranteed
    prediction intervals. The calibration set is used to compute
    nonconformity scores, which define the interval width.

    For binary classification probabilities:
    - Nonconformity score = |predicted_prob - actual_outcome|
    - Intervals are centered on point prediction with width from quantile

    Key guarantee: At coverage level 1-alpha, the prediction interval
    will contain the true probability approximately (1-alpha)*100% of the time.
    """

    def __init__(
        self,
        model: BaseModel,
        alpha: float = 0.1,
    ):
        """
        Initialize conformal predictor.

        Args:
            model: Fitted BaseModel to wrap
            alpha: Significance level (0.1 = 90% coverage)
        """
        if not model.is_fitted:
            raise ValueError("Model must be fitted before creating conformal predictor")

        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.model = model
        self.alpha = alpha
        self._nonconformity_scores: Optional[np.ndarray] = None
        self._calibration_quantile: Optional[float] = None
        self._is_calibrated = False

    def calibrate(
        self,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
    ) -> "ConformalPredictor":
        """
        Calibrate the conformal predictor on a calibration set.

        Uses the calibration set to compute nonconformity scores
        and the quantile needed for the desired coverage.

        Warning:
            This method is not thread-safe. Do not call calibrate()
            from multiple threads simultaneously.

        Args:
            X_cal: Calibration features
            y_cal: Calibration labels (0 or 1)

        Returns:
            Self for method chaining
        """
        if len(X_cal) == 0:
            raise ValueError("Calibration set cannot be empty")

        # Warn about small calibration sets
        min_recommended = int(np.ceil(2 / self.alpha))  # Rule of thumb
        if len(X_cal) < min_recommended:
            logger.warning(
                f"Calibration set size ({len(X_cal)}) is below recommended minimum "
                f"({min_recommended}) for alpha={self.alpha}. "
                "Coverage guarantees may be unreliable."
            )
        elif len(X_cal) < 100:
            logger.warning(
                f"Small calibration set ({len(X_cal)} samples). "
                "Consider using at least 100 samples for stable coverage."
            )

        # Reset calibration state before recomputing
        self._nonconformity_scores = None
        self._calibration_quantile = None
        self._is_calibrated = False

        # Get model predictions on calibration set
        cal_probs = self.model.predict_proba(X_cal)

        # Compute nonconformity scores: |prediction - actual|
        # For binary classification, this measures how wrong the prediction was
        y_cal_values = y_cal.values if hasattr(y_cal, 'values') else np.array(y_cal)
        nonconformity_scores = np.abs(cal_probs - y_cal_values)

        # Check for degenerate nonconformity scores
        if np.all(nonconformity_scores == 0):
            logger.warning(
                "All nonconformity scores are zero (perfect calibration predictions). "
                "This may indicate overfitting or data leakage. Intervals will be degenerate."
            )
        elif np.max(nonconformity_scores) < 1e-6:
            logger.warning(
                f"All nonconformity scores very small (max={np.max(nonconformity_scores):.6f}). "
                "Intervals may be unrealistically narrow."
            )

        # Compute the (1-alpha) quantile of nonconformity scores
        # This will be the half-width of our prediction interval
        n = len(nonconformity_scores)

        # Handle n=1 edge case explicitly
        if n == 1:
            logger.warning(
                "Calibration set has only 1 sample. Using that sample's score. "
                "Coverage guarantees are not valid with n=1."
            )
            calibration_quantile = nonconformity_scores[0]
        else:
            quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            # Ensure quantile_level is strictly less than 1.0 to avoid edge cases
            quantile_level = min(quantile_level, 1.0 - 1e-10)

            calibration_quantile = np.quantile(
                nonconformity_scores,
                quantile_level
            )

        # All computations succeeded - commit state atomically
        self._nonconformity_scores = nonconformity_scores
        self._calibration_quantile = calibration_quantile
        self._is_calibrated = True

        logger.info(
            f"Conformal predictor calibrated: "
            f"n_cal={n}, alpha={self.alpha}, "
            f"quantile={self._calibration_quantile:.4f}"
        )

        return self

    def predict_interval(
        self,
        X: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get prediction intervals for new data.

        Args:
            X: Features for prediction

        Returns:
            Tuple of (point_predictions, lower_bounds, upper_bounds)
        """
        if not self._is_calibrated:
            raise ValueError("Predictor not calibrated. Call calibrate() first.")

        # Get point predictions
        point_preds = self.model.predict_proba(X)

        # Construct intervals using calibration quantile
        half_width = self._calibration_quantile

        lower = np.clip(point_preds - half_width, 0, 1)
        upper = np.clip(point_preds + half_width, 0, 1)

        return point_preds, lower, upper

    def predict_with_intervals(
        self,
        X: pd.DataFrame,
    ) -> List[PredictionInterval]:
        """
        Get predictions with full interval objects.

        Args:
            X: Features for prediction

        Returns:
            List of PredictionInterval objects
        """
        point_preds, lower, upper = self.predict_interval(X)

        return [
            PredictionInterval(
                point_estimate=p,
                lower_bound=l,
                upper_bound=u,
                confidence_level=1 - self.alpha,
            )
            for p, l, u in zip(point_preds, lower, upper)
        ]

    def get_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get calibrated uncertainty estimates.

        Uses the interval width as uncertainty measure.

        Args:
            X: Features for prediction

        Returns:
            Array of uncertainty values (interval widths)
        """
        _, lower, upper = self.predict_interval(X)
        return upper - lower

    def filter_confident_predictions(
        self,
        X: pd.DataFrame,
        max_interval_width: float = 0.2,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Filter to only confident predictions.

        Args:
            X: Features for prediction
            max_interval_width: Maximum interval width to consider confident

        Returns:
            Tuple of (filtered_X, filtered_predictions, confident_mask)
        """
        point_preds, lower, upper = self.predict_interval(X)
        interval_width = upper - lower

        confident_mask = interval_width <= max_interval_width

        return (
            X[confident_mask],
            point_preds[confident_mask],
            confident_mask,
        )

    def evaluate_coverage(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """
        Evaluate the coverage of prediction intervals on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with coverage statistics
        """
        if not self._is_calibrated:
            raise ValueError("Predictor not calibrated. Call calibrate() first.")

        if len(X_test) != len(y_test):
            raise ValueError(
                f"X_test and y_test length mismatch: {len(X_test)} vs {len(y_test)}"
            )

        point_preds, lower, upper = self.predict_interval(X_test)
        y_test_values = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

        # For binary outcomes treated as 0.0 or 1.0, check if within interval
        # This is the correct approach: check if the actual outcome falls within bounds
        in_interval = (y_test_values >= lower) & (y_test_values <= upper)
        empirical_coverage = in_interval.mean()

        # Also check nonconformity score approach (same as calibration)
        nonconformity = np.abs(point_preds - y_test_values)
        direct_coverage = (nonconformity <= self._calibration_quantile).mean()

        avg_interval_width = (upper - lower).mean()

        return {
            "target_coverage": 1 - self.alpha,
            "empirical_coverage": empirical_coverage,
            "direct_coverage": direct_coverage,
            "coverage_gap": empirical_coverage - (1 - self.alpha),
            "avg_interval_width": avg_interval_width,
            "calibration_quantile": self._calibration_quantile,
        }

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for reproducibility."""
        return {
            "model_name": self.model.name,
            "alpha": self.alpha,
            "is_calibrated": self._is_calibrated,
            "calibration_quantile": self._calibration_quantile,
            "n_calibration_samples": len(self._nonconformity_scores)
                if self._nonconformity_scores is not None else 0,
        }


class AdaptiveConformalPredictor(ConformalPredictor):
    """
    Adaptive conformal prediction with local adjustments.

    Extends basic conformal prediction by adjusting interval width
    based on local difficulty (using model's uncertainty estimate).
    """

    def __init__(
        self,
        model: BaseModel,
        alpha: float = 0.1,
        use_model_uncertainty: bool = True,
    ):
        """
        Initialize adaptive conformal predictor.

        Args:
            model: Fitted BaseModel to wrap
            alpha: Significance level
            use_model_uncertainty: Whether to use model's uncertainty for adaptation
        """
        super().__init__(model, alpha)
        self.use_model_uncertainty = use_model_uncertainty
        self._uncertainty_scale: Optional[float] = None

    def calibrate(
        self,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
    ) -> "AdaptiveConformalPredictor":
        """
        Calibrate with uncertainty-based adaptation.
        """
        # Standard calibration first
        super().calibrate(X_cal, y_cal)

        if self.use_model_uncertainty:
            # Compute correlation between model uncertainty and nonconformity
            model_uncertainty = self.model.get_uncertainty(X_cal)

            # Normalize uncertainties - check both stds to ensure correlation is meaningful
            uncertainty_std = model_uncertainty.std()
            nonconformity_std = self._nonconformity_scores.std()

            if uncertainty_std > 1e-6 and nonconformity_std > 1e-6:
                correlation = np.corrcoef(
                    model_uncertainty,
                    self._nonconformity_scores
                )[0, 1]

                if not np.isnan(correlation) and correlation > 0.1:
                    # Use uncertainty to scale interval widths
                    raw_scale = nonconformity_std / uncertainty_std
                    # Clamp to reasonable range [0.1, 10] to prevent extreme scaling
                    self._uncertainty_scale = np.clip(raw_scale, 0.1, 10.0)
                    logger.info(
                        f"Adaptive scaling enabled: correlation={correlation:.3f}, "
                        f"scale={self._uncertainty_scale:.3f}"
                    )
                else:
                    self._uncertainty_scale = None
                    logger.info(
                        f"Adaptive scaling disabled: correlation={correlation:.3f} too low"
                    )
            else:
                self._uncertainty_scale = None
                if uncertainty_std <= 1e-6:
                    logger.info("Adaptive scaling disabled: model uncertainty has near-zero variance")
                if nonconformity_std <= 1e-6:
                    logger.info("Adaptive scaling disabled: nonconformity scores have near-zero variance")

        return self

    def predict_interval(
        self,
        X: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get adaptive prediction intervals.

        Intervals are wider where model is uncertain, narrower where confident.
        """
        if not self._is_calibrated:
            raise ValueError("Predictor not calibrated. Call calibrate() first.")

        point_preds = self.model.predict_proba(X)

        if self._uncertainty_scale is not None and self.use_model_uncertainty:
            # Adaptive width based on model uncertainty
            model_uncertainty = self.model.get_uncertainty(X)

            # Scale uncertainty to nonconformity space
            scaled_uncertainty = model_uncertainty * self._uncertainty_scale

            # Use scaled uncertainty as half-width, capped by calibration quantile
            # Also cap at 0.5 to ensure intervals fit within [0,1] probability range
            adaptive_half_width = np.minimum(
                scaled_uncertainty + self._calibration_quantile * 0.5,
                self._calibration_quantile * 2
            )
            half_width = np.clip(adaptive_half_width, 0, 0.5)

            # Warn if many intervals are capped
            capped_count = np.sum(adaptive_half_width > 0.5)
            if capped_count > len(X) * 0.1:  # >10% capped
                logger.warning(
                    f"{capped_count}/{len(X)} intervals capped at max width 0.5. "
                    "Model uncertainty may be too high for meaningful intervals."
                )
        else:
            # Fall back to standard conformal
            half_width = np.full(len(point_preds), self._calibration_quantile)

        lower = np.clip(point_preds - half_width, 0, 1)
        upper = np.clip(point_preds + half_width, 0, 1)

        return point_preds, lower, upper


def create_conformal_wrapper(
    model: BaseModel,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    alpha: float = 0.1,
    adaptive: bool = False,
) -> Union[ConformalPredictor, AdaptiveConformalPredictor]:
    """
    Factory function to create and calibrate a conformal predictor.

    Args:
        model: Fitted BaseModel to wrap
        X_cal: Calibration features
        y_cal: Calibration labels
        alpha: Significance level (0.1 = 90% coverage)
        adaptive: Whether to use adaptive conformal prediction

    Returns:
        Calibrated conformal predictor
    """
    if adaptive:
        predictor = AdaptiveConformalPredictor(model, alpha)
    else:
        predictor = ConformalPredictor(model, alpha)

    predictor.calibrate(X_cal, y_cal)
    return predictor


class BayesianConformalPredictor:
    """
    Combines Bayesian uncertainty with conformal calibration.

    Uses Bayesian credible intervals as the base uncertainty estimate,
    then applies conformal calibration to ensure finite-sample coverage
    guarantees. This gives the best of both worlds:
    - Bayesian: Principled uncertainty from posterior
    - Conformal: Coverage-guaranteed intervals

    The calibration adjusts the Bayesian intervals to achieve the
    desired coverage level on held-out data.
    """

    def __init__(
        self,
        bayesian_model: BaseModel,
        alpha: float = 0.1,
    ):
        """
        Initialize Bayesian conformal predictor.

        Args:
            bayesian_model: A model with get_credible_intervals() method
                           (e.g., BayesianLinearModel)
            alpha: Significance level (0.1 = 90% coverage)
        """
        if not bayesian_model.is_fitted:
            raise ValueError("Model must be fitted before creating predictor")

        if not hasattr(bayesian_model, "get_credible_intervals"):
            raise ValueError(
                "Model must have get_credible_intervals() method. "
                "Use BayesianLinearModel or similar."
            )

        self.model = bayesian_model
        self.alpha = alpha
        self._calibration_factor: Optional[float] = None
        self._is_calibrated = False

    def calibrate(
        self,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
    ) -> "BayesianConformalPredictor":
        """
        Calibrate the Bayesian intervals using conformal approach.

        Computes a scaling factor for the Bayesian intervals to
        achieve the desired coverage on calibration data.

        Args:
            X_cal: Calibration features
            y_cal: Calibration labels

        Returns:
            Self for method chaining
        """
        if len(X_cal) == 0:
            raise ValueError("Calibration set cannot be empty")

        # Warn about small calibration sets
        min_recommended = int(np.ceil(2 / self.alpha))  # Rule of thumb
        if len(X_cal) < min_recommended:
            logger.warning(
                f"Calibration set size ({len(X_cal)}) is below recommended minimum "
                f"({min_recommended}) for alpha={self.alpha}. "
                "Coverage guarantees may be unreliable."
            )
        elif len(X_cal) < 100:
            logger.warning(
                f"Small calibration set ({len(X_cal)} samples). "
                "Consider using at least 100 samples for stable coverage."
            )

        # Get Bayesian credible intervals at the target alpha level
        lower_bayes, upper_bayes = self.model.get_credible_intervals(
            X_cal, alpha=self.alpha
        )

        # Get point predictions
        point_preds = self.model.predict_proba(X_cal)
        y_cal_values = y_cal.values if hasattr(y_cal, 'values') else np.array(y_cal)

        # Compute how much we need to scale the intervals
        # to achieve desired coverage
        bayes_width = upper_bayes - lower_bayes
        center = (upper_bayes + lower_bayes) / 2

        # Binary search for calibration factor
        # We need factor such that scaled intervals cover (1-alpha) of data
        target_coverage = 1 - self.alpha

        # Compute nonconformity scores: distance from center to actual
        nonconformity = np.abs(y_cal_values - center)

        # Half-width of current Bayesian intervals
        half_width = bayes_width / 2

        # Find the factor that achieves target coverage
        # Score = nonconformity / half_width (how many half-widths away)
        # We want: P(score <= factor) >= 1 - alpha
        # When half_width is near zero, use large score to indicate degenerate interval
        scores = np.where(
            half_width > 1e-6,
            nonconformity / half_width,
            100.0 * np.ones_like(nonconformity)
        )

        n = len(scores)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        quantile_level = min(quantile_level, 1.0 - 1e-10)  # Consistent epsilon bound

        self._calibration_factor = np.quantile(scores, quantile_level)

        # Ensure minimum factor of 1.0 (never shrink intervals)
        self._calibration_factor = max(1.0, self._calibration_factor)

        self._is_calibrated = True

        logger.info(
            f"Bayesian conformal predictor calibrated: "
            f"n_cal={n}, alpha={self.alpha}, "
            f"calibration_factor={self._calibration_factor:.4f}"
        )

        return self

    def predict_interval(
        self,
        X: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get calibrated Bayesian prediction intervals.

        Args:
            X: Features for prediction

        Returns:
            Tuple of (point_predictions, lower_bounds, upper_bounds)
        """
        if not self._is_calibrated:
            raise ValueError("Predictor not calibrated. Call calibrate() first.")

        # Get Bayesian credible intervals
        lower_bayes, upper_bayes = self.model.get_credible_intervals(
            X, alpha=self.alpha
        )

        # Get point predictions
        point_preds = self.model.predict_proba(X)

        # Scale intervals by calibration factor
        center = (upper_bayes + lower_bayes) / 2
        half_width = (upper_bayes - lower_bayes) / 2

        scaled_half_width = half_width * self._calibration_factor

        lower = np.clip(center - scaled_half_width, 0, 1)
        upper = np.clip(center + scaled_half_width, 0, 1)

        return point_preds, lower, upper

    def predict_with_intervals(
        self,
        X: pd.DataFrame,
    ) -> List[PredictionInterval]:
        """
        Get predictions with full interval objects.

        Args:
            X: Features for prediction

        Returns:
            List of PredictionInterval objects
        """
        point_preds, lower, upper = self.predict_interval(X)

        return [
            PredictionInterval(
                point_estimate=p,
                lower_bound=l,
                upper_bound=u,
                confidence_level=1 - self.alpha,
            )
            for p, l, u in zip(point_preds, lower, upper)
        ]

    def get_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get calibrated uncertainty estimates.

        Returns the calibrated interval width as uncertainty.
        """
        _, lower, upper = self.predict_interval(X)
        return upper - lower

    def evaluate_coverage(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """
        Evaluate the coverage of calibrated Bayesian intervals.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with coverage statistics
        """
        if not self._is_calibrated:
            raise ValueError("Predictor not calibrated. Call calibrate() first.")

        point_preds, lower, upper = self.predict_interval(X_test)
        y_test_values = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

        # Get uncalibrated Bayesian intervals for comparison
        lower_bayes, upper_bayes = self.model.get_credible_intervals(
            X_test, alpha=self.alpha
        )

        # Check coverage for calibrated intervals
        in_interval = (y_test_values >= lower) & (y_test_values <= upper)
        calibrated_coverage = in_interval.mean()

        # Check coverage for uncalibrated Bayesian intervals
        in_bayes = (y_test_values >= lower_bayes) & (y_test_values <= upper_bayes)
        bayes_coverage = in_bayes.mean()

        # Interval widths
        calibrated_width = (upper - lower).mean()
        bayes_width = (upper_bayes - lower_bayes).mean()

        # Compute width ratio with robust handling for all edge cases
        if calibrated_width < 1e-6 and bayes_width < 1e-6:
            width_ratio = 1.0  # Both negligible
        elif bayes_width < 1e-6:
            # Bayesian width near zero but calibrated is not - ratio is undefined
            width_ratio = np.nan
            logger.warning("Bayesian interval width near zero, width ratio undefined")
        elif calibrated_width < 1e-6:
            # Calibrated width near zero - essentially no interval
            width_ratio = 0.0
        else:
            width_ratio = calibrated_width / bayes_width

        return {
            "target_coverage": 1 - self.alpha,
            "calibrated_coverage": calibrated_coverage,
            "bayesian_coverage": bayes_coverage,
            "coverage_improvement": calibrated_coverage - bayes_coverage,
            "calibrated_avg_width": calibrated_width,
            "bayesian_avg_width": bayes_width,
            "width_increase_ratio": width_ratio,
            "calibration_factor": self._calibration_factor,
        }

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for reproducibility."""
        return {
            "model_name": self.model.name,
            "alpha": self.alpha,
            "is_calibrated": self._is_calibrated,
            "calibration_factor": self._calibration_factor,
        }


def create_bayesian_conformal_wrapper(
    bayesian_model: BaseModel,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    alpha: float = 0.1,
) -> BayesianConformalPredictor:
    """
    Factory function to create and calibrate a Bayesian conformal predictor.

    Args:
        bayesian_model: Fitted model with get_credible_intervals() method
        X_cal: Calibration features
        y_cal: Calibration labels
        alpha: Significance level (0.1 = 90% coverage)

    Returns:
        Calibrated Bayesian conformal predictor
    """
    predictor = BayesianConformalPredictor(bayesian_model, alpha)
    predictor.calibrate(X_cal, y_cal)
    return predictor
