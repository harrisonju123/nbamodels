"""
Model Calibration

Implements probability calibration methods to ensure well-calibrated predictions
for optimal Kelly betting.
"""

from typing import Optional, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from loguru import logger


class CalibratedModel(BaseEstimator, ClassifierMixin):
    """
    Wrapper that adds probability calibration to any classifier.

    Implements:
    - Platt Scaling (sigmoid/logistic)
    - Isotonic Regression
    - Temperature Scaling
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        method: Literal["sigmoid", "isotonic", "temperature"] = "isotonic",
    ):
        self.base_model = base_model
        self.method = method
        self.calibrator = None
        self.temperature = 1.0

    def fit(self, X, y, X_calib=None, y_calib=None):
        """
        Fit the base model and calibration.

        Args:
            X: Training features
            y: Training labels
            X_calib: Calibration features (if None, uses X)
            y_calib: Calibration labels (if None, uses y)
        """
        # Fit base model
        self.base_model.fit(X, y)

        # Use separate calibration set if provided
        if X_calib is None:
            X_calib = X
            y_calib = y

        # Get uncalibrated probabilities
        if hasattr(self.base_model, "predict_proba"):
            probs = self.base_model.predict_proba(X_calib)[:, 1]
        else:
            probs = self.base_model.decision_function(X_calib)

        # Fit calibrator
        if self.method == "sigmoid":
            self.calibrator = LogisticRegression()
            self.calibrator.fit(probs.reshape(-1, 1), y_calib)

        elif self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(probs, y_calib)

        elif self.method == "temperature":
            self.temperature = self._find_temperature(probs, y_calib)

        return self

    def _find_temperature(self, probs: np.ndarray, y: np.ndarray) -> float:
        """Find optimal temperature for temperature scaling."""
        # Convert probs to logits
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        logits = np.log(probs / (1 - probs))

        # Grid search for best temperature
        best_temp = 1.0
        best_nll = float("inf")

        for temp in np.linspace(0.1, 5.0, 50):
            scaled_probs = 1 / (1 + np.exp(-logits / temp))
            nll = -np.mean(y * np.log(scaled_probs + 1e-10) +
                          (1 - y) * np.log(1 - scaled_probs + 1e-10))

            if nll < best_nll:
                best_nll = nll
                best_temp = temp

        logger.info(f"Optimal temperature: {best_temp:.3f}")
        return best_temp

    def predict_proba(self, X) -> np.ndarray:
        """Get calibrated probability predictions."""
        # Get base model probabilities
        if hasattr(self.base_model, "predict_proba"):
            probs = self.base_model.predict_proba(X)[:, 1]
        else:
            probs = self.base_model.decision_function(X)

        # Apply calibration
        if self.method == "sigmoid":
            calibrated = self.calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]

        elif self.method == "isotonic":
            calibrated = self.calibrator.predict(probs)

        elif self.method == "temperature":
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            logits = np.log(probs / (1 - probs))
            calibrated = 1 / (1 + np.exp(-logits / self.temperature))

        # Return in sklearn format
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, X) -> np.ndarray:
        """Get class predictions."""
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Evaluate model calibration.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with calibration metrics
    """
    from sklearn.metrics import brier_score_loss, log_loss

    # Brier score (lower is better)
    brier = brier_score_loss(y_true, y_prob)

    # Log loss (lower is better)
    logloss = log_loss(y_true, y_prob)

    # Calibration curve - use strategy='uniform' to ensure consistent bins
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

    # Expected Calibration Error (ECE)
    # Calculate bin boundaries matching the calibration curve
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_boundaries[1:-1])

    # Count samples in each bin that appears in prob_pred
    ece = 0.0
    for i, (pt, pp) in enumerate(zip(prob_true, prob_pred)):
        # Find samples in this bin range
        bin_mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if i == n_bins - 1:  # Last bin includes right edge
            bin_mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        bin_count = np.sum(bin_mask)
        ece += np.abs(pt - pp) * bin_count / len(y_true)

    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0.0

    return {
        "brier_score": brier,
        "log_loss": logloss,
        "ece": ece,
        "mce": mce,
        "calibration_curve": (prob_true, prob_pred),
    }


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    n_bins: int = 10,
):
    """
    Plot calibration curve.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        model_name: Name for legend
        n_bins: Number of bins
    """
    import matplotlib.pyplot as plt

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration plot
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax1.plot(prob_pred, prob_true, "s-", label=model_name)
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title("Calibration Curve")
    ax1.legend()

    # Histogram of predictions
    ax2.hist(y_prob, bins=50, range=(0, 1), alpha=0.7)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")

    plt.tight_layout()
    return fig


def compare_calibration_methods(
    base_model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> pd.DataFrame:
    """
    Compare different calibration methods.

    Args:
        base_model: Base classifier
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels

    Returns:
        DataFrame comparing calibration metrics
    """
    results = []

    # Uncalibrated
    base_model.fit(X_train, y_train)
    probs = base_model.predict_proba(X_val)[:, 1]
    metrics = evaluate_calibration(y_val, probs)
    results.append({"method": "uncalibrated", **metrics})

    # Calibrated methods
    for method in ["sigmoid", "isotonic", "temperature"]:
        model = CalibratedModel(base_model, method=method)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_calibration(y_val, probs)
        results.append({"method": method, **metrics})

    df = pd.DataFrame(results)
    df = df.drop(columns=["calibration_curve"])

    return df
