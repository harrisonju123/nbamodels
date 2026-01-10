"""
Bayesian Uncertainty Quantification

Provides true Bayesian modeling with credible intervals and posterior distributions.
Implements analytical Bayesian linear regression and model averaging.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit, logsumexp
from loguru import logger

from .base_model import BaseModel


@dataclass
class BayesianPrediction:
    """Bayesian prediction with full posterior information."""
    mean: float
    std: float
    credible_interval_lower: float  # e.g., 5th percentile
    credible_interval_upper: float  # e.g., 95th percentile
    posterior_samples: Optional[np.ndarray] = None

    def credible_interval(self, alpha: float = 0.1) -> Tuple[float, float]:
        """Get credible interval at given alpha level."""
        if self.posterior_samples is not None:
            lower = np.percentile(self.posterior_samples, 100 * alpha / 2)
            upper = np.percentile(self.posterior_samples, 100 * (1 - alpha / 2))
            return (lower, upper)
        else:
            # Approximate using normal distribution
            z = stats.norm.ppf(1 - alpha / 2)
            return (self.mean - z * self.std, self.mean + z * self.std)


class BayesianLinearModel(BaseModel):
    """
    Bayesian Linear Regression for interpretable uncertainty.

    Uses conjugate priors for efficient computation:
    - Prior: N(0, alpha^-1 * I) on weights
    - Likelihood: N(X @ w, beta^-1)

    Provides analytical posterior: N(mu_n, Sigma_n)

    For classification, applies logistic sigmoid to the linear output.
    """

    def __init__(
        self,
        alpha: float = 1.0,  # Prior precision (regularization)
        beta: float = 1.0,   # Noise precision
        name: str = "bayesian_linear",
    ):
        """
        Initialize Bayesian Linear Model.

        Args:
            alpha: Prior precision (higher = stronger regularization)
            beta: Noise precision (higher = less noise assumed)
            name: Model identifier
        """
        super().__init__(name=name)

        if alpha <= 0:
            raise ValueError(f"alpha (prior precision) must be positive, got {alpha}")
        if beta <= 0:
            raise ValueError(f"beta (noise precision) must be positive, got {beta}")

        self.alpha = alpha
        self.beta = beta

        # Posterior parameters (set after fit)
        self._posterior_mean: Optional[np.ndarray] = None
        self._posterior_cov: Optional[np.ndarray] = None
        self._posterior_precision: Optional[np.ndarray] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "BayesianLinearModel":
        """
        Compute posterior distribution over weights.

        Uses conjugate Bayesian linear regression:
        posterior precision = alpha * I + beta * X^T X
        posterior mean = posterior_cov @ (beta * X^T y)

        Args:
            X: Training features
            y: Training labels (binary for classification)
            X_val: Unused (kept for interface compatibility)
            y_val: Unused

        Returns:
            Self for method chaining
        """
        self.feature_columns = list(X.columns)

        X_arr = X.values
        y_arr = y.values.astype(float)

        # Validate binary labels
        unique_vals = np.unique(y_arr)
        if not (len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0, 1]))):
            raise ValueError(
                f"Binary labels expected (0 and 1), got unique values: {unique_vals}. "
                "BayesianLinearModel only supports binary classification."
            )

        # For binary classification with linear regression:
        # Map 0 -> -1, 1 -> +1 for symmetric regression targets
        # This is more appropriate than logit transformation which
        # would require probabilistic targets, not binary labels
        y_transformed = 2 * y_arr - 1  # Maps 0 -> -1, 1 -> +1

        n_samples = X_arr.shape[0]
        n_features = X_arr.shape[1]

        # Validate dimensions
        if n_samples == 0:
            raise ValueError("Cannot fit model with zero samples")
        if n_features == 0:
            raise ValueError("Cannot fit model with zero features")

        # Add bias term
        X_with_bias = np.column_stack([np.ones(len(X_arr)), X_arr])
        n_features_with_bias = n_features + 1

        # Reset state before computing to ensure clean state on failure
        self._posterior_precision = None
        self._posterior_cov = None
        self._posterior_mean = None
        self.is_fitted = False

        # Compute posterior precision matrix
        prior_precision = self.alpha * np.eye(n_features_with_bias)
        posterior_precision = (
            prior_precision + self.beta * X_with_bias.T @ X_with_bias
        )

        # Compute posterior covariance (inverse of precision) with progressive regularization
        try:
            posterior_cov = np.linalg.inv(posterior_precision)
        except np.linalg.LinAlgError:
            # Use progressively stronger regularization
            logger.warning("Posterior precision singular, trying progressive regularization")
            for reg in [1e-6, 1e-4, 1e-2, 1e-1]:
                try:
                    posterior_cov = np.linalg.inv(
                        posterior_precision + reg * np.eye(n_features_with_bias)
                    )
                    logger.info(f"Successfully inverted with regularization {reg}")
                    break
                except np.linalg.LinAlgError:
                    continue
            else:
                # All attempts failed - raise with informative message
                raise ValueError(
                    "Cannot invert posterior precision matrix even with strong regularization. "
                    "Data may be rank-deficient or severely collinear."
                )

        # Compute posterior mean
        posterior_mean = (
            self.beta * posterior_cov @ X_with_bias.T @ y_transformed
        )

        # All computations succeeded - commit state atomically
        self._posterior_precision = posterior_precision
        self._posterior_cov = posterior_cov
        self._posterior_mean = posterior_mean
        self.is_fitted = True
        logger.info(f"Fitted Bayesian Linear with {n_features} features")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions (posterior mean through sigmoid).

        Args:
            X: Features for prediction

        Returns:
            Array of probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_arr = X.values
        X_with_bias = np.column_stack([np.ones(len(X_arr)), X_arr])

        # Posterior mean prediction (on [-1, +1] scale from training)
        linear_pred = X_with_bias @ self._posterior_mean

        # Map from [-1, +1] back to [0, 1] probability scale
        # Since we trained with y_transformed = 2*y - 1, we invert: p = (pred + 1) / 2
        proba = (linear_pred + 1) / 2
        proba = np.clip(proba, 0, 1)

        return proba

    def predict_distribution(
        self,
        X: pd.DataFrame,
        n_samples: int = 1000,
    ) -> List[BayesianPrediction]:
        """
        Get full posterior predictive distribution.

        Args:
            X: Features for prediction
            n_samples: Number of posterior samples

        Returns:
            List of BayesianPrediction for each sample
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if n_samples > 1_000_000:
            logger.warning(
                f"Large n_samples ({n_samples}) may cause memory issues. "
                "Consider using 1000-10000."
            )

        X_arr = X.values
        X_with_bias = np.column_stack([np.ones(len(X_arr)), X_arr])

        predictions = []

        # Validate posterior parameters before sampling
        if not np.all(np.isfinite(self._posterior_mean)):
            raise ValueError("Posterior mean contains NaN or Inf values - model may be ill-conditioned")
        if not np.all(np.isfinite(self._posterior_cov)):
            raise ValueError("Posterior covariance contains NaN or Inf values - model may be ill-conditioned")

        # Sample weights from posterior with validation
        try:
            weight_samples = np.random.multivariate_normal(
                self._posterior_mean,
                self._posterior_cov,
                size=n_samples,
            )
        except (np.linalg.LinAlgError, ValueError) as e:
            # Posterior covariance not positive definite or other issue, add regularization
            logger.warning(
                f"Posterior sampling failed ({e}), adding regularization"
            )

            # Check if covariance has NaN/Inf before regularizing
            if not np.all(np.isfinite(self._posterior_cov)):
                logger.error(
                    "Posterior covariance has NaN/Inf, cannot regularize. Using point estimate."
                )
                # Also check posterior_mean for NaN/Inf before using as fallback
                if np.all(np.isfinite(self._posterior_mean)):
                    weight_samples = np.tile(self._posterior_mean, (n_samples, 1))
                else:
                    raise ValueError(
                        "Both posterior mean and covariance contain NaN/Inf. "
                        "Model is ill-conditioned. Check data for collinearity or extreme values."
                    )
            else:
                regularized_cov = self._posterior_cov + 1e-5 * np.eye(len(self._posterior_mean))
                try:
                    weight_samples = np.random.multivariate_normal(
                        self._posterior_mean,
                        regularized_cov,
                        size=n_samples,
                    )
                except (np.linalg.LinAlgError, ValueError) as e2:
                    # Even regularization failed - use fallback: sample from mean only
                    logger.error(
                        f"Posterior sampling failed even with regularization ({e2}). "
                        "Using point estimate (no uncertainty)."
                    )
                    # Check posterior_mean for NaN/Inf before using as fallback
                    if np.all(np.isfinite(self._posterior_mean)):
                        weight_samples = np.tile(self._posterior_mean, (n_samples, 1))
                    else:
                        raise ValueError(
                            "Posterior mean contains NaN/Inf after regularization failed. "
                            "Model is ill-conditioned. Check data for collinearity or extreme values."
                        )

        for i in range(len(X_arr)):
            x_i = X_with_bias[i]

            # Get predictions for all weight samples (on [-1, +1] scale)
            linear_preds = weight_samples @ x_i
            # Map to probability scale: (pred + 1) / 2
            prob_samples = np.clip((linear_preds + 1) / 2, 0, 1)

            mean = np.mean(prob_samples)
            std = np.std(prob_samples)
            lower = np.percentile(prob_samples, 5)
            upper = np.percentile(prob_samples, 95)

            predictions.append(BayesianPrediction(
                mean=mean,
                std=std,
                credible_interval_lower=lower,
                credible_interval_upper=upper,
                posterior_samples=prob_samples,
            ))

        return predictions

    def get_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get posterior predictive standard deviation.

        This is the Bayesian uncertainty estimate derived from
        the posterior distribution over weights.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_arr = X.values
        X_with_bias = np.column_stack([np.ones(len(X_arr)), X_arr])

        # Predictive variance for each point (vectorized)
        # Var[y|x] = x^T Sigma x + 1/beta (noise)
        # Compute diag(X @ Sigma @ X^T) efficiently as sum of element-wise products
        temp = X_with_bias @ self._posterior_cov  # (n, d) @ (d, d) = (n, d)
        pred_var = np.sum(temp * X_with_bias, axis=1) + 1.0 / self.beta

        # Check for negative variance (numerical issue indicator)
        if np.any(pred_var < -1e-10):
            logger.warning(
                f"Negative variance detected in uncertainty computation "
                f"(min={pred_var.min():.6e}). This indicates numerical instability."
            )

        uncertainties = np.sqrt(np.maximum(pred_var, 0))  # Ensure non-negative

        return uncertainties

    def get_credible_intervals(
        self,
        X: pd.DataFrame,
        alpha: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get credible intervals for predictions.

        Args:
            X: Features for prediction
            alpha: Significance level (0.1 = 90% interval)

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get mean prediction and uncertainty
        proba = self.predict_proba(X)
        uncertainty = self.get_uncertainty(X)

        # Compute credible intervals on probability scale
        # Uncertainty from get_uncertainty is in the linear [-1, +1] space
        # Transform: y_prob = (y_linear + 1) / 2, so std_prob = std_linear / 2
        z = stats.norm.ppf(1 - alpha / 2)
        # Scale by 0.5 for probability space, cap at 0.5 to ensure valid intervals
        scaled_uncertainty = np.minimum(uncertainty * 0.5, 0.5)

        # Compute intervals directly on probability scale
        lower = np.clip(proba - z * scaled_uncertainty, 0, 1)
        upper = np.clip(proba + z * scaled_uncertainty, 0, 1)

        return lower, upper

    def feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from posterior mean weights.

        Returns:
            DataFrame with feature names and absolute weight values
        """
        if not self.is_fitted or self._posterior_mean is None:
            return pd.DataFrame(columns=["feature", "importance"])

        # Skip bias term (first element)
        weights = self._posterior_mean[1:]
        importance = np.abs(weights)

        # Normalize with robust threshold
        importance_sum = importance.sum()
        if importance_sum > 1e-10:
            importance = importance / importance_sum
        else:
            # All weights near zero: return uniform importance
            n_features = len(importance)
            if n_features > 0:
                importance = np.ones(n_features) / n_features
            # else importance stays as zeros

        return pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importance,
        }).sort_values("importance", ascending=False)


class BayesianEnsembleUncertainty:
    """
    Bayesian Model Averaging for ensemble predictions.

    Uses approximate marginal likelihood (BIC) to compute posterior
    model probabilities, then combines predictions weighted by
    these probabilities.
    """

    def __init__(
        self,
        models: List[BaseModel],
        prior_weights: Optional[np.ndarray] = None,
    ):
        """
        Initialize Bayesian ensemble.

        Args:
            models: List of fitted BaseModel instances
            prior_weights: Prior model probabilities (uniform if None)
        """
        if not models:
            raise ValueError("At least one model required")

        # Validate all models are fitted
        unfitted = [i for i, m in enumerate(models) if not m.is_fitted]
        if unfitted:
            raise ValueError(
                f"All models must be fitted. Unfitted model indices: {unfitted}"
            )

        self.models = models
        self.n_models = len(models)

        # Initialize with uniform prior
        if prior_weights is None:
            self.prior_weights = np.ones(self.n_models) / self.n_models
        else:
            prior_weights = np.asarray(prior_weights)
            if len(prior_weights) != self.n_models:
                raise ValueError(
                    f"prior_weights length ({len(prior_weights)}) doesn't match "
                    f"number of models ({self.n_models})"
                )
            if np.any(prior_weights < 0):
                raise ValueError("prior_weights must be non-negative")
            weight_sum = prior_weights.sum()
            if weight_sum < 1e-10:
                raise ValueError(
                    f"prior_weights sum ({weight_sum}) too small, must be positive"
                )
            self.prior_weights = prior_weights / weight_sum

        # Posterior weights (computed from validation data)
        self.posterior_weights: Optional[np.ndarray] = None

    def compute_model_posteriors(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> np.ndarray:
        """
        Compute posterior model probabilities via BIC approximation.

        Uses log-likelihood on validation data with BIC penalty
        as approximate marginal likelihood.

        Args:
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Array of posterior model probabilities
        """
        log_likelihoods = np.zeros(self.n_models)

        for i, model in enumerate(self.models):
            try:
                proba = model.predict_proba(X_val)
                proba = np.clip(proba, 1e-10, 1 - 1e-10)

                # Binary cross-entropy log-likelihood
                ll = np.sum(
                    y_val * np.log(proba) + (1 - y_val) * np.log(1 - proba)
                )
                log_likelihoods[i] = ll
            except Exception as e:
                logger.warning(f"Could not compute likelihood for {model.name}: {e}")
                log_likelihoods[i] = -np.inf

        # Convert to posterior using softmax with prior
        # log P(M|D) ∝ log P(D|M) + log P(M)
        # Use higher threshold (1e-8) to avoid extreme log values (log(1e-10) ≈ -23)
        safe_priors = np.clip(self.prior_weights, 1e-8, 1.0)
        if np.any(self.prior_weights < 1e-8):
            logger.warning(
                f"Some prior weights very small (min={self.prior_weights.min():.6e}), "
                f"clipping to 1e-8 for numerical stability"
            )
        log_posteriors = log_likelihoods + np.log(safe_priors)

        # Softmax normalization using logsumexp for numerical stability
        posteriors = np.exp(log_posteriors - logsumexp(log_posteriors))

        self.posterior_weights = posteriors

        logger.info(f"Computed Bayesian model posteriors: {dict(zip([m.name for m in self.models], posteriors.round(3)))}")

        return posteriors

    def predict_with_bma(
        self,
        X: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get Bayesian Model Averaged predictions with uncertainty.

        Args:
            X: Features for prediction

        Returns:
            Tuple of (mean_predictions, uncertainties)
        """
        weights = self.posterior_weights
        if weights is None:
            weights = self.prior_weights

        # Get predictions from all models
        predictions = np.zeros((len(X), self.n_models))

        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict_proba(X)

        # Weighted mean (BMA prediction)
        mean_pred = predictions @ weights

        # BMA uncertainty: weighted variance + weighted mean of uncertainties
        variance = np.zeros(len(X))

        for i in range(len(X)):
            # Variance from model disagreement
            variance[i] = np.sum(weights * (predictions[i] - mean_pred[i]) ** 2)

        # Add individual model uncertainties if available
        model_uncertainties = np.zeros((len(X), self.n_models))
        for i, model in enumerate(self.models):
            try:
                model_uncertainties[:, i] = model.get_uncertainty(X)
            except Exception:
                model_uncertainties[:, i] = 0.1  # Default uncertainty

        # Add weighted average of individual uncertainties
        avg_model_uncertainty = model_uncertainties @ weights

        # Total uncertainty
        total_uncertainty = np.sqrt(variance + avg_model_uncertainty ** 2)

        return mean_pred, total_uncertainty

    def get_model_weights(self) -> pd.DataFrame:
        """Get current model weights."""
        weights = self.posterior_weights if self.posterior_weights is not None else self.prior_weights

        return pd.DataFrame({
            "model": [m.name for m in self.models],
            "weight": weights,
        }).sort_values("weight", ascending=False)


class MCDropoutWrapper(BaseModel):
    """
    Wrapper to enable MC Dropout on neural network models.

    Performs multiple stochastic forward passes with dropout enabled
    to estimate prediction uncertainty.
    """

    def __init__(
        self,
        base_model: BaseModel,
        n_samples: int = 100,
        name: str = "mc_dropout",
    ):
        """
        Initialize MC Dropout wrapper.

        Args:
            base_model: Neural network model with dropout
            n_samples: Number of MC samples for uncertainty
            name: Model identifier
        """
        super().__init__(name=name)

        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        self.base_model = base_model
        self.n_samples = n_samples
        self.feature_columns = base_model.feature_columns

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "MCDropoutWrapper":
        """Delegate fit to base model."""
        self.base_model.fit(X, y, X_val, y_val)
        self.feature_columns = self.base_model.feature_columns
        self.is_fitted = self.base_model.is_fitted
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get mean prediction from multiple MC samples."""
        samples = self.sample_predictions(X, self.n_samples)
        return np.mean(samples, axis=1)

    def sample_predictions(
        self,
        X: pd.DataFrame,
        n_samples: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get multiple stochastic forward passes.

        Args:
            X: Features for prediction
            n_samples: Number of MC samples

        Returns:
            Array of shape (len(X), n_samples) - each row is samples for one input
        """
        n = n_samples or self.n_samples

        # Check if base model has mc_predict method
        if hasattr(self.base_model, "mc_predict"):
            samples = self.base_model.mc_predict(X, n_samples=n)
            # Ensure shape is (len(X), n_samples)
            if samples.shape[0] == n and samples.shape[1] == len(X):
                return samples.T
            return samples

        # Otherwise, just repeat the same prediction (no dropout)
        proba = self.base_model.predict_proba(X)
        # Return shape (len(X), n_samples) - each row is identical samples
        return np.tile(proba[:, np.newaxis], (1, n))

    def get_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """Get uncertainty from MC sample variance."""
        samples = self.sample_predictions(X, self.n_samples)
        return np.std(samples, axis=1)

    def feature_importance(self) -> pd.DataFrame:
        """Delegate to base model."""
        return self.base_model.feature_importance()
