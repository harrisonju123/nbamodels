"""
Tests for Archived Models

These tests are for models that have been archived during the 2026-01-09 consolidation:
- ConformalPredictor (archive/unused_models/conformal.py)
- AdaptiveConformalPredictor (archive/unused_models/conformal.py)
- OnlineUpdater (archive/unused_models/online_learning.py)

Models were archived because:
- ConformalPredictor: Only used in tests, not production
- AdaptiveConformalPredictor: Only used in tests, not production
- OnlineUpdater: Only used in tests, not production

These tests are kept for reference and can be run if models are restored.
To run: pytest tests/archive/test_archived_models.py --archived-models-available
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# Skip all tests by default since models are archived
pytestmark = pytest.mark.skipif(
    not pytest.config.getoption("--archived-models-available", default=False),
    reason="Archived models not available. Use --archived-models-available to run."
)


def test_conformal_predictor():
    """Test ConformalPredictor with XGBSpreadModel."""
    from src.models.conformal import ConformalPredictor
    from src.models.xgb_model import XGBSpreadModel

    # Generate sample data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(500, 10), columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series((X['feature_0'] + X['feature_1'] > 0).astype(int))

    # Split data
    X_train, X_cal, X_test = X.iloc[:300], X.iloc[300:400], X.iloc[400:]
    y_train, y_cal, y_test = y.iloc[:300], y.iloc[300:400], y.iloc[400:]

    # Train base model (for classification, convert y to binary)
    y_train_bin = pd.Series((y_train > 0).astype(int))
    y_cal_bin = pd.Series((y_cal > 0).astype(int))

    base_model = XGBSpreadModel(params={'n_estimators': 50, 'max_depth': 3})
    base_model.fit(X_train, y_train_bin)

    # Create conformal predictor (alpha set at creation, not predict_interval)
    conformal = ConformalPredictor(base_model, alpha=0.1)
    conformal.calibrate(X_cal, y_cal_bin)

    # Get prediction intervals (returns tuple: lower, point, upper)
    lower, point, upper = conformal.predict_interval(X_test)

    # Assertions
    assert len(lower) == len(X_test), "Lower bounds length mismatch"
    assert len(point) == len(X_test), "Point predictions length mismatch"
    assert len(upper) == len(X_test), "Upper bounds length mismatch"
    assert (upper >= lower).all(), "Upper bounds should be >= lower bounds"

    interval_width = (upper - lower).mean()
    print(f"  ConformalPredictor: OK")
    print(f"    Interval width mean: {interval_width:.4f}")


def test_adaptive_conformal_predictor():
    """Test AdaptiveConformalPredictor."""
    from src.models.conformal import AdaptiveConformalPredictor
    from src.models.xgb_model import XGBSpreadModel

    # Generate sample data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(500, 10), columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series((X['feature_0'] + X['feature_1'] > 0).astype(int))

    # Split data
    X_train, X_cal, X_test = X.iloc[:300], X.iloc[300:400], X.iloc[400:]
    y_train, y_cal, y_test = y.iloc[:300], y.iloc[300:400], y.iloc[400:]

    y_train_bin = (y_train > 0).astype(int)
    y_cal_bin = (y_cal > 0).astype(int)

    base_model = XGBSpreadModel(params={'n_estimators': 50, 'max_depth': 3})
    base_model.fit(X_train, y_train_bin)

    adaptive = AdaptiveConformalPredictor(base_model)
    adaptive.calibrate(X_cal, y_cal_bin)

    # Test predictions
    predictions = adaptive.predict_proba(X_test)

    # Assertions
    assert len(predictions) == len(X_test), "Predictions length mismatch"
    print(f"  AdaptiveConformalPredictor: OK")


def test_online_updater():
    """Test OnlineUpdater for retraining detection."""
    from src.models.online_learning import OnlineUpdater, RetrainTrigger
    from src.models.xgb_model import XGBSpreadModel

    # Create a model for the updater
    model = XGBSpreadModel(params={'n_estimators': 50, 'max_depth': 3})

    # Create sample data and fit the model first
    np.random.seed(42)
    feature_names = [f'feature_{i}' for i in range(10)]
    X_train = pd.DataFrame(np.random.randn(200, 10), columns=feature_names)
    y_train = pd.Series((X_train['feature_0'] > 0).astype(int))
    model.fit(X_train, y_train)

    # Create updater with the fitted model
    updater = OnlineUpdater(
        model=model,
        clv_threshold=-0.01,
        win_rate_threshold=0.50,
        min_samples_for_retrain=100,
    )

    # Add some prediction results to build history
    for i in range(60):
        clv = np.random.normal(-0.015, 0.03)
        updater.add_prediction_result(
            prediction=0.55,
            actual=1 if np.random.random() > 0.52 else 0,
            clv=clv,
        )

    # Test should_retrain
    should_retrain, reason = updater.should_retrain()

    # Assertions
    assert isinstance(should_retrain, bool), "should_retrain must be bool"
    assert isinstance(reason, str), "reason must be string"

    print(f"  OnlineUpdater: OK")
    print(f"    Should retrain: {should_retrain}")
    print(f"    Reason: {reason}")


def pytest_addoption(parser):
    """Add custom pytest option for archived models."""
    parser.addoption(
        "--archived-models-available",
        action="store_true",
        default=False,
        help="Run tests for archived models (requires models in archive/unused_models/)"
    )


if __name__ == "__main__":
    print("These tests are for archived models and will be skipped by default.")
    print("To run: pytest tests/archive/test_archived_models.py --archived-models-available")
