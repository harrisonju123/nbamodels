"""
Build Stacked Ensemble Model

Trains a stacking ensemble using XGBoost and LightGBM as base models
with a logistic regression meta-learner for optimal probability calibration.

Usage:
    python scripts/build_stacked_ensemble.py
    python scripts/build_stacked_ensemble.py --meta xgboost  # Use XGBoost meta-learner
    python scripts/build_stacked_ensemble.py --folds 10       # Use 10-fold CV
"""

import sys
import os
import argparse
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

from src.models import (
    create_xgb_model,
    create_lgbm_model,
    create_stacked_ensemble,
    StackingConfig,
    StackedEnsembleModel,
)


def load_training_data():
    """Load features and prepare training data."""
    features_path = Path("data/features/game_features.parquet")

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features not found at {features_path}. "
            "Run 'python scripts/retrain_models.py' first."
        )

    df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(df)} games with {len(df.columns)} columns")

    # Need target column - home_win (1 if home team won)
    if 'home_score' in df.columns and 'away_score' in df.columns:
        df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    elif 'home_win' not in df.columns:
        raise ValueError("No target column found (home_win, home_score/away_score)")

    # Remove rows with missing target
    df = df.dropna(subset=['home_win'])

    # Feature columns (exclude metadata and targets)
    exclude_cols = [
        'game_id', 'date', 'home_team', 'away_team', 'season',
        'home_score', 'away_score', 'home_win', 'total_points',
        'home_team_id', 'away_team_id', 'status',
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    # Remove any columns with all NaN
    feature_cols = [c for c in feature_cols if df[c].notna().any()]

    logger.info(f"Using {len(feature_cols)} feature columns")

    X = df[feature_cols].fillna(0)
    y = df['home_win']

    return X, y, feature_cols


def build_stacked_ensemble(
    meta_learner_type: str = "logistic",
    n_folds: int = 5,
    output_path: str = "models/stacked_ensemble.pkl",
) -> StackedEnsembleModel:
    """
    Build and train a stacked ensemble model.

    Args:
        meta_learner_type: Type of meta-learner ("logistic", "xgboost", "lightgbm")
        n_folds: Number of folds for out-of-fold predictions
        output_path: Path to save the trained model

    Returns:
        Trained StackedEnsembleModel
    """
    # Load data
    X, y, feature_cols = load_training_data()

    # Split for final validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Training set: {len(X_train)} games")
    logger.info(f"Validation set: {len(X_val)} games")

    # Create base models
    logger.info("Creating base models...")

    base_models = []

    # XGBoost
    try:
        xgb_model = create_xgb_model(name="xgb_base")
        base_models.append(xgb_model)
        logger.info("  Added XGBoost model")
    except Exception as e:
        logger.warning(f"  Could not create XGBoost: {e}")

    # LightGBM
    try:
        lgb_model = create_lgbm_model(name="lgb_base")
        base_models.append(lgb_model)
        logger.info("  Added LightGBM model")
    except Exception as e:
        logger.warning(f"  Could not create LightGBM: {e}")

    # Try to add CatBoost
    try:
        from src.models import create_catboost_model
        if create_catboost_model is not None:
            cb_model = create_catboost_model(name="catboost_base")
            base_models.append(cb_model)
            logger.info("  Added CatBoost model")
    except Exception as e:
        logger.info(f"  CatBoost not available: {e}")

    if len(base_models) < 2:
        raise ValueError("Need at least 2 base models for stacking")

    # Create stacking configuration
    config = StackingConfig(
        meta_learner_type=meta_learner_type,
        n_folds=n_folds,
        use_probas=True,
        include_original_features=False,  # Only use base model predictions
        calibrate_meta=True,
        regularization=1.0,
    )

    # Create stacked ensemble
    stacked = StackedEnsembleModel(
        base_models=base_models,
        config=config,
        name=f"stacked_{meta_learner_type}",
    )

    # Train
    logger.info(f"Training stacked ensemble with {n_folds}-fold CV...")
    logger.info(f"Meta-learner: {meta_learner_type}")

    stacked.fit(X_train, y_train, X_val, y_val)

    # Evaluate on validation set
    logger.info("\nEvaluating on validation set...")

    val_probs = stacked.predict_proba(X_val)
    val_preds = (val_probs > 0.5).astype(int)

    auc = roc_auc_score(y_val, val_probs)
    accuracy = accuracy_score(y_val, val_preds)
    brier = brier_score_loss(y_val, val_probs)

    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Brier Score: {brier:.4f}")

    # Get model importance
    logger.info("\nBase Model Importance:")
    importance_df = stacked.get_meta_feature_importance()
    for _, row in importance_df.iterrows():
        logger.info(f"  {row['model']}: {row['importance']:.3f}")

    # Get uncertainty estimates
    uncertainty = stacked.get_uncertainty(X_val)
    logger.info(f"\nUncertainty stats:")
    logger.info(f"  Mean: {uncertainty.mean():.4f}")
    logger.info(f"  Std: {uncertainty.std():.4f}")
    logger.info(f"  Min: {uncertainty.min():.4f}")
    logger.info(f"  Max: {uncertainty.max():.4f}")

    # Save model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(stacked, f)

    logger.info(f"\nSaved stacked ensemble to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("STACKED ENSEMBLE TRAINING COMPLETE")
    print("=" * 60)
    print(f"Base models: {len(base_models)}")
    print(f"Meta-learner: {meta_learner_type}")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Brier: {brier:.4f}")
    print("=" * 60)

    return stacked


def verify_model(model_path: str = "models/stacked_ensemble.pkl"):
    """Verify the stacked ensemble model loads and works."""
    logger.info(f"Verifying model at {model_path}...")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    if not model.is_fitted:
        logger.error("Model is not fitted!")
        return False

    logger.info(f"Model name: {model.name}")
    logger.info(f"Base models: {len(model.base_models)}")

    # Try a prediction
    X, y, _ = load_training_data()
    sample = X.head(10)

    probs = model.predict_proba(sample)
    uncertainty = model.get_uncertainty(sample)

    logger.info(f"Sample predictions: {probs[:5]}")
    logger.info(f"Sample uncertainty: {uncertainty[:5]}")

    logger.info("Model verification passed!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Stacked Ensemble Model")
    parser.add_argument("--meta", type=str, default="logistic",
                        choices=["logistic", "xgboost", "lightgbm"],
                        help="Meta-learner type")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--output", type=str, default="models/stacked_ensemble.pkl",
                        help="Output path for model")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing model instead of training")

    args = parser.parse_args()

    if args.verify:
        verify_model(args.output)
    else:
        build_stacked_ensemble(
            meta_learner_type=args.meta,
            n_folds=args.folds,
            output_path=args.output,
        )
