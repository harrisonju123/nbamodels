"""
Train Bayesian Meta-Learner for True Posterior Uncertainty

Trains a BayesianLinearModel on base model predictions to provide
true Bayesian credible intervals and posterior uncertainty estimates.

Usage:
    python scripts/train_bayesian_meta.py
    python scripts/train_bayesian_meta.py --alpha 0.5  # Stronger regularization
"""

import sys
import os
import pickle
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

from src.models import create_xgb_model, create_lgbm_model
from src.models.bayesian import BayesianLinearModel


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


def get_oof_predictions(X, y, n_folds=5):
    """
    Generate out-of-fold predictions from base models.

    Returns:
        DataFrame with base model predictions as features
    """
    logger.info("Generating out-of-fold predictions from base models...")

    # Create base models
    xgb_model = create_xgb_model(name="xgb_oof")
    lgb_model = create_lgbm_model(name="lgb_oof")

    base_models = [xgb_model, lgb_model]

    # Try to add CatBoost
    try:
        from src.models import create_catboost_model
        if create_catboost_model is not None:
            cb_model = create_catboost_model(name="catboost_oof")
            base_models.append(cb_model)
            logger.info("  Added CatBoost to base models")
    except Exception as e:
        logger.info(f"  CatBoost not available: {e}")

    # Initialize OOF predictions array
    oof_predictions = np.zeros((len(X), len(base_models)))

    # Generate OOF predictions using cross-validation
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        logger.info(f"  Fold {fold + 1}/{n_folds}")

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]

        # Train each base model and get OOF predictions
        for i, model in enumerate(base_models):
            # Create fresh model for this fold
            if model.name.startswith("xgb"):
                fold_model = create_xgb_model(name=f"{model.name}_fold{fold}")
            elif model.name.startswith("lgb"):
                fold_model = create_lgbm_model(name=f"{model.name}_fold{fold}")
            else:  # CatBoost
                from src.models import create_catboost_model
                fold_model = create_catboost_model(name=f"{model.name}_fold{fold}")

            # Train on fold
            fold_model.fit(X_train_fold, y_train_fold)

            # Predict on validation fold
            oof_predictions[val_idx, i] = fold_model.predict_proba(X_val_fold)

    # Convert to DataFrame
    oof_df = pd.DataFrame(
        oof_predictions,
        columns=[f"{m.name}_pred" for m in base_models]
    )

    logger.info(f"Generated OOF predictions: {oof_df.shape}")
    logger.info(f"  Mean predictions: {oof_df.mean().to_dict()}")

    return oof_df


def train_bayesian_meta(
    alpha: float = 1.0,
    beta: float = 1.0,
    n_folds: int = 5,
    output_path: str = "models/bayesian_meta.pkl",
) -> BayesianLinearModel:
    """
    Train Bayesian meta-learner on base model predictions.

    Args:
        alpha: Prior precision (regularization strength)
        beta: Noise precision
        n_folds: Number of folds for OOF predictions
        output_path: Path to save trained model

    Returns:
        Trained BayesianLinearModel
    """
    # Load full feature set for base model training
    full_features, y, feature_cols = load_training_data()

    # Generate out-of-fold base model predictions (meta-features)
    logger.info(f"Generating out-of-fold predictions from {len(full_features)} games...")
    oof_predictions = get_oof_predictions(full_features, y, n_folds=n_folds)

    # Split meta-features for validation
    meta_X_train, meta_X_val, y_train, y_val = train_test_split(
        oof_predictions, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Training Bayesian meta-learner on {oof_predictions.shape[1]} base model predictions...")
    logger.info(f"  Training set: {len(meta_X_train)} games")
    logger.info(f"  Validation set: {len(meta_X_val)} games")
    logger.info(f"  Alpha (prior precision): {alpha}")
    logger.info(f"  Beta (noise precision): {beta}")

    # Create and train Bayesian model
    bayesian_model = BayesianLinearModel(
        alpha=alpha,
        beta=beta,
        name="bayesian_meta"
    )

    bayesian_model.fit(meta_X_train, y_train)

    # Evaluate on validation set
    logger.info("\nEvaluating on validation set...")

    val_probs = bayesian_model.predict_proba(meta_X_val)
    val_preds = (val_probs > 0.5).astype(int)

    auc = roc_auc_score(y_val, val_probs)
    accuracy = accuracy_score(y_val, val_preds)
    brier = brier_score_loss(y_val, val_probs)

    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Brier Score: {brier:.4f}")

    # Get uncertainty estimates
    uncertainty = bayesian_model.get_uncertainty(meta_X_val)
    logger.info(f"\nPosterior Uncertainty Stats:")
    logger.info(f"  Mean: {uncertainty.mean():.4f}")
    logger.info(f"  Std: {uncertainty.std():.4f}")
    logger.info(f"  Min: {uncertainty.min():.4f}")
    logger.info(f"  Max: {uncertainty.max():.4f}")

    # Get credible intervals
    ci_lower, ci_upper = bayesian_model.get_credible_intervals(meta_X_val, alpha=0.1)
    ci_width = ci_upper - ci_lower
    logger.info(f"\n90% Credible Interval Stats:")
    logger.info(f"  Mean width: {ci_width.mean():.4f}")
    logger.info(f"  Median width: {np.median(ci_width):.4f}")

    # Check calibration (what % of true values fall in CI)
    in_ci = (y_val >= ci_lower) & (y_val <= ci_upper)
    coverage = in_ci.mean()
    logger.info(f"  Empirical coverage: {coverage:.2%} (target: 90%)")

    # Feature importance
    logger.info("\nBase Model Importance:")
    importance_df = bayesian_model.feature_importance()
    for _, row in importance_df.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.3f}")

    # Save model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bayesian_model.save(str(output_path))
    logger.info(f"\nSaved Bayesian meta-learner to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("BAYESIAN META-LEARNER TRAINING COMPLETE")
    print("=" * 60)
    print(f"Base models used: {len(meta_X_train.columns)}")
    print(f"Alpha (regularization): {alpha}")
    print(f"Beta (noise precision): {beta}")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Brier: {brier:.4f}")
    print(f"CI Coverage: {coverage:.2%}")
    print("=" * 60)

    return bayesian_model


def verify_model(model_path: str = "models/bayesian_meta.pkl"):
    """Verify the Bayesian model loads and works."""
    logger.info(f"Verifying model at {model_path}...")

    model = BayesianLinearModel.load(model_path)

    if not model.is_fitted:
        logger.error("Model is not fitted!")
        return False

    logger.info(f"Model name: {model.name}")
    logger.info(f"Features: {len(model.feature_columns)}")

    # Try a prediction with dummy data
    sample = pd.DataFrame(
        np.random.rand(10, len(model.feature_columns)),
        columns=model.feature_columns
    )

    probs = model.predict_proba(sample)
    uncertainty = model.get_uncertainty(sample)
    ci_lower, ci_upper = model.get_credible_intervals(sample, alpha=0.1)

    logger.info(f"Sample predictions: {probs[:5]}")
    logger.info(f"Sample uncertainty: {uncertainty[:5]}")
    logger.info(f"Sample CI lower: {ci_lower[:5]}")
    logger.info(f"Sample CI upper: {ci_upper[:5]}")

    logger.info("Model verification passed!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bayesian Meta-Learner")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Prior precision (regularization)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Noise precision")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of CV folds for OOF predictions")
    parser.add_argument("--output", type=str, default="models/bayesian_meta.pkl",
                        help="Output path for model")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing model instead of training")

    args = parser.parse_args()

    if args.verify:
        verify_model(args.output)
    else:
        train_bayesian_meta(
            alpha=args.alpha,
            beta=args.beta,
            n_folds=args.folds,
            output_path=args.output,
        )
