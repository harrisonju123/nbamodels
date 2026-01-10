"""
Retrain Point Spread Model with Elo Features

Ensures Elo differential and other important features are included.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger


def retrain_spread_model(
    features_path: str = "data/features/game_features.parquet",
    output_path: str = "models/point_spread_model_tuned.pkl",
):
    """Retrain spread model ensuring Elo features are included."""

    # Load features
    df = pd.read_parquet(features_path)
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} games")

    # Check for Elo features
    elo_cols = [c for c in df.columns if 'elo' in c.lower()]
    logger.info(f"Elo features found: {elo_cols}")

    # Exclude leaky and non-feature columns
    exclude = {
        'game_id', 'date', 'home_team', 'away_team', 'home_score',
        'away_score', 'point_diff', 'total_points', 'home_win', 'season',
        'away_win', 'home_total_points', 'away_total_points',
        'home_point_diff', 'away_point_diff',
    }

    # Get all numeric feature columns
    feature_cols = [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in ['float64', 'int64', 'int32', 'float32']
    ]

    # Ensure Elo is included
    for col in ['elo_diff', 'home_elo', 'away_elo', 'elo_prob']:
        if col in df.columns and col not in feature_cols:
            feature_cols.append(col)
            logger.info(f"Added missing Elo feature: {col}")

    logger.info(f"Total features: {len(feature_cols)}")

    # Check if Elo is in features
    elo_in_features = [c for c in feature_cols if 'elo' in c.lower()]
    logger.info(f"Elo features in training: {elo_in_features}")

    # Time-based split: 70% train, 15% val, 15% test
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))

    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["point_diff"]
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df["point_diff"]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["point_diff"]

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Use same hyperparameters as the tuned model but train on new features
    params = {
        'max_depth': 3,
        'learning_rate': 0.012,
        'n_estimators': 471,
        'subsample': 0.635,
        'colsample_bytree': 0.83,
        'min_child_weight': 5,
        'reg_alpha': 0.009,
        'reg_lambda': 0.245,
        'gamma': 1.85,
        'random_state': 42,
        'n_jobs': -1,
    }

    # Train model
    logger.info("Training XGBoost model...")
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Evaluate
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    val_mae = np.mean(np.abs(y_val - val_preds))
    val_rmse = np.sqrt(np.mean((y_val - val_preds) ** 2))
    test_mae = np.mean(np.abs(y_test - test_preds))
    test_rmse = np.sqrt(np.mean((y_test - test_preds) ** 2))
    calibrated_std = np.std(y_test - test_preds)

    logger.info(f"Validation - MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}")
    logger.info(f"Test - MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}")
    logger.info(f"Calibrated std: {calibrated_std:.2f}")

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nTop 20 Features:")
    print(importance.head(20).to_string())

    print("\nElo Feature Importance:")
    elo_imp = importance[importance['feature'].str.contains('elo', case=False)]
    print(elo_imp.to_string())

    # Save model
    model_data = {
        "model": model,
        "feature_columns": feature_cols,
        "feature_importance": importance,
        "calibrated_std": calibrated_std,
        "params": params,
        "metrics": {
            "mae": test_mae,
            "rmse": test_rmse,
            "std_error": calibrated_std,
        },
        "tuned_at": datetime.now().isoformat(),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)

    logger.info(f"Model saved to {output_path}")

    return model, feature_cols, importance


if __name__ == "__main__":
    model, features, importance = retrain_spread_model()
