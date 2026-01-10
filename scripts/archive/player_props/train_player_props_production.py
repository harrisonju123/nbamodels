#!/usr/bin/env python3
"""
Train Player Props Models - Production Version

Trains final models on the FULL dataset (no train/test split) for production deployment.
Includes model versioning and metadata tracking.

Usage:
    python scripts/train_player_props_production.py
    python scripts/train_player_props_production.py --prop-types pts reb
"""

import argparse
import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def prepare_training_data(df: pd.DataFrame, prop_type: str) -> tuple:
    """
    Prepare features and target for training.

    Args:
        df: Feature dataframe
        prop_type: Target variable (pts, reb, ast, fg3m)

    Returns:
        X, y (features, target)
    """
    logger.info(f"Preparing training data for {prop_type.upper()}...")

    # Sort by date
    df = df.sort_values('game_date').copy()

    # Define target
    target_col = prop_type.lower()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    # Remove rows with missing target
    df = df[df[target_col].notna()].copy()

    # Feature columns (exclude metadata and target)
    exclude_cols = [
        'game_id', 'player_id', 'player_name', 'team_abbreviation', 'game_date',
        'pts', 'reb', 'ast', 'fg3m', 'fgm', 'fga', 'ftm', 'fta',
        'oreb', 'dreb', 'stl', 'blk', 'to', 'pf', 'plus_minus',
        'team_id', 'team_city', 'nickname', 'start_position', 'comment',
        'fg_pct', 'fg3a', 'fg3_pct', 'ft_pct', 'min',
        'home_team', 'away_team', 'opponent_team', 'is_home',
        'season', 'game_date_dt', 'day_of_season'
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Remove any non-numeric features
    X = df[feature_cols].copy()
    X = X.select_dtypes(include=[np.number])

    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with 0
    X = X.fillna(0)

    y = df[target_col].values

    logger.info(f"  Training samples: {len(X):,}")
    logger.info(f"  Features: {len(X.columns)}")
    logger.info(f"  Target range: {y.min():.1f} to {y.max():.1f} (mean: {y.mean():.1f})")
    logger.info(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")

    return X, y, df['game_date']


def train_production_model(
    X: pd.DataFrame,
    y: np.ndarray,
    prop_type: str,
    hyperparams: dict = None
) -> xgb.XGBRegressor:
    """
    Train XGBoost model on full dataset.

    Args:
        X: Features
        y: Target
        prop_type: Prop type name
        hyperparams: Optional hyperparameters (uses defaults if None)

    Returns:
        Trained model
    """
    logger.info(f"Training production model for {prop_type.upper()}...")

    # Default hyperparameters (optimized from backtest)
    if hyperparams is None:
        hyperparams = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }

    model = xgb.XGBRegressor(**hyperparams)

    # Train on full dataset
    model.fit(X, y, verbose=False)

    logger.success(f"  ✓ Model trained ({model.n_estimators} trees)")

    # Calculate in-sample metrics (for reference only)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    logger.info(f"  In-sample MAE: {mae:.2f}")
    logger.info(f"  In-sample RMSE: {rmse:.2f}")
    logger.info(f"  In-sample R²: {r2:.3f}")

    return model


def get_feature_importance(model: xgb.XGBRegressor, feature_names: list, top_n: int = 20) -> pd.DataFrame:
    """Get top N most important features."""
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return feature_importance.head(top_n)


def save_production_model(
    model: xgb.XGBRegressor,
    prop_type: str,
    X: pd.DataFrame,
    y: np.ndarray,
    dates: pd.Series,
    output_dir: str = "models/player_props_production"
) -> str:
    """
    Save model with versioning and metadata.

    Args:
        model: Trained model
        prop_type: Prop type
        X: Training features
        y: Training target
        dates: Training dates
        output_dir: Output directory

    Returns:
        Path to saved model
    """
    logger.info(f"Saving production model for {prop_type.upper()}...")

    # Create versioned directory
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = os.path.join(output_dir, f"v_{version}")
    os.makedirs(version_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(version_dir, f"{prop_type}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    logger.info(f"  Saved model to {model_path}")

    # Get feature importance
    feature_importance = get_feature_importance(model, X.columns.tolist(), top_n=50)
    feature_importance.to_csv(
        os.path.join(version_dir, f"{prop_type}_feature_importance.csv"),
        index=False
    )

    # Calculate in-sample metrics
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    # Create metadata
    metadata = {
        'prop_type': prop_type,
        'version': version,
        'created_at': datetime.now().isoformat(),
        'model_type': 'XGBRegressor',
        'hyperparameters': model.get_params(),
        'training_data': {
            'num_samples': len(X),
            'num_features': len(X.columns),
            'date_range': {
                'start': dates.min().isoformat(),
                'end': dates.max().isoformat()
            },
            'target_stats': {
                'min': float(y.min()),
                'max': float(y.max()),
                'mean': float(y.mean()),
                'std': float(y.std())
            }
        },
        'performance': {
            'in_sample_mae': float(mae),
            'in_sample_rmse': float(rmse),
            'in_sample_r2': float(r2)
        },
        'features': X.columns.tolist(),
        'top_features': feature_importance.head(10).to_dict('records')
    }

    # Save metadata
    metadata_path = os.path.join(version_dir, f"{prop_type}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  Saved metadata to {metadata_path}")

    # Also save to latest directory (for easy access)
    latest_dir = os.path.join(output_dir, "latest")
    os.makedirs(latest_dir, exist_ok=True)

    # Copy model to latest
    latest_model_path = os.path.join(latest_dir, f"{prop_type}_model.pkl")
    with open(latest_model_path, 'wb') as f:
        pickle.dump(model, f)

    # Copy metadata to latest
    latest_metadata_path = os.path.join(latest_dir, f"{prop_type}_metadata.json")
    with open(latest_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.success(f"  ✓ Also saved to {latest_dir} for production use")

    return model_path


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Train production player props models")
    parser.add_argument(
        "--features",
        type=str,
        default="data/features/player_game_features_advanced.parquet",
        help="Path to features"
    )
    parser.add_argument(
        "--prop-types",
        nargs="+",
        default=["pts", "reb", "ast", "fg3m"],
        help="Prop types to train"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/player_props_production",
        help="Output directory for models"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("TRAINING PRODUCTION PLAYER PROPS MODELS")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Training on FULL dataset (no train/test split)")
    logger.info(f"Prop types: {', '.join([p.upper() for p in args.prop_types])}")
    logger.info("")

    # Load features
    logger.info(f"Loading features from {args.features}...")
    df = pd.read_parquet(args.features)
    logger.info(f"  Loaded {len(df):,} player-game records")
    logger.info(f"  Features: {len(df.columns)}")
    logger.info("")

    # Train each prop type
    trained_models = {}

    for prop_type in args.prop_types:
        logger.info("=" * 80)
        logger.info(f"TRAINING: {prop_type.upper()}")
        logger.info("=" * 80)
        logger.info("")

        try:
            # Prepare data
            X, y, dates = prepare_training_data(df, prop_type)

            # Train model
            model = train_production_model(X, y, prop_type)

            # Save model
            model_path = save_production_model(
                model, prop_type, X, y, dates, args.output_dir
            )

            # Store for summary
            trained_models[prop_type] = {
                'model': model,
                'path': model_path,
                'num_features': len(X.columns),
                'num_samples': len(X)
            }

            logger.info("")

        except Exception as e:
            logger.error(f"Error training {prop_type}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info("")

    summary_rows = []
    for prop_type, info in trained_models.items():
        summary_rows.append({
            'Prop Type': prop_type.upper(),
            'Samples': f"{info['num_samples']:,}",
            'Features': info['num_features'],
            'Status': '✅ Trained'
        })

    summary_df = pd.DataFrame(summary_rows)
    logger.info(summary_df.to_string(index=False))
    logger.info("")

    # Production deployment info
    latest_dir = os.path.join(args.output_dir, "latest")
    logger.success("✅ Production models saved!")
    logger.info("")
    logger.info("Production models location:")
    logger.info(f"  {latest_dir}")
    logger.info("")
    logger.info("Model files:")
    for prop_type in trained_models.keys():
        logger.info(f"  - {prop_type}_model.pkl")
        logger.info(f"  - {prop_type}_metadata.json")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Test predictions: python scripts/test_player_props_predictions.py")
    logger.info("  2. Add to daily pipeline: Edit config/multi_strategy_config.yaml")
    logger.info("  3. Run paper trading: python scripts/daily_multi_strategy_pipeline.py")
    logger.info("")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
