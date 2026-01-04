#!/usr/bin/env python3
"""
Model Retraining with Alternative Data Features

Retrains the NBA betting model with:
1. Latest data through January 2026
2. All alternative data features (referee, lineup, news, sentiment)
3. Improved train/test split
4. Performance comparison with old model

Usage:
    python scripts/retrain_model_with_alt_data.py

    # Save model to specific path
    python scripts/retrain_model_with_alt_data.py --output-path models/model_2026_01_04.pkl

    # Use custom train/test split date
    python scripts/retrain_model_with_alt_data.py --split-date 2023-10-01
"""

import sys
sys.path.insert(0, '.')

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

from src.models.dual_model import DualPredictionModel
from src.features.game_features import GameFeatureBuilder
from src.features.team_features import TeamFeatureBuilder

# Constants
DEFAULT_SPLIT_DATE = '2023-10-01'  # Updated to include 2022-23 season in training
DEFAULT_MODEL_PATH = 'models/nba_model_retrained.pkl'

# Skip Four Factors (has data leakage in historical data)
original_add_four_factors = TeamFeatureBuilder.add_four_factors
TeamFeatureBuilder.add_four_factors = lambda self, df: df


def load_and_prepare_data():
    """Load games and odds data."""
    logger.info("Loading data...")

    # Load games
    games = pd.read_parquet('data/raw/games.parquet')
    logger.info(f"Loaded {len(games)} games")
    logger.info(f"Date range: {games['date'].min()} to {games['date'].max()}")

    # Load odds
    odds = pd.read_csv('data/raw/historical_odds.csv')

    # Team mapping
    TEAM_MAP = {
        'atl': 'ATL', 'bkn': 'BKN', 'bos': 'BOS', 'cha': 'CHA', 'chi': 'CHI',
        'cle': 'CLE', 'dal': 'DAL', 'den': 'DEN', 'det': 'DET', 'gs': 'GSW',
        'hou': 'HOU', 'ind': 'IND', 'lac': 'LAC', 'lal': 'LAL', 'mem': 'MEM',
        'mia': 'MIA', 'mil': 'MIL', 'min': 'MIN', 'no': 'NOP', 'nop': 'NOP',
        'ny': 'NYK', 'nyk': 'NYK', 'okc': 'OKC', 'orl': 'ORL', 'phi': 'PHI',
        'phx': 'PHX', 'por': 'POR', 'sac': 'SAC', 'sa': 'SAS', 'tor': 'TOR',
        'uta': 'UTA', 'was': 'WAS',
    }

    odds['home_team'] = odds['home'].map(TEAM_MAP)
    odds['away_team'] = odds['away'].map(TEAM_MAP)
    odds['date'] = pd.to_datetime(odds['date'])

    # Merge odds
    games['date'] = pd.to_datetime(games['date'])
    games_with_odds = games.merge(
        odds[['date', 'home_team', 'away_team', 'spread', 'total']],
        on=['date', 'home_team', 'away_team'],
        how='left'
    ).rename(columns={'spread': 'spread_home'})

    logger.info(f"Games with spreads: {games_with_odds['spread_home'].notna().sum()}")

    return games_with_odds


def build_features(games_with_odds):
    """Build all features including alternative data."""
    logger.info("Building features with alternative data...")

    builder = GameFeatureBuilder()
    features = builder.build_game_features(games_with_odds.copy())

    # Preserve odds
    features['spread_home'] = games_with_odds['spread_home']
    features['total'] = games_with_odds['total']

    # Get feature columns
    feature_cols = builder.get_feature_columns(features)
    logger.info(f"Total features: {len(feature_cols)}")

    # Log which alternative data features are present
    alt_data_features = {
        'Referee': [c for c in feature_cols if 'ref_' in c],
        'Lineup': [c for c in feature_cols if 'lineup' in c or 'confirmed_' in c],
        'News': [c for c in feature_cols if 'news_' in c],
        'Sentiment': [c for c in feature_cols if 'sentiment' in c]
    }

    logger.info("\nAlternative Data Features:")
    for category, feats in alt_data_features.items():
        if feats:
            logger.info(f"  {category}: {len(feats)} features")
            logger.debug(f"    {feats}")
        else:
            logger.warning(f"  {category}: No features found")

    return features, feature_cols


def create_targets_and_filter(features, games_with_odds):
    """Create training targets and filter data."""
    logger.info("Creating targets and filtering...")

    # Add targets
    features['point_diff'] = games_with_odds['home_score'] - games_with_odds['away_score']
    features['home_win'] = (features['point_diff'] > 0).astype(int)

    # Validate spread_home column exists
    if 'spread_home' not in features.columns:
        raise ValueError("Missing 'spread_home' column in features DataFrame")

    # Check for missing spread data
    missing_spreads = features['spread_home'].isna().sum()
    if missing_spreads > 0:
        logger.warning(f"{missing_spreads} games missing spread data - will be filtered out")

    # CRITICAL: Train on spread coverage, not game wins!
    features['home_covers'] = (features['point_diff'] + features['spread_home'] > 0).astype(int)

    # Filter to games with scores and spreads
    features = features[
        features['point_diff'].notna() &
        features['spread_home'].notna()
    ].copy()

    # Validate we have data after filtering
    if features.empty:
        raise ValueError("No valid game data available for training")

    logger.info(f"Valid games after filtering: {len(features)}")

    return features


def train_test_split(features, split_date):
    """Split data into train and test sets."""
    logger.info(f"\nSplitting data at {split_date}...")

    split_date = pd.to_datetime(split_date)

    train_df = features[features['date'] < split_date].copy()
    test_df = features[features['date'] >= split_date].copy()

    logger.info(f"Train period: {train_df['date'].min()} to {train_df['date'].max()}")
    logger.info(f"Test period: {test_df['date'].min()} to {test_df['date'].max()}")
    logger.info(f"Train games: {len(train_df)}")
    logger.info(f"Test games: {len(test_df)}")

    # Check class balance
    train_balance = train_df['home_covers'].mean()
    test_balance = test_df['home_covers'].mean()
    logger.info(f"Train home_covers rate: {train_balance:.1%}")
    logger.info(f"Test home_covers rate: {test_balance:.1%}")

    return train_df, test_df


def train_model(train_df, feature_cols):
    """Train the dual prediction model."""
    logger.info("\nTraining model...")

    X_train = train_df[feature_cols]
    y_train = train_df['home_covers']

    # Check for any NaN values in features
    nan_cols = X_train.columns[X_train.isna().any()].tolist()
    if nan_cols:
        logger.warning(f"Features with NaN values: {nan_cols[:10]}...")
        logger.info("Filling NaN values with 0")
        X_train = X_train.fillna(0)

    model = DualPredictionModel()
    model.fit(X_train, y_train)

    logger.success("✓ Model trained successfully")

    return model


def evaluate_model(model, test_df, feature_cols):
    """Evaluate model on test set."""
    logger.info("\nEvaluating model on test set...")

    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['home_covers']

    # Get predictions
    probs = model.predict_proba(X_test)
    if isinstance(probs, dict):
        predictions = probs['ensemble']
    else:
        predictions = probs

    # Calculate metrics
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

    binary_preds = (predictions > 0.5).astype(int)
    accuracy = accuracy_score(y_test, binary_preds)
    auc = roc_auc_score(y_test, predictions)
    logloss = log_loss(y_test, predictions)

    logger.info(f"Test Set Performance:")
    logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"  AUC-ROC: {auc:.4f}")
    logger.info(f"  Log Loss: {logloss:.4f}")

    # Calibration check
    avg_pred = predictions.mean()
    actual_rate = y_test.mean()
    calibration_error = abs(avg_pred - actual_rate)
    logger.info(f"  Avg Prediction: {avg_pred:.4f}")
    logger.info(f"  Actual Rate: {actual_rate:.4f}")
    logger.info(f"  Calibration Error: {calibration_error:.4f}")

    return {
        'accuracy': accuracy,
        'auc': auc,
        'log_loss': logloss,
        'calibration_error': calibration_error,
        'n_test': len(test_df)
    }


def save_model(model, output_path):
    """Save trained model to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    logger.success(f"✓ Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Retrain NBA betting model with latest data')
    parser.add_argument('--split-date', type=str, default=DEFAULT_SPLIT_DATE,
                        help='Train/test split date (YYYY-MM-DD)')
    parser.add_argument('--output-path', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to save trained model')
    parser.add_argument('--skip-save', action='store_true',
                        help='Skip saving model (evaluation only)')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("NBA MODEL RETRAINING WITH ALTERNATIVE DATA")
    logger.info("=" * 80)
    logger.info(f"Split date: {args.split_date}")
    logger.info(f"Output path: {args.output_path}")

    try:
        # Load data
        games_with_odds = load_and_prepare_data()

        # Build features
        features, feature_cols = build_features(games_with_odds)

        # Create targets and filter
        features = create_targets_and_filter(features, games_with_odds)

        # Train/test split
        train_df, test_df = train_test_split(features, args.split_date)

        # Train model
        model = train_model(train_df, feature_cols)

        # Evaluate model
        metrics = evaluate_model(model, test_df, feature_cols)

        # Save model
        if not args.skip_save:
            save_model(model, args.output_path)

            # Save metadata
            metadata = {
                'trained_at': datetime.now().isoformat(),
                'split_date': args.split_date,
                'n_train': len(train_df),
                'n_test': len(test_df),
                'n_features': len(feature_cols),
                'metrics': metrics,
                'feature_columns': feature_cols
            }

            metadata_path = Path(args.output_path).with_suffix('.metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.success(f"✓ Metadata saved to {metadata_path}")

        logger.info("\n" + "=" * 80)
        logger.success("✅ MODEL RETRAINING COMPLETE")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.exception(f"Error during model retraining: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
