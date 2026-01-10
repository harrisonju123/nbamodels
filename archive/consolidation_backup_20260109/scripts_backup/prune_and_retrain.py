#!/usr/bin/env python3
"""
Feature Pruning and Model Retraining

Removes duplicate/highly correlated features and retrains model for better efficiency.

Usage:
    python scripts/prune_and_retrain.py

    # Use custom correlation threshold
    python scripts/prune_and_retrain.py --correlation-threshold 0.95

    # Skip retraining, just show features to remove
    python scripts/prune_and_retrain.py --dry-run
"""

import sys
sys.path.insert(0, '.')

import argparse
import pickle
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from loguru import logger


# Features to remove based on perfect correlation (r = 1.0) or redundancy
FEATURES_TO_REMOVE = [
    # Perfect duplicates (r = 1.0)
    'home_is_b2b',           # Duplicate of home_b2b
    'away_is_b2b',           # Duplicate of away_b2b
    'home_total_impact',     # Duplicate of home_lineup_impact
    'away_total_impact',     # Duplicate of away_lineup_impact
    'missing_impact_diff',   # Low importance, redundant with lineup_impact_diff

    # Win streak features (5g is duplicate of 20g with r=1.0)
    'home_win_streak_5g',    # Keep 20g version
    'away_win_streak_5g',    # Keep 20g version
    'diff_win_streak_5g',    # Keep 20g version

    # High correlation features (r > 0.95)
    'rest_advantage',        # r=0.955 with rest_diff
    'elo_prob',              # r=0.979 with elo_diff
    'h2h_home_margin',       # r=0.979 with h2h_recency_weighted_margin

    # Injury features (low importance, redundant)
    'home_injury_pct',       # Captured by lineup_impact
    'away_injury_pct',       # Captured by lineup_impact
    'home_n_injured',        # Captured by lineup_impact
    'away_n_injured',        # Captured by lineup_impact
    'home_missing_impact',   # Captured by lineup_impact
    'away_missing_impact',   # Captured by lineup_impact
]


def load_current_model():
    """Load current model and metadata."""
    logger.info("Loading current model...")

    with open('models/spread_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('models/spread_model.metadata.json', 'r') as f:
        metadata = json.load(f)

    logger.info(f"Current model: {metadata['n_features']} features")

    return model, metadata


def create_pruned_feature_list(original_features, features_to_remove):
    """Create pruned feature list."""
    logger.info("\nCreating pruned feature list...")

    pruned_features = [f for f in original_features if f not in features_to_remove]

    removed_count = len(original_features) - len(pruned_features)
    reduction_pct = (removed_count / len(original_features)) * 100

    logger.info(f"Original features: {len(original_features)}")
    logger.info(f"Pruned features: {len(pruned_features)}")
    logger.info(f"Removed: {removed_count} features ({reduction_pct:.1f}% reduction)")

    # Show what was removed
    logger.info("\nRemoved features:")
    for feat in features_to_remove:
        if feat in original_features:
            logger.info(f"  - {feat}")

    # Show features that weren't in original list
    not_found = [f for f in features_to_remove if f not in original_features]
    if not_found:
        logger.warning(f"\nFeatures not found in original list: {not_found}")

    return pruned_features


def retrain_with_pruned_features(pruned_features, split_date='2023-10-01'):
    """Retrain model with pruned features."""
    logger.info("\n" + "=" * 80)
    logger.info("RETRAINING MODEL WITH PRUNED FEATURES")
    logger.info("=" * 80)

    # Import training dependencies
    from src.models.dual_model import DualPredictionModel
    from src.features.game_features import GameFeatureBuilder
    from src.features.team_features import TeamFeatureBuilder

    # Load data
    logger.info("Loading data...")
    games = pd.read_parquet('data/raw/games.parquet')
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

    # Merge
    games['date'] = pd.to_datetime(games['date'])
    games_with_odds = games.merge(
        odds[['date', 'home_team', 'away_team', 'spread', 'total']],
        on=['date', 'home_team', 'away_team'],
        how='left'
    ).rename(columns={'spread': 'spread_home'})

    logger.info(f"Loaded {len(games)} games")

    # Build features (disable Four Factors)
    original_add_four_factors = TeamFeatureBuilder.add_four_factors
    TeamFeatureBuilder.add_four_factors = lambda self, df: df

    builder = GameFeatureBuilder()
    features = builder.build_game_features(games_with_odds.copy())

    # Add targets
    features['spread_home'] = games_with_odds['spread_home']
    features['total'] = games_with_odds['total']
    features['point_diff'] = games_with_odds['home_score'] - games_with_odds['away_score']
    features['home_covers'] = (features['point_diff'] + features['spread_home'] > 0).astype(int)

    # Filter
    features = features[
        features['point_diff'].notna() &
        features['spread_home'].notna()
    ].copy()

    logger.info(f"Valid games: {len(features)}")

    # Verify pruned features exist
    missing_features = [f for f in pruned_features if f not in features.columns]
    if missing_features:
        logger.error(f"Missing features: {missing_features[:10]}...")
        raise ValueError(f"Missing {len(missing_features)} features in data")

    # Split
    split_date = pd.to_datetime(split_date)
    train_df = features[features['date'] < split_date].copy()
    test_df = features[features['date'] >= split_date].copy()

    logger.info(f"Train: {len(train_df)} games")
    logger.info(f"Test: {len(test_df)} games")

    # Train model
    logger.info("\nTraining pruned model...")

    X_train = train_df[pruned_features].fillna(0)
    y_train = train_df['home_covers']

    X_test = test_df[pruned_features].fillna(0)
    y_test = test_df['home_covers']

    model = DualPredictionModel()
    model.fit(X_train, y_train)

    logger.success("✓ Model trained")

    # Evaluate
    logger.info("\nEvaluating pruned model...")

    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

    pred = model.predict_proba(X_test)
    if isinstance(pred, dict):
        pred = pred['ensemble']

    metrics = {
        'accuracy': accuracy_score(y_test, (pred > 0.5).astype(int)),
        'auc': roc_auc_score(y_test, pred),
        'log_loss': log_loss(y_test, pred),
        'calibration_error': abs(pred.mean() - y_test.mean()),
    }

    logger.info(f"Pruned model accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"AUC-ROC: {metrics['auc']:.4f}")
    logger.info(f"Log Loss: {metrics['log_loss']:.4f}")
    logger.info(f"Calibration Error: {metrics['calibration_error']:.4f}")

    return model, metrics, train_df, test_df


def compare_models(original_metadata, pruned_metrics):
    """Compare original and pruned models."""
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 80)

    orig_metrics = original_metadata['metrics']

    comparison = {
        'features': {
            'original': original_metadata['n_features'],
            'pruned': original_metadata['n_features'] - len(FEATURES_TO_REMOVE),
            'change': -len(FEATURES_TO_REMOVE),
        },
        'accuracy': {
            'original': orig_metrics['accuracy'],
            'pruned': pruned_metrics['accuracy'],
            'change': pruned_metrics['accuracy'] - orig_metrics['accuracy'],
        },
        'auc': {
            'original': orig_metrics['auc'],
            'pruned': pruned_metrics['auc'],
            'change': pruned_metrics['auc'] - orig_metrics['auc'],
        },
        'log_loss': {
            'original': orig_metrics['log_loss'],
            'pruned': pruned_metrics['log_loss'],
            'change': pruned_metrics['log_loss'] - orig_metrics['log_loss'],
        },
        'calibration': {
            'original': orig_metrics['calibration_error'],
            'pruned': pruned_metrics['calibration_error'],
            'change': pruned_metrics['calibration_error'] - orig_metrics['calibration_error'],
        },
    }

    # Print comparison
    print("\n" + "=" * 80)
    print("ORIGINAL vs PRUNED MODEL")
    print("=" * 80)
    print(f"\nFeatures:     {comparison['features']['original']} → {comparison['features']['pruned']} "
          f"({comparison['features']['change']:+d})")
    print(f"Accuracy:     {comparison['accuracy']['original']:.4f} → {comparison['accuracy']['pruned']:.4f} "
          f"({comparison['accuracy']['change']:+.4f})")
    print(f"AUC-ROC:      {comparison['auc']['original']:.4f} → {comparison['auc']['pruned']:.4f} "
          f"({comparison['auc']['change']:+.4f})")
    print(f"Log Loss:     {comparison['log_loss']['original']:.4f} → {comparison['log_loss']['pruned']:.4f} "
          f"({comparison['log_loss']['change']:+.4f})")
    print(f"Calibration:  {comparison['calibration']['original']:.4f} → {comparison['calibration']['pruned']:.4f} "
          f"({comparison['calibration']['change']:+.4f})")

    # Recommendation
    print("\n" + "=" * 80)

    if comparison['accuracy']['change'] >= 0:
        print("✅ RECOMMENDATION: Deploy pruned model")
        print(f"   - Same or better accuracy with {abs(comparison['features']['change'])} fewer features")
        print(f"   - {abs(comparison['features']['change']/comparison['features']['original']*100):.1f}% feature reduction")
        print("   - Faster predictions, lower overfitting risk")
    else:
        acc_loss_pct = abs(comparison['accuracy']['change']) * 100
        if acc_loss_pct < 1.0:
            print("⚠️  RECOMMENDATION: Deploy pruned model (minor accuracy trade-off)")
            print(f"   - {acc_loss_pct:.2f}% accuracy loss is acceptable")
            print(f"   - {abs(comparison['features']['change'])} fewer features")
            print("   - Simpler, more robust model")
        else:
            print("❌ RECOMMENDATION: Keep original model")
            print(f"   - Accuracy loss of {acc_loss_pct:.2f}% is too high")
            print("   - Feature reduction not worth the performance cost")

    print("=" * 80)

    return comparison


def save_pruned_model(model, pruned_features, metrics, comparison, output_path='models/spread_model_pruned.pkl'):
    """Save pruned model."""
    logger.info(f"\nSaving pruned model to {output_path}...")

    output_path = Path(output_path)

    # Save model
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    # Save metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'model_type': 'pruned',
        'split_date': '2023-10-01',
        'n_features': len(pruned_features),
        'n_features_removed': len(FEATURES_TO_REMOVE),
        'features_removed': FEATURES_TO_REMOVE,
        'feature_columns': pruned_features,
        'metrics': metrics,
        'comparison': comparison,
    }

    metadata_path = output_path.with_suffix('.metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.success(f"✓ Model saved to {output_path}")
    logger.success(f"✓ Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Prune features and retrain model')
    parser.add_argument('--dry-run', action='store_true', help='Show features to remove without retraining')
    parser.add_argument('--output', type=str, default='models/spread_model_pruned.pkl', help='Output path')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("FEATURE PRUNING AND MODEL RETRAINING")
    logger.info("=" * 80)

    try:
        # Load current model
        original_model, original_metadata = load_current_model()
        original_features = original_metadata['feature_columns']

        # Create pruned feature list
        pruned_features = create_pruned_feature_list(original_features, FEATURES_TO_REMOVE)

        if args.dry_run:
            logger.info("\nDry run complete - no retraining performed")
            return 0

        # Retrain with pruned features
        pruned_model, pruned_metrics, train_df, test_df = retrain_with_pruned_features(pruned_features)

        # Compare models
        comparison = compare_models(original_metadata, pruned_metrics)

        # Save pruned model
        save_pruned_model(pruned_model, pruned_features, pruned_metrics, comparison, args.output)

        logger.info("\n" + "=" * 80)
        logger.success("✅ FEATURE PRUNING COMPLETE")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.exception(f"Error during pruning: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
