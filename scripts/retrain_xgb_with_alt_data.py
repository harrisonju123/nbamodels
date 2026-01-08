#!/usr/bin/env python3
"""
Retrain XGBoost Spread Model WITH Alternative Data

Creates a new spread model that:
1. Uses XGBoost (proven architecture - 42% ROI)
2. Includes alternative data (lineup, news, sentiment, referee)
3. Trains on spread coverage (not just wins)
4. Saves as spread_model_alt_data.pkl

Usage:
    python scripts/retrain_xgb_with_alt_data.py

    # Then backtest before using in production:
    python scripts/run_rigorous_backtest.py --model spread_model_alt_data.pkl
"""

import sys
sys.path.insert(0, '.')

import pickle
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

from src.features.game_features import GameFeatureBuilder
from src.features.team_features import TeamFeatureBuilder

# Skip Four Factors (data leakage issues)
TeamFeatureBuilder.add_four_factors = lambda self, df: df


def load_and_prepare_data():
    """Load games and odds data."""
    logger.info("=" * 70)
    logger.info("LOADING DATA")
    logger.info("=" * 70)

    # Load games
    games = pd.read_parquet('data/raw/games.parquet')
    logger.info(f"✓ Loaded {len(games):,} games")
    logger.info(f"  Date range: {games['date'].min()} to {games['date'].max()}")

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
    merged = games.merge(
        odds[['date', 'home_team', 'away_team', 'spread', 'total']],
        on=['date', 'home_team', 'away_team'],
        how='left'
    ).rename(columns={'spread': 'spread_home'})

    logger.info(f"✓ Games with spreads: {merged['spread_home'].notna().sum():,}")

    return merged


def build_features(games_with_odds):
    """Build all features including alternative data."""
    logger.info("\n" + "=" * 70)
    logger.info("BUILDING FEATURES (WITH ALTERNATIVE DATA)")
    logger.info("=" * 70)

    builder = GameFeatureBuilder()
    features = builder.build_game_features(games_with_odds.copy())

    # Preserve odds
    features['spread_home'] = games_with_odds['spread_home']
    features['total'] = games_with_odds['total']

    # Get feature columns
    feature_cols = builder.get_feature_columns(features)
    logger.info(f"\n✓ Total features: {len(feature_cols)}")

    # Check for alternative data
    alt_data = {
        'Lineup/Impact': [c for c in feature_cols if 'lineup' in c.lower() or 'impact' in c.lower()],
        'News': [c for c in feature_cols if 'news' in c.lower()],
        'Sentiment': [c for c in feature_cols if 'sentiment' in c.lower()],
        'Referee': [c for c in feature_cols if 'ref' in c.lower() or 'official' in c.lower()]
    }

    logger.info("\nAlternative Data Features:")
    total_alt = 0
    for category, feats in alt_data.items():
        if feats:
            logger.info(f"  ✓ {category}: {len(feats)} features")
            total_alt += len(feats)
        else:
            logger.warning(f"  ✗ {category}: No features found")

    logger.info(f"\n✓ Total alternative data features: {total_alt}")
    logger.info(f"✓ Traditional features: {len(feature_cols) - total_alt}")

    return features, feature_cols


def train_model(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier with isotonic calibration."""
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING XGBOOST MODEL")
    logger.info("=" * 70)

    # Train XGBoost
    logger.info("\nTraining XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train, verbose=False)
    logger.success(f"✓ Model trained ({model.n_estimators} trees)")

    # Get probabilities for calibration
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    # Isotonic calibration
    logger.info("\nApplying isotonic calibration...")
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(train_probs, y_train)

    calibrated_train_probs = calibrator.predict(train_probs)
    calibrated_test_probs = calibrator.predict(test_probs)

    logger.success("✓ Calibration complete")

    return model, calibrator, calibrated_test_probs


def evaluate_model(model, calibrator, X_train, y_train, X_test, y_test):
    """Evaluate model performance."""
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION")
    logger.info("=" * 70)

    # Get predictions
    train_probs_raw = model.predict_proba(X_train)[:, 1]
    test_probs_raw = model.predict_proba(X_test)[:, 1]

    train_probs_cal = calibrator.predict(train_probs_raw)
    test_probs_cal = calibrator.predict(test_probs_raw)

    # Calculate metrics
    train_acc = accuracy_score(y_train, (train_probs_cal > 0.5).astype(int))
    test_acc = accuracy_score(y_test, (test_probs_cal > 0.5).astype(int))

    train_auc = roc_auc_score(y_train, train_probs_cal)
    test_auc = roc_auc_score(y_test, test_probs_cal)

    train_brier = brier_score_loss(y_train, train_probs_cal)
    test_brier = brier_score_loss(y_test, test_probs_cal)

    # Calculate calibration error (ECE)
    def expected_calibration_error(y_true, y_prob, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins[1:-1])

        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_acc = y_true[mask].mean()
                bin_conf = y_prob[mask].mean()
                ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)

        return ece

    test_ece = expected_calibration_error(y_test, test_probs_cal)

    logger.info("\nTrain Set:")
    logger.info(f"  Accuracy: {train_acc:.3f}")
    logger.info(f"  AUC: {train_auc:.3f}")
    logger.info(f"  Brier: {train_brier:.4f}")

    logger.info("\nTest Set:")
    logger.info(f"  Accuracy: {test_acc:.3f}")
    logger.info(f"  AUC: {test_auc:.3f}")
    logger.info(f"  Brier: {test_brier:.4f}")
    logger.info(f"  ECE: {test_ece:.4f}")

    metrics = {
        'accuracy': float(test_acc),
        'auc': float(test_auc),
        'brier_score': float(test_brier),
        'calibration_error': float(test_ece),
        'n_train': len(X_train),
        'n_test': len(X_test)
    }

    return metrics


def main():
    """Main execution."""
    logger.info("=" * 70)
    logger.info("RETRAIN XGBOOST WITH ALTERNATIVE DATA")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now()}")
    logger.info("")

    # Load data
    games_with_odds = load_and_prepare_data()

    # Build features
    features, feature_cols = build_features(games_with_odds)

    # Create targets
    logger.info("\n" + "=" * 70)
    logger.info("PREPARING TRAINING DATA")
    logger.info("=" * 70)

    features['point_diff'] = games_with_odds['home_score'] - games_with_odds['away_score']

    # CRITICAL: Train on SPREAD COVERAGE, not game wins!
    features['home_covers'] = (features['point_diff'] + features['spread_home'] > 0).astype(int)

    # Filter to valid games
    features = features[
        features['point_diff'].notna() &
        features['spread_home'].notna()
    ].copy()

    logger.info(f"✓ Valid games: {len(features):,}")

    # Train/test split (chronological)
    split_date = pd.Timestamp('2023-10-01')
    train = features[features['date'] < split_date].copy()
    test = features[features['date'] >= split_date].copy()

    logger.info(f"\nTrain: {len(train):,} games ({train['date'].min()} to {train['date'].max()})")
    logger.info(f"Test:  {len(test):,} games ({test['date'].min()} to {test['date'].max()})")

    # Prepare X, y
    X_train = train[feature_cols].fillna(0).values
    y_train = train['home_covers'].values

    X_test = test[feature_cols].fillna(0).values
    y_test = test['home_covers'].values

    # Train model
    model, calibrator, calibrated_probs = train_model(X_train, y_train, X_test, y_test)

    # Evaluate
    metrics = evaluate_model(model, calibrator, X_train, y_train, X_test, y_test)

    # Save model
    output_path = 'models/spread_model_alt_data.pkl'
    logger.info("\n" + "=" * 70)
    logger.info(f"SAVING MODEL: {output_path}")
    logger.info("=" * 70)

    model_data = {
        'model': model,
        'calibrated_model': calibrator,
        'feature_columns': feature_cols,
        'train_median': train[feature_cols].median().to_dict()
    }

    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)

    logger.success(f"✓ Model saved to {output_path}")

    # Save metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'model_type': 'XGBoost_with_alt_data',
        'split_date': '2023-10-01',
        'n_features': len(feature_cols),
        'n_train': len(train),
        'n_test': len(test),
        'metrics': metrics,
        'feature_columns': feature_cols
    }

    metadata_path = 'models/spread_model_alt_data.metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.success(f"✓ Metadata saved to {metadata_path}")

    # Final instructions
    logger.info("\n" + "=" * 70)
    logger.info("NEXT STEPS")
    logger.info("=" * 70)
    logger.info("")
    logger.info("⚠️  DO NOT USE FOR BETTING YET!")
    logger.info("")
    logger.info("1. Run rigorous backtest:")
    logger.info("   python scripts/run_rigorous_backtest.py")
    logger.info("")
    logger.info("2. Verify ROI > 10% with statistical significance")
    logger.info("")
    logger.info("3. If validated, copy to production:")
    logger.info("   cp models/spread_model_alt_data.pkl models/spread_model.pkl")
    logger.info("")
    logger.info(f"Completed: {datetime.now()}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
