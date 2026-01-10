"""
Retrain Spread Model with Tuned Hyperparameters

Uses optimized hyperparameters from Optuna tuning to retrain the production spread model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
from loguru import logger

from src.models.spread_model import SpreadPredictionModel
from src.features.game_features import GameFeatureBuilder


def load_tuned_params(params_path: str = "models/tuned_params_spread.json") -> dict:
    """Load tuned hyperparameters from JSON file."""
    with open(params_path, 'r') as f:
        data = json.load(f)

    params = data['best_params'].copy()

    # Add fixed params
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['random_state'] = 42
    params['n_jobs'] = -1

    logger.info(f"Loaded tuned params from {params_path}")
    logger.info(f"Tuning date: {data['tuning_date']}")
    logger.info(f"Training samples: {data['training_samples']}")
    logger.info(f"Composite score: {data['metrics']['composite_score']:.4f}")
    logger.info(f"AUC: {data['metrics']['auc']:.4f}")
    logger.info(f"Log loss: {data['metrics']['log_loss']:.4f}")

    return params


def train_with_tuned_params(
    output_path: str = "models/spread_model_tuned.pkl",
    compare_to_baseline: bool = True
):
    """
    Train spread model with tuned hyperparameters.

    Args:
        output_path: Where to save the tuned model
        compare_to_baseline: If True, also train baseline for comparison
    """
    logger.info("=" * 60)
    logger.info("RETRAINING SPREAD MODEL WITH TUNED HYPERPARAMETERS")
    logger.info("=" * 60)

    # Load data
    logger.info("\n1. Loading training data...")
    games = pd.read_parquet("data/raw/games_with_spread_coverage.parquet")
    logger.info(f"Using {len(games)} games with spread coverage")

    # Build features
    logger.info("\n2. Building features...")
    feature_builder = GameFeatureBuilder(
        use_referee_features=False,
        use_news_features=False,
        use_sentiment_features=False
    )
    features_df = feature_builder.build_game_features(games)

    # DEDUPLICATE by game_id (feature builder creates multiple rows per game)
    features_df = features_df.drop_duplicates(subset=['game_id'], keep='first')
    logger.info(f"Deduplicated features to {len(features_df)} unique games")

    # Add spread and coverage target from games (inner join to only keep games with spread data)
    features_df = features_df.merge(
        games[['game_id', 'spread', 'home_cover']],
        on='game_id',
        how='inner'  # Only keep games that have spread coverage
    )
    logger.info(f"Merged to {len(features_df)} games with spread coverage (should be 641)")

    # Get feature columns (CRITICAL: Exclude ALL target-derived features to prevent leakage)
    exclude_cols = [
        # Identifiers
        'game_id', 'date', 'season', 'home_team', 'away_team',
        'home_team_id', 'away_team_id', 'status',
        # Direct target leakage
        'home_score', 'away_score', 'home_win', 'away_win',
        'point_diff', 'home_point_diff', 'away_point_diff',
        'total_points', 'home_margin', 'away_margin', 'point_margin',
        # Derived from scores
        'home_cover', 'away_cover', 'over', 'under',
        'actual_total', 'actual_spread', 'actual_margin',
        # Spread data (don't use spread as a feature)
        'spread', 'spread_odds', 'spread_implied_prob'
    ]
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if features_df[c].notna().any()]

    logger.info(f"Features: {len(feature_cols)}")

    # Prepare X and y
    X = features_df[feature_cols].fillna(0)
    y = features_df['home_cover']  # CHANGED: Predict spread coverage, not game winner

    # Split into train/val (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(f"Train: {len(X_train)} games")
    logger.info(f"Val: {len(X_val)} games")

    # Load tuned parameters
    logger.info("\n3. Loading tuned hyperparameters...")
    tuned_params = load_tuned_params()

    # Train model with tuned params
    logger.info("\n4. Training model with tuned parameters...")
    tuned_model = SpreadPredictionModel(params=tuned_params)
    tuned_model.fit(X_train, y_train, X_val, y_val, feature_columns=feature_cols)

    # Evaluate tuned model
    logger.info("\n5. Evaluating tuned model...")
    train_metrics = tuned_model.evaluate(X_train, y_train)
    val_metrics = tuned_model.evaluate(X_val, y_val)

    logger.info("\n" + "-" * 60)
    logger.info("TUNED MODEL RESULTS:")
    logger.info("-" * 60)
    logger.info(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
    logger.info(f"Train AUC: {train_metrics['roc_auc']:.4f}")
    logger.info(f"Train Log Loss: {train_metrics['log_loss']:.4f}")
    logger.info(f"Train Brier: {train_metrics['brier_score']:.6f}")
    logger.info("")
    logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
    logger.info(f"Val AUC: {val_metrics['roc_auc']:.4f}")
    logger.info(f"Val Log Loss: {val_metrics['log_loss']:.4f}")
    logger.info(f"Val Brier: {val_metrics['brier_score']:.6f}")

    # Compare to baseline if requested
    if compare_to_baseline:
        logger.info("\n6. Training baseline model for comparison...")
        baseline_model = SpreadPredictionModel()  # Uses DEFAULT_PARAMS
        baseline_model.fit(X_train, y_train, X_val, y_val, feature_columns=feature_cols)

        baseline_train = baseline_model.evaluate(X_train, y_train)
        baseline_val = baseline_model.evaluate(X_val, y_val)

        logger.info("\n" + "-" * 60)
        logger.info("BASELINE MODEL RESULTS:")
        logger.info("-" * 60)
        logger.info(f"Train Accuracy: {baseline_train['accuracy']:.4f}")
        logger.info(f"Train AUC: {baseline_train['roc_auc']:.4f}")
        logger.info(f"Train Log Loss: {baseline_train['log_loss']:.4f}")
        logger.info(f"Train Brier: {baseline_train['brier_score']:.6f}")
        logger.info("")
        logger.info(f"Val Accuracy: {baseline_val['accuracy']:.4f}")
        logger.info(f"Val AUC: {baseline_val['roc_auc']:.4f}")
        logger.info(f"Val Log Loss: {baseline_val['log_loss']:.4f}")
        logger.info(f"Val Brier: {baseline_val['brier_score']:.6f}")

        # Calculate improvements
        logger.info("\n" + "=" * 60)
        logger.info("IMPROVEMENT OVER BASELINE:")
        logger.info("=" * 60)
        logger.info(f"Val Accuracy: {val_metrics['accuracy'] - baseline_val['accuracy']:+.4f} ({(val_metrics['accuracy'] / baseline_val['accuracy'] - 1) * 100:+.2f}%)")
        logger.info(f"Val AUC: {val_metrics['roc_auc'] - baseline_val['roc_auc']:+.4f} ({(val_metrics['roc_auc'] / baseline_val['roc_auc'] - 1) * 100:+.2f}%)")
        logger.info(f"Val Log Loss: {val_metrics['log_loss'] - baseline_val['log_loss']:+.4f} ({(val_metrics['log_loss'] / baseline_val['log_loss'] - 1) * 100:+.2f}%) [lower is better]")
        logger.info(f"Val Brier: {val_metrics['brier_score'] - baseline_val['brier_score']:+.6f} ({(val_metrics['brier_score'] / baseline_val['brier_score'] - 1) * 100:+.2f}%) [lower is better]")

    # Save tuned model
    logger.info(f"\n7. Saving tuned model to {output_path}...")
    tuned_model.save(output_path)

    # Save metadata
    metadata_path = output_path.replace('.pkl', '_metadata.json')
    metadata = {
        'tuned_params': tuned_params,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'n_features': len(feature_cols),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'training_date': pd.Timestamp.now().isoformat()
    }

    if compare_to_baseline:
        metadata['baseline_val_metrics'] = baseline_val
        metadata['improvement'] = {
            'accuracy': val_metrics['accuracy'] - baseline_val['accuracy'],
            'roc_auc': val_metrics['roc_auc'] - baseline_val['roc_auc'],
            'log_loss': val_metrics['log_loss'] - baseline_val['log_loss'],
            'brier_score': val_metrics['brier_score'] - baseline_val['brier_score']
        }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_path}")

    # Print top features
    logger.info("\n" + "=" * 60)
    logger.info("TOP 15 FEATURE IMPORTANCES:")
    logger.info("=" * 60)
    print(tuned_model.feature_importance.head(15).to_string(index=False))

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"✅ Tuned model saved to: {output_path}")
    logger.info(f"✅ Metadata saved to: {metadata_path}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Review validation metrics above")
    logger.info("2. Run backtest: python scripts/backtest_tuned_model.py")
    logger.info("3. If backtest shows improvement, deploy to production")
    logger.info("=" * 60)

    return tuned_model, val_metrics


if __name__ == "__main__":
    train_with_tuned_params()
