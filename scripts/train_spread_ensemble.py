"""
Train Ensemble Model for Spread Coverage Prediction

Combines XGBoost, LightGBM, and CatBoost with optimized weighting.
Uses proper temporal split to prevent data leakage.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from src.models.xgb_model import XGBSpreadModel
from src.models.lgbm_model import LGBMSpreadModel
from src.models.catboost_model import CatBoostSpreadModel
from src.models.ensemble import EnsembleModel
from src.features.game_features import GameFeatureBuilder


def load_tuned_xgb_params(params_path: str = "models/tuned_params_spread.json") -> dict:
    """Load tuned hyperparameters for XGBoost model."""
    with open(params_path, 'r') as f:
        data = json.load(f)

    params = data['best_params'].copy()

    # Add fixed params
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['random_state'] = 42

    logger.info(f"Loaded tuned XGBoost params from {params_path}")
    logger.info(f"Tuning AUC: {data['metrics']['auc']:.4f}")

    return params


def build_features_for_games(games: pd.DataFrame) -> pd.DataFrame:
    """Build features for a set of games."""
    feature_builder = GameFeatureBuilder(
        use_referee_features=False,
        use_news_features=False,
        use_sentiment_features=False
    )
    features_df = feature_builder.build_game_features(games)

    # Deduplicate by game_id
    features_df = features_df.drop_duplicates(subset=['game_id'], keep='first')

    # Add spread and coverage target from games
    features_df = features_df.merge(
        games[['game_id', 'spread', 'home_cover']],
        on='game_id',
        how='inner'
    )

    return features_df


def get_feature_columns(features_df: pd.DataFrame) -> list:
    """Get feature columns excluding targets and identifiers."""
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
        # Spread data
        'spread', 'spread_odds', 'spread_implied_prob'
    ]
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if features_df[c].notna().any()]

    return feature_cols


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, name: str) -> dict:
    """Evaluate a model and return metrics."""
    pred_proba = model.predict_proba(X)
    pred_binary = (pred_proba > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y, pred_binary),
        'auc': roc_auc_score(y, pred_proba),
        'log_loss': log_loss(y, pred_proba),
    }

    logger.info(f"\n{name}:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  AUC: {metrics['auc']:.4f}")
    logger.info(f"  Log Loss: {metrics['log_loss']:.4f}")

    return metrics


def train_ensemble():
    """Train ensemble model with proper temporal split."""
    logger.info("=" * 80)
    logger.info("TRAINING ENSEMBLE MODEL FOR SPREAD COVERAGE")
    logger.info("=" * 80)

    # Load all data
    logger.info("\n1. Loading data...")
    all_games = pd.read_parquet("data/raw/games_with_spread_coverage.parquet")
    logger.info(f"Loaded {len(all_games)} total games")
    logger.info(f"Seasons available: {sorted(all_games['season'].unique())}")

    # PROPER TEMPORAL SPLIT (prevent data leakage)
    logger.info("\n2. Creating proper temporal split...")

    # Train: 2022 season only
    train_games = all_games[all_games['season'] == 2022].copy()

    # Validation + Test: 2023+ seasons
    future_games = all_games[all_games['season'] >= 2023].copy()
    future_games = future_games.sort_values('date')

    # Validation: First 100 games of 2023 (for weight optimization)
    val_games = future_games.head(100)

    # Test: Remaining games
    test_games = future_games.iloc[100:]

    logger.info(f"Train: {len(train_games)} games (2022 season)")
    logger.info(f"Val: {len(val_games)} games (first 100 of 2023+)")
    logger.info(f"Test: {len(test_games)} games (remaining 2023+)")

    # Build features for each split
    logger.info("\n3. Building features...")

    logger.info("Building training features...")
    train_features = build_features_for_games(train_games)
    logger.info(f"  Train features: {len(train_features)} games")

    logger.info("Building validation features...")
    val_features = build_features_for_games(val_games)
    logger.info(f"  Val features: {len(val_features)} games")

    logger.info("Building test features...")
    test_features = build_features_for_games(test_games)
    logger.info(f"  Test features: {len(test_features)} games")

    # Get feature columns
    feature_cols = get_feature_columns(train_features)
    logger.info(f"\nFeature count: {len(feature_cols)}")

    # Prepare X and y
    X_train = train_features[feature_cols].fillna(0)
    y_train = train_features['home_cover']

    X_val = val_features[feature_cols].fillna(0)
    y_val = val_features['home_cover']

    X_test = test_features[feature_cols].fillna(0)
    y_test = test_features['home_cover']

    logger.info(f"\nTarget distribution:")
    logger.info(f"  Train cover rate: {y_train.mean():.1%}")
    logger.info(f"  Val cover rate: {y_val.mean():.1%}")
    logger.info(f"  Test cover rate: {y_test.mean():.1%}")

    # Load tuned XGBoost params
    logger.info("\n4. Loading tuned XGBoost parameters...")
    xgb_params = load_tuned_xgb_params()

    # Create individual models
    logger.info("\n5. Initializing models...")
    models = [
        XGBSpreadModel(params=xgb_params, name="xgb_spread"),
        LGBMSpreadModel(name="lgbm_spread"),
        CatBoostSpreadModel(name="catboost_spread"),
    ]
    logger.info(f"Created {len(models)} models: {[m.name for m in models]}")

    # Create ensemble with optimized weighting
    logger.info("\n6. Creating ensemble with optimized weighting...")
    ensemble = EnsembleModel(
        models=models,
        weighting="optimized",  # Minimize log-loss on validation
        min_weight=0.1,  # Ensure all models contribute
        name="spread_ensemble"
    )

    # Train ensemble (will train individual models + optimize weights)
    logger.info("\n7. Training ensemble...")
    logger.info("This will:")
    logger.info("  1. Train XGBoost with tuned params")
    logger.info("  2. Train LightGBM with default params")
    logger.info("  3. Train CatBoost with default params")
    logger.info("  4. Optimize ensemble weights on validation set")
    logger.info("")

    ensemble.fit(X_train, y_train, X_val, y_val)

    logger.info(f"\nOptimized weights: {ensemble.weights}")

    # Evaluate on validation set
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SET PERFORMANCE")
    logger.info("=" * 80)

    # Individual models
    for model in models:
        evaluate_model(model, X_val, y_val, model.name)

    # Ensemble
    ensemble_metrics_val = evaluate_model(ensemble, X_val, y_val, "Ensemble")

    # Model diversity (correlation)
    logger.info("\n8. Analyzing model diversity...")
    corr_matrix = ensemble.get_model_correlations(X_val)
    logger.info("\nModel prediction correlations (validation set):")
    print(corr_matrix.to_string())

    # Get average pairwise correlation
    n = len(models)
    if n > 1:
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        pairwise_corrs = corr_matrix.values[mask]
        avg_corr = pairwise_corrs.mean()
        logger.info(f"\nAverage pairwise correlation: {avg_corr:.4f}")
        if avg_corr < 0.85:
            logger.success("✅ Good model diversity (correlation < 0.85)")
        else:
            logger.warning("⚠️ Models may be too similar (correlation >= 0.85)")

    # Evaluate on test set
    logger.info("\n" + "=" * 80)
    logger.info("TEST SET PERFORMANCE (FINAL EVALUATION)")
    logger.info("=" * 80)

    # Individual models
    for model in models:
        evaluate_model(model, X_test, y_test, model.name)

    # Ensemble
    ensemble_metrics_test = evaluate_model(ensemble, X_test, y_test, "Ensemble")

    # Check success criteria
    logger.info("\n" + "=" * 80)
    logger.info("SUCCESS CRITERIA CHECK")
    logger.info("=" * 80)

    # Best individual model AUC on test
    individual_aucs = [evaluate_model(m, X_test, y_test, m.name)['auc'] for m in models]
    best_individual_auc = max(individual_aucs)

    criteria_met = 0
    total_criteria = 4

    # 1. Higher AUC than best individual
    if ensemble_metrics_test['auc'] > best_individual_auc:
        logger.success(f"✅ Ensemble AUC ({ensemble_metrics_test['auc']:.4f}) > Best individual ({best_individual_auc:.4f})")
        criteria_met += 1
    else:
        logger.warning(f"❌ Ensemble AUC ({ensemble_metrics_test['auc']:.4f}) <= Best individual ({best_individual_auc:.4f})")

    # 2. Lower log-loss than best individual
    individual_logloss = [evaluate_model(m, X_test, y_test, m.name)['log_loss'] for m in models]
    best_individual_logloss = min(individual_logloss)

    if ensemble_metrics_test['log_loss'] < best_individual_logloss:
        logger.success(f"✅ Ensemble log-loss ({ensemble_metrics_test['log_loss']:.4f}) < Best individual ({best_individual_logloss:.4f})")
        criteria_met += 1
    else:
        logger.warning(f"❌ Ensemble log-loss ({ensemble_metrics_test['log_loss']:.4f}) >= Best individual ({best_individual_logloss:.4f})")

    # 3. Model diversity
    if avg_corr < 0.85:
        logger.success(f"✅ Model diversity: avg correlation ({avg_corr:.4f}) < 0.85")
        criteria_met += 1
    else:
        logger.warning(f"❌ Model diversity: avg correlation ({avg_corr:.4f}) >= 0.85")

    # 4. AUC > 0.558 (baseline from properly split single model)
    if ensemble_metrics_test['auc'] > 0.558:
        logger.success(f"✅ Ensemble AUC ({ensemble_metrics_test['auc']:.4f}) > Baseline (0.558)")
        criteria_met += 1
    else:
        logger.warning(f"❌ Ensemble AUC ({ensemble_metrics_test['auc']:.4f}) <= Baseline (0.558)")

    logger.info(f"\nCriteria met: {criteria_met}/{total_criteria}")

    # Save ensemble
    if criteria_met >= 3:
        logger.info("\n9. Saving ensemble (criteria met)...")
        output_path = "models/spread_ensemble"
        ensemble.save(output_path)
        logger.success(f"✅ Ensemble saved to {output_path}_ensemble.pkl")
        logger.success(f"✅ Individual models saved to {output_path}_*.pkl")

        # Save metadata
        metadata = {
            'ensemble_weights': ensemble.weights,
            'val_metrics': ensemble_metrics_val,
            'test_metrics': ensemble_metrics_test,
            'model_diversity': {
                'avg_pairwise_correlation': float(avg_corr),
                'correlation_matrix': corr_matrix.to_dict()
            },
            'success_criteria_met': criteria_met,
            'n_features': len(feature_cols),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(X_test),
            'training_date': pd.Timestamp.now().isoformat()
        }

        metadata_path = "models/spread_ensemble_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.success(f"✅ Metadata saved to {metadata_path}")
    else:
        logger.warning(f"\n⚠️ Not saving ensemble (only {criteria_met}/{total_criteria} criteria met)")
        logger.info("Ensemble did not significantly improve over individual models")

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)

    return ensemble, ensemble_metrics_test


if __name__ == "__main__":
    train_ensemble()
