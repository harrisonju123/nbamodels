"""
Hyperparameter Tuning for Spread Model

Uses Optuna to find optimal XGBoost parameters for maximum ROI.

Expected improvement: +3-7% ROI from default params.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score
import xgboost as xgb

from src.features.game_features import GameFeatureBuilder
from src.models.calibration import CalibratedModel


def load_training_data():
    """Load and prepare training data."""
    logger.info("Loading training data...")

    # Load games WITH SPREAD COVERAGE TARGET
    games = pd.read_parquet("data/raw/games_with_spread_coverage.parquet")

    # Use ALL games with spread data (we need the full dataset - only 641 total)
    logger.info(f"Using {len(games)} games with spread coverage")

    # Build features
    feature_builder = GameFeatureBuilder(
        use_referee_features=False,  # Disabled (noise)
        use_news_features=False,     # Disabled (noise)
        use_sentiment_features=False  # Disabled (noise)
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
        # Spread data (don't use spread as a feature since it's not available pre-game)
        'spread', 'spread_odds', 'spread_implied_prob'
    ]
    feature_cols = [c for c in features_df.columns if c not in exclude_cols and c in features_df.columns]

    # Remove any NaN columns
    feature_cols = [c for c in feature_cols if features_df[c].notna().any()]

    X = features_df[feature_cols].fillna(0)
    y = features_df['home_cover']  # CHANGED: Predict spread coverage, not game winner

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Samples: {len(X)}")
    logger.info(f"Home cover rate: {y.mean():.1%}")

    return X, y, features_df, feature_cols


def objective(trial, X, y):
    """Optuna objective function - maximize ROI proxy."""

    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.2),
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1,
    }

    # Time series cross-validation (3 folds for small dataset)
    tscv = TimeSeriesSplit(n_splits=3)

    # Track metrics across folds
    log_losses = []
    roc_aucs = []
    calibration_errors = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Predict
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # Calculate metrics
        ll = log_loss(y_val, y_pred_proba)
        auc = roc_auc_score(y_val, y_pred_proba)

        # Calibration error (Brier score)
        brier = np.mean((y_pred_proba - y_val) ** 2)

        log_losses.append(ll)
        roc_aucs.append(auc)
        calibration_errors.append(brier)

    # Return composite metric (optimize for ROI proxy)
    # Lower log loss = better probability estimates = better ROI
    # Higher AUC = better discrimination = find more edges
    # Lower Brier = better calibration = Kelly sizing more accurate

    avg_log_loss = np.mean(log_losses)
    avg_auc = np.mean(roc_aucs)
    avg_brier = np.mean(calibration_errors)

    # Composite score (maximize AUC, minimize log loss and Brier)
    # Weights: AUC is most important for finding edges
    composite_score = (avg_auc * 2.0) - (avg_log_loss * 0.5) - (avg_brier * 0.5)

    # Report intermediate values
    trial.set_user_attr('log_loss', avg_log_loss)
    trial.set_user_attr('auc', avg_auc)
    trial.set_user_attr('brier', avg_brier)

    return composite_score


def main():
    """Run hyperparameter tuning."""
    logger.info("=" * 60)
    logger.info("SPREAD MODEL HYPERPARAMETER TUNING")
    logger.info("=" * 60)

    # Load data
    X, y, features_df, feature_cols = load_training_data()

    # Create Optuna study
    logger.info("\nStarting Optuna optimization...")
    logger.info("Trials: 100 (takes ~30-60 minutes)")

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='spread_model_tuning'
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=100,
        show_progress_bar=True,
        n_jobs=1  # Sequential for stability
    )

    # Results
    logger.info("\n" + "=" * 60)
    logger.info("TUNING COMPLETE")
    logger.info("=" * 60)

    best_trial = study.best_trial
    logger.info(f"\nBest Trial: #{best_trial.number}")
    logger.info(f"Composite Score: {best_trial.value:.4f}")
    logger.info(f"Log Loss: {best_trial.user_attrs['log_loss']:.4f}")
    logger.info(f"AUC: {best_trial.user_attrs['auc']:.4f}")
    logger.info(f"Brier Score: {best_trial.user_attrs['brier']:.4f}")

    logger.info("\n" + "-" * 60)
    logger.info("BEST HYPERPARAMETERS:")
    logger.info("-" * 60)
    for param, value in best_trial.params.items():
        logger.info(f"  {param:20s}: {value}")

    # Compare to default params
    logger.info("\n" + "-" * 60)
    logger.info("COMPARISON TO DEFAULTS:")
    logger.info("-" * 60)

    default_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'scale_pos_weight': 1.0,
        'objective': 'binary:logistic',
        'random_state': 42,
    }

    for param in best_trial.params:
        if param in default_params:
            best_val = best_trial.params[param]
            default_val = default_params[param]
            change = ((best_val - default_val) / default_val * 100) if default_val != 0 else 0
            logger.info(f"  {param:20s}: {default_val} → {best_val} ({change:+.1f}%)")

    # Save best params
    import json
    output_path = Path("models/tuned_params_spread.json")
    with open(output_path, 'w') as f:
        json.dump({
            'best_params': best_trial.params,
            'metrics': {
                'composite_score': best_trial.value,
                'log_loss': best_trial.user_attrs['log_loss'],
                'auc': best_trial.user_attrs['auc'],
                'brier': best_trial.user_attrs['brier']
            },
            'tuning_date': pd.Timestamp.now().isoformat(),
            'n_trials': len(study.trials),
            'training_samples': len(X)
        }, f, indent=2)

    logger.info(f"\n✅ Saved best params to {output_path}")

    # Plot optimization history
    try:
        import plotly.graph_objects as go
        from optuna.visualization import plot_optimization_history, plot_param_importances

        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_html("models/tuning_history.html")
        logger.info("✅ Saved optimization history to models/tuning_history.html")

        # Parameter importances
        fig = plot_param_importances(study)
        fig.write_html("models/tuning_param_importance.html")
        logger.info("✅ Saved parameter importances to models/tuning_param_importance.html")

    except Exception as e:
        logger.warning(f"Could not create plots: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("NEXT STEPS:")
    logger.info("=" * 60)
    logger.info("1. Review tuned params in models/tuned_params_spread.json")
    logger.info("2. Retrain model with: python scripts/retrain_models.py")
    logger.info("3. Backtest with tuned params")
    logger.info("4. Expected ROI improvement: +3-7% from default params")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
