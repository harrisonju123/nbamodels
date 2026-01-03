"""
NBA Betting Model - Getting Started

This script demonstrates the full pipeline:
1. Fetch NBA game data
2. Engineer features
3. Train a spread prediction model
4. Backtest the model

Run this after setting up your API keys in config/.env
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv(project_root / "config" / ".env")


def main():
    """Run the full pipeline."""

    # =========================================================================
    # Step 1: Fetch or Load Data
    # =========================================================================
    logger.info("Step 1: Loading data...")

    data_path = project_root / "data" / "raw" / "games.parquet"

    if data_path.exists():
        games_df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(games_df)} games from cache")
    else:
        # Fetch from API (requires API key)
        from src.data import NBAStatsClient

        client = NBAStatsClient()
        games_df = pd.DataFrame()

        # Fetch last 3 seasons as example
        for season in [2022, 2023, 2024]:
            logger.info(f"Fetching season {season}...")
            season_games = client.get_season_games(season)
            games_df = pd.concat([games_df, season_games], ignore_index=True)

        # Save for later
        os.makedirs(data_path.parent, exist_ok=True)
        games_df.to_parquet(data_path)
        logger.info(f"Saved {len(games_df)} games to {data_path}")

    print(f"\nGames data shape: {games_df.shape}")
    print(games_df.head())

    # =========================================================================
    # Step 2: Engineer Features
    # =========================================================================
    logger.info("\nStep 2: Engineering features...")

    from src.features import GameFeatureBuilder

    builder = GameFeatureBuilder()
    features_df = builder.build_game_features(games_df)

    # Get feature columns
    feature_cols = builder.get_feature_columns(features_df)

    print(f"\nFeatures shape: {features_df.shape}")
    print(f"Number of features: {len(feature_cols)}")
    print("\nFeature columns:")
    for col in feature_cols[:10]:
        print(f"  - {col}")
    print(f"  ... and {len(feature_cols) - 10} more")

    # =========================================================================
    # Step 3: Train Model
    # =========================================================================
    logger.info("\nStep 3: Training model...")

    from src.models import SpreadPredictionModel

    # Split by season
    train_seasons = [2022]
    val_seasons = [2023]
    test_seasons = [2024]

    train_df = features_df[features_df["season"].isin(train_seasons)]
    val_df = features_df[features_df["season"].isin(val_seasons)]
    test_df = features_df[features_df["season"].isin(test_seasons)]

    print(f"\nTrain: {len(train_df)} games")
    print(f"Val: {len(val_df)} games")
    print(f"Test: {len(test_df)} games")

    # Create and train model
    model = SpreadPredictionModel(calibration_method="isotonic")

    # Prepare target (home team win for now, since we don't have spread data)
    y_train = train_df["home_win"]
    y_val = val_df["home_win"]
    y_test = test_df["home_win"]

    # Train
    model.fit(
        train_df[feature_cols],
        y_train,
        val_df[feature_cols],
        y_val,
        feature_columns=feature_cols,
    )

    # Evaluate
    print("\nModel Performance:")
    print("-" * 40)

    train_metrics = model.evaluate(train_df[feature_cols], y_train)
    print(f"Train Accuracy: {train_metrics['accuracy']:.3f}")
    print(f"Train Brier:    {train_metrics['brier_score']:.4f}")

    val_metrics = model.evaluate(val_df[feature_cols], y_val)
    print(f"Val Accuracy:   {val_metrics['accuracy']:.3f}")
    print(f"Val Brier:      {val_metrics['brier_score']:.4f}")

    if len(test_df) > 0:
        test_metrics = model.evaluate(test_df[feature_cols], y_test)
        print(f"Test Accuracy:  {test_metrics['accuracy']:.3f}")
        print(f"Test Brier:     {test_metrics['brier_score']:.4f}")

    # =========================================================================
    # Step 4: Feature Importance
    # =========================================================================
    print("\nTop 10 Features:")
    print("-" * 40)
    print(model.feature_importance.head(10).to_string(index=False))

    # =========================================================================
    # Step 5: Simple Backtest Simulation
    # =========================================================================
    logger.info("\nStep 4: Running backtest simulation...")

    from src.betting import BacktestEngine

    # Create synthetic odds (since we don't have real odds data)
    # In production, you'd use actual odds from The Odds API
    test_predictions = test_df[["game_id", "date", "home_team", "away_team"]].copy()
    test_predictions["model_prob"] = model.predict_proba(test_df[feature_cols])
    test_predictions["implied_prob"] = 0.524  # Standard -110 odds
    test_predictions["odds"] = -110

    test_outcomes = test_df[["game_id"]].copy()
    test_outcomes["result"] = y_test.values

    # Run backtest
    engine = BacktestEngine(
        initial_bankroll=10000,
        kelly_fraction=0.2,
        min_edge=0.03,
    )

    results = engine.run(test_predictions, test_outcomes)

    print("\nBacktest Results (Simulated):")
    print("-" * 40)
    print(f"Number of bets: {results.num_bets}")
    print(f"Win rate: {results.win_rate:.1%}")
    print(f"ROI: {results.roi:.1%}")
    print(f"Total profit: ${results.total_profit:,.2f}")
    print(f"Final bankroll: ${results.final_bankroll:,.2f}")
    print(f"Max drawdown: {results.max_drawdown:.1%}")

    # =========================================================================
    # Step 6: Save Model
    # =========================================================================
    model_path = project_root / "models" / "spread_model.pkl"
    os.makedirs(model_path.parent, exist_ok=True)
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # =========================================================================
    # Next Steps
    # =========================================================================
    print("\n" + "=" * 50)
    print("NEXT STEPS")
    print("=" * 50)
    print("""
1. Get API keys:
   - BallDontLie: https://www.balldontlie.io/
   - The Odds API: https://the-odds-api.com/

2. Add keys to config/.env

3. Fetch more historical data (2015-present)

4. Add real odds data for proper backtesting

5. Tune model hyperparameters

6. Set up paper trading to validate in real-time
""")


if __name__ == "__main__":
    main()
