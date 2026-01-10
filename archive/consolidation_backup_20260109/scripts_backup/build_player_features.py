#!/usr/bin/env python3
"""
Build Player Features

Fetches player box scores from NBA API and builds player-level features.
This is a slow operation due to API rate limiting.

The script caches progress so it can be stopped and resumed.

Usage:
    python scripts/build_player_features.py [--seasons 2022 2023 2024] [--delay 0.6]
"""

import argparse
import os
import sys
import time
from datetime import datetime

import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.player_features import (
    PlayerStatsClient,
    PlayerFeatureBuilder,
    NBA_API_AVAILABLE,
)


def fetch_player_box_scores(
    games_df: pd.DataFrame,
    output_path: str = "data/cache/player_box_scores.parquet",
    delay: float = 0.6,
    batch_save: int = 50,
) -> pd.DataFrame:
    """
    Fetch player box scores for all games.

    Args:
        games_df: DataFrame with game data (needs 'game_id' column)
        output_path: Path to save/load cached data
        delay: Delay between API calls
        batch_save: Save progress every N games

    Returns:
        DataFrame with all player box scores
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load existing cache
    existing_df = None
    existing_game_ids = set()

    if os.path.exists(output_path):
        existing_df = pd.read_parquet(output_path)
        existing_game_ids = set(existing_df["game_id"].unique())
        logger.info(f"Loaded {len(existing_game_ids)} cached games")

    # Get all game IDs we need
    all_game_ids = games_df["game_id"].unique().tolist()
    new_game_ids = [g for g in all_game_ids if g not in existing_game_ids]

    if not new_game_ids:
        logger.info("All games already cached")
        return existing_df

    logger.info(f"Need to fetch {len(new_game_ids)} new games")

    client = PlayerStatsClient()
    new_box_scores = []
    failed_games = []

    for i, game_id in enumerate(new_game_ids):
        try:
            box = client.get_player_box_scores(game_id, delay=delay)

            if box is not None and not box.empty:
                new_box_scores.append(box)
            else:
                failed_games.append(game_id)

        except Exception as e:
            logger.warning(f"Error fetching {game_id}: {e}")
            failed_games.append(game_id)

        # Progress logging
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i + 1}/{len(new_game_ids)} ({100*(i+1)/len(new_game_ids):.1f}%)")

        # Batch save
        if (i + 1) % batch_save == 0 and new_box_scores:
            logger.info(f"Saving batch at {i + 1} games...")
            batch_df = pd.concat(new_box_scores, ignore_index=True)

            if existing_df is not None:
                combined = pd.concat([existing_df, batch_df], ignore_index=True)
            else:
                combined = batch_df

            combined.to_parquet(output_path)
            existing_df = combined
            new_box_scores = []  # Reset for next batch

    # Final save
    if new_box_scores:
        batch_df = pd.concat(new_box_scores, ignore_index=True)

        if existing_df is not None:
            combined = pd.concat([existing_df, batch_df], ignore_index=True)
        else:
            combined = batch_df

        combined.to_parquet(output_path)
        logger.info(f"Saved {len(combined)} total box scores to {output_path}")
    else:
        combined = existing_df

    if failed_games:
        logger.warning(f"Failed to fetch {len(failed_games)} games")

    return combined


def build_player_features_from_box_scores(
    box_scores_df: pd.DataFrame,
    games_df: pd.DataFrame,
    output_path: str = "data/features/player_features.parquet",
) -> pd.DataFrame:
    """
    Build player features from cached box scores.

    Args:
        box_scores_df: DataFrame with player box scores
        games_df: DataFrame with game data
        output_path: Path to save player features

    Returns:
        DataFrame with player features for each game
    """
    logger.info("Building player features...")

    # Prepare box scores for feature building
    df = box_scores_df.copy()

    # Ensure required columns
    if "game_date" not in df.columns:
        # Try to get from games_df
        game_dates = games_df[["game_id", "date"]].copy()
        game_dates.columns = ["game_id", "game_date"]
        df = df.merge(game_dates, on="game_id", how="left")

    # Ensure player_id column
    if "player_id" not in df.columns and "player_id" in df.columns:
        df["player_id"] = df["player_id"]

    # Initialize builder
    builder = PlayerFeatureBuilder()

    # Calculate player rolling stats
    logger.info("Calculating player rolling stats...")
    player_stats = builder.build_player_rolling_stats(df)

    # Aggregate to team level
    logger.info("Aggregating to team level...")
    team_player_stats = builder.aggregate_to_team(player_stats)

    if team_player_stats.empty:
        logger.warning("No team player stats generated")
        return pd.DataFrame()

    # Build game-level features
    logger.info("Building game-level player features...")
    player_features = builder.build_game_player_features(games_df, df)

    # Save
    if not player_features.empty:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        player_features.to_parquet(output_path)
        logger.info(f"Saved {len(player_features)} player feature records to {output_path}")

    return player_features


def main():
    parser = argparse.ArgumentParser(description="Build player features from NBA API")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=None,
        help="Seasons to process (default: last 3 in data)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.6,
        help="Delay between API calls in seconds (default: 0.6)"
    )
    parser.add_argument(
        "--batch-save",
        type=int,
        default=50,
        help="Save progress every N games (default: 50)"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetching and just build features from cache"
    )

    args = parser.parse_args()

    if not NBA_API_AVAILABLE:
        logger.error("nba_api not installed. Run: pip install nba_api")
        return

    # Load games data
    games_path = "data/raw/games.parquet"
    if not os.path.exists(games_path):
        logger.error(f"Games data not found at {games_path}")
        logger.error("Run the data pipeline first to fetch games")
        return

    games_df = pd.read_parquet(games_path)
    logger.info(f"Loaded {len(games_df)} games")

    # Filter to seasons
    if args.seasons:
        games_df = games_df[games_df["season"].isin(args.seasons)]
    else:
        # Default to last 3 seasons
        seasons = sorted(games_df["season"].unique())[-3:]
        games_df = games_df[games_df["season"].isin(seasons)]

    logger.info(f"Processing {len(games_df)} games from seasons {games_df['season'].unique().tolist()}")

    # Fetch box scores
    box_scores_path = "data/cache/player_box_scores.parquet"

    if not args.skip_fetch:
        start_time = time.time()
        box_scores = fetch_player_box_scores(
            games_df,
            output_path=box_scores_path,
            delay=args.delay,
            batch_save=args.batch_save,
        )
        elapsed = time.time() - start_time
        logger.info(f"Fetching took {elapsed/60:.1f} minutes")
    else:
        if not os.path.exists(box_scores_path):
            logger.error(f"Cache not found at {box_scores_path}")
            return
        box_scores = pd.read_parquet(box_scores_path)
        logger.info(f"Loaded {len(box_scores)} cached box scores")

    if box_scores is None or box_scores.empty:
        logger.error("No box scores available")
        return

    # Build features
    player_features = build_player_features_from_box_scores(
        box_scores,
        games_df,
        output_path="data/features/player_features.parquet",
    )

    if not player_features.empty:
        logger.info(f"\nPlayer features built successfully!")
        logger.info(f"  Games: {len(player_features)}")
        logger.info(f"  Features: {len(player_features.columns) - 3}")

        # Show sample feature columns
        feature_cols = [c for c in player_features.columns if "player_" in c][:10]
        logger.info(f"  Sample features: {feature_cols}")


if __name__ == "__main__":
    main()
