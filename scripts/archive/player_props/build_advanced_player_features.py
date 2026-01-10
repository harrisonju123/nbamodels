#!/usr/bin/env python3
"""
Build Advanced Player Features

Uses sophisticated feature engineering to create 80+ features per player-game.
Replaces simple rolling averages with:
- Exponentially weighted stats
- Performance trends
- Matchup history
- Usage patterns
- Minute load analysis
- Game context

Usage:
    python scripts/build_advanced_player_features.py [--seasons 2023 2024 2025]
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.advanced_player_features import build_advanced_player_features


def main():
    parser = argparse.ArgumentParser(description="Build advanced player game features")
    parser.add_argument(
        "--box-scores",
        type=str,
        default="data/cache/player_box_scores.parquet",
        help="Path to player box scores cache"
    )
    parser.add_argument(
        "--games",
        type=str,
        default="data/raw/games.parquet",
        help="Path to games data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/features/player_game_features_advanced.parquet",
        help="Output path for advanced features"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2023, 2024, 2025],
        help="Seasons to process"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Building Advanced Player Features")
    logger.info("=" * 80)
    logger.info("")

    # 1. Load player box scores
    if not os.path.exists(args.box_scores):
        logger.error(f"Box scores not found at {args.box_scores}")
        logger.error("Run: python scripts/build_player_features.py first")
        return

    logger.info(f"Loading box scores from {args.box_scores}...")
    box_scores = pd.read_parquet(args.box_scores)
    logger.info(f"  Loaded {len(box_scores):,} player-game records")

    # 2. Load games data
    games_df = None
    if os.path.exists(args.games):
        logger.info(f"Loading games from {args.games}...")
        games_df = pd.read_parquet(args.games)
        logger.info(f"  Loaded {len(games_df):,} games")
    else:
        logger.warning(f"Games data not found at {args.games}")

    # 3. Filter to seasons if specified
    if args.seasons and 'season' in box_scores.columns:
        box_scores = box_scores[box_scores['season'].isin(args.seasons)]
        logger.info(f"  Filtered to {len(box_scores):,} records from seasons {args.seasons}")

    # 4. First build basic features (using existing pipeline)
    logger.info("")
    logger.info("Step 1: Building basic features...")
    logger.info("")

    from scripts.build_player_game_features import (
        calculate_player_rolling_stats,
        add_matchup_features,
        add_context_features
    )

    # Build basic rolling stats first
    basic_features = calculate_player_rolling_stats(box_scores)
    basic_features = add_matchup_features(basic_features, games_df if games_df is not None else pd.DataFrame())
    basic_features = add_context_features(basic_features, games_df if games_df is not None else pd.DataFrame())

    logger.info(f"✓ Basic features built ({len(basic_features.columns)} columns)")

    # 5. Now add advanced features on top
    logger.info("")
    logger.info("Step 2: Adding advanced features...")
    logger.info("")

    advanced_features = build_advanced_player_features(
        player_box_scores=basic_features,
        games_df=games_df
    )

    # 5. Filter to players with sufficient data
    logger.info("")
    logger.info("Filtering to valid players...")
    player_game_counts = advanced_features.groupby('player_id').size()
    valid_players = player_game_counts[player_game_counts >= 5].index

    advanced_features = advanced_features[advanced_features['player_id'].isin(valid_players)]
    logger.info(f"  Filtered to {len(valid_players):,} players with ≥5 games")

    # 6. Remove rows with missing critical features
    # (first few games per player won't have rolling stats)
    if 'pts_roll10' in advanced_features.columns:
        advanced_features = advanced_features[advanced_features['pts_roll10'].notna()]
        logger.info(f"  Removed games without sufficient history: {len(advanced_features):,} remaining")

    # 7. Save
    if not advanced_features.empty:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        advanced_features.to_parquet(args.output)

        logger.info("")
        logger.success(f"✅ Saved {len(advanced_features):,} player-game records")
        logger.success(f"   Output: {args.output}")
        logger.info("")
        logger.info("Summary:")
        logger.info(f"  Unique players: {advanced_features['player_id'].nunique():,}")
        logger.info(f"  Unique games: {advanced_features['game_id'].nunique():,}")
        logger.info(f"  Total features: {len(advanced_features.columns)}")
        logger.info(f"  Date range: {advanced_features['game_date'].min()} to {advanced_features['game_date'].max()}")

        # Show sample of new advanced features
        advanced_cols = [c for c in advanced_features.columns if any([
            '_ewma_' in c, '_trend_' in c, '_vs_opp_' in c,
            'usage_trend' in c, 'min_last_3g' in c, 'season_progress' in c
        ])]

        if advanced_cols:
            logger.info("")
            logger.info(f"Sample advanced features ({len(advanced_cols)} total):")
            for col in sorted(advanced_cols)[:15]:
                logger.info(f"    {col}")
            if len(advanced_cols) > 15:
                logger.info(f"    ... and {len(advanced_cols) - 15} more")

        logger.info("")
        logger.success("✅ Advanced player features built successfully!")
        logger.info("")
        logger.info("Next step: Retrain models with advanced features:")
        logger.info("  python scripts/train_player_props_advanced.py")

    else:
        logger.error("❌ No features generated")


if __name__ == "__main__":
    main()
