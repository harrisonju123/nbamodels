#!/usr/bin/env python3
"""
Build Player Game Features for Props Training

Creates individual player-level features for each game, suitable for
training player props models (PTS, REB, AST, 3PM).

This script:
1. Loads cached player box scores
2. Calculates rolling averages for each player
3. Adds matchup/context features
4. Outputs player_game_features.parquet for training

Usage:
    python scripts/build_player_game_features.py [--seasons 2022 2023 2024]
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.nba_stats import NBAStatsClient


def calculate_player_rolling_stats(
    box_scores_df: pd.DataFrame,
    window_sizes: list = [3, 5, 10],
) -> pd.DataFrame:
    """
    Calculate rolling averages for each player.

    Args:
        box_scores_df: Player box scores with game_date
        window_sizes: Rolling windows to calculate

    Returns:
        DataFrame with rolling stats added
    """
    logger.info("Calculating player rolling statistics...")

    # Ensure sorted by date
    df = box_scores_df.sort_values(['player_id', 'game_date']).copy()

    # Convert minutes from MM:SS to decimal if needed
    if 'min' in df.columns and df['min'].dtype == 'object':
        def convert_minutes(min_str):
            if pd.isna(min_str) or min_str == '':
                return 0.0
            try:
                if ':' in str(min_str):
                    parts = str(min_str).split(':')
                    return float(parts[0]) + float(parts[1]) / 60
                return float(min_str)
            except:
                return 0.0

        df['min'] = df['min'].apply(convert_minutes)
        logger.info("  Converted minutes from MM:SS to decimal")

    # Rename 'to' to 'tov' for consistency
    if 'to' in df.columns:
        df['tov'] = df['to']

    # Stats to roll - expanded to include all needed features
    roll_stats = [
        'pts', 'reb', 'ast', 'fg3m', 'min', 'fga', 'fta', 'stl', 'blk', 'tov',
        'oreb', 'dreb', 'fg3a', 'fgm', 'ftm', 'pf'
    ]

    # Group by player and calculate rolling averages
    for window in window_sizes:
        logger.info(f"  Calculating {window}-game rolling averages...")

        for stat in roll_stats:
            if stat in df.columns:
                df[f'{stat}_roll{window}'] = (
                    df.groupby('player_id')[stat]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )

    # Calculate rolling percentages
    logger.info("  Calculating rolling percentage stats...")
    for window in [5, 10]:
        # FG%
        df[f'fg_pct_roll{window}'] = (
            df.groupby('player_id').apply(
                lambda x: (x['fgm'].shift(1).rolling(window, min_periods=1).sum() /
                          x['fga'].shift(1).rolling(window, min_periods=1).sum()).fillna(0)
            ).reset_index(level=0, drop=True)
        )

        # FG3%
        df[f'fg3_pct_roll{window}'] = (
            df.groupby('player_id').apply(
                lambda x: (x['fg3m'].shift(1).rolling(window, min_periods=1).sum() /
                          x['fg3a'].shift(1).rolling(window, min_periods=1).sum()).fillna(0)
            ).reset_index(level=0, drop=True)
        )

        # FT%
        df[f'ft_pct_roll{window}'] = (
            df.groupby('player_id').apply(
                lambda x: (x['ftm'].shift(1).rolling(window, min_periods=1).sum() /
                          x['fta'].shift(1).rolling(window, min_periods=1).sum()).fillna(0)
            ).reset_index(level=0, drop=True)
        )

    # Calculate usage rate (simplified version)
    # Usage = (FGA + 0.44*FTA + TOV) / minutes * 48 (normalized per 48 min)
    logger.info("  Calculating usage rate...")
    df['usage_rate'] = ((df['fga'] + 0.44 * df['fta'] + df['tov']) / df['min'] * 48).fillna(0)

    # Cap unrealistic values
    df['usage_rate'] = df['usage_rate'].clip(0, 50)

    logger.success(f"Added rolling stats for {len(roll_stats)} metrics plus percentages and usage rate")
    return df


def add_matchup_features(
    player_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add team-level and opponent defensive stats.

    Args:
        player_df: Player game features
        team_stats_df: Team-level defensive stats (currently unused)

    Returns:
        DataFrame with matchup features
    """
    logger.info("Adding team and opponent features...")

    # Calculate team-level rolling stats
    # Group by team + date to aggregate team performance
    team_cols = ['team_abbreviation', 'game_date']

    # Calculate team totals per game
    team_game_stats = player_df.groupby(team_cols + ['game_id']).agg({
        'pts': 'sum',
        'reb': 'sum',
        'ast': 'sum',
        'fg3m': 'sum',
        'tov': 'sum',
        'min': 'sum',
    }).reset_index()

    # Calculate team pace (possessions per 48 minutes)
    # Simplified: pace = 48 / (total_minutes / 5 players) * possessions
    # Possessions ≈ FGA + 0.44*FTA - OReb + TOV
    # For simplicity, use a normalized minutes-based pace
    team_game_stats['team_pace'] = 100.0  # Placeholder - league average

    # Rolling team stats
    team_game_stats = team_game_stats.sort_values(['team_abbreviation', 'game_date'])

    for window in [5]:
        team_game_stats[f'team_pts_roll{window}'] = (
            team_game_stats.groupby('team_abbreviation')['pts']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        team_game_stats[f'team_reb_roll{window}'] = (
            team_game_stats.groupby('team_abbreviation')['reb']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        team_game_stats[f'team_ast_roll{window}'] = (
            team_game_stats.groupby('team_abbreviation')['ast']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        team_game_stats[f'team_3pm_roll{window}'] = (
            team_game_stats.groupby('team_abbreviation')['fg3m']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    # Merge team stats back to player data
    merge_cols = ['game_id', 'team_abbreviation']
    player_df = player_df.merge(
        team_game_stats[['game_id', 'team_abbreviation', 'team_pace',
                         'team_pts_roll5', 'team_reb_roll5', 'team_ast_roll5', 'team_3pm_roll5']],
        on=merge_cols,
        how='left'
    )

    # Fill missing team stats with league averages
    player_df['team_pace'] = player_df['team_pace'].fillna(100.0)
    player_df['team_pts_roll5'] = player_df['team_pts_roll5'].fillna(110.0)
    player_df['team_reb_roll5'] = player_df['team_reb_roll5'].fillna(45.0)
    player_df['team_ast_roll5'] = player_df['team_ast_roll5'].fillna(25.0)
    player_df['team_3pm_roll5'] = player_df['team_3pm_roll5'].fillna(12.0)

    # Opponent defensive stats (DISABLED - placeholders provided no signal)
    # These were hardcoded to league averages and added noise, not signal
    # logger.info("  Adding opponent defensive stats (using league averages)...")
    # player_df['opp_def_rating'] = 110.0         # League average defensive rating
    # player_df['opp_pace'] = 100.0               # League average pace
    # player_df['opp_pts_allowed_roll5'] = 110.0  # League average points allowed
    # player_df['opp_reb_allowed_roll5'] = 45.0   # League average rebounds allowed
    # player_df['opp_3pt_defense'] = 12.0         # League average 3PM allowed

    logger.success("Added team features")
    return player_df


def add_context_features(
    player_df: pd.DataFrame,
    games_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add game context (home/away, rest days, etc.).

    Args:
        player_df: Player game features
        games_df: Game-level data

    Returns:
        DataFrame with context features
    """
    logger.info("Adding context features...")

    # Merge with games to get home/away
    # Simplified for now
    player_df['is_home'] = 0.5  # Placeholder
    player_df['days_rest'] = 1  # Placeholder

    return player_df


def build_player_game_features(
    box_scores_path: str = "data/cache/player_box_scores.parquet",
    games_path: str = "data/raw/games.parquet",
    output_path: str = "data/features/player_game_features.parquet",
    seasons: list = None,
) -> pd.DataFrame:
    """
    Build complete player game features dataset.

    Args:
        box_scores_path: Path to cached player box scores
        games_path: Path to games data
        output_path: Where to save features
        seasons: Optional list of seasons to include

    Returns:
        DataFrame with player game features
    """
    logger.info("=" * 80)
    logger.info("Building Player Game Features for Props Training")
    logger.info("=" * 80)

    # 1. Load player box scores
    if not os.path.exists(box_scores_path):
        logger.error(f"Player box scores not found at {box_scores_path}")
        logger.error("Run: python scripts/build_player_features.py first")
        return pd.DataFrame()

    logger.info(f"Loading box scores from {box_scores_path}...")
    box_scores = pd.read_parquet(box_scores_path)
    logger.info(f"  Loaded {len(box_scores)} player-game records")

    # 2. Load games data for context
    if os.path.exists(games_path):
        games_df = pd.read_parquet(games_path)
        logger.info(f"  Loaded {len(games_df)} games")
    else:
        logger.warning(f"Games data not found at {games_path}")
        games_df = pd.DataFrame()

    # 3. Filter to seasons if specified
    if seasons and 'season' in box_scores.columns:
        box_scores = box_scores[box_scores['season'].isin(seasons)]
        logger.info(f"  Filtered to {len(box_scores)} records from seasons {seasons}")

    # 4. Ensure required columns
    required_cols = ['player_id', 'game_id', 'game_date', 'pts', 'reb', 'ast']
    missing_cols = [c for c in required_cols if c not in box_scores.columns]

    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return pd.DataFrame()

    # 5. Calculate rolling statistics
    player_features = calculate_player_rolling_stats(box_scores)

    # 6. Add matchup features (opponent defense, pace)
    # Note: This is simplified - in production, merge with team stats
    player_features = add_matchup_features(player_features, games_df)

    # 7. Add context features (home/away, rest)
    player_features = add_context_features(player_features, games_df)

    # 8. Filter to players with sufficient data
    # Require at least 5 games played
    player_game_counts = player_features.groupby('player_id').size()
    valid_players = player_game_counts[player_game_counts >= 5].index

    player_features = player_features[player_features['player_id'].isin(valid_players)]
    logger.info(f"  Filtered to {len(valid_players)} players with ≥5 games")

    # 9. Remove rows with missing rolling stats (first few games for each player)
    player_features = player_features[player_features['pts_roll3'].notna()]
    logger.info(f"  Removed games without rolling stats: {len(player_features)} remaining")

    # 10. Save
    if not player_features.empty:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        player_features.to_parquet(output_path)
        logger.success(f"\nSaved {len(player_features)} player-game records to {output_path}")

        logger.info("\nFeature Summary:")
        logger.info(f"  Unique players: {player_features['player_id'].nunique()}")
        logger.info(f"  Unique games: {player_features['game_id'].nunique()}")
        logger.info(f"  Total features: {len(player_features.columns)}")

        # Show feature columns
        feature_cols = [c for c in player_features.columns if '_roll' in c]
        logger.info(f"\nRolling features ({len(feature_cols)}):")
        for col in sorted(feature_cols)[:20]:
            logger.info(f"    {col}")

        return player_features
    else:
        logger.error("No player features generated")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Build player game features for props training")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2022, 2023, 2024, 2025],
        help="Seasons to process (default: 2022 2023 2024 2025)"
    )
    parser.add_argument(
        "--box-scores",
        type=str,
        default="data/cache/player_box_scores.parquet",
        help="Path to player box scores cache"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/features/player_game_features.parquet",
        help="Output path for player game features"
    )

    args = parser.parse_args()

    # Build features
    features_df = build_player_game_features(
        box_scores_path=args.box_scores,
        output_path=args.output,
        seasons=args.seasons,
    )

    if not features_df.empty:
        logger.success("\n✅ Player game features built successfully!")
        logger.info(f"\nNext step: Train models with:")
        logger.info(f"  python scripts/train_player_props.py")
    else:
        logger.error("\n❌ Failed to build player features")
        logger.info("\nTo fix:")
        logger.info("  1. Run: python scripts/build_player_features.py")
        logger.info("  2. This will fetch box scores from NBA API (takes ~1 hour)")
        logger.info("  3. Then run this script again")


if __name__ == "__main__":
    main()
