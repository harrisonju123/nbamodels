#!/usr/bin/env python3
"""
Add opponent_team column to player box scores.

This enables matchup history features (e.g., Giannis vs LAL career stats).
"""

import sys
from pathlib import Path
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def add_opponent_team_column(
    box_scores_path: str = "data/cache/player_box_scores.parquet",
    games_path: str = "data/raw/games.parquet",
    output_path: str = None
) -> pd.DataFrame:
    """
    Add opponent_team column to box scores.

    Infers opponent from home/away teams:
    - If player's team = home_team, opponent = away_team
    - If player's team = away_team, opponent = home_team

    Args:
        box_scores_path: Path to player box scores
        games_path: Path to games data (for home/away team info)
        output_path: Where to save updated box scores (default: overwrite input)

    Returns:
        Updated DataFrame
    """
    logger.info("=" * 80)
    logger.info("Adding opponent_team column to player box scores")
    logger.info("=" * 80)

    # Load box scores
    logger.info(f"Loading box scores from {box_scores_path}...")
    df = pd.read_parquet(box_scores_path)
    logger.info(f"  Loaded {len(df):,} player-game records")

    # Load games data to get home/away teams
    logger.info(f"Loading games from {games_path}...")
    games = pd.read_parquet(games_path)
    logger.info(f"  Loaded {len(games):,} games")

    # Merge to get home/away team info
    if 'home_team' not in df.columns or 'away_team' not in df.columns:
        logger.info("Merging with games data to get home/away teams...")
        game_teams = games[['game_id', 'home_team', 'away_team']].copy()

        df = df.merge(game_teams, on='game_id', how='left')
        logger.info(f"  ✓ Merged {len(df):,} records")

    # Check required columns
    required_cols = ['team_abbreviation', 'home_team', 'away_team']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        return df

    # Add opponent_team if not already present
    if 'opponent_team' in df.columns:
        logger.warning("opponent_team column already exists")
        logger.info(f"  Sample values: {df['opponent_team'].head().tolist()}")
        return df

    logger.info("Inferring opponent_team from home/away teams...")

    # Infer opponent
    df['opponent_team'] = df.apply(
        lambda row: row['away_team'] if row['team_abbreviation'] == row['home_team']
                    else row['home_team'],
        axis=1
    )

    # Validate
    non_null = df['opponent_team'].notna().sum()
    logger.info(f"  ✓ Added opponent_team to {non_null:,} records ({non_null/len(df)*100:.1f}%)")

    # Show sample
    sample = df[['player_name', 'team_abbreviation', 'home_team', 'away_team', 'opponent_team']].head(10)
    logger.info("")
    logger.info("Sample matchups:")
    for idx, row in sample.iterrows():
        logger.info(f"  {row['player_name']:20s} ({row['team_abbreviation']}) vs {row['opponent_team']}")

    # Check for any issues
    null_count = df['opponent_team'].isna().sum()
    if null_count > 0:
        logger.warning(f"  ⚠ {null_count} records with null opponent_team")

    # Check opponent_team values match team_abbreviation values (should be same set)
    unique_teams = set(df['team_abbreviation'].unique())
    unique_opponents = set(df['opponent_team'].unique())
    if unique_teams != unique_opponents:
        logger.warning(f"  ⚠ Team sets don't match:")
        logger.warning(f"    Teams: {sorted(unique_teams)}")
        logger.warning(f"    Opponents: {sorted(unique_opponents)}")

    # Save
    output_path = output_path or box_scores_path
    logger.info("")
    logger.info(f"Saving to {output_path}...")
    df.to_parquet(output_path)
    logger.success("✓ Saved!")

    logger.info("")
    logger.success("✅ opponent_team column added successfully!")
    logger.info(f"  Total records: {len(df):,}")
    logger.info(f"  Unique players: {df['player_id'].nunique():,}")
    logger.info(f"  Unique teams: {df['team_abbreviation'].nunique()}")
    logger.info(f"  Unique opponents: {df['opponent_team'].nunique()}")

    return df


if __name__ == "__main__":
    add_opponent_team_column()
