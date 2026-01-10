"""
Build Player Impact Model

Two approaches available:
1. Box Plus-Minus (BPM) - Quick, uses box score +/- data (default)
2. RAPM (Regularized Adjusted Plus-Minus) - Slower, needs stint data

Run with --rapm flag to use full RAPM model (requires API calls).
"""

import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.player_impact import PlayerImpact, PlayerImpactModel


def parse_minutes(min_str: str) -> float:
    """Convert MM:SS string to decimal minutes."""
    if pd.isna(min_str) or min_str == '' or min_str is None:
        return 0.0
    if isinstance(min_str, (int, float)):
        return float(min_str)
    try:
        if ':' in str(min_str):
            parts = str(min_str).split(':')
            return float(parts[0]) + float(parts[1]) / 60
        return float(min_str)
    except (ValueError, IndexError):
        return 0.0


def build_bpm_model(
    box_scores_path: str = "data/cache/player_box_scores.parquet",
    games_path: str = "data/raw/games.parquet",
    output_path: str = "data/cache/player_impact/player_impact_model.parquet",
    min_minutes: float = 100,
    seasons: list = None,
) -> pd.DataFrame:
    """
    Build player impact model using Box Plus-Minus approach.

    This uses the +/- from box scores, weighted by minutes played,
    and normalizes to per-48-minute rates.

    Args:
        box_scores_path: Path to player box scores
        games_path: Path to games data (for season filtering)
        output_path: Path to save model
        min_minutes: Minimum minutes played to include player
        seasons: List of seasons to include (default: last 2)

    Returns:
        DataFrame with player impacts
    """
    logger.info("Building Box Plus-Minus player impact model...")

    # Load box scores
    box_df = pd.read_parquet(box_scores_path)
    logger.info(f"Loaded {len(box_df)} player-game records")

    # Load games for season info and filtering
    games_df = pd.read_parquet(games_path)

    # Merge to get season
    box_df = box_df.merge(
        games_df[['game_id', 'season', 'date']],
        on='game_id',
        how='left'
    )

    # Filter by seasons
    if seasons is None:
        all_seasons = sorted(box_df['season'].dropna().unique())
        seasons = all_seasons[-2:] if len(all_seasons) >= 2 else all_seasons

    logger.info(f"Using seasons: {seasons}")
    box_df = box_df[box_df['season'].isin(seasons)]

    # Parse minutes
    box_df['minutes'] = box_df['min'].apply(parse_minutes)

    # Filter to players who actually played
    box_df = box_df[box_df['minutes'] > 0].copy()

    # Get player aggregate stats
    player_stats = box_df.groupby(['player_id', 'player_name', 'team_id', 'team_abbreviation']).agg({
        'minutes': 'sum',
        'plus_minus': 'sum',
        'game_id': 'count',  # Number of games
        'pts': 'mean',
        'reb': 'mean',
        'ast': 'mean',
    }).reset_index()

    player_stats.columns = ['player_id', 'player_name', 'team_id', 'team_abbrev',
                            'total_minutes', 'total_plus_minus', 'n_games',
                            'ppg', 'rpg', 'apg']

    # Filter by minimum minutes
    player_stats = player_stats[player_stats['total_minutes'] >= min_minutes].copy()
    logger.info(f"Found {len(player_stats)} players with {min_minutes}+ minutes")

    # Calculate per-48-minute impact
    # Raw BPM = total +/- per 48 minutes
    player_stats['impact'] = (
        player_stats['total_plus_minus'] / player_stats['total_minutes'] * 48
    )

    # Apply regularization: shrink toward 0 based on sample size
    # More minutes = less shrinkage
    avg_minutes = player_stats['total_minutes'].median()
    shrinkage = avg_minutes / (player_stats['total_minutes'] + avg_minutes)
    player_stats['impact'] = player_stats['impact'] * (1 - shrinkage)

    # Sort by impact
    player_stats = player_stats.sort_values('impact', ascending=False).reset_index(drop=True)

    # Create output in format compatible with PlayerImpactModel
    output_df = player_stats[[
        'player_id', 'player_name', 'team_id', 'team_abbrev',
        'impact', 'total_minutes', 'n_games'
    ]].copy()
    output_df.columns = ['player_id', 'player_name', 'team_id', 'team_abbrev',
                         'impact', 'minutes', 'n_stints']

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_parquet(output_path)

    logger.info(f"Saved {len(output_df)} player impacts to {output_path}")

    # Print top/bottom players
    print("\n" + "="*60)
    print("TOP 20 PLAYERS BY IMPACT (per 48 min)")
    print("="*60)
    for i, row in output_df.head(20).iterrows():
        print(f"{row['player_name']:25s} ({row['team_abbrev']}) {row['impact']:+6.2f}")

    print("\n" + "="*60)
    print("BOTTOM 10 PLAYERS BY IMPACT (per 48 min)")
    print("="*60)
    for i, row in output_df.tail(10).iterrows():
        print(f"{row['player_name']:25s} ({row['team_abbrev']}) {row['impact']:+6.2f}")

    # Print team aggregates
    print("\n" + "="*60)
    print("TEAM ROSTER IMPACT (sum of top 8 players)")
    print("="*60)
    team_impacts = []
    for team in output_df['team_abbrev'].unique():
        team_players = output_df[output_df['team_abbrev'] == team].nlargest(8, 'minutes')
        total_impact = team_players['impact'].sum()
        team_impacts.append({'team': team, 'impact': total_impact})

    team_df = pd.DataFrame(team_impacts).sort_values('impact', ascending=False)
    for _, row in team_df.iterrows():
        print(f"{row['team']:5s} {row['impact']:+6.2f}")

    return output_df


def build_rapm_model(
    games_path: str = "data/raw/games.parquet",
    output_path: str = "data/cache/player_impact/player_impact_model.parquet",
    max_games: int = None,
    seasons: list = None,
) -> pd.DataFrame:
    """
    Build full RAPM model from stint-level data.

    WARNING: This requires fetching stint data from NBA API, which can take
    several hours for a full season of games.

    Args:
        games_path: Path to games data
        output_path: Path to save model
        max_games: Max games to process (for testing)
        seasons: Seasons to include

    Returns:
        DataFrame with player impacts
    """
    from src.features.player_impact import build_player_impact_model

    logger.info("Building full RAPM model (this may take a while)...")

    model = build_player_impact_model(
        games_path=games_path,
        output_path=output_path,
        seasons=seasons,
        max_games=max_games,
    )

    if model.is_fitted:
        return model.get_impacts_df()
    else:
        logger.error("Failed to build RAPM model")
        return pd.DataFrame()


def verify_model(model_path: str = "data/cache/player_impact/player_impact_model.parquet"):
    """Verify the player impact model loads correctly."""
    model = PlayerImpactModel()
    model.load(model_path)

    if not model.is_fitted:
        logger.error("Model failed to load")
        return False

    impacts = model.get_impacts_df()
    logger.info(f"Model loaded: {len(impacts)} players")

    # Test team impact calculation
    test_teams = impacts['team_id'].unique()[:3]
    for team_id in test_teams:
        team_players = impacts[impacts['team_id'] == team_id].head(5)
        player_ids = team_players['player_id'].tolist()
        total_impact = model.get_team_impact(player_ids)
        logger.info(f"Team {team_id} top 5 players impact: {total_impact:.2f}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build player impact model")
    parser.add_argument("--rapm", action="store_true", help="Use full RAPM model (slow)")
    parser.add_argument("--min-minutes", type=float, default=100, help="Minimum minutes threshold")
    parser.add_argument("--seasons", type=int, nargs="+", help="Seasons to include")
    parser.add_argument("--max-games", type=int, help="Max games for RAPM (testing)")
    parser.add_argument("--verify", action="store_true", help="Verify existing model")

    args = parser.parse_args()

    output_path = "data/cache/player_impact/player_impact_model.parquet"

    if args.verify:
        verify_model(output_path)
    elif args.rapm:
        build_rapm_model(
            output_path=output_path,
            max_games=args.max_games,
            seasons=args.seasons,
        )
    else:
        build_bpm_model(
            output_path=output_path,
            min_minutes=args.min_minutes,
            seasons=args.seasons,
        )

    # Always verify after building
    if not args.verify:
        print("\nVerifying model...")
        verify_model(output_path)
