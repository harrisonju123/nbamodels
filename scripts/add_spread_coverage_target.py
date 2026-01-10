#!/usr/bin/env python3
"""
Add Spread Coverage Target to Games Data

Merges historical spread data with games to create a 'home_cover' target variable
for training spread coverage prediction models.

Usage:
    python scripts/add_spread_coverage_target.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from glob import glob
from loguru import logger

# Team name mapping (full name -> abbreviation)
TEAM_NAME_MAP = {
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC',
    'Los Angeles Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS'
}


def load_historical_spreads():
    """Load and aggregate historical spread data."""
    logger.info("Loading historical odds...")

    odds_files = glob("data/historical_odds/odds_*.parquet")
    if not odds_files:
        logger.error("No historical odds files found!")
        return pd.DataFrame()

    logger.info(f"Found {len(odds_files)} odds files")

    # Load all odds
    odds_list = []
    for file in odds_files:
        try:
            df = pd.read_parquet(file)
            odds_list.append(df)
        except Exception as e:
            logger.warning(f"Could not load {file}: {e}")

    if not odds_list:
        return pd.DataFrame()

    odds = pd.concat(odds_list, ignore_index=True)

    # Filter to spread market only
    odds = odds[odds['market'] == 'spread'].copy()

    # Get home team spreads
    home_spreads = odds[odds['team'] == 'home'].copy()
    home_spreads = home_spreads.rename(columns={'line': 'spread'})

    # Take median spread across bookmakers for each game
    median_spreads = home_spreads.groupby('game_id').agg({
        'commence_time': 'first',
        'home_team': 'first',
        'away_team': 'first',
        'spread': 'median'
    }).reset_index()

    # Add game date
    median_spreads['game_date'] = pd.to_datetime(median_spreads['commence_time']).dt.date

    # Map team names to abbreviations
    median_spreads['home_team'] = median_spreads['home_team'].map(TEAM_NAME_MAP)
    median_spreads['away_team'] = median_spreads['away_team'].map(TEAM_NAME_MAP)

    # Remove rows where mapping failed
    median_spreads = median_spreads.dropna(subset=['home_team', 'away_team'])

    logger.success(f"Loaded spreads for {len(median_spreads)} games")

    return median_spreads[['game_date', 'home_team', 'away_team', 'spread']]


def add_spread_coverage(games, spreads):
    """
    Add spread coverage target to games.

    Args:
        games: DataFrame with game data
        spreads: DataFrame with historical spreads

    Returns:
        DataFrame with spread coverage added
    """
    logger.info(f"Merging {len(games)} games with {len(spreads)} spreads...")

    # Add game_date to games
    games['game_date'] = pd.to_datetime(games['date']).dt.date

    # Merge on date and teams
    games_with_spreads = games.merge(
        spreads,
        on=['game_date', 'home_team', 'away_team'],
        how='left'
    )

    # Calculate spread coverage
    # Home team covers if: (home_score - away_score) > -spread
    # Example: If spread is -3.5 (home favored), home covers if they win by 4+
    games_with_spreads['point_margin'] = (
        games_with_spreads['home_score'] - games_with_spreads['away_score']
    )
    games_with_spreads['home_cover'] = (
        games_with_spreads['point_margin'] > -games_with_spreads['spread']
    ).astype(int)

    # Drop rows without spreads (keep only games where we have spread data)
    games_with_coverage = games_with_spreads[games_with_spreads['spread'].notna()].copy()

    # Drop temporary game_date column
    games_with_coverage = games_with_coverage.drop(columns=['game_date'])

    logger.success(f"Added spread coverage to {len(games_with_coverage)} games")
    logger.info(f"Home cover rate: {games_with_coverage['home_cover'].mean():.1%}")

    return games_with_coverage


def main():
    logger.info("=" * 80)
    logger.info("ADD SPREAD COVERAGE TARGET")
    logger.info("=" * 80)

    # Load games
    logger.info("\n1. Loading games...")
    games = pd.read_parquet("data/raw/games.parquet")
    logger.info(f"Loaded {len(games)} total games")

    # Filter to completed games
    games = games[games['status'] == 'Final'].copy()
    logger.info(f"Filtered to {len(games)} completed games")

    # Load spreads
    logger.info("\n2. Loading historical spreads...")
    spreads = load_historical_spreads()

    if spreads.empty:
        logger.error("No spreads loaded - cannot continue")
        return 1

    # Add coverage
    logger.info("\n3. Adding spread coverage target...")
    games_with_coverage = add_spread_coverage(games, spreads)

    # Save
    output_path = "data/raw/games_with_spread_coverage.parquet"
    logger.info(f"\n4. Saving to {output_path}...")
    games_with_coverage.to_parquet(output_path)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nSummary:")
    logger.info(f"  Total games: {len(games)}")
    logger.info(f"  Games with spreads: {len(games_with_coverage)} ({len(games_with_coverage)/len(games)*100:.1f}%)")
    logger.info(f"  Home cover rate: {games_with_coverage['home_cover'].mean():.1%}")
    logger.info(f"  Output: {output_path}")

    # Show sample
    logger.info(f"\n📋 Sample data:")
    sample = games_with_coverage[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'spread', 'point_margin', 'home_cover']].head(5)
    print(sample.to_string(index=False))

    logger.info(f"\n✅ Ready to retrain models with spread coverage target!")
    logger.info(f"   Next step: python scripts/retrain_with_tuned_params.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
