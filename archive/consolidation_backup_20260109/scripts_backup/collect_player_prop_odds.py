#!/usr/bin/env python3
"""
Collect Player Prop Odds Daily

Fetches player prop odds from The Odds API and stores them for backtesting.
Run daily at 3 PM ET (before 4 PM betting pipeline).

Usage:
    python scripts/collect_player_prop_odds.py

Cron:
    0 20 * * * /path/to/venv/bin/python /path/to/scripts/collect_player_prop_odds.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.odds_api import OddsAPIClient


def collect_todays_player_props(output_dir: str = "data/historical_player_props") -> pd.DataFrame:
    """
    Collect player prop odds for today's games.

    Returns:
        DataFrame with player prop odds
    """
    logger.info("=" * 80)
    logger.info("Collecting Player Prop Odds")
    logger.info("=" * 80)

    # Initialize Odds API client
    odds_client = OddsAPIClient()

    # Get today's games first
    logger.info("Fetching today's games...")
    try:
        games_odds = odds_client.get_odds(
            sport='basketball_nba',
            markets=['h2h']  # Just to get game IDs
        )
    except Exception as e:
        logger.error(f"Error fetching games: {e}")
        return pd.DataFrame()

    if 'games' not in games_odds or len(games_odds['games']) == 0:
        logger.warning("No games found for today")
        return pd.DataFrame()

    games = games_odds['games']
    logger.info(f"Found {len(games)} games today")

    # Collect props for each game
    all_props = []
    prop_markets = ['player_points', 'player_rebounds', 'player_assists', 'player_threes']

    for i, game in enumerate(games):
        game_id = game['id']
        home_team = game.get('home_team', 'Unknown')
        away_team = game.get('away_team', 'Unknown')
        commence_time = game.get('commence_time')

        logger.info(f"  [{i+1}/{len(games)}] {away_team} @ {home_team}")

        try:
            # Fetch player props for this game
            props = odds_client.get_player_props(
                event_id=game_id,
                markets=prop_markets
            )

            if not props.empty:
                # Add game metadata
                props['game_id'] = game_id
                props['home_team'] = home_team
                props['away_team'] = away_team
                props['commence_time'] = commence_time
                props['collected_at'] = datetime.now().isoformat()

                all_props.append(props)
                logger.info(f"    ✓ Collected {len(props)} prop lines")
            else:
                logger.warning(f"    ⚠ No props available for this game")

        except Exception as e:
            logger.error(f"    ✗ Error collecting props: {e}")
            continue

    if not all_props:
        logger.warning("No player props collected")
        return pd.DataFrame()

    # Combine all props
    all_props_df = pd.concat(all_props, ignore_index=True)

    logger.info("")
    logger.success(f"Total props collected: {len(all_props_df)}")
    logger.info(f"  Unique players: {all_props_df['player_name'].nunique() if 'player_name' in all_props_df else 0}")
    logger.info(f"  Markets: {all_props_df['market'].unique().tolist() if 'market' in all_props_df else []}")

    # Save to parquet
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.now().date()
    output_file = os.path.join(output_dir, f"props_{today}.parquet")

    all_props_df.to_parquet(output_file)
    logger.success(f"Saved to: {output_file}")

    return all_props_df


def cleanup_old_props(output_dir: str = "data/historical_player_props", keep_days: int = 90):
    """
    Delete prop odds files older than keep_days.

    Args:
        output_dir: Directory containing prop files
        keep_days: Keep files from last N days
    """
    if not os.path.exists(output_dir):
        return

    cutoff_date = datetime.now() - timedelta(days=keep_days)

    deleted_count = 0
    for file in Path(output_dir).glob("props_*.parquet"):
        try:
            # Extract date from filename: props_2026-01-05.parquet
            date_str = file.stem.replace("props_", "")
            file_date = datetime.strptime(date_str, "%Y-%m-%d")

            if file_date < cutoff_date:
                file.unlink()
                deleted_count += 1
                logger.info(f"Deleted old file: {file.name}")
        except Exception as e:
            logger.warning(f"Error processing {file.name}: {e}")

    if deleted_count > 0:
        logger.info(f"Cleaned up {deleted_count} old files (older than {keep_days} days)")


def main():
    """Main execution."""
    # Collect today's props
    props_df = collect_todays_player_props()

    # Cleanup old files
    cleanup_old_props(keep_days=90)

    # Summary
    logger.info("")
    logger.info("=" * 80)
    if not props_df.empty:
        logger.success("✅ Player prop odds collection complete!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Props will be used for backtesting after 30+ days")
        logger.info("  2. Run this script daily via cron")
        logger.info("  3. Check data/historical_player_props/ for collected odds")
    else:
        logger.warning("⚠ No props collected today")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
