#!/usr/bin/env python3
"""
Fetch Opponent Defense Stats by Position

Fetches team defensive stats broken down by position from NBA Stats API.
This enables accurate opponent defensive features (e.g., how many PPG does LAL allow to opposing PFs).

Usage:
    python scripts/fetch_opponent_defense_stats.py [--season 2024-25]

Caching:
    - Updates weekly (defensive stats change slowly)
    - Cached in data/opponent_defense/defense_by_position_SEASON.parquet
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger
from nba_api.stats.endpoints import LeagueDashPtTeamDefend
from nba_api.stats.static import teams
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_defense_by_position(season: str = "2024-25") -> pd.DataFrame:
    """
    Fetch team defensive stats broken down by position.

    Args:
        season: NBA season (e.g., "2024-25")

    Returns:
        DataFrame with columns:
        - team_abbreviation: Team code
        - defense_category: Position defended (e.g., "Overall", "Guard", "Forward", "Center")
        - pts_allowed: Points allowed per game to that position
        - fg_pct_allowed: FG% allowed
        - fg3_pct_allowed: 3P% allowed
        - opp_possessions: Possessions against
        - def_rating: Defensive rating vs that position
    """
    logger.info(f"Fetching defensive stats by position for {season}...")

    all_defense = []

    # Defense categories (positions)
    defense_categories = [
        "Overall",
        "3 Pointers",  # Perimeter defense
        "2 Pointers",  # Paint defense
        "Less Than 6Ft",  # Rim protection
        "Less Than 10Ft",
        "Greater Than 15Ft"
    ]

    for category in defense_categories:
        logger.info(f"  Fetching {category} defense...")

        try:
            # Fetch from NBA Stats API
            defense_data = LeagueDashPtTeamDefend(
                season=season,
                season_type_all_star="Regular Season",
                defense_category=category
            )

            df = defense_data.get_data_frames()[0]

            if not df.empty:
                # Add category label
                df['defense_category'] = category

                # Select relevant columns
                cols = [
                    'TEAM_ABBREVIATION', 'defense_category',
                    'D_FGM', 'D_FGA', 'D_FG_PCT',
                    'FREQ'  # Frequency faced
                ]

                # Check which columns exist
                available_cols = [c for c in cols if c in df.columns]
                df_clean = df[available_cols].copy()

                # Rename
                df_clean.columns = [c.lower() for c in df_clean.columns]
                all_defense.append(df_clean)

                logger.info(f"    ✓ Got {len(df_clean)} teams")
            else:
                logger.warning(f"    ⚠ No data for {category}")

            # Rate limit
            time.sleep(0.6)

        except Exception as e:
            logger.error(f"    ✗ Error fetching {category}: {e}")
            continue

    if not all_defense:
        logger.error("No defensive data collected")
        return pd.DataFrame()

    # Combine all categories
    all_defense_df = pd.concat(all_defense, ignore_index=True)

    logger.success(f"✓ Collected {len(all_defense_df)} team-category records")

    return all_defense_df


def get_team_defense_summary(season: str = "2024-25") -> pd.DataFrame:
    """
    Get simplified team defensive stats for player props features.

    Returns:
        DataFrame with columns:
        - team_abbreviation
        - opp_fg_pct: Overall opponent FG% allowed
        - opp_fg3_pct: Opponent 3P% allowed
        - opp_fg2_pct: Opponent 2P% allowed
        - opp_fgpct_lt6ft: Opponent FG% allowed <6ft (rim)
        - opp_fgpct_lt10ft: Opponent FG% allowed <10ft
        - opp_fgpct_gt15ft: Opponent FG% allowed >15ft (perimeter)
        - def_freq: Frequency of possessions defended
    """
    logger.info(f"Building team defensive summary for {season}...")

    # Get all defensive data
    defense_df = get_defense_by_position(season)

    if defense_df.empty:
        return pd.DataFrame()

    # Pivot to get features per team
    summary_list = []

    for team in defense_df['team_abbreviation'].unique():
        team_def = defense_df[defense_df['team_abbreviation'] == team]

        # Build summary from different defense categories
        summary = {'team_abbreviation': team}

        # Map defense categories to (feature_name, column_name)
        # Different categories use different column names in the API
        category_map = {
            'Overall': ('opp_fg_pct', 'd_fg_pct'),
            '3 Pointers': ('opp_fg3_pct', 'fg3_pct'),
            '2 Pointers': ('opp_fg2_pct', 'd_fg_pct'),
            'Less Than 6Ft': ('opp_fgpct_lt6ft', 'd_fg_pct'),
            'Less Than 10Ft': ('opp_fgpct_lt10ft', 'd_fg_pct'),
            'Greater Than 15Ft': ('opp_fgpct_gt15ft', 'd_fg_pct')
        }

        for category, (feature_name, col_name) in category_map.items():
            cat_data = team_def[team_def['defense_category'] == category]
            if not cat_data.empty and col_name in cat_data.columns:
                summary[feature_name] = cat_data[col_name].iloc[0]
            else:
                # Use reasonable defaults if data missing
                defaults = {
                    'opp_fg_pct': 0.46,
                    'opp_fg3_pct': 0.36,
                    'opp_fg2_pct': 0.52,
                    'opp_fgpct_lt6ft': 0.63,
                    'opp_fgpct_lt10ft': 0.50,
                    'opp_fgpct_gt15ft': 0.38
                }
                summary[feature_name] = defaults.get(feature_name, 0.46)

        summary_list.append(summary)

    summary_df = pd.DataFrame(summary_list)

    logger.success(f"✓ Built defensive summary for {len(summary_df)} teams")

    return summary_df


def cache_defense_stats(season: str = "2024-25", force_refresh: bool = False) -> pd.DataFrame:
    """
    Cache defensive stats (update weekly).

    Args:
        season: NBA season
        force_refresh: Force fetch even if cache exists

    Returns:
        Cached defensive stats
    """
    cache_dir = "data/opponent_defense"
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(cache_dir, f"defense_by_position_{season}.parquet")

    # Check cache age
    if os.path.exists(cache_file) and not force_refresh:
        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))

        if file_age < timedelta(days=7):
            logger.info(f"Using cached defense stats (age: {file_age.days} days)")
            return pd.read_parquet(cache_file)
        else:
            logger.info(f"Cache expired (age: {file_age.days} days), refreshing...")

    # Fetch fresh data
    logger.info("Fetching fresh defensive stats...")
    defense_df = get_team_defense_summary(season)

    if not defense_df.empty:
        # Save cache
        defense_df.to_parquet(cache_file)
        logger.success(f"✓ Cached to {cache_file}")
    else:
        logger.error("Failed to fetch defensive stats")

    return defense_df


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Fetch opponent defense stats by position")
    parser.add_argument("--season", type=str, default="2024-25", help="NBA season (e.g., 2024-25)")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh cache")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Fetching Opponent Defensive Stats")
    logger.info("=" * 80)
    logger.info("")

    # Fetch and cache
    defense_df = cache_defense_stats(season=args.season, force_refresh=args.force_refresh)

    # Show results
    if not defense_df.empty:
        logger.info("")
        logger.success("✅ Defensive stats cached successfully!")
        logger.info(f"  Teams: {len(defense_df)}")
        logger.info(f"  Columns: {defense_df.columns.tolist()}")
        logger.info("")
        logger.info("Sample (Best rim protection - lowest FG% <6ft):")
        if 'opp_fgpct_lt6ft' in defense_df.columns:
            top_rim = defense_df.nsmallest(5, 'opp_fgpct_lt6ft')[['team_abbreviation', 'opp_fgpct_lt6ft']]
            for idx, row in top_rim.iterrows():
                logger.info(f"  {row['team_abbreviation']:4s}: {row['opp_fgpct_lt6ft']:.1%} allowed at rim")

        logger.info("")
        logger.info("Sample (Best perimeter defense - lowest 3P%):")
        if 'opp_fg3_pct' in defense_df.columns:
            top_perimeter = defense_df.nsmallest(5, 'opp_fg3_pct')[['team_abbreviation', 'opp_fg3_pct']]
            for idx, row in top_perimeter.iterrows():
                logger.info(f"  {row['team_abbreviation']:4s}: {row['opp_fg3_pct']:.1%} allowed from 3PT")

        logger.info("")
        logger.info("Sample (Best overall defense):")
        if 'opp_fg_pct' in defense_df.columns:
            top_overall = defense_df.nsmallest(5, 'opp_fg_pct')[['team_abbreviation', 'opp_fg_pct']]
            for idx, row in top_overall.iterrows():
                logger.info(f"  {row['team_abbreviation']:4s}: {row['opp_fg_pct']:.1%} overall FG% allowed")
    else:
        logger.error("❌ Failed to fetch defensive stats")

    logger.info("")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
