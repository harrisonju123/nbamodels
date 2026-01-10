"""
NBA Stats Data Fetcher

Uses nba_api to fetch comprehensive NBA statistics from NBA.com.
No rate limits, fast and reliable.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from nba_api.stats.endpoints import leaguegamefinder, commonteamroster
from nba_api.stats.static import teams


class NBAStatsClient:
    """Client for fetching NBA game and player statistics using nba_api."""

    def __init__(self):
        pass  # No API key needed for nba_api

    def get_season_games(self, season: int) -> pd.DataFrame:
        """
        Fetch all games for an NBA season.

        Args:
            season: NBA season year (e.g., 2024 for 2024-25 season)

        Returns:
            DataFrame with all games for the season
        """
        season_str = f"{season}-{str(season + 1)[-2:]}"  # e.g., "2024-25"
        logger.info(f"Fetching games for {season_str} season...")

        # Fetch all games for the season
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season_str,
            league_id_nullable='00'  # NBA
        )
        df = gamefinder.get_data_frames()[0]

        if df.empty:
            logger.warning(f"No games found for {season_str} season")
            return pd.DataFrame()

        # Convert team-game records to game-level records
        return self._convert_to_game_level(df, season)

    def _convert_to_game_level(self, team_games: pd.DataFrame, season: int) -> pd.DataFrame:
        """Convert team-game records to game-level records."""
        # Each game appears twice (once per team), need to merge
        games = {}

        for _, row in team_games.iterrows():
            game_id = row['GAME_ID']
            matchup = row['MATCHUP']
            team = row['TEAM_ABBREVIATION']
            pts = row['PTS']

            if game_id not in games:
                games[game_id] = {
                    'game_id': game_id,
                    'date': row['GAME_DATE'],
                    'season': season,
                }

            # Determine if home or away based on matchup format
            # Home games: "LAL vs. BOS", Away games: "LAL @ BOS"
            is_home = ' vs. ' in matchup

            if is_home:
                games[game_id]['home_team'] = team
                games[game_id]['home_score'] = pts
                games[game_id]['home_team_id'] = row['TEAM_ID']
            else:
                games[game_id]['away_team'] = team
                games[game_id]['away_score'] = pts
                games[game_id]['away_team_id'] = row['TEAM_ID']

        # Convert to DataFrame
        records = list(games.values())
        df = pd.DataFrame(records)

        if df.empty:
            return df

        # Filter out incomplete games (missing team data)
        required_cols = ['home_team', 'away_team', 'home_score', 'away_score']
        df = df.dropna(subset=required_cols)

        # Add computed columns
        df['date'] = pd.to_datetime(df['date'])
        df['home_score'] = df['home_score'].astype(int)
        df['away_score'] = df['away_score'].astype(int)
        df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
        df['total_score'] = df['home_score'] + df['away_score']
        df['point_diff'] = df['home_score'] - df['away_score']
        df['status'] = 'Final'

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        logger.info(f"Processed {len(df)} games for season {season}")
        return df

    def get_games(
        self,
        start_date: str,
        end_date: str,
        per_page: int = 100,  # Ignored, kept for compatibility
    ) -> pd.DataFrame:
        """
        Fetch games between two dates.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            per_page: Ignored (for API compatibility)

        Returns:
            DataFrame with game data
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Determine which seasons to fetch
        seasons = set()
        current = start
        while current <= end:
            # NBA season starts in October
            if current.month >= 10:
                seasons.add(current.year)
            else:
                seasons.add(current.year - 1)
            current += timedelta(days=365)

        # Fetch each season
        all_games = []
        for season in sorted(seasons):
            games = self.get_season_games(season)
            if not games.empty:
                all_games.append(games)

        if not all_games:
            return pd.DataFrame()

        df = pd.concat(all_games, ignore_index=True)

        # Filter to date range
        df = df[(df['date'] >= start) & (df['date'] <= end)]
        df = df.drop_duplicates(subset=['game_id']).reset_index(drop=True)

        return df

    def get_teams(self) -> pd.DataFrame:
        """Fetch all NBA teams."""
        nba_teams = teams.get_teams()

        records = []
        for team in nba_teams:
            records.append({
                "team_id": team["id"],
                "abbreviation": team["abbreviation"],
                "city": team["city"],
                "name": team["nickname"],
                "full_name": team["full_name"],
                "conference": "",  # Not provided by static data
                "division": "",
            })

        return pd.DataFrame(records)


def fetch_historical_games(
    start_season: int = 2015,
    end_season: int = 2024,
    output_path: str = "data/raw/games.parquet",
) -> pd.DataFrame:
    """
    Fetch historical game data for multiple seasons.

    Args:
        start_season: First season to fetch
        end_season: Last season to fetch
        output_path: Path to save the data

    Returns:
        DataFrame with all historical games
    """
    client = NBAStatsClient()
    all_games = []

    for season in range(start_season, end_season + 1):
        logger.info(f"Fetching season {season}...")
        games = client.get_season_games(season)
        all_games.append(games)
        time.sleep(1)  # Be nice to the API

    df = pd.concat(all_games, ignore_index=True)
    df = df.drop_duplicates(subset=["game_id"])
    df = df.sort_values("date").reset_index(drop=True)

    # Save to parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
    logger.info(f"Saved {len(df)} games to {output_path}")

    return df


if __name__ == "__main__":
    # Example usage
    client = NBAStatsClient()

    # Fetch current season games
    games = client.get_season_games(2024)
    print(f"Fetched {len(games)} games for 2024-25 season")
    print(games.head())
