"""
News Feature Builder

Generates features from NBA news article volume and recency.
Tracks news attention for each team to detect potential market inefficiencies.
"""

from datetime import datetime, timedelta
from typing import Dict

import pandas as pd
from loguru import logger

from src.data.news_scrapers import NBANewsClient
from src.utils.constants import BETS_DB_PATH


class NewsFeatureBuilder:
    """
    Build news-related features for NBA predictions.

    Features generated:
    - home_news_volume_24h: Articles mentioning home team (24h)
    - away_news_volume_24h: Articles mentioning away team (24h)
    - home_news_recency: Hours since last article about home team
    - away_news_recency: Hours since last article about away team
    - news_volume_diff: Home - Away volume difference
    """

    def __init__(self, db_path: str = None):
        """
        Initialize news feature builder.

        Args:
            db_path: Path to database (defaults to BETS_DB_PATH)
        """
        self.client = NBANewsClient(db_path=db_path or BETS_DB_PATH)
        self._cache = {}
        self._cache_time = None
        self._cache_duration = timedelta(hours=1)

    def get_game_features(
        self,
        home_team: str,
        away_team: str,
        lookback_hours: int = 24
    ) -> Dict[str, float]:
        """
        Get news features for a game matchup.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            lookback_hours: Hours to look back for news volume

        Returns:
            Dictionary of news features
        """
        # Get news volume for each team
        home_volume = self.client.get_team_news_volume(home_team, hours=lookback_hours)
        away_volume = self.client.get_team_news_volume(away_team, hours=lookback_hours)

        # Get recency (hours since last article)
        home_recency = self.client.get_team_news_recency(home_team)
        away_recency = self.client.get_team_news_recency(away_team)

        # Handle None values (no articles found)
        if home_recency is None:
            home_recency = 999.0  # Large number = no recent news
        if away_recency is None:
            away_recency = 999.0

        return {
            "home_news_volume_24h": home_volume,
            "away_news_volume_24h": away_volume,
            "home_news_recency": round(home_recency, 1),
            "away_news_recency": round(away_recency, 1),
            "news_volume_diff": home_volume - away_volume,
        }

    def get_features_for_games(
        self,
        games_df: pd.DataFrame,
        home_team_col: str = "home_team",
        away_team_col: str = "away_team",
        lookback_hours: int = 24
    ) -> pd.DataFrame:
        """
        Generate news features for multiple games.

        Args:
            games_df: DataFrame with game information
            home_team_col: Column name for home team
            away_team_col: Column name for away team
            lookback_hours: Hours to look back for news

        Returns:
            DataFrame with news features added
        """
        features_list = []

        for _, game in games_df.iterrows():
            features = self.get_game_features(
                home_team=game[home_team_col],
                away_team=game[away_team_col],
                lookback_hours=lookback_hours
            )
            features_list.append(features)

        features_df = pd.DataFrame(features_list)
        return pd.concat([games_df.reset_index(drop=True), features_df], axis=1)

    def get_team_summary(self, team_abbrev: str, hours: int = 24) -> Dict:
        """
        Get news summary for a team.

        Args:
            team_abbrev: Team abbreviation
            hours: Lookback window

        Returns:
            Dictionary with team news summary
        """
        volume = self.client.get_team_news_volume(team_abbrev, hours=hours)
        recency = self.client.get_team_news_recency(team_abbrev)

        return {
            "team": team_abbrev,
            "volume_24h": volume,
            "hours_since_last": recency if recency is not None else 999.0,
            "has_recent_news": recency is not None and recency < 2.0,  # Within 2 hours
        }


if __name__ == "__main__":
    # Test the news feature builder
    builder = NewsFeatureBuilder()

    print("News Feature Builder Test")
    print("=" * 60)

    # Example: Get features for a matchup
    features = builder.get_game_features("LAL", "BOS")
    print("\nFeatures for LAL vs BOS:")
    for k, v in features.items():
        print(f"  {k}: {v}")

    # Get team summary
    print("\nTeam Summary for LAL:")
    summary = builder.get_team_summary("LAL")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
