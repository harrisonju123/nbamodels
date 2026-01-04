"""
Sentiment Feature Builder

Aggregates public betting sentiment from free web scraping sources.
Uses Reddit's public JSON API (no authentication required).
"""

from typing import Dict
import pandas as pd
from loguru import logger

from src.data.public_sentiment_scraper import PublicSentimentScraper


class SentimentFeatureBuilder:
    """
    Build social sentiment features for NBA predictions.

    Features generated:
    - home_sentiment: Public sentiment for home team (-1 to 1)
    - away_sentiment: Public sentiment for away team (-1 to 1)
    - home_sentiment_volume: Number of mentions for home team
    - away_sentiment_volume: Number of mentions for away team
    - sentiment_diff: Difference in sentiment (home - away)
    - sentiment_enabled: Always True (free scraping, no API keys needed)
    """

    def __init__(self):
        """Initialize sentiment feature builder."""
        self.scraper = PublicSentimentScraper()
        self.enabled = True  # Always enabled - uses free public data
        logger.debug("Sentiment feature builder initialized with free public scraping")

    def get_game_features(
        self,
        home_team: str,
        away_team: str
    ) -> Dict[str, float]:
        """
        Get sentiment features for a game matchup.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation

        Returns:
            Dictionary of sentiment features
        """
        # Get sentiment from public scraping
        home_data = self.scraper.get_team_sentiment(home_team)
        away_data = self.scraper.get_team_sentiment(away_team)

        home_sentiment = home_data.get("sentiment", 0.0)
        away_sentiment = away_data.get("sentiment", 0.0)
        home_volume = home_data.get("volume", 0)
        away_volume = away_data.get("volume", 0)

        return {
            "home_sentiment": round(home_sentiment, 3),
            "away_sentiment": round(away_sentiment, 3),
            "home_sentiment_volume": home_volume,
            "away_sentiment_volume": away_volume,
            "sentiment_diff": round(home_sentiment - away_sentiment, 3),
            "sentiment_enabled": self.enabled,
        }

    def get_features_for_games(
        self,
        games_df: pd.DataFrame,
        home_team_col: str = "home_team",
        away_team_col: str = "away_team"
    ) -> pd.DataFrame:
        """
        Generate sentiment features for multiple games.

        Args:
            games_df: DataFrame with game information
            home_team_col: Column name for home team
            away_team_col: Column name for away team

        Returns:
            DataFrame with sentiment features added
        """
        features_list = []

        for _, game in games_df.iterrows():
            features = self.get_game_features(
                home_team=game[home_team_col],
                away_team=game[away_team_col]
            )
            features_list.append(features)

        features_df = pd.DataFrame(features_list)
        return pd.concat([games_df.reset_index(drop=True), features_df], axis=1)


if __name__ == "__main__":
    # Test the sentiment feature builder
    builder = SentimentFeatureBuilder()

    print("Sentiment Feature Builder Test")
    print("=" * 60)
    print(f"Sentiment enabled: {builder.enabled}")
    print("Using FREE public web scraping (no API keys required)")

    # Get features for a matchup
    features = builder.get_game_features("LAL", "BOS")
    print("\nFeatures for LAL vs BOS:")
    for k, v in features.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Sentiment values based on r/sportsbook public discussions")
