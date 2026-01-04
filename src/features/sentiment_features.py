"""
Sentiment Feature Builder

Aggregates social sentiment from Reddit and Twitter.
Returns neutral defaults when API credentials not configured.
"""

from typing import Dict
import pandas as pd
from loguru import logger

from src.data.reddit_client import RedditClient
from src.data.twitter_client import TwitterClient


class SentimentFeatureBuilder:
    """
    Build social sentiment features for NBA predictions.

    Features generated:
    - home_reddit_sentiment: Reddit sentiment for home team (-1 to 1)
    - away_reddit_sentiment: Reddit sentiment for away team (-1 to 1)
    - home_twitter_sentiment: Twitter sentiment for home team (-1 to 1)
    - away_twitter_sentiment: Twitter sentiment for away team (-1 to 1)
    - home_sentiment_avg: Average of Reddit + Twitter
    - away_sentiment_avg: Average of Reddit + Twitter
    - sentiment_enabled: Boolean flag for API availability
    """

    def __init__(self):
        """Initialize sentiment feature builder."""
        self.reddit = RedditClient()
        self.twitter = TwitterClient()
        self.enabled = self.reddit.enabled or self.twitter.enabled

        if not self.enabled:
            logger.info("Social sentiment APIs not configured - features will return neutral defaults")

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
        # Get Reddit sentiment
        reddit_data = self.reddit.get_game_sentiment(home_team, away_team)
        home_reddit = reddit_data.get("home_sentiment", 0.0)
        away_reddit = reddit_data.get("away_sentiment", 0.0)

        # Get Twitter sentiment
        twitter_data = self.twitter.get_game_sentiment(home_team, away_team)
        home_twitter = twitter_data.get("home_sentiment", 0.0)
        away_twitter = twitter_data.get("away_sentiment", 0.0)

        # Calculate averages
        home_avg = (home_reddit + home_twitter) / 2
        away_avg = (away_reddit + away_twitter) / 2

        return {
            "home_reddit_sentiment": round(home_reddit, 3),
            "away_reddit_sentiment": round(away_reddit, 3),
            "home_twitter_sentiment": round(home_twitter, 3),
            "away_twitter_sentiment": round(away_twitter, 3),
            "home_sentiment_avg": round(home_avg, 3),
            "away_sentiment_avg": round(away_avg, 3),
            "sentiment_diff": round(home_avg - away_avg, 3),
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
    print(f"Sentiment APIs enabled: {builder.enabled}")

    # Get features for a matchup
    features = builder.get_game_features("LAL", "BOS")
    print("\nFeatures for LAL vs BOS:")
    for k, v in features.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("All sentiment values are neutral (0.0) until APIs configured")
