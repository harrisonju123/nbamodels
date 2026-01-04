"""
Reddit Client - STUBBED

Reddit sentiment analysis client for NBA discussions.
Requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET.

Status: Stubbed until API credentials are obtained.
Returns neutral defaults when not configured.
"""

import os
from typing import Dict
from loguru import logger


class RedditClient:
    """
    Client for Reddit NBA sentiment analysis.

    STUBBED: Returns neutral defaults until API credentials configured.

    To activate:
    1. Get Reddit API credentials from https://www.reddit.com/prefs/apps
    2. Add to .env:
       REDDIT_CLIENT_ID=your_client_id
       REDDIT_CLIENT_SECRET=your_client_secret
    3. Install praw: pip install praw
    """

    def __init__(self):
        """Initialize Reddit client (stubbed)."""
        self.client_id = os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.enabled = bool(self.client_id and self.client_secret)

        if not self.enabled:
            logger.warning(
                "Reddit API not configured - sentiment features disabled. "
                "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to enable."
            )

    def get_team_sentiment(
        self,
        team_abbrev: str,
        hours: int = 24
    ) -> Dict:
        """
        Get Reddit sentiment for a team (stubbed).

        Args:
            team_abbrev: Team abbreviation
            hours: Lookback window

        Returns:
            Dict with neutral sentiment values
        """
        if not self.enabled:
            return {
                "team": team_abbrev,
                "sentiment": 0.0,  # Neutral
                "volume": 0,
                "enabled": False,
            }

        # TODO: Implement actual Reddit API calls when credentials available
        # Would use PRAW to:
        # 1. Search r/nba for team mentions
        # 2. Analyze comment sentiment
        # 3. Track discussion volume

        return {
            "team": team_abbrev,
            "sentiment": 0.0,
            "volume": 0,
            "enabled": True,
        }

    def get_game_sentiment(
        self,
        home_team: str,
        away_team: str
    ) -> Dict:
        """
        Get Reddit sentiment for a game matchup (stubbed).

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation

        Returns:
            Dict with neutral sentiment values
        """
        if not self.enabled:
            return {
                "home_sentiment": 0.0,
                "away_sentiment": 0.0,
                "sentiment_diff": 0.0,
                "enabled": False,
            }

        # TODO: Implement game-specific sentiment analysis
        return {
            "home_sentiment": 0.0,
            "away_sentiment": 0.0,
            "sentiment_diff": 0.0,
            "enabled": True,
        }


if __name__ == "__main__":
    # Test the stubbed client
    client = RedditClient()
    print(f"Reddit client enabled: {client.enabled}")

    # Get stubbed sentiment
    sentiment = client.get_team_sentiment("LAL")
    print(f"\nLAL sentiment (stubbed): {sentiment}")
