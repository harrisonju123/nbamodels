"""
Twitter/X Client - STUBBED

Twitter/X sentiment analysis client for NBA discussions.
Requires TWITTER_BEARER_TOKEN.

Status: Stubbed until API credentials are obtained.
Returns neutral defaults when not configured.
"""

import os
from typing import Dict
from loguru import logger


class TwitterClient:
    """
    Client for Twitter/X NBA sentiment analysis.

    STUBBED: Returns neutral defaults until API credentials configured.

    To activate:
    1. Get Twitter API Bearer Token from https://developer.twitter.com/
    2. Add to .env:
       TWITTER_BEARER_TOKEN=your_bearer_token
    3. Install tweepy: pip install tweepy
    """

    def __init__(self):
        """Initialize Twitter client (stubbed)."""
        self.bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        self.enabled = bool(self.bearer_token)

        if not self.enabled:
            logger.warning(
                "Twitter API not configured - sentiment features disabled. "
                "Set TWITTER_BEARER_TOKEN to enable."
            )

    def get_team_sentiment(
        self,
        team_abbrev: str,
        hours: int = 24
    ) -> Dict:
        """
        Get Twitter sentiment for a team (stubbed).

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

        # TODO: Implement actual Twitter API calls when credentials available
        # Would use Tweepy to:
        # 1. Search for team hashtags/mentions
        # 2. Analyze tweet sentiment
        # 3. Track tweet volume and engagement

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
        Get Twitter sentiment for a game matchup (stubbed).

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
    client = TwitterClient()
    print(f"Twitter client enabled: {client.enabled}")

    # Get stubbed sentiment
    sentiment = client.get_team_sentiment("LAL")
    print(f"\nLAL sentiment (stubbed): {sentiment}")
