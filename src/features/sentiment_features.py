"""
Sentiment Feature Builder

Aggregates public betting sentiment from free web scraping sources.
Uses Reddit's HTML scraping (no authentication required).
"""

from typing import Dict
from datetime import datetime, timedelta
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
        """Initialize sentiment feature builder with caching (fixes Issue #15)."""
        self.scraper = PublicSentimentScraper()
        self.enabled = True  # Always enabled - uses free public data

        # Cache for batch operations
        self._cache = {}
        self._cache_time = None
        self._cache_ttl = timedelta(minutes=15)  # Match cron schedule

        logger.debug("Sentiment feature builder initialized with free public scraping")

    def _get_cached_sentiments(self, date: str) -> Dict[str, Dict]:
        """
        Get all team sentiments with caching (fixes Issue #15).

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            Dictionary mapping team abbreviations to sentiment data
        """
        now = datetime.now()

        # Check cache validity
        if self._cache_time and (now - self._cache_time) < self._cache_ttl:
            logger.debug(f"Using cached sentiment data (age: {(now - self._cache_time).seconds}s)")
            return self._cache

        # Refresh cache - batch query all teams at once
        logger.debug(f"Refreshing sentiment cache for {date}")

        conn = self.scraper._get_connection()
        try:
            cursor = conn.execute("""
                SELECT team_abbrev, sentiment_score, total_mentions
                FROM sentiment_scores
                WHERE date = ?
            """, (date,))

            cache = {}
            for row in cursor.fetchall():
                cache[row['team_abbrev']] = {
                    'sentiment': row['sentiment_score'],
                    'volume': row['total_mentions'],
                    'enabled': True
                }

            self._cache = cache
            self._cache_time = now

            logger.debug(f"Cached sentiment for {len(cache)} teams")
            return cache

        finally:
            conn.close()

    def get_game_features(
        self,
        home_team: str,
        away_team: str
    ) -> Dict[str, float]:
        """
        Get sentiment features for a game matchup.

        Uses caching for better performance (fixes Issue #15).

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation

        Returns:
            Dictionary of sentiment features
        """
        date = datetime.now().strftime("%Y-%m-%d")

        # Use cached batch query
        sentiments = self._get_cached_sentiments(date)

        # Get team data with defaults
        home_data = sentiments.get(home_team, {'sentiment': 0.0, 'volume': 0, 'enabled': True})
        away_data = sentiments.get(away_team, {'sentiment': 0.0, 'volume': 0, 'enabled': True})

        home_sentiment = home_data['sentiment']
        away_sentiment = away_data['sentiment']
        home_volume = home_data['volume']
        away_volume = away_data['volume']

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

        Optimized batch query (fixes Issue #7 - N+1 problem).

        Args:
            games_df: DataFrame with game information
            home_team_col: Column name for home team
            away_team_col: Column name for away team

        Returns:
            DataFrame with sentiment features added
        """
        date = datetime.now().strftime("%Y-%m-%d")

        # Single batch query for all teams (avoids N+1 problem)
        sentiments = self._get_cached_sentiments(date)

        # Build features using cached data
        features_list = []
        for _, game in games_df.iterrows():
            home = sentiments.get(game[home_team_col], {'sentiment': 0.0, 'volume': 0})
            away = sentiments.get(game[away_team_col], {'sentiment': 0.0, 'volume': 0})

            features_list.append({
                "home_sentiment": round(home['sentiment'], 3),
                "away_sentiment": round(away['sentiment'], 3),
                "home_sentiment_volume": home['volume'],
                "away_sentiment_volume": away['volume'],
                "sentiment_diff": round(home['sentiment'] - away['sentiment'], 3),
                "sentiment_enabled": self.enabled,
            })

        features_df = pd.DataFrame(features_list)
        return pd.concat([games_df.reset_index(drop=True), features_df], axis=1)


if __name__ == "__main__":
    # Test the sentiment feature builder
    builder = SentimentFeatureBuilder()

    print("Sentiment Feature Builder Test")
    print("=" * 60)
    print(f"Sentiment enabled: {builder.enabled}")
    print("Using FREE HTML scraping (no API keys required)")

    # Get features for a matchup
    features = builder.get_game_features("LAL", "BOS")
    print("\nFeatures for LAL vs BOS:")
    for k, v in features.items():
        print(f"  {k}: {v}")

    # Test batch operation
    print("\n" + "=" * 60)
    print("Testing batch operation (multiple games)...")

    test_games = pd.DataFrame({
        "home_team": ["LAL", "GSW", "BOS"],
        "away_team": ["BOS", "LAC", "PHI"],
    })

    result = builder.get_features_for_games(test_games)
    print(f"\nProcessed {len(result)} games in single batch query")
    print(result[["home_team", "away_team", "home_sentiment", "away_sentiment", "sentiment_diff"]])

    print("\n" + "=" * 60)
    print("Sentiment values based on r/sportsbook HTML scraping")
