"""
News Feature Builder

Generates features from NBA news article volume and recency.
Tracks news attention for each team to detect potential market inefficiencies.

Optimized with batch queries and caching to avoid N+1 query problems.
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

    Optimized with:
    - Batch queries to avoid N+1 problem
    - 1-hour caching for news data
    """

    def __init__(self, db_path: str = None):
        """
        Initialize news feature builder.

        Args:
            db_path: Path to database (defaults to BETS_DB_PATH)
        """
        self.client = NBANewsClient(db_path=db_path or BETS_DB_PATH)

        # Caching for all team news (1-hour TTL)
        self._volume_cache = {}
        self._recency_cache = {}
        self._cache_time = None
        self._cache_ttl = timedelta(hours=1)

    def _get_all_team_news(self, lookback_hours: int = 24) -> Dict[str, Dict]:
        """
        Get news volume and recency for all teams with caching.

        Loads all team news data once and caches for 1 hour.
        Avoids N+1 queries when processing multiple games.

        Args:
            lookback_hours: Hours to look back for news volume

        Returns:
            Dictionary mapping team abbreviation to news data
        """
        now = datetime.now()

        # Check cache validity
        if self._cache_time and (now - self._cache_time) < self._cache_ttl:
            logger.debug(f"Using cached news data (age: {(now - self._cache_time).seconds}s)")
            return self._volume_cache

        # Refresh cache - batch query all team news
        logger.debug(f"Refreshing news cache (lookback: {lookback_hours}h)")

        try:
            # Get connection from client
            conn = self.client._get_connection()

            # Calculate cutoff time for lookback
            cutoff = datetime.now() - timedelta(hours=lookback_hours)
            cutoff_str = cutoff.isoformat()

            # Single query to get volume for all teams
            volume_cursor = conn.execute("""
                SELECT
                    ne.team_abbrev,
                    COUNT(DISTINCT na.id) as article_count
                FROM news_entities ne
                JOIN news_articles na ON ne.article_id = na.id
                WHERE na.published_at >= ?
                  AND ne.team_abbrev IS NOT NULL
                GROUP BY ne.team_abbrev
            """, (cutoff_str,))

            volume_cache = {}
            for row in volume_cursor.fetchall():
                team = row['team_abbrev']
                volume_cache[team] = {
                    'volume': row['article_count'],
                    'recency': None,  # Will be filled next
                }

            # Single query to get recency for all teams
            recency_cursor = conn.execute("""
                SELECT
                    ne.team_abbrev,
                    MAX(na.published_at) as latest_article
                FROM news_entities ne
                JOIN news_articles na ON ne.article_id = na.id
                WHERE ne.team_abbrev IS NOT NULL
                GROUP BY ne.team_abbrev
            """)

            for row in recency_cursor.fetchall():
                team = row['team_abbrev']
                latest = row['latest_article']

                if latest:
                    try:
                        latest_dt = datetime.fromisoformat(latest)
                        hours_since = (datetime.now() - latest_dt).total_seconds() / 3600
                    except (ValueError, TypeError):
                        hours_since = 999.0
                else:
                    hours_since = 999.0

                # Update cache with recency
                if team in volume_cache:
                    volume_cache[team]['recency'] = hours_since
                else:
                    volume_cache[team] = {
                        'volume': 0,
                        'recency': hours_since,
                    }

            conn.close()

            self._volume_cache = volume_cache
            self._cache_time = now

            logger.debug(f"Cached news data for {len(volume_cache)} teams")
            return volume_cache

        except Exception as e:
            logger.error(f"Failed to load news data: {e}")
            return {}

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
        # Get all team news (cached)
        all_news = self._get_all_team_news(lookback_hours)

        # Get data for home and away teams
        home_data = all_news.get(home_team, {'volume': 0, 'recency': 999.0})
        away_data = all_news.get(away_team, {'volume': 0, 'recency': 999.0})

        home_volume = home_data['volume']
        away_volume = away_data['volume']
        home_recency = home_data['recency'] if home_data['recency'] is not None else 999.0
        away_recency = away_data['recency'] if away_data['recency'] is not None else 999.0

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

        Optimized with batch queries to avoid N+1 problem.

        Args:
            games_df: DataFrame with game information
            home_team_col: Column name for home team
            away_team_col: Column name for away team
            lookback_hours: Hours to look back for news

        Returns:
            DataFrame with news features added
        """
        if games_df.empty:
            return games_df

        # Pre-load all team news data (single set of queries, cached)
        all_news = self._get_all_team_news(lookback_hours)

        # Build features for each game using cached data
        features_list = []

        for _, game in games_df.iterrows():
            home_team = game[home_team_col]
            away_team = game[away_team_col]

            # Get data from cache
            home_data = all_news.get(home_team, {'volume': 0, 'recency': 999.0})
            away_data = all_news.get(away_team, {'volume': 0, 'recency': 999.0})

            home_volume = home_data['volume']
            away_volume = away_data['volume']
            home_recency = home_data['recency'] if home_data['recency'] is not None else 999.0
            away_recency = away_data['recency'] if away_data['recency'] is not None else 999.0

            features_list.append({
                "home_news_volume_24h": home_volume,
                "away_news_volume_24h": away_volume,
                "home_news_recency": round(home_recency, 1),
                "away_news_recency": round(away_recency, 1),
                "news_volume_diff": home_volume - away_volume,
            })

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
        # Get all team news (cached)
        all_news = self._get_all_team_news(hours)

        team_data = all_news.get(team_abbrev, {'volume': 0, 'recency': 999.0})

        volume = team_data['volume']
        recency = team_data['recency'] if team_data['recency'] is not None else 999.0

        return {
            "team": team_abbrev,
            "volume_24h": volume,
            "hours_since_last": recency,
            "has_recent_news": recency < 2.0,  # Within 2 hours
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

    # Test batch operation
    print("\n" + "=" * 60)
    print("Testing batch operation (multiple games)...")

    test_games = pd.DataFrame({
        "home_team": ["LAL", "GSW", "BOS"],
        "away_team": ["BOS", "LAC", "PHI"],
    })

    result = builder.get_features_for_games(test_games)
    print(f"\nProcessed {len(result)} games in single batch query")
    print(result[["home_team", "away_team", "home_news_volume_24h", "away_news_volume_24h", "news_volume_diff"]])

    print("\n" + "=" * 60)
    print("News features based on RSS feed scraping")
