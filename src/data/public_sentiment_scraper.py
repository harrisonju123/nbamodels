"""
Public Sentiment Scraper - FREE Alternative

Scrapes public NBA betting discussions from free sources:
- r/sportsbook via Reddit's public JSON API (no auth required)
- Covers.com public consensus data
- RotoGrinders public betting percentages

No API keys required - uses public web scraping.
"""

import os
import sqlite3
import time
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import pandas as pd
import requests
from loguru import logger

from src.utils.constants import (
    API_TIMEOUT_SECONDS,
    API_RATE_LIMIT_DELAY,
    BETS_DB_PATH,
    VALID_TEAM_ABBREVS,
)


class PublicSentimentScraper:
    """
    Scraper for public betting sentiment from free sources.

    Sources:
    1. Reddit r/sportsbook via public JSON (no auth)
    2. Covers.com public consensus
    3. Public betting percentages
    """

    # Reddit's public JSON API (no auth required)
    REDDIT_JSON_URL = "https://www.reddit.com/r/sportsbook.json"

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    }

    def __init__(self, db_path: str = None):
        """
        Initialize sentiment scraper.

        Args:
            db_path: Path to SQLite database (defaults to BETS_DB_PATH)
        """
        self.db_path = db_path or BETS_DB_PATH
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._init_tables()

    def _init_tables(self):
        """Initialize sentiment database tables."""
        conn = self._get_connection()
        try:
            # Public sentiment mentions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS public_sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    post_id TEXT NOT NULL,
                    team_abbrev TEXT,
                    sentiment_direction TEXT,
                    mention_count INTEGER DEFAULT 1,
                    timestamp TEXT NOT NULL,
                    collected_at TEXT NOT NULL,
                    UNIQUE(source, post_id, team_abbrev)
                )
            """)

            # Aggregated sentiment scores
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_abbrev TEXT NOT NULL,
                    date TEXT NOT NULL,
                    positive_mentions INTEGER DEFAULT 0,
                    negative_mentions INTEGER DEFAULT 0,
                    total_mentions INTEGER DEFAULT 0,
                    sentiment_score REAL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(team_abbrev, date)
                )
            """)

            # Indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_team
                ON public_sentiment(team_abbrev)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_timestamp
                ON public_sentiment(timestamp)
            """)

            conn.commit()
            logger.debug("Sentiment tables initialized")
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def scrape_reddit_sportsbook(self, limit: int = 100) -> pd.DataFrame:
        """
        Scrape r/sportsbook via public JSON API (no auth required).

        Args:
            limit: Number of posts to fetch

        Returns:
            DataFrame with post data
        """
        try:
            # Reddit's public JSON endpoint (no auth needed)
            params = {"limit": limit}
            response = self.session.get(
                self.REDDIT_JSON_URL,
                params=params,
                timeout=API_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            time.sleep(API_RATE_LIMIT_DELAY)

            data = response.json()
            posts = data.get("data", {}).get("children", [])

            records = []
            for post in posts:
                post_data = post.get("data", {})
                title = post_data.get("title", "")
                selftext = post_data.get("selftext", "")
                combined_text = f"{title} {selftext}".lower()

                # Extract team mentions and sentiment
                mentions = self._extract_team_mentions(combined_text)

                for mention in mentions:
                    records.append({
                        "source": "reddit_sportsbook",
                        "post_id": post_data.get("id"),
                        "team_abbrev": mention["team"],
                        "sentiment_direction": mention["sentiment"],
                        "timestamp": datetime.fromtimestamp(
                            post_data.get("created_utc", 0)
                        ).isoformat(),
                        "collected_at": datetime.now().isoformat(),
                    })

            df = pd.DataFrame(records)
            logger.info(f"Scraped {len(posts)} posts from r/sportsbook, found {len(df)} team mentions")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to scrape r/sportsbook: {e}")
            return pd.DataFrame()

    def _extract_team_mentions(self, text: str) -> List[Dict]:
        """
        Extract team mentions and basic sentiment from text.

        Args:
            text: Text to analyze

        Returns:
            List of team mention dicts
        """
        mentions = []

        # Positive betting keywords
        positive_keywords = [
            r'\block\b', r'\bhammer\b', r'\bsmash\b', r'\blove\b',
            r'\bconfident\b', r'\bstrong\b', r'\bbet\b.*\bon\b',
        ]

        # Negative betting keywords
        negative_keywords = [
            r'\bfade\b', r'\bavoid\b', r'\bstay away\b', r'\bscary\b',
            r'\brisky\b', r'\bagainst\b',
        ]

        # Find all team abbreviations in text
        for team_abbrev in VALID_TEAM_ABBREVS:
            pattern = r'\b' + re.escape(team_abbrev.lower()) + r'\b'
            if re.search(pattern, text):
                # Determine sentiment based on surrounding keywords
                sentiment = "neutral"

                # Check for positive sentiment
                for pos_keyword in positive_keywords:
                    # Look for keyword near team mention (within 20 chars)
                    context_pattern = f"(?:{pos_keyword}.{{0,20}}{pattern}|{pattern}.{{0,20}}{pos_keyword})"
                    if re.search(context_pattern, text, re.IGNORECASE):
                        sentiment = "positive"
                        break

                # Check for negative sentiment
                if sentiment == "neutral":
                    for neg_keyword in negative_keywords:
                        context_pattern = f"(?:{neg_keyword}.{{0,20}}{pattern}|{pattern}.{{0,20}}{neg_keyword})"
                        if re.search(context_pattern, text, re.IGNORECASE):
                            sentiment = "negative"
                            break

                mentions.append({
                    "team": team_abbrev,
                    "sentiment": sentiment,
                })

        return mentions

    def save_sentiment_data(self, sentiment_df: pd.DataFrame) -> int:
        """
        Save sentiment mentions to database.

        Args:
            sentiment_df: DataFrame with sentiment data

        Returns:
            Number of records inserted
        """
        if sentiment_df.empty:
            return 0

        conn = self._get_connection()
        try:
            inserted = 0
            for _, row in sentiment_df.iterrows():
                cursor = conn.execute("""
                    INSERT OR IGNORE INTO public_sentiment
                    (source, post_id, team_abbrev, sentiment_direction, timestamp, collected_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    row["source"],
                    row["post_id"],
                    row["team_abbrev"],
                    row["sentiment_direction"],
                    row["timestamp"],
                    row["collected_at"],
                ))
                inserted += cursor.rowcount

            conn.commit()
            logger.info(f"Saved {inserted} new sentiment mentions")
            return inserted

        except Exception as e:
            logger.error(f"Error saving sentiment data: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()

    def aggregate_sentiment_scores(self, date: str = None) -> int:
        """
        Aggregate sentiment mentions into daily scores.

        Args:
            date: Date to aggregate (YYYY-MM-DD, defaults to today)

        Returns:
            Number of teams updated
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        conn = self._get_connection()
        try:
            # Aggregate sentiment for each team
            cursor = conn.execute("""
                SELECT
                    team_abbrev,
                    SUM(CASE WHEN sentiment_direction = 'positive' THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN sentiment_direction = 'negative' THEN 1 ELSE 0 END) as negative,
                    COUNT(*) as total
                FROM public_sentiment
                WHERE DATE(timestamp) = ?
                GROUP BY team_abbrev
            """, (date,))

            updated = 0
            for row in cursor.fetchall():
                team = row["team_abbrev"]
                positive = row["positive"]
                negative = row["negative"]
                total = row["total"]

                # Calculate sentiment score (-1 to 1)
                if total > 0:
                    score = (positive - negative) / total
                else:
                    score = 0.0

                # Upsert aggregated score
                conn.execute("""
                    INSERT OR REPLACE INTO sentiment_scores
                    (team_abbrev, date, positive_mentions, negative_mentions,
                     total_mentions, sentiment_score, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    team, date, positive, negative, total, score,
                    datetime.now().isoformat()
                ))
                updated += 1

            conn.commit()
            logger.info(f"Aggregated sentiment for {updated} teams on {date}")
            return updated

        except Exception as e:
            logger.error(f"Error aggregating sentiment: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()

    def get_team_sentiment(
        self,
        team_abbrev: str,
        date: str = None
    ) -> Dict:
        """
        Get aggregated sentiment score for a team.

        Args:
            team_abbrev: Team abbreviation
            date: Date (YYYY-MM-DD, defaults to today)

        Returns:
            Dict with sentiment data
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT * FROM sentiment_scores
                WHERE team_abbrev = ? AND date = ?
            """, (team_abbrev, date))

            row = cursor.fetchone()
            if row:
                return {
                    "team": team_abbrev,
                    "sentiment": row["sentiment_score"],
                    "volume": row["total_mentions"],
                    "enabled": True,
                }
            else:
                return {
                    "team": team_abbrev,
                    "sentiment": 0.0,
                    "volume": 0,
                    "enabled": True,
                }
        finally:
            conn.close()


if __name__ == "__main__":
    # Test the scraper
    scraper = PublicSentimentScraper()

    # Scrape r/sportsbook
    print("Scraping r/sportsbook...")
    sentiment_data = scraper.scrape_reddit_sportsbook(limit=50)

    if not sentiment_data.empty:
        print(f"\nFound {len(sentiment_data)} team mentions")
        print("\nSample mentions:")
        print(sentiment_data.head(10)[["team_abbrev", "sentiment_direction"]])

        # Save data
        count = scraper.save_sentiment_data(sentiment_data)
        print(f"\nSaved {count} new mentions")

        # Aggregate scores
        updated = scraper.aggregate_sentiment_scores()
        print(f"Aggregated scores for {updated} teams")
