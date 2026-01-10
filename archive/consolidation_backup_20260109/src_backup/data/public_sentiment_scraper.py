"""
Public Sentiment Scraper - FREE Alternative

Scrapes public NBA betting discussions from free sources:
- r/sportsbook via HTML scraping (no API keys required)
- No authentication needed

Uses HTML scraping to avoid Reddit's API authentication requirements.
"""

import os
import sqlite3
import time
import re
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from bs4 import BeautifulSoup

import pandas as pd
import requests
from requests.exceptions import RequestException, Timeout, HTTPError
from loguru import logger

from src.utils.constants import (
    API_TIMEOUT_SECONDS,
    BETS_DB_PATH,
    VALID_TEAM_ABBREVS,
)


class PublicSentimentScraper:
    """
    Scraper for public betting sentiment from free sources.

    Uses HTML scraping of r/sportsbook (no API keys required).
    """

    # Reddit HTML URL (no auth required, unlike JSON API)
    REDDIT_HTML_URL = "https://old.reddit.com/r/sportsbook"

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    # Rate limit: 1 request per second (conservative for HTML scraping)
    RATE_LIMIT_DELAY = 1.0

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
        with self._get_connection_context() as conn:
            # Public sentiment mentions - add date column for better indexing
            conn.execute("""
                CREATE TABLE IF NOT EXISTS public_sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    post_id TEXT NOT NULL,
                    team_abbrev TEXT,
                    sentiment_direction TEXT,
                    mention_count INTEGER DEFAULT 1,
                    date TEXT NOT NULL,
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

            # Composite index for common query pattern (fixes Issue #11)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_date_team
                ON public_sentiment(date, team_abbrev, sentiment_direction)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_team
                ON public_sentiment(team_abbrev)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_timestamp
                ON public_sentiment(timestamp)
            """)

            logger.debug("Sentiment tables initialized")

    @contextmanager
    def _get_connection_context(self):
        """
        Context manager for database connections (fixes Issue #5).

        Ensures proper connection cleanup and automatic commit/rollback.
        """
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection (legacy method for backward compatibility)."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _make_request_with_backoff(self, url: str, params: dict = None, max_retries: int = 3) -> requests.Response:
        """
        Make HTTP request with exponential backoff for rate limits (fixes Issue #4).

        Args:
            url: URL to request
            params: Query parameters
            max_retries: Maximum retry attempts

        Returns:
            Response object

        Raises:
            HTTPError: If request fails after retries
        """
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=API_TIMEOUT_SECONDS)
                response.raise_for_status()

                # Respect rate limit: 1 request per second
                time.sleep(self.RATE_LIMIT_DELAY)

                return response

            except HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limited - exponential backoff
                    retry_after = int(e.response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {retry_after}s before retry {attempt+1}/{max_retries}")
                    time.sleep(retry_after)
                elif e.response.status_code == 403:
                    logger.error("Access forbidden - possible IP ban or bot detection")
                    raise
                else:
                    raise

        raise Exception(f"Max retries ({max_retries}) exceeded for {url}")

    def scrape_reddit_sportsbook(self, limit: int = 100) -> pd.DataFrame:
        """
        Scrape r/sportsbook via HTML (no auth required) - fixes Issue #1.

        Uses HTML scraping instead of JSON API to avoid authentication requirements.

        Args:
            limit: Number of posts to fetch (max 100)

        Returns:
            DataFrame with post data
        """
        try:
            # Fetch HTML from old.reddit.com (no auth needed)
            params = {"limit": min(limit, 100)}  # Reddit limits to 100 per page
            response = self._make_request_with_backoff(self.REDDIT_HTML_URL, params=params)

            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            posts = soup.find_all('div', class_='thing')

            records = []
            seen_teams_per_post = {}  # Track teams per post for deduplication (fixes Issue #13)

            for post in posts:
                try:
                    # Extract post data
                    post_id = post.get('data-fullname', '').replace('t3_', '')
                    if not post_id:
                        continue

                    # Get title and selftext
                    title_elem = post.find('a', class_='title')
                    title = title_elem.get_text() if title_elem else ""

                    # Get timestamp (fixes Issue #8 - validate timestamps)
                    time_elem = post.find('time')
                    if time_elem and time_elem.get('datetime'):
                        timestamp_str = time_elem['datetime']
                        try:
                            # Parse timezone-aware datetime and convert to naive local time
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            # Convert to naive datetime (remove timezone info)
                            timestamp = timestamp.replace(tzinfo=None)
                        except (ValueError, AttributeError):
                            timestamp = datetime.now()
                    else:
                        timestamp = datetime.now()

                    # Validate timestamp is reasonable (2020 to now+1day)
                    min_timestamp = datetime(2020, 1, 1)
                    max_timestamp = datetime.now() + timedelta(days=1)

                    if timestamp < min_timestamp or timestamp > max_timestamp:
                        logger.warning(f"Invalid timestamp {timestamp} for post {post_id}, using current time")
                        timestamp = datetime.now()

                    combined_text = title.lower()

                    # Extract team mentions and sentiment (improved patterns - fixes Issue #10)
                    mentions = self._extract_team_mentions(combined_text)

                    # Deduplicate teams within same post
                    if post_id not in seen_teams_per_post:
                        seen_teams_per_post[post_id] = set()

                    for mention in mentions:
                        team = mention["team"]
                        if team in seen_teams_per_post[post_id]:
                            continue  # Skip duplicate team in same post

                        seen_teams_per_post[post_id].add(team)

                        records.append({
                            "source": "reddit_sportsbook",
                            "post_id": post_id,
                            "team_abbrev": team,
                            "sentiment_direction": mention["sentiment"],
                            "date": timestamp.strftime("%Y-%m-%d"),
                            "timestamp": timestamp.isoformat(),
                            "collected_at": datetime.now().isoformat(),
                        })

                except Exception as e:
                    logger.debug(f"Error parsing post: {e}")
                    continue

            df = pd.DataFrame(records)
            logger.info(f"Scraped {len(posts)} posts from r/sportsbook, found {len(df)} unique team mentions")

            # Debug logging for sentiment extraction (fixes Issue #14)
            if not df.empty:
                logger.debug(f"Sentiment breakdown: "
                           f"{len(df[df['sentiment_direction']=='positive'])} positive, "
                           f"{len(df[df['sentiment_direction']=='negative'])} negative, "
                           f"{len(df[df['sentiment_direction']=='neutral'])} neutral")

            return df

        except HTTPError as e:
            if e.response.status_code == 429:
                logger.error(f"Rate limited by Reddit: {e}")
            elif e.response.status_code == 403:
                logger.error(f"Access forbidden by Reddit: {e}")
            else:
                logger.error(f"HTTP error scraping r/sportsbook: {e}")
            return pd.DataFrame()

        except Timeout:
            logger.error("Request to Reddit timed out")
            return pd.DataFrame()

        except RequestException as e:
            logger.error(f"Network error scraping r/sportsbook: {e}")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Unexpected error scraping r/sportsbook: {e}")
            return pd.DataFrame()

    def _extract_team_mentions(self, text: str) -> List[Dict]:
        """
        Extract team mentions and basic sentiment from text.

        Improved keyword patterns to reduce false positives (fixes Issue #10).

        Args:
            text: Text to analyze (already lowercased)

        Returns:
            List of team mention dicts
        """
        mentions = []

        logger.debug(f"Analyzing text ({len(text)} chars): {text[:100]}...")

        # Improved positive betting keywords (more specific patterns)
        positive_keywords = [
            r'\block\s+(in|it|them|the)\b',  # "lock in", "lock it", not "roadblock"
            r'\bhammer(ing)?\s+(the\s+)?\b',  # "hammer the Lakers"
            r'\bsmash(ing)?\s+(the\s+)?\b',   # "smashing the over"
            r'\blove\s+(the\s+)?\b',          # "love the Lakers"
            r'\bconfident\s+(in|on|about)\b', # "confident in Lakers"
            r'\bstrong\s+(on|play)\b',        # "strong on Lakers"
            r'\b(betting|bet)\s+on\b',        # "betting on Lakers"
            r'\btailing\b',                   # "tailing this pick"
            r'\bpounding\b',                  # "pounding Lakers ML"
        ]

        # Improved negative betting keywords
        negative_keywords = [
            r'\bfade\s+(the\s+)?\b',          # "fade the Lakers"
            r'\bavoid(ing)?\s+(the\s+)?\b',   # "avoiding the Lakers"
            r'\bstay\s+away\s+from\b',        # "stay away from"
            r'\bscary\s+\b',                  # "scary bet"
            r'\brisky\s+\b',                  # "risky play"
            r'\bbetting\s+against\b',         # "betting against"
            r'\bno\s+(way|bet)\b',           # "no way", "no bet"
        ]

        # Find all team abbreviations in text
        for team_abbrev in VALID_TEAM_ABBREVS:
            # Case-insensitive pattern (team already uppercase in VALID_TEAM_ABBREVS)
            pattern = r'\b' + re.escape(team_abbrev.lower()) + r'\b'

            if re.search(pattern, text, re.IGNORECASE):
                # Determine sentiment based on surrounding keywords
                sentiment = "neutral"

                # Check for positive sentiment (within 30 chars of team mention)
                for pos_keyword in positive_keywords:
                    context_pattern = f"(?:{pos_keyword}.{{0,30}}{pattern}|{pattern}.{{0,30}}{pos_keyword})"
                    if re.search(context_pattern, text, re.IGNORECASE):
                        sentiment = "positive"
                        break

                # Check for negative sentiment (only if not positive)
                if sentiment == "neutral":
                    for neg_keyword in negative_keywords:
                        context_pattern = f"(?:{neg_keyword}.{{0,30}}{pattern}|{pattern}.{{0,30}}{neg_keyword})"
                        if re.search(context_pattern, text, re.IGNORECASE):
                            sentiment = "negative"
                            break

                mentions.append({
                    "team": team_abbrev,  # Already uppercase
                    "sentiment": sentiment,
                })

        if mentions:
            logger.debug(f"Found {len(mentions)} team mentions: {[(m['team'], m['sentiment']) for m in mentions]}")

        return mentions

    def save_sentiment_data(self, sentiment_df: pd.DataFrame) -> int:
        """
        Save sentiment mentions to database.

        Uses context manager for proper connection handling (fixes Issue #5).

        Args:
            sentiment_df: DataFrame with sentiment data

        Returns:
            Number of records inserted
        """
        if sentiment_df.empty:
            return 0

        with self._get_connection_context() as conn:
            inserted = 0
            for _, row in sentiment_df.iterrows():
                cursor = conn.execute("""
                    INSERT OR IGNORE INTO public_sentiment
                    (source, post_id, team_abbrev, sentiment_direction, date, timestamp, collected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    row["source"],
                    row["post_id"],
                    row["team_abbrev"],
                    row["sentiment_direction"],
                    row["date"],
                    row["timestamp"],
                    row["collected_at"],
                ))
                inserted += cursor.rowcount

            logger.info(f"Saved {inserted} new sentiment mentions")
            return inserted

    def aggregate_sentiment_scores(self, date: str = None) -> int:
        """
        Aggregate sentiment mentions into daily scores.

        Fixes Issue #2 (SQL injection) with date validation.
        Fixes Issue #6 (race condition) with proper UPSERT.

        Args:
            date: Date to aggregate (YYYY-MM-DD, defaults to today)

        Returns:
            Number of teams updated
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # Validate date format to prevent SQL injection (fixes Issue #2)
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format: {date}. Expected YYYY-MM-DD")
            return 0

        with self._get_connection_context() as conn:
            # Use date column instead of DATE() function for better performance
            cursor = conn.execute("""
                SELECT
                    team_abbrev,
                    SUM(CASE WHEN sentiment_direction = 'positive' THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN sentiment_direction = 'negative' THEN 1 ELSE 0 END) as negative,
                    COUNT(*) as total
                FROM public_sentiment
                WHERE date = ?
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

                # Proper UPSERT to avoid race condition (fixes Issue #6)
                conn.execute("""
                    INSERT INTO sentiment_scores
                    (team_abbrev, date, positive_mentions, negative_mentions,
                     total_mentions, sentiment_score, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(team_abbrev, date) DO UPDATE SET
                        positive_mentions = excluded.positive_mentions,
                        negative_mentions = excluded.negative_mentions,
                        total_mentions = excluded.total_mentions,
                        sentiment_score = excluded.sentiment_score,
                        updated_at = excluded.updated_at
                """, (
                    team, date, positive, negative, total, score,
                    datetime.now().isoformat()
                ))
                updated += 1

            logger.info(f"Aggregated sentiment for {updated} teams on {date}")
            return updated

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

        # Validate date format
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format: {date}")
            return {
                "team": team_abbrev,
                "sentiment": 0.0,
                "volume": 0,
                "enabled": True,
            }

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
