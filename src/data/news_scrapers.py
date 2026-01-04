"""
NBA News Scraper

Fetches NBA news from RSS feeds (ESPN, NBA.com, Yahoo Sports).
Extracts team and player mentions for news volume tracking.
"""

import os
import sqlite3
import time
import hashlib
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from xml.etree import ElementTree

import pandas as pd
import requests
from loguru import logger

from src.utils.constants import (
    API_TIMEOUT_SECONDS,
    API_RATE_LIMIT_DELAY,
    BETS_DB_PATH,
    TEAM_NAME_TO_ABBREV,
    VALID_TEAM_ABBREVS,
)


class NBANewsClient:
    """Client for scraping NBA news from RSS feeds."""

    # RSS feed URLs (no API key required)
    RSS_FEEDS = {
        "espn": "https://www.espn.com/espn/rss/nba/news",
        "nba": "https://www.nba.com/news/rss.xml",
        "yahoo": "https://sports.yahoo.com/nba/rss.xml",
    }

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    }

    def __init__(self, db_path: str = None):
        """
        Initialize news scraper.

        Args:
            db_path: Path to SQLite database (defaults to BETS_DB_PATH)
        """
        self.db_path = db_path or BETS_DB_PATH
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._init_tables()

    def _init_tables(self):
        """Initialize news database tables."""
        conn = self._get_connection()
        try:
            # News articles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    article_id TEXT NOT NULL,
                    url TEXT,
                    title TEXT NOT NULL,
                    summary TEXT,
                    published_at TEXT,
                    collected_at TEXT NOT NULL,
                    UNIQUE(source, article_id)
                )
            """)

            # News entities (team/player mentions)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id INTEGER NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_value TEXT NOT NULL,
                    team_abbrev TEXT,
                    FOREIGN KEY (article_id) REFERENCES news_articles(id)
                )
            """)

            # Indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_news_published
                ON news_articles(published_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_news_entities_team
                ON news_entities(team_abbrev)
            """)

            conn.commit()
            logger.debug("News tables initialized")
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _make_request(self, url: str) -> str:
        """
        Fetch RSS feed content.

        Args:
            url: RSS feed URL

        Returns:
            RSS XML content (empty string on error)
        """
        try:
            response = self.session.get(url, timeout=API_TIMEOUT_SECONDS)
            response.raise_for_status()
            time.sleep(API_RATE_LIMIT_DELAY)
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"RSS feed request failed for {url}: {e}")
            return ""

    def fetch_feed(self, source: str) -> pd.DataFrame:
        """
        Fetch and parse a single RSS feed.

        Args:
            source: Feed source name ('espn', 'nba', 'yahoo')

        Returns:
            DataFrame with article records
        """
        if source not in self.RSS_FEEDS:
            logger.error(f"Unknown feed source: {source}")
            return pd.DataFrame()

        url = self.RSS_FEEDS[source]
        logger.info(f"Fetching {source} RSS feed from {url}")

        xml_content = self._make_request(url)
        if not xml_content:
            return pd.DataFrame()

        # Parse RSS XML
        try:
            root = ElementTree.fromstring(xml_content)
        except ElementTree.ParseError as e:
            logger.error(f"Failed to parse RSS XML from {source}: {e}")
            return pd.DataFrame()

        # Extract items
        records = []
        items = root.findall(".//item")

        for item in items:
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            description = item.findtext("description", "")
            pub_date = item.findtext("pubDate", "")

            # Create unique article ID from URL
            article_id = hashlib.md5(link.encode()).hexdigest()

            records.append({
                "source": source,
                "article_id": article_id,
                "url": link,
                "title": title,
                "summary": description,
                "published_at": pub_date,
                "collected_at": datetime.now().isoformat(),
            })

        df = pd.DataFrame(records)
        logger.info(f"Fetched {len(df)} articles from {source}")
        return df

    def fetch_all_feeds(self) -> pd.DataFrame:
        """
        Fetch all RSS feeds.

        Returns:
            Combined DataFrame with all articles
        """
        all_articles = []

        for source in self.RSS_FEEDS.keys():
            articles = self.fetch_feed(source)
            if not articles.empty:
                all_articles.append(articles)

        if not all_articles:
            return pd.DataFrame()

        combined = pd.concat(all_articles, ignore_index=True)
        logger.info(f"Total articles fetched: {len(combined)}")
        return combined

    def extract_entities(self, article: Dict) -> List[Dict]:
        """
        Extract team and player mentions from article text.

        Args:
            article: Article dictionary with 'title' and 'summary'

        Returns:
            List of entity dictionaries
        """
        text = f"{article.get('title', '')} {article.get('summary', '')}".lower()
        entities = []

        # Extract team mentions
        for team_name, abbrev in TEAM_NAME_TO_ABBREV.items():
            if team_name.lower() in text:
                entities.append({
                    "entity_type": "team",
                    "entity_value": team_name,
                    "team_abbrev": abbrev,
                })

        # Also check for team abbreviations
        for abbrev in VALID_TEAM_ABBREVS:
            # Look for abbreviation as whole word
            pattern = r'\b' + re.escape(abbrev.lower()) + r'\b'
            if re.search(pattern, text):
                entities.append({
                    "entity_type": "team",
                    "entity_value": abbrev,
                    "team_abbrev": abbrev,
                })

        # Remove duplicates
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity["entity_type"], entity["team_abbrev"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities

    def save_articles(self, articles_df: pd.DataFrame) -> int:
        """
        Save articles and extracted entities to database.

        Args:
            articles_df: DataFrame with articles

        Returns:
            Number of articles inserted
        """
        if articles_df.empty:
            return 0

        conn = self._get_connection()
        try:
            inserted_count = 0

            for _, article in articles_df.iterrows():
                # Insert article
                cursor = conn.execute("""
                    INSERT OR IGNORE INTO news_articles
                    (source, article_id, url, title, summary, published_at, collected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    article["source"],
                    article["article_id"],
                    article.get("url"),
                    article["title"],
                    article.get("summary"),
                    article.get("published_at"),
                    article["collected_at"],
                ))

                if cursor.rowcount > 0:
                    inserted_count += 1
                    article_pk = cursor.lastrowid

                    # Extract and save entities
                    entities = self.extract_entities(article)
                    for entity in entities:
                        conn.execute("""
                            INSERT INTO news_entities
                            (article_id, entity_type, entity_value, team_abbrev)
                            VALUES (?, ?, ?, ?)
                        """, (
                            article_pk,
                            entity["entity_type"],
                            entity["entity_value"],
                            entity.get("team_abbrev"),
                        ))

            conn.commit()
            logger.info(f"Saved {inserted_count} new articles")
            return inserted_count

        except Exception as e:
            logger.error(f"Error saving articles: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()

    def get_team_news_volume(
        self,
        team_abbrev: str,
        hours: int = 24
    ) -> int:
        """
        Get number of articles mentioning a team in the last N hours.

        Args:
            team_abbrev: Team abbreviation
            hours: Lookback window in hours

        Returns:
            Number of articles
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff_time.isoformat()

        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT COUNT(DISTINCT a.id)
                FROM news_articles a
                JOIN news_entities e ON a.id = e.article_id
                WHERE e.team_abbrev = ?
                AND a.collected_at >= ?
            """, (team_abbrev, cutoff_str))

            count = cursor.fetchone()[0]
            return count
        finally:
            conn.close()

    def get_team_news_recency(self, team_abbrev: str) -> Optional[float]:
        """
        Get hours since last article mentioning a team.

        Args:
            team_abbrev: Team abbreviation

        Returns:
            Hours since last article (None if no articles)
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT MAX(a.collected_at)
                FROM news_articles a
                JOIN news_entities e ON a.id = e.article_id
                WHERE e.team_abbrev = ?
            """, (team_abbrev,))

            latest = cursor.fetchone()[0]
            if not latest:
                return None

            latest_dt = datetime.fromisoformat(latest)
            hours_ago = (datetime.now() - latest_dt).total_seconds() / 3600
            return hours_ago

        finally:
            conn.close()


if __name__ == "__main__":
    # Test the news scraper
    client = NBANewsClient()

    # Fetch all feeds
    articles = client.fetch_all_feeds()
    print(f"\nFetched {len(articles)} articles")

    if not articles.empty:
        print("\nSample articles:")
        print(articles[["source", "title"]].head())

        # Save articles
        count = client.save_articles(articles)
        print(f"\nSaved {count} new articles to database")

        # Test entity extraction
        sample = articles.iloc[0] if len(articles) > 0 else {}
        entities = client.extract_entities(sample)
        print(f"\nExtracted {len(entities)} entities from first article")
        for entity in entities[:5]:
            print(f"  - {entity}")
