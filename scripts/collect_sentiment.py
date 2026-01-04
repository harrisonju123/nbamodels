#!/usr/bin/env python3
"""
Collect Public Sentiment - Cron Script

Scrapes public betting sentiment from free sources (no API keys required):
- Reddit r/sportsbook via HTML scraping
- No authentication needed

Cron schedule:
*/15 17-23 * * * cd /Users/harrisonju/PycharmProjects/nbamodels && python scripts/collect_sentiment.py >> logs/sentiment.log 2>&1
"""

import sys
import os
import fcntl
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from requests.exceptions import RequestException, Timeout, HTTPError

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from loguru import logger

from src.data.public_sentiment_scraper import PublicSentimentScraper

# Load environment variables
load_dotenv()

# Lock file to prevent concurrent execution
LOCK_FILE = "/tmp/nba_sentiment_collector.lock"
LOCK_TIMEOUT = 300  # 5 minutes max runtime (fixes Issue #9)


def acquire_lock():
    """
    Acquire lock with stale lock detection (fixes Issue #9).

    Returns:
        File descriptor if lock acquired, None otherwise
    """
    if os.path.exists(LOCK_FILE):
        # Check if lock is stale
        lock_age = time.time() - os.path.getmtime(LOCK_FILE)
        if lock_age > LOCK_TIMEOUT:
            logger.warning(f"Removing stale lock (age: {lock_age:.0f}s)")
            try:
                os.remove(LOCK_FILE)
            except OSError as e:
                logger.error(f"Failed to remove stale lock: {e}")
                return None
        else:
            # Lock is active
            return None

    # Try to acquire lock
    lock_fd = open(LOCK_FILE, 'w')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd.write(str(os.getpid()))
        lock_fd.flush()
        return lock_fd
    except IOError:
        lock_fd.close()
        return None


def release_lock(lock_fd):
    """
    Release lock and remove lock file.

    Args:
        lock_fd: File descriptor to release
    """
    if lock_fd:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)
        except Exception as e:
            logger.error(f"Error releasing lock: {e}")


def main():
    """Main collection function with granular error handling (fixes Issue #3)."""
    logger.info("=" * 60)
    logger.info("Starting public sentiment collection")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # File locking to prevent concurrent cron runs
    lock_fd = acquire_lock()
    if not lock_fd:
        logger.warning("Another sentiment collection process is running - skipping")
        return 0

    try:
        scraper = PublicSentimentScraper()

        # Scrape r/sportsbook with specific error handling
        logger.info("Scraping r/sportsbook via HTML...")

        try:
            sentiment_data = scraper.scrape_reddit_sportsbook(limit=100)

        except HTTPError as e:
            if e.response.status_code == 429:
                logger.error(f"Rate limited by Reddit API - back off: {e}")
                logger.info("Try increasing RATE_LIMIT_DELAY or reducing scrape frequency")
                return 1
            elif e.response.status_code == 403:
                logger.error(f"Reddit API authentication failed or IP banned: {e}")
                logger.info("Possible bot detection - consider rotating user agent or IP")
                return 1
            else:
                logger.error(f"HTTP error from Reddit (status {e.response.status_code}): {e}")
                return 1

        except Timeout:
            logger.error("Request to Reddit timed out - network issue or Reddit downtime")
            logger.info("This is usually temporary - next cron run should succeed")
            return 1

        except RequestException as e:
            logger.error(f"Network error during scraping: {e}")
            logger.info("Check network connectivity or Reddit availability")
            return 1

        # Process scraped data
        if not sentiment_data.empty:
            logger.info(f"Found {len(sentiment_data)} team mentions")

            # Save to database with error handling
            try:
                inserted = scraper.save_sentiment_data(sentiment_data)
                logger.info(f"Saved {inserted} new sentiment mentions")

            except sqlite3.Error as e:
                logger.error(f"Database error saving sentiment data: {e}")
                logger.info("Check database permissions and disk space")
                return 1

            # Aggregate scores with error handling
            try:
                updated = scraper.aggregate_sentiment_scores()
                logger.info(f"Aggregated sentiment for {updated} teams")

            except sqlite3.Error as e:
                logger.error(f"Database error aggregating scores: {e}")
                logger.info("Check database permissions and disk space")
                return 1

        else:
            logger.warning("No sentiment data found - possibly no NBA-related posts")
            logger.info("This is normal during off-season or late night")

        logger.info("=" * 60)
        logger.info("Sentiment collection completed successfully")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Unexpected error during sentiment collection: {e}", exc_info=True)
        logger.info("This is a bug - please report with traceback")
        return 1

    finally:
        release_lock(lock_fd)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
