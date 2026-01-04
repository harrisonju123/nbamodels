#!/usr/bin/env python3
"""
Collect Public Sentiment - Cron Script

Scrapes public betting sentiment from free sources (no API keys required):
- Reddit r/sportsbook via public JSON API
- No authentication needed

Cron schedule:
*/15 17-23 * * * cd /Users/harrisonju/PycharmProjects/nbamodels && python scripts/collect_sentiment.py >> logs/sentiment.log 2>&1
"""

import sys
import os
import fcntl
from datetime import datetime
from pathlib import Path

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


def main():
    """Main collection function."""
    logger.info("=" * 60)
    logger.info("Starting public sentiment collection")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # File locking to prevent concurrent cron runs
    lock_fd = open(LOCK_FILE, 'w')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        logger.warning("Another sentiment collection process is running - skipping")
        return 0

    try:
        scraper = PublicSentimentScraper()

        # Scrape r/sportsbook public JSON
        logger.info("Scraping r/sportsbook...")
        sentiment_data = scraper.scrape_reddit_sportsbook(limit=100)

        if not sentiment_data.empty:
            logger.info(f"Found {len(sentiment_data)} team mentions")

            # Save to database
            inserted = scraper.save_sentiment_data(sentiment_data)
            logger.info(f"Saved {inserted} new sentiment mentions")

            # Aggregate daily scores
            updated = scraper.aggregate_sentiment_scores()
            logger.info(f"Aggregated sentiment for {updated} teams")
        else:
            logger.warning("No sentiment data found")

        logger.info("=" * 60)
        logger.info("Sentiment collection completed successfully")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Sentiment collection failed: {e}")
        return 1

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
