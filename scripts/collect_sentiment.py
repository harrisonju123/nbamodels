#!/usr/bin/env python3
"""
Collect Social Sentiment - Cron Script (STUBBED)

Collects NBA sentiment from Reddit and Twitter.
Currently stubbed - exits gracefully if API credentials not configured.

Cron schedule (disabled until APIs configured):
# */15 17-23 * * * cd /Users/harrisonju/PycharmProjects/nbamodels && python scripts/collect_sentiment.py >> logs/sentiment.log 2>&1
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from loguru import logger

from src.data.reddit_client import RedditClient
from src.data.twitter_client import TwitterClient

# Load environment variables
load_dotenv()


def main():
    """Main collection function."""
    logger.info("=" * 60)
    logger.info("Starting social sentiment collection")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Check if APIs are configured
    reddit = RedditClient()
    twitter = TwitterClient()

    if not reddit.enabled and not twitter.enabled:
        logger.warning("Neither Reddit nor Twitter APIs are configured")
        logger.info("To enable sentiment collection:")
        logger.info("  1. Get Reddit API credentials: https://www.reddit.com/prefs/apps")
        logger.info("  2. Get Twitter Bearer Token: https://developer.twitter.com/")
        logger.info("  3. Add to .env file (see .env.example)")
        logger.info("Exiting gracefully")
        return 0

    logger.info(f"Reddit enabled: {reddit.enabled}")
    logger.info(f"Twitter enabled: {twitter.enabled}")

    # TODO: Implement actual sentiment collection when APIs configured
    # Would:
    # 1. Get today's games
    # 2. Collect sentiment for each team
    # 3. Store in database

    logger.info("=" * 60)
    logger.info("Sentiment collection completed (stubbed)")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
