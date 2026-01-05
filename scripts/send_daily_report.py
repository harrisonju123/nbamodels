#!/usr/bin/env python3
"""
Send Daily Discord Report

Automated script for sending daily performance updates to Discord.
Designed to be run via cron for automated daily reporting.

Usage:
    python scripts/send_daily_report.py

Cron setup (run at 11 PM daily):
    0 23 * * * cd /path/to/nbamodels && /path/to/python scripts/send_daily_report.py >> logs/daily_report.log 2>&1

Or use the orchestrator:
    0 23 * * * cd /path/to/nbamodels && /path/to/python nba.py report daily >> logs/daily_report.log 2>&1
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.discord_notifier import DiscordNotifier


def configure_logging():
    """Configure logging for daily report script."""
    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Add file handler
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    logger.add(
        log_dir / "daily_report_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="1 week",
        retention="1 month"
    )


def send_daily_report() -> bool:
    """
    Send daily performance report to Discord.

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("DAILY DISCORD REPORT")
    logger.info("=" * 80)
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check for webhook URL
    webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    if not webhook_url:
        logger.error("DISCORD_WEBHOOK_URL environment variable not set")
        logger.error("Please set it in your environment or .env file")
        return False

    try:
        # Initialize notifier
        notifier = DiscordNotifier(webhook_url=webhook_url)

        # Send daily update
        logger.info("Sending daily update to Discord...")
        success = notifier.send_daily_update()

        if success:
            logger.success("Daily update sent successfully!")
            return True
        else:
            logger.error("Failed to send daily update")
            return False

    except Exception as e:
        logger.error(f"Error sending daily report: {e}")
        logger.exception("Detailed error:")
        return False


def main():
    """Main entry point."""
    configure_logging()

    try:
        success = send_daily_report()

        if success:
            logger.info("=" * 80)
            logger.info("Daily report completed successfully")
            logger.info("=" * 80)
            return 0
        else:
            logger.error("=" * 80)
            logger.error("Daily report failed")
            logger.error("=" * 80)
            return 1

    except KeyboardInterrupt:
        logger.warning("\nScript interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.exception("Detailed error:")
        return 1


if __name__ == '__main__':
    sys.exit(main())
