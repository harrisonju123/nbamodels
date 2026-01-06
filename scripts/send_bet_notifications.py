#!/usr/bin/env python3
"""
Send Bet Notifications to Discord

Sends immediate notification after bets are placed by the pipeline.
Different from daily_report.py which sends end-of-day summaries.

Usage:
    python scripts/send_bet_notifications.py

This should be called right after the betting pipeline completes.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.discord_notifier import DiscordNotifier
from src.bet_tracker import get_bet_history


def send_bet_notifications() -> bool:
    """
    Send notifications for bets placed today.

    Returns:
        True if successful, False otherwise
    """
    logger.info("Checking for new bets to notify...")

    # Check for webhook URL
    webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    if not webhook_url:
        logger.warning("DISCORD_WEBHOOK_URL not set - skipping notification")
        return False

    try:
        # Get all bet history
        bet_history_df = get_bet_history()

        if bet_history_df.empty:
            logger.info("No bets in history - skipping notification")
            return True

        # Filter to today only
        today = datetime.now().date()
        bet_history_df['placed_date'] = pd.to_datetime(bet_history_df['placed_at']).dt.date
        today_bets_df = bet_history_df[bet_history_df['placed_date'] == today]

        if today_bets_df.empty:
            logger.info("No new bets today - skipping notification")
            return True

        logger.info(f"Found {len(today_bets_df)} bets placed today")

        # Initialize notifier
        notifier = DiscordNotifier(webhook_url=webhook_url)

        # Create notification message
        message = f"🎯 **{len(today_bets_df)} Bets Placed Today** ({datetime.now().strftime('%Y-%m-%d')})\n\n"

        for i, (_, bet) in enumerate(today_bets_df.iterrows(), 1):
            strategy = bet.get('strategy_type', 'unknown')
            bet_type = bet.get('bet_type', 'spread')
            side = bet.get('bet_side', 'unknown')
            amount = bet.get('bet_amount', 0)
            odds = bet.get('odds', -110)
            edge = bet.get('edge', 0)

            game = f"{bet.get('away_team', '???')} @ {bet.get('home_team', '???')}"

            message += f"**{i}.** {strategy.upper()} - {game}\n"
            message += f"   • {bet_type.title()} {side} ${amount:.2f} @ {odds:+d}\n"
            message += f"   • Edge: {edge:.2%}\n\n"

        message += "\n📊 Full report coming at 11 PM"

        # Send to Discord
        success = notifier.send_message(message)

        if success:
            logger.success("Bet notifications sent successfully!")
            return True
        else:
            logger.error("Failed to send bet notifications")
            return False

    except Exception as e:
        logger.error(f"Error sending bet notifications: {e}")
        logger.exception("Detailed error:")
        return False


def main():
    """Main entry point."""
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    try:
        success = send_bet_notifications()
        return 0 if success else 1

    except KeyboardInterrupt:
        logger.warning("\nScript interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.exception("Detailed error:")
        return 1


if __name__ == '__main__':
    sys.exit(main())
