"""
Discord Notifier

Sends automated performance updates to Discord via webhook.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Optional
import requests
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class DiscordNotifier:
    """Send automated Discord notifications for betting performance."""

    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize Discord notifier.

        Args:
            webhook_url: Discord webhook URL (or set DISCORD_WEBHOOK_URL env var)
        """
        self.webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL')
        if not self.webhook_url:
            logger.warning("No Discord webhook URL configured. Set DISCORD_WEBHOOK_URL environment variable.")

    def send_message(self, content: str = None, embed: Dict = None) -> bool:
        """
        Send a message to Discord.

        Args:
            content: Plain text message
            embed: Rich embed object

        Returns:
            True if successful, False otherwise
        """
        if not self.webhook_url:
            logger.error("Cannot send Discord message: No webhook URL configured")
            return False

        payload = {}
        if content:
            payload['content'] = content
        if embed:
            payload['embeds'] = [embed]

        try:
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            logger.info("Discord message sent successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send Discord message: {e}")
            return False

    def get_daily_stats(self, db_path: str = "data/bets/bets.db") -> Dict:
        """Get daily performance statistics."""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get today's date
            today = datetime.now().date()

            # Bets placed today
            cursor.execute("""
                SELECT COUNT(*), SUM(bet_amount)
                FROM bets
                WHERE DATE(logged_at) = ?
            """, (today,))
            bets_today, volume_today = cursor.fetchone()
            bets_today = bets_today or 0
            volume_today = volume_today or 0

            # Bets settled today
            cursor.execute("""
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END),
                    SUM(COALESCE(profit, 0))
                FROM bets
                WHERE DATE(settled_at) = ? AND outcome IS NOT NULL
            """, (today,))
            settled_today, wins_today, profit_today = cursor.fetchone()
            settled_today = settled_today or 0
            wins_today = wins_today or 0
            profit_today = profit_today or 0

            # Current bankroll - calculate from all settled bets
            # Starting bankroll + total profit
            starting_bankroll = 1000.0  # Default starting bankroll
            cursor.execute("""
                SELECT SUM(COALESCE(profit, 0))
                FROM bets
                WHERE outcome IS NOT NULL
            """)
            row = cursor.fetchone()
            total_profit_all_time = row[0] if row and row[0] else 0
            current_bankroll = starting_bankroll + total_profit_all_time

            # Overall stats (last 30 days)
            thirty_days_ago = today - timedelta(days=30)
            cursor.execute("""
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END),
                    SUM(COALESCE(profit, 0)),
                    SUM(bet_amount)
                FROM bets
                WHERE outcome IS NOT NULL
                  AND DATE(settled_at) >= ?
            """, (thirty_days_ago,))
            total_bets, total_wins, total_profit, total_wagered = cursor.fetchone()
            total_bets = total_bets or 0
            total_wins = total_wins or 0
            total_profit = total_profit or 0
            total_wagered = total_wagered or 1  # Avoid division by zero

            # Strategy breakdown (last 30 days)
            strategy_breakdown = []
            cursor.execute("""
                SELECT
                    COALESCE(strategy_type, 'spread') as strategy,
                    COUNT(*) as bets,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                    SUM(COALESCE(profit, 0)) as profit,
                    SUM(bet_amount) as wagered
                FROM bets
                WHERE outcome IS NOT NULL
                  AND DATE(settled_at) >= ?
                GROUP BY strategy_type
                ORDER BY profit DESC
            """, (thirty_days_ago,))

            for row in cursor.fetchall():
                strategy, bets, wins, profit, wagered = row
                win_rate = (wins / bets * 100) if bets > 0 else 0
                roi = (profit / wagered * 100) if wagered > 0 else 0
                strategy_breakdown.append({
                    'strategy': strategy,
                    'bets': bets,
                    'wins': wins,
                    'win_rate': win_rate,
                    'profit': profit,
                    'roi': roi
                })

            conn.close()

            return {
                'bets_placed_today': bets_today,
                'volume_today': volume_today,
                'bets_settled_today': settled_today,
                'wins_today': wins_today,
                'profit_today': profit_today,
                'win_rate_today': (wins_today / settled_today * 100) if settled_today > 0 else 0,
                'current_bankroll': current_bankroll,
                'total_bets_30d': total_bets,
                'total_wins_30d': total_wins,
                'win_rate_30d': (total_wins / total_bets * 100) if total_bets > 0 else 0,
                'total_profit_30d': total_profit,
                'roi_30d': (total_profit / total_wagered * 100) if total_wagered > 0 else 0,
                'strategy_breakdown': strategy_breakdown,
            }
        except Exception as e:
            logger.error(f"Error getting daily stats: {e}")
            return {}

    def send_daily_update(self) -> bool:
        """Send daily performance update to Discord."""
        stats = self.get_daily_stats()

        if not stats:
            return self.send_message(content="⚠️ Unable to generate daily update - database error")

        # Determine color based on today's profit
        profit_today = stats['profit_today']
        if profit_today > 0:
            color = 0x00FF00  # Green
            emoji = "📈"
        elif profit_today < 0:
            color = 0xFF0000  # Red
            emoji = "📉"
        else:
            color = 0xFFFF00  # Yellow
            emoji = "➖"

        # Build embed fields
        fields = [
            {
                "name": "📊 Today's Activity",
                "value": (
                    f"**Bets Placed**: {stats['bets_placed_today']}\n"
                    f"**Volume**: ${stats['volume_today']:,.2f}\n"
                    f"**Bets Settled**: {stats['bets_settled_today']}\n"
                    f"**Wins**: {stats['wins_today']} ({stats['win_rate_today']:.1f}%)"
                ),
                "inline": True
            },
            {
                "name": "💰 Today's Results",
                "value": (
                    f"**Profit/Loss**: ${profit_today:+,.2f}\n"
                    f"**Current Bankroll**: ${stats['current_bankroll']:,.2f}"
                ),
                "inline": True
            },
            {
                "name": "📈 Last 30 Days",
                "value": (
                    f"**Total Bets**: {stats['total_bets_30d']}\n"
                    f"**Win Rate**: {stats['win_rate_30d']:.1f}%\n"
                    f"**Total Profit**: ${stats['total_profit_30d']:+,.2f}\n"
                    f"**ROI**: {stats['roi_30d']:+.1f}%"
                ),
                "inline": False
            }
        ]

        # Add strategy breakdown if available
        if stats.get('strategy_breakdown') and len(stats['strategy_breakdown']) > 0:
            breakdown_text = ""
            for strategy in stats['strategy_breakdown']:
                profit_symbol = "+" if strategy['profit'] >= 0 else ""
                breakdown_text += (
                    f"**{strategy['strategy'].title()}**: "
                    f"{strategy['bets']} bets | "
                    f"{strategy['win_rate']:.1f}% WR | "
                    f"{profit_symbol}${strategy['profit']:.2f} ({strategy['roi']:+.1f}% ROI)\n"
                )

            fields.append({
                "name": "🎯 Performance by Strategy",
                "value": breakdown_text.strip(),
                "inline": False
            })

        # Build embed
        embed = {
            "title": f"{emoji} Daily Betting Performance",
            "description": f"Report for {datetime.now().strftime('%B %d, %Y')}",
            "color": color,
            "fields": fields,
            "footer": {
                "text": "🤖 Generated by NBA Betting System"
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        return self.send_message(embed=embed)

    def send_bet_placed(self, bet_details: Dict) -> bool:
        """Send notification when a bet is placed."""
        embed = {
            "title": "🎲 New Bet Placed",
            "color": 0x3498db,  # Blue
            "fields": [
                {
                    "name": "Game",
                    "value": f"{bet_details.get('home_team')} vs {bet_details.get('away_team')}",
                    "inline": False
                },
                {
                    "name": "Bet",
                    "value": f"{bet_details.get('bet_side')} {bet_details.get('line'):+.1f}",
                    "inline": True
                },
                {
                    "name": "Odds",
                    "value": f"{bet_details.get('odds'):.2f}",
                    "inline": True
                },
                {
                    "name": "Amount",
                    "value": f"${bet_details.get('bet_amount', 0):,.2f}",
                    "inline": True
                },
                {
                    "name": "Edge",
                    "value": f"{bet_details.get('edge', 0):.1%}",
                    "inline": True
                }
            ],
            "timestamp": datetime.utcnow().isoformat()
        }

        return self.send_message(embed=embed)

    def send_bet_result(self, bet_id: int, outcome: str, profit: float) -> bool:
        """Send notification when a bet is settled."""
        emoji = "✅" if outcome == "win" else "❌" if outcome == "loss" else "➖"
        color = 0x00FF00 if outcome == "win" else 0xFF0000 if outcome == "loss" else 0x95A5A6

        embed = {
            "title": f"{emoji} Bet Settled",
            "color": color,
            "fields": [
                {
                    "name": "Outcome",
                    "value": outcome.upper(),
                    "inline": True
                },
                {
                    "name": "Profit/Loss",
                    "value": f"${profit:+,.2f}",
                    "inline": True
                }
            ],
            "timestamp": datetime.utcnow().isoformat()
        }

        return self.send_message(embed=embed)

    def send_alert(self, message: str, level: str = "info") -> bool:
        """
        Send alert message to Discord.

        Args:
            message: Alert message
            level: Alert level (info, warning, error)
        """
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "🚨"
        }

        color_map = {
            "info": 0x3498db,
            "warning": 0xf39c12,
            "error": 0xe74c3c
        }

        embed = {
            "title": f"{emoji_map.get(level, 'ℹ️')} System Alert",
            "description": message,
            "color": color_map.get(level, 0x3498db),
            "timestamp": datetime.utcnow().isoformat()
        }

        return self.send_message(embed=embed)
