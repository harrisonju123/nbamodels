"""
Live Game Monitor

Monitor live NBA games and detect betting edges in real-time.

Usage:
    python scripts/live_game_monitor.py [--dry-run] [--min-edge 0.05]

Options:
    --dry-run: Don't save to database, just print
    --min-edge: Minimum edge threshold (default 0.05 = 5%)
    --game-poll: Game polling interval in seconds (default 30)
    --odds-poll: Odds polling interval in seconds (default 180)

Run this during game hours (5-11 PM ET) to monitor live games.
"""

import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.data.live_game_client import LiveGameClient
from src.data.live_odds_tracker import LiveOddsTracker
from src.models.live_win_probability import LiveWinProbModel
from src.betting.live_edge_detector import LiveEdgeDetector

# Configuration
DEFAULT_GAME_POLL_INTERVAL = 30    # Check games every 30 seconds
DEFAULT_ODDS_POLL_INTERVAL = 180   # Check odds every 3 minutes
DEFAULT_MIN_EDGE = 0.05            # 5% minimum edge

# Set up logging
logger.add(
    "logs/live_monitor_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO"
)


class LiveGameMonitor:
    """Monitor live games and detect betting edges."""

    def __init__(
        self,
        min_edge: float = DEFAULT_MIN_EDGE,
        game_poll_interval: int = DEFAULT_GAME_POLL_INTERVAL,
        odds_poll_interval: int = DEFAULT_ODDS_POLL_INTERVAL,
        dry_run: bool = False
    ):
        """
        Initialize the monitor.

        Args:
            min_edge: Minimum edge to alert (default 5%)
            game_poll_interval: Seconds between game checks
            odds_poll_interval: Seconds between odds checks
            dry_run: If True, don't save to database
        """
        self.game_poll_interval = game_poll_interval
        self.odds_poll_interval = odds_poll_interval
        self.dry_run = dry_run

        # Initialize components
        self.game_client = LiveGameClient()
        self.odds_tracker = LiveOddsTracker()
        self.win_prob_model = LiveWinProbModel()
        self.edge_detector = LiveEdgeDetector(min_edge=min_edge)

        # Track last odds check per game
        self.last_odds_check = {}

        # Track alerts to avoid duplicates
        self.recent_alerts = set()

        logger.info(f"🚀 Live Game Monitor initialized")
        logger.info(f"   Min edge: {min_edge:.1%}")
        logger.info(f"   Game poll: {game_poll_interval}s")
        logger.info(f"   Odds poll: {odds_poll_interval}s")
        logger.info(f"   Dry run: {dry_run}")

    def run(self):
        """Main monitoring loop."""
        logger.info("=" * 60)
        logger.info("Live Game Monitor Started")
        logger.info("=" * 60)

        try:
            iteration = 0
            while True:
                iteration += 1
                logger.info(f"\n--- Iteration {iteration} at {datetime.now().strftime('%I:%M:%S %p')} ---")

                # Get live games
                live_games = self.game_client.get_live_games()

                if live_games.empty:
                    logger.info("No live games currently")
                    time.sleep(60)  # Wait longer when no games
                    continue

                logger.info(f"Monitoring {len(live_games)} live games")

                # Process each live game
                for _, game in live_games.iterrows():
                    self._process_game(game.to_dict())

                # Sleep before next check
                time.sleep(self.game_poll_interval)

        except KeyboardInterrupt:
            logger.info("\n" + "=" * 60)
            logger.info("Monitoring stopped by user")
            logger.info("=" * 60)
            self._print_summary()

        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _process_game(self, game: Dict):
        """
        Process a single live game.

        Args:
            game: Game state dict
        """
        game_id = game['game_id']
        away_team = game.get('away_team', 'AWAY')
        home_team = game.get('home_team', 'HOME')

        logger.info(f"\n📊 {away_team} @ {home_team}")
        logger.info(f"   Score: {game['away_score']}-{game['home_score']}")
        logger.info(f"   Q{game['period']} {game.get('game_clock', '')}")

        # Save game state
        if not self.dry_run:
            self.game_client.save_game_state(game)
            logger.debug(f"   ✓ Saved game state")

        # Check if we should fetch odds
        now = datetime.now()
        last_check = self.last_odds_check.get(game_id)

        should_check_odds = (
            not last_check or
            (now - last_check).total_seconds() >= self.odds_poll_interval
        )

        if not should_check_odds:
            time_until_next = self.odds_poll_interval - (now - last_check).total_seconds()
            logger.debug(f"   Odds check in {time_until_next:.0f}s")
            return

        # Get live odds for this game (match by teams since APIs use different game IDs)
        logger.info(f"   Fetching odds...")
        odds = self.odds_tracker.get_live_odds(teams=[(home_team, away_team)])

        if odds.empty:
            logger.warning(f"   ⚠️  No odds available")
            self.last_odds_check[game_id] = now
            return

        # Save odds snapshot
        if not self.dry_run:
            saved = self.odds_tracker.save_odds_snapshot(odds)
            logger.debug(f"   ✓ Saved {saved} odds records")

        # Get the Odds API game_id (different from NBA Stats game_id)
        odds_game_id = odds['game_id'].iloc[0] if not odds.empty else None
        if not odds_game_id:
            logger.warning(f"   ⚠️  Could not determine odds game_id")
            self.last_odds_check[game_id] = now
            return

        # Get latest odds for edge detection
        latest_odds = self.odds_tracker.get_latest_odds(odds_game_id)

        if not latest_odds:
            logger.warning(f"   ⚠️  Could not parse odds")
            self.last_odds_check[game_id] = now
            return

        logger.info(f"   Odds: Spread={latest_odds.get('spread', {}).get('spread_value', 'N/A')}")

        # Calculate win probability
        win_prob = self.win_prob_model.predict(game)
        logger.info(
            f"   Win Prob: Home {win_prob['home_win_prob']:.1%} "
            f"(confidence: {win_prob['confidence']:.1%})"
        )

        # Detect edges
        edges = self.edge_detector.find_edges(game, latest_odds, win_prob)

        if edges:
            logger.success(f"   🎯 Found {len(edges)} edge(s)!")

            for edge in edges:
                self._handle_edge(edge)
        else:
            logger.info(f"   No edges detected")

        self.last_odds_check[game_id] = now

    def _handle_edge(self, edge: Dict):
        """
        Handle detected edge.

        Args:
            edge: Edge dict
        """
        # Create alert signature to avoid duplicates
        alert_sig = (
            edge['game_id'],
            edge['alert_type'],
            edge['bet_side'],
            edge.get('line_value')
        )

        # Check if we've alerted recently (within 5 minutes)
        if alert_sig in self.recent_alerts:
            logger.debug(f"   (Already alerted for this edge)")
            return

        # Log the edge
        logger.success(
            f"\n   🎯 EDGE DETECTED - {edge['alert_type'].upper()} {edge['bet_side'].upper()}"
        )
        logger.success(f"      Edge: {edge['edge']:.1%}")
        logger.success(f"      Confidence: {edge['confidence']}")
        logger.success(f"      Model: {edge['model_prob']:.1%}, Market: {edge['market_prob']:.1%}")

        if edge.get('odds'):
            logger.success(f"      Odds: {edge['odds']:+d}")

        if edge.get('line_value') is not None:
            logger.success(f"      Line: {edge['line_value']:+.1f}")

        logger.success(
            f"      Game State: Q{edge['quarter']}, "
            f"Score {edge['home_score']}-{edge['away_score']}, "
            f"{edge['time_remaining']}"
        )

        # Save to database
        if not self.dry_run:
            alert_id = self.edge_detector.save_alert(edge)
            logger.success(f"      Alert ID: {alert_id}")

        # Add to recent alerts
        self.recent_alerts.add(alert_sig)

        # Clean old alerts from cache (older than 5 min)
        # (In production, would use a more sophisticated approach)

    def _print_summary(self):
        """Print monitoring summary."""
        if self.dry_run:
            logger.info("Dry run mode - no data saved")
            return

        from src.data.live_betting_db import get_stats

        stats = get_stats()

        logger.info("\n📊 Monitoring Summary:")
        logger.info(f"   Game states saved: {stats.get('live_game_state', 0)}")
        logger.info(f"   Odds snapshots: {stats.get('live_odds_snapshot', 0)}")
        logger.info(f"   Edge alerts: {stats.get('live_edge_alerts', 0)}")

        if stats.get('date_range'):
            logger.info(f"   Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor live NBA games for betting edges")

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Don't save to database, just print"
    )

    parser.add_argument(
        '--min-edge',
        type=float,
        default=DEFAULT_MIN_EDGE,
        help=f"Minimum edge threshold (default {DEFAULT_MIN_EDGE:.1%})"
    )

    parser.add_argument(
        '--game-poll',
        type=int,
        default=DEFAULT_GAME_POLL_INTERVAL,
        help=f"Game polling interval in seconds (default {DEFAULT_GAME_POLL_INTERVAL})"
    )

    parser.add_argument(
        '--odds-poll',
        type=int,
        default=DEFAULT_ODDS_POLL_INTERVAL,
        help=f"Odds polling interval in seconds (default {DEFAULT_ODDS_POLL_INTERVAL})"
    )

    args = parser.parse_args()

    # Create monitor
    monitor = LiveGameMonitor(
        min_edge=args.min_edge,
        game_poll_interval=args.game_poll,
        odds_poll_interval=args.odds_poll,
        dry_run=args.dry_run
    )

    # Run
    monitor.run()


if __name__ == "__main__":
    main()
