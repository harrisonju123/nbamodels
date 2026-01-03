"""
Line Snapshot Collector

Collects hourly odds snapshots for market microstructure analysis.
Designed for batch processing via cron/scheduler.
"""

import os
import sqlite3
import fcntl
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

import pandas as pd
import requests
from loguru import logger

from src.data.odds_api import OddsAPIClient


# Database path (matches bet_tracker.py)
DB_PATH = "data/bets/bets.db"
LOCK_FILE = "/tmp/nba_snapshot_collector.lock"


class LineSnapshotCollector:
    """
    Collects hourly odds snapshots for all upcoming games.

    API Rate Limit Considerations:
    - The Odds API: 500 requests/month on free tier, 10K on Starter ($19/mo)
    - Each call to get_current_odds() = 1 request
    - With hourly collection: ~720 requests/month (need Starter tier)
    - Strategy: Collect only for games within collection_window_hours
    """

    def __init__(
        self,
        collection_window_hours: int = 24,  # Only collect for games within 24hrs
        snapshot_interval_hours: int = 1,   # Hourly collection
    ):
        """
        Initialize the snapshot collector.

        Args:
            collection_window_hours: Only collect snapshots for games within this window
            snapshot_interval_hours: Minimum hours between snapshots for same game
        """
        self.collection_window = collection_window_hours
        self.interval = snapshot_interval_hours
        self.odds_client = OddsAPIClient()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def collect_snapshot(self) -> int:
        """
        Collect current odds snapshot for upcoming games.

        Returns:
            Count of snapshot records saved.

        Workflow:
            1. Acquire file lock to prevent concurrent execution
            2. Fetch current odds from API
            3. Filter to games within collection window
            4. Save snapshots to line_snapshots table
        """
        # Acquire file lock to prevent concurrent cron jobs
        lock_fd = open(LOCK_FILE, 'w')
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            logger.warning("Another snapshot collection process is already running - skipping")
            lock_fd.close()
            return 0

        try:
            logger.info("Starting line snapshot collection...")

        try:
            # Fetch current odds
            odds_df = self.odds_client.get_current_odds(
                markets=["h2h", "spreads", "totals"]
            )

            if odds_df.empty:
                logger.warning("No odds data returned from API")
                return 0

        except requests.exceptions.Timeout as e:
            logger.error(f"API timeout during snapshot collection: {e}")
            raise  # Re-raise to alert monitoring systems
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.error("Rate limit exceeded - consider upgrading API tier")
            elif e.response.status_code == 401:
                logger.error("Authentication failed - check API key")
            else:
                logger.error(f"HTTP error during snapshot collection: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during snapshot collection: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching odds: {e}")
            raise

        try:

            # Filter to games within collection window (use UTC for consistency)
            now = datetime.now(timezone.utc)
            cutoff = now + timedelta(hours=self.collection_window)

            odds_df['commence_time'] = pd.to_datetime(odds_df['commence_time'], utc=True)
            filtered = odds_df[
                (odds_df['commence_time'] >= now) &
                (odds_df['commence_time'] <= cutoff)
            ]

            if filtered.empty:
                logger.info(f"No games within {self.collection_window}hr window")
                return 0

            # Save snapshots
            count = self.save_snapshot(filtered)
            logger.info(f"Saved {count} snapshot records")

            return count

        except Exception as e:
            logger.error(f"Error during snapshot collection: {e}")
            raise

        finally:
            # Release file lock
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def get_games_needing_snapshot(self) -> List[str]:
        """
        Return game_ids that need snapshots based on:
        - Game is within collection_window_hours
        - Haven't collected snapshot in last interval_hours

        Returns:
            List of game_ids
        """
        conn = self._get_connection()

        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=self.collection_window)
        last_snapshot_cutoff = now - timedelta(hours=self.interval)

        # Query games that need snapshots
        query = """
            SELECT DISTINCT game_id
            FROM line_snapshots
            WHERE snapshot_time >= ?
            AND game_id NOT IN (
                SELECT game_id
                FROM line_snapshots
                WHERE snapshot_time >= ?
            )
        """

        games = conn.execute(
            query,
            (last_snapshot_cutoff.isoformat(), now.isoformat())
        ).fetchall()

        conn.close()

        return [game['game_id'] for game in games]

    def save_snapshot(self, odds_df: pd.DataFrame) -> int:
        """
        Save odds snapshot to line_snapshots table (optimized - bulk insert).

        Args:
            odds_df: DataFrame from OddsAPIClient.get_current_odds()

        Returns:
            Number of records inserted
        """
        conn = self._get_connection()
        snapshot_time = datetime.now(timezone.utc).isoformat()

        # Prepare all records for bulk insert (much faster than row-by-row)
        records = []

        for _, row in odds_df.iterrows():
            game_id = row['game_id']
            bookmaker = row['bookmaker']
            market = row['market']

            # Map market to bet_type
            if market == 'h2h':
                bet_type = 'moneyline'
            elif market == 'spreads':
                bet_type = 'spread'
            elif market == 'totals':
                bet_type = 'totals'
            else:
                continue

            # Collect records based on bet type
            if bet_type == 'moneyline':
                # Home odds
                if 'home_odds' in row and pd.notna(row['home_odds']):
                    records.append((
                        game_id, snapshot_time, bet_type, 'home', bookmaker,
                        int(row['home_odds']), None,
                        row.get('home_implied_prob'), row.get('home_no_vig_prob')
                    ))
                # Away odds
                if 'away_odds' in row and pd.notna(row['away_odds']):
                    records.append((
                        game_id, snapshot_time, bet_type, 'away', bookmaker,
                        int(row['away_odds']), None,
                        row.get('away_implied_prob'), row.get('away_no_vig_prob')
                    ))

            elif bet_type == 'spread' and 'line' in row and pd.notna(row['line']):
                side = row.get('team', 'home')
                odds = int(row['odds']) if 'odds' in row and pd.notna(row['odds']) else None
                records.append((
                    game_id, snapshot_time, bet_type, side, bookmaker,
                    odds, float(row['line']),
                    row.get('implied_prob'), row.get('no_vig_prob')
                ))

            elif bet_type == 'totals':
                line_value = row.get('line')
                if pd.notna(line_value):
                    # Over
                    if 'over_odds' in row and pd.notna(row['over_odds']):
                        records.append((
                            game_id, snapshot_time, bet_type, 'over', bookmaker,
                            int(row['over_odds']), float(line_value),
                            row.get('over_implied_prob'), row.get('over_no_vig_prob')
                        ))
                    # Under
                    if 'under_odds' in row and pd.notna(row['under_odds']):
                        records.append((
                            game_id, snapshot_time, bet_type, 'under', bookmaker,
                            int(row['under_odds']), float(line_value),
                            row.get('under_implied_prob'), row.get('under_no_vig_prob')
                        ))

        # Bulk insert all records (10-100x faster than individual inserts)
        if records:
            try:
                conn.executemany("""
                    INSERT OR REPLACE INTO line_snapshots (
                        game_id, snapshot_time, bet_type, side, bookmaker,
                        odds, line, implied_prob, no_vig_prob
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
                conn.commit()
                count = len(records)
                logger.debug(f"Bulk inserted {count} snapshot records")
            except Exception as e:
                logger.error(f"Error during bulk insert: {e}")
                conn.rollback()
                count = 0
        else:
            count = 0

        conn.close()
        return count

    def _insert_snapshot(
        self,
        conn: sqlite3.Connection,
        game_id: str,
        snapshot_time: str,
        bet_type: str,
        side: str,
        bookmaker: str,
        odds: Optional[int],
        line: Optional[float],
        implied_prob: Optional[float],
        no_vig_prob: Optional[float]
    ) -> bool:
        """
        Insert a single snapshot record.

        Uses INSERT OR REPLACE to handle duplicates (same game/time/type/side/bookmaker).

        Returns:
            True if successful, False if error occurred
        """
        try:
            conn.execute("""
                INSERT OR REPLACE INTO line_snapshots (
                    game_id, snapshot_time, bet_type, side, bookmaker,
                    odds, line, implied_prob, no_vig_prob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id, snapshot_time, bet_type, side, bookmaker,
                odds, line, implied_prob, no_vig_prob
            ))
            return True
        except Exception as e:
            logger.error(f"Error inserting snapshot for {game_id}/{bookmaker}: {e}")
            return False

    def get_snapshot_stats(self) -> Dict:
        """Get statistics about collected snapshots."""
        conn = self._get_connection()

        stats = {}

        # Total snapshots
        total = conn.execute("SELECT COUNT(*) as count FROM line_snapshots").fetchone()
        stats['total_snapshots'] = total['count']

        # Unique games
        games = conn.execute("SELECT COUNT(DISTINCT game_id) as count FROM line_snapshots").fetchone()
        stats['unique_games'] = games['count']

        # Date range
        date_range = conn.execute("""
            SELECT
                MIN(snapshot_time) as earliest,
                MAX(snapshot_time) as latest
            FROM line_snapshots
        """).fetchone()
        stats['earliest_snapshot'] = date_range['earliest']
        stats['latest_snapshot'] = date_range['latest']

        # Snapshots by bookmaker
        by_book = pd.read_sql_query("""
            SELECT bookmaker, COUNT(*) as count
            FROM line_snapshots
            GROUP BY bookmaker
            ORDER BY count DESC
        """, conn)
        stats['by_bookmaker'] = by_book.to_dict('records')

        conn.close()

        return stats


if __name__ == "__main__":
    # Test the collector
    collector = LineSnapshotCollector()

    print("Collecting snapshots...")
    count = collector.collect_snapshot()
    print(f"Collected {count} snapshots")

    print("\nSnapshot stats:")
    stats = collector.get_snapshot_stats()
    for key, value in stats.items():
        if key != 'by_bookmaker':
            print(f"  {key}: {value}")
        else:
            print(f"  {key}:")
            for book in value:
                print(f"    {book['bookmaker']}: {book['count']}")
