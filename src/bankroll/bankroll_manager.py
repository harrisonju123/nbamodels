"""
Bankroll Manager

Tracks bankroll progression over time and calculates dynamic bet sizing.
Enables compounding growth and better risk management.
"""

import sqlite3
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path
import pandas as pd
from loguru import logger


DB_PATH = "data/bets/bets.db"


class BankrollManager:
    """
    Manages bankroll tracking and progression.

    Features:
    - Track bankroll changes over time
    - Calculate current bankroll
    - Monitor drawdowns
    - Enable dynamic bet sizing
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_tables()

    def _get_connection(self):
        """Get database connection with foreign key enforcement."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")  # Enforce foreign key constraints
        return conn

    def _init_tables(self):
        """Initialize bankroll tracking tables."""
        conn = self._get_connection()

        # Bankroll history table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bankroll_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                amount REAL NOT NULL,
                change_amount REAL,
                change_type TEXT,
                bet_id TEXT,
                notes TEXT,
                peak_bankroll REAL,
                drawdown_pct REAL,
                FOREIGN KEY (bet_id) REFERENCES bets(id)
            )
        """)

        # Index for performance
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_bankroll_timestamp
            ON bankroll_history(timestamp)
        """)

        conn.commit()
        conn.close()

        logger.debug("Bankroll tables initialized")

    def initialize_bankroll(self, starting_amount: float, notes: str = "Initial bankroll") -> Dict:
        """
        Initialize bankroll with starting amount.

        Args:
            starting_amount: Starting bankroll amount
            notes: Optional notes

        Returns:
            Dict with bankroll record

        Raises:
            ValueError: If starting_amount is invalid
        """
        # Input validation
        if not isinstance(starting_amount, (int, float)):
            raise ValueError(f"starting_amount must be numeric, got {type(starting_amount)}")

        if starting_amount <= 0:
            raise ValueError(f"starting_amount must be positive, got {starting_amount}")

        if starting_amount > 1_000_000_000:
            raise ValueError(f"starting_amount {starting_amount} exceeds reasonable limit")

        conn = self._get_connection()

        # Check if already initialized
        existing = conn.execute(
            "SELECT * FROM bankroll_history ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()

        if existing:
            logger.warning(
                f"Bankroll already initialized on {existing['timestamp']} "
                f"at ${existing['amount']:.2f}. Skipping initialization."
            )
            logger.info("Use record_adjustment() if you need to add/remove funds")
            conn.close()
            return dict(existing)

        timestamp = datetime.now().isoformat()

        conn.execute("""
            INSERT INTO bankroll_history (
                timestamp, amount, change_amount, change_type, notes, peak_bankroll, drawdown_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, starting_amount, 0, 'initial', notes, starting_amount, 0))

        conn.commit()

        result = conn.execute(
            "SELECT * FROM bankroll_history ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()

        conn.close()

        logger.info(f"✓ Initialized bankroll: ${starting_amount:.2f}")
        return dict(result)

    def get_current_bankroll(self) -> float:
        """
        Get current bankroll amount.

        Returns:
            Current bankroll (or 0 if not initialized)
        """
        conn = self._get_connection()

        result = conn.execute(
            "SELECT amount FROM bankroll_history ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()

        conn.close()

        if result:
            return result['amount']
        else:
            logger.warning("Bankroll not initialized - returning 0")
            return 0

    def record_bet_outcome(
        self,
        bet_id: str,
        profit: float,
        notes: str = None
    ) -> Dict:
        """
        Record bankroll change from bet outcome.

        Args:
            bet_id: Bet identifier
            profit: Profit/loss amount (positive for win, negative for loss)
            notes: Optional notes

        Returns:
            Dict with updated bankroll record

        Raises:
            ValueError: If bankroll not initialized or invalid inputs
        """
        # Input validation
        if not bet_id or not isinstance(bet_id, str):
            raise ValueError(f"bet_id must be non-empty string, got {bet_id}")

        if not isinstance(profit, (int, float)):
            raise ValueError(f"profit must be numeric, got {type(profit)}")

        conn = self._get_connection()

        try:
            # Start exclusive transaction to prevent race conditions
            conn.execute("BEGIN EXCLUSIVE")

            # Check for duplicate bet_id (idempotency)
            existing = conn.execute(
                "SELECT * FROM bankroll_history WHERE bet_id = ?",
                (bet_id,)
            ).fetchone()

            if existing:
                logger.warning(f"Bet {bet_id} already recorded in bankroll - skipping")
                conn.rollback()
                return dict(existing)

            # Get current bankroll within transaction
            current = conn.execute(
                "SELECT amount FROM bankroll_history ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()

            if not current:
                conn.rollback()
                raise ValueError("Bankroll not initialized - cannot record bet outcome")

            current_bankroll = current['amount']

            # Sanity check on profit magnitude
            if abs(profit) > current_bankroll * 2:
                logger.warning(f"Profit {profit} exceeds 2x bankroll {current_bankroll}")

            new_bankroll = current_bankroll + profit
            timestamp = datetime.now().isoformat()

            # Get peak bankroll
            peak = conn.execute(
                "SELECT MAX(peak_bankroll) as peak FROM bankroll_history"
            ).fetchone()
            peak_bankroll = max(peak['peak'] if peak['peak'] else 0, new_bankroll)

            # Calculate drawdown
            drawdown_pct = (peak_bankroll - new_bankroll) / peak_bankroll if peak_bankroll > 0 else 0

            # Determine change type
            if profit > 0:
                change_type = 'win'
            elif profit < 0:
                change_type = 'loss'
            else:
                change_type = 'push'

            conn.execute("""
                INSERT INTO bankroll_history (
                    timestamp, amount, change_amount, change_type, bet_id, notes,
                    peak_bankroll, drawdown_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, new_bankroll, profit, change_type, bet_id, notes, peak_bankroll, drawdown_pct))

            result = conn.execute(
                "SELECT * FROM bankroll_history ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()

            conn.commit()

            logger.info(f"Bankroll updated: ${current_bankroll:.2f} → ${new_bankroll:.2f} ({profit:+.2f})")

            return dict(result)

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to record bet outcome: {e}")
            raise
        finally:
            conn.close()

    def record_adjustment(
        self,
        amount: float,
        adjustment_type: str,
        notes: str = None
    ) -> Dict:
        """
        Record manual bankroll adjustment (deposit/withdrawal).

        Args:
            amount: Adjustment amount (positive for deposit, negative for withdrawal)
            adjustment_type: 'deposit' or 'withdrawal'
            notes: Optional notes

        Returns:
            Dict with updated bankroll record
        """
        current_bankroll = self.get_current_bankroll()
        new_bankroll = current_bankroll + amount
        timestamp = datetime.now().isoformat()

        conn = self._get_connection()

        # Get peak
        peak = conn.execute(
            "SELECT MAX(peak_bankroll) as peak FROM bankroll_history"
        ).fetchone()
        peak_bankroll = max(peak['peak'] if peak['peak'] else 0, new_bankroll)

        drawdown_pct = (peak_bankroll - new_bankroll) / peak_bankroll if peak_bankroll > 0 else 0

        conn.execute("""
            INSERT INTO bankroll_history (
                timestamp, amount, change_amount, change_type, notes,
                peak_bankroll, drawdown_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, new_bankroll, amount, adjustment_type, notes, peak_bankroll, drawdown_pct))

        conn.commit()

        result = conn.execute(
            "SELECT * FROM bankroll_history ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()

        conn.close()

        logger.info(f"{adjustment_type.title()}: ${abs(amount):.2f} - New bankroll: ${new_bankroll:.2f}")

        return dict(result)

    def get_bankroll_history(self, limit: int = None) -> pd.DataFrame:
        """
        Get bankroll history.

        Args:
            limit: Maximum number of records to return

        Returns:
            DataFrame with bankroll history
        """
        conn = self._get_connection()

        query = "SELECT * FROM bankroll_history ORDER BY timestamp DESC"
        params = []
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        df = pd.read_sql_query(query, conn, params=params if params else None)
        conn.close()

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    def get_bankroll_stats(self) -> Dict:
        """
        Get bankroll statistics.

        Returns:
            Dict with stats: current, peak, starting, total_profit, roi, max_drawdown
        """
        conn = self._get_connection()

        # Get current
        current = conn.execute(
            "SELECT amount FROM bankroll_history ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()

        if not current:
            conn.close()
            return {
                'current': 0,
                'starting': 0,
                'peak': 0,
                'total_profit': 0,
                'roi': 0,
                'max_drawdown': 0,
            }

        current_amount = current['amount']

        # Get starting
        starting = conn.execute(
            "SELECT amount FROM bankroll_history ORDER BY timestamp ASC LIMIT 1"
        ).fetchone()
        starting_amount = starting['amount']

        # Get peak
        peak = conn.execute(
            "SELECT MAX(amount) as peak FROM bankroll_history"
        ).fetchone()
        peak_amount = peak['peak']

        # Get max drawdown
        max_dd = conn.execute(
            "SELECT MAX(drawdown_pct) as max_dd FROM bankroll_history"
        ).fetchone()
        max_drawdown = max_dd['max_dd'] if max_dd['max_dd'] else 0

        conn.close()

        total_profit = current_amount - starting_amount
        roi = (total_profit / starting_amount * 100) if starting_amount > 0 else 0

        return {
            'current': current_amount,
            'starting': starting_amount,
            'peak': peak_amount,
            'total_profit': total_profit,
            'roi': roi,
            'max_drawdown': max_drawdown,
        }

    def sync_with_bets(self, starting_bankroll: float = 1000.0):
        """
        Sync bankroll history with existing bet records.

        Useful for backfilling bankroll from historical bets.

        Args:
            starting_bankroll: Starting bankroll amount
        """
        logger.info("Syncing bankroll with bet history...")

        # Initialize if needed
        current = self.get_current_bankroll()
        if current == 0:
            self.initialize_bankroll(starting_bankroll, "Initial bankroll (synced from bets)")

        conn = self._get_connection()

        # Get all settled bets not in bankroll history
        settled_bets = conn.execute("""
            SELECT id, profit, settled_at, outcome
            FROM bets
            WHERE outcome IS NOT NULL
            AND id NOT IN (SELECT bet_id FROM bankroll_history WHERE bet_id IS NOT NULL)
            ORDER BY settled_at ASC
        """).fetchall()

        conn.close()

        logger.info(f"Found {len(settled_bets)} settled bets to sync")

        for bet in settled_bets:
            self.record_bet_outcome(
                bet_id=bet['id'],
                profit=bet['profit'],
                notes=f"Synced from bet outcome: {bet['outcome']}"
            )

        stats = self.get_bankroll_stats()
        logger.info(f"✓ Sync complete - Current bankroll: ${stats['current']:.2f}")


if __name__ == "__main__":
    # Test the bankroll manager
    print("Bankroll Manager - Test Mode")
    print("=" * 70)

    manager = BankrollManager()

    # Get stats
    stats = manager.get_bankroll_stats()

    print("\nCurrent Bankroll Stats:")
    print(f"  Current:      ${stats['current']:.2f}")
    print(f"  Starting:     ${stats['starting']:.2f}")
    print(f"  Peak:         ${stats['peak']:.2f}")
    print(f"  Total Profit: ${stats['total_profit']:+.2f}")
    print(f"  ROI:          {stats['roi']:.2f}%")
    print(f"  Max Drawdown: {stats['max_drawdown']:.2%}")
