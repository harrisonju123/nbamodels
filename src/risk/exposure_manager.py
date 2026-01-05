"""
Exposure Manager

Manages daily/weekly exposure limits and tracks unsettled exposure.
Queries existing bet_tracker database to calculate wagered amounts.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Tuple, List
from loguru import logger

from .config import RiskConfig
from .models import BetCorrelationContext


DB_PATH = "data/bets/bets.db"


class ExposureManager:
    """
    Manages daily/weekly exposure limits and pending exposure tracking.

    Queries the bets database to calculate:
    - Daily wagered amount
    - Weekly wagered amount
    - Pending (unsettled) exposure
    """

    def __init__(self, config: RiskConfig, db_path: str = DB_PATH):
        """
        Initialize exposure manager.

        Args:
            config: Risk configuration
            db_path: Path to bets database
        """
        self.config = config
        self.db_path = db_path

        logger.debug("ExposureManager initialized")

    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_daily_wagered(self, date: str = None) -> float:
        """
        Get total wagered today (or on specified date).

        Args:
            date: Optional date in YYYY-MM-DD format. Defaults to today.

        Returns:
            Total wagered amount
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        conn = self._get_connection()

        result = conn.execute("""
            SELECT COALESCE(SUM(bet_amount), 0) as total
            FROM bets
            WHERE DATE(logged_at) = ?
        """, (date,)).fetchone()

        conn.close()

        total = result['total'] if result else 0.0
        logger.debug(f"Daily wagered ({date}): ${total:.2f}")

        return total

    def get_weekly_wagered(self, end_date: str = None) -> float:
        """
        Get total wagered this week (last 7 days including today).

        Args:
            end_date: Optional end date in YYYY-MM-DD format. Defaults to today.

        Returns:
            Total wagered amount
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # Calculate start date (7 days ago)
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=6)  # Last 7 days including end_date
        start_date = start_dt.strftime('%Y-%m-%d')

        conn = self._get_connection()

        result = conn.execute("""
            SELECT COALESCE(SUM(bet_amount), 0) as total
            FROM bets
            WHERE DATE(logged_at) BETWEEN ? AND ?
        """, (start_date, end_date)).fetchone()

        conn.close()

        total = result['total'] if result else 0.0
        logger.debug(f"Weekly wagered ({start_date} to {end_date}): ${total:.2f}")

        return total

    def get_pending_exposure(self) -> float:
        """
        Get total exposure in unsettled bets.

        Returns:
            Total amount in pending bets
        """
        conn = self._get_connection()

        result = conn.execute("""
            SELECT COALESCE(SUM(bet_amount), 0) as total
            FROM bets
            WHERE outcome IS NULL
        """).fetchone()

        conn.close()

        total = result['total'] if result else 0.0
        logger.debug(f"Pending exposure: ${total:.2f}")

        return total

    def get_remaining_daily_budget(self, bankroll: float, date: str = None) -> float:
        """
        Calculate remaining daily betting budget.

        Args:
            bankroll: Current bankroll
            date: Optional date (defaults to today)

        Returns:
            Remaining daily budget
        """
        daily_limit = bankroll * self.config.max_daily_exposure
        daily_wagered = self.get_daily_wagered(date)
        remaining = max(0, daily_limit - daily_wagered)

        logger.debug(
            f"Daily budget: ${remaining:.2f} remaining "
            f"(limit: ${daily_limit:.2f}, wagered: ${daily_wagered:.2f})"
        )

        return remaining

    def get_remaining_weekly_budget(self, bankroll: float, end_date: str = None) -> float:
        """
        Calculate remaining weekly betting budget.

        Args:
            bankroll: Current bankroll
            end_date: Optional end date (defaults to today)

        Returns:
            Remaining weekly budget
        """
        weekly_limit = bankroll * self.config.max_weekly_exposure
        weekly_wagered = self.get_weekly_wagered(end_date)
        remaining = max(0, weekly_limit - weekly_wagered)

        logger.debug(
            f"Weekly budget: ${remaining:.2f} remaining "
            f"(limit: ${weekly_limit:.2f}, wagered: ${weekly_wagered:.2f})"
        )

        return remaining

    def cap_to_limits(
        self,
        proposed_size: float,
        bankroll: float,
        context: BetCorrelationContext = None
    ) -> Tuple[float, List[str]]:
        """
        Cap bet size to all applicable exposure limits.

        Args:
            proposed_size: Proposed bet amount
            bankroll: Current bankroll
            context: Bet correlation context (optional)

        Returns:
            (capped_size, list_of_limiting_factors)
        """
        limiting_factors = []
        capped_size = proposed_size

        # 1. Daily exposure limit
        daily_remaining = self.get_remaining_daily_budget(bankroll)
        if capped_size > daily_remaining:
            capped_size = daily_remaining
            limiting_factors.append(f"daily_limit (${daily_remaining:.2f} remaining)")

        # 2. Weekly exposure limit
        weekly_remaining = self.get_remaining_weekly_budget(bankroll)
        if capped_size > weekly_remaining:
            capped_size = weekly_remaining
            limiting_factors.append(f"weekly_limit (${weekly_remaining:.2f} remaining)")

        # 3. Pending exposure limit
        pending_limit = bankroll * self.config.max_pending_exposure
        pending_exposure = self.get_pending_exposure()
        pending_remaining = max(0, pending_limit - pending_exposure)
        if capped_size > pending_remaining:
            capped_size = pending_remaining
            limiting_factors.append(
                f"pending_limit (${pending_remaining:.2f} remaining)"
            )

        # Minimum bet size
        if capped_size < 1.0:
            capped_size = 0.0
            limiting_factors.append("below_minimum ($1)")

        return capped_size, limiting_factors

    def check_daily_limit(self, bankroll: float, proposed_amount: float) -> bool:
        """
        Check if proposed bet would exceed daily limit.

        Args:
            bankroll: Current bankroll
            proposed_amount: Proposed bet amount

        Returns:
            True if within limit, False if would exceed
        """
        remaining = self.get_remaining_daily_budget(bankroll)
        return proposed_amount <= remaining

    def check_weekly_limit(self, bankroll: float, proposed_amount: float) -> bool:
        """
        Check if proposed bet would exceed weekly limit.

        Args:
            bankroll: Current bankroll
            proposed_amount: Proposed bet amount

        Returns:
            True if within limit, False if would exceed
        """
        remaining = self.get_remaining_weekly_budget(bankroll)
        return proposed_amount <= remaining

    def reset_daily(self, date: str = None):
        """
        Reset daily tracking (for testing/simulation).

        Note: In live operation, this is automatic based on date queries.

        Args:
            date: Date to reset (defaults to today)
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"Daily exposure reset (informational only) for {date}")

    def reset_weekly(self, end_date: str = None):
        """
        Reset weekly tracking (for testing/simulation).

        Note: In live operation, this is automatic based on date queries.

        Args:
            end_date: End date to reset (defaults to today)
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"Weekly exposure reset (informational only) for {end_date}")

    def get_exposure_summary(self, bankroll: float) -> dict:
        """
        Get comprehensive exposure summary.

        Args:
            bankroll: Current bankroll

        Returns:
            Dict with all exposure metrics
        """
        daily_limit = bankroll * self.config.max_daily_exposure
        weekly_limit = bankroll * self.config.max_weekly_exposure
        pending_limit = bankroll * self.config.max_pending_exposure

        daily_wagered = self.get_daily_wagered()
        weekly_wagered = self.get_weekly_wagered()
        pending_exposure = self.get_pending_exposure()

        return {
            'daily': {
                'limit': daily_limit,
                'wagered': daily_wagered,
                'remaining': max(0, daily_limit - daily_wagered),
                'utilization_pct': (daily_wagered / daily_limit * 100) if daily_limit > 0 else 0
            },
            'weekly': {
                'limit': weekly_limit,
                'wagered': weekly_wagered,
                'remaining': max(0, weekly_limit - weekly_wagered),
                'utilization_pct': (weekly_wagered / weekly_limit * 100) if weekly_limit > 0 else 0
            },
            'pending': {
                'limit': pending_limit,
                'exposure': pending_exposure,
                'remaining': max(0, pending_limit - pending_exposure),
                'utilization_pct': (pending_exposure / pending_limit * 100) if pending_limit > 0 else 0
            }
        }

    def __repr__(self) -> str:
        daily = self.get_daily_wagered()
        weekly = self.get_weekly_wagered()
        pending = self.get_pending_exposure()
        return (
            f"ExposureManager("
            f"daily=${daily:.2f}, "
            f"weekly=${weekly:.2f}, "
            f"pending=${pending:.2f}"
            f")"
        )
