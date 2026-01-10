"""
Line History Query Module

Provides functions for querying line movement history, opening lines,
and movement pattern analysis.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import sqlite3

import pandas as pd
import numpy as np
from loguru import logger

DB_PATH = "data/bets/bets.db"


@dataclass
class LineMovement:
    """Represents a single line movement."""
    from_line: float
    to_line: float
    from_odds: int
    to_odds: int
    movement_pts: float
    movement_time: str
    bookmaker: str


@dataclass
class MovementPattern:
    """Describes the overall movement pattern for a game."""
    game_id: str
    bet_type: str
    pattern: str  # 'stable', 'trending_up', 'trending_down', 'volatile', 'late_steam'
    total_movement: float
    velocity: float
    reversals: int
    sharp_direction: Optional[str]
    confidence: float


class LineHistoryManager:
    """Manages line history queries and analysis."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # === CORE QUERY FUNCTIONS ===

    def get_line_history(
        self,
        game_id: str,
        bet_type: str = 'spread',
        side: str = 'home',
        bookmaker: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get full line history timeline for a game.

        Args:
            game_id: Game identifier
            bet_type: 'spread', 'totals', or 'moneyline'
            side: 'home', 'away', 'over', or 'under'
            bookmaker: Specific bookmaker or None for all

        Returns:
            DataFrame with columns:
            - snapshot_time, bookmaker, odds, line, implied_prob, no_vig_prob
            - line_change, odds_change (from previous snapshot)
        """
        conn = self._get_connection()

        query = """
            SELECT
                ls.snapshot_time,
                ls.bookmaker,
                ls.odds,
                ls.line,
                ls.implied_prob,
                ls.no_vig_prob
            FROM line_snapshots ls
            WHERE ls.game_id = ?
            AND ls.bet_type = ?
            AND ls.side = ?
        """
        params = [game_id, bet_type, side]

        if bookmaker:
            query += " AND ls.bookmaker = ?"
            params.append(bookmaker)

        query += " ORDER BY ls.snapshot_time ASC"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            return df

        # Add derived columns
        df['snapshot_time'] = pd.to_datetime(df['snapshot_time'])

        # Calculate line and odds changes
        df['line_change'] = df.groupby('bookmaker')['line'].diff()
        df['odds_change'] = df.groupby('bookmaker')['odds'].diff()

        return df

    def get_opening_line(
        self,
        game_id: str,
        bet_type: str,
        bookmaker: Optional[str] = None
    ) -> Dict:
        """
        Get opening line for a game.

        Args:
            game_id: Game identifier
            bet_type: 'spread', 'totals', or 'moneyline'
            bookmaker: Specific bookmaker or None for consensus

        Returns:
            Dict with: opening_odds, opening_line, opening_implied_prob,
            first_seen_at, is_true_opener, bookmaker
        """
        conn = self._get_connection()

        # Try opening_lines table first
        if bookmaker:
            query = """
                SELECT * FROM opening_lines
                WHERE game_id = ? AND bet_type = ? AND bookmaker = ?
            """
            params = [game_id, bet_type, bookmaker]
        else:
            # Get consensus opener (median across books)
            query = """
                SELECT
                    AVG(opening_odds) as opening_odds,
                    AVG(opening_line) as opening_line,
                    AVG(opening_implied_prob) as opening_implied_prob,
                    MIN(first_seen_at) as first_seen_at,
                    'consensus' as bookmaker
                FROM opening_lines
                WHERE game_id = ? AND bet_type = ?
            """
            params = [game_id, bet_type]

        result = conn.execute(query, params).fetchone()

        if result and result['opening_odds'] is not None:
            conn.close()
            return dict(result)

        # Fallback: earliest snapshot as opener
        fallback_query = """
            SELECT
                odds as opening_odds,
                line as opening_line,
                implied_prob as opening_implied_prob,
                snapshot_time as first_seen_at,
                bookmaker
            FROM line_snapshots
            WHERE game_id = ? AND bet_type = ?
            ORDER BY snapshot_time ASC
            LIMIT 1
        """
        fallback = conn.execute(fallback_query, [game_id, bet_type]).fetchone()
        conn.close()

        if fallback:
            result = dict(fallback)
            result['is_true_opener'] = False
            return result

        return {}

    def get_line_at_time(
        self,
        game_id: str,
        bet_type: str,
        hours_before_game: float,
        side: str = 'home',
        bookmaker: Optional[str] = None
    ) -> Dict:
        """
        Get line at specific hours before game time.

        Args:
            game_id: Game identifier
            bet_type: 'spread', 'totals', or 'moneyline'
            hours_before_game: Hours before commence_time (e.g., 1.0, 4.0, 12.0)
            side: 'home', 'away', 'over', or 'under'
            bookmaker: Specific bookmaker or None for consensus

        Returns:
            Dict with odds, line, implied_prob at that time
        """
        conn = self._get_connection()

        # Get game commence time
        commence_query = """
            SELECT DISTINCT commence_time FROM bets WHERE game_id = ?
            UNION
            SELECT commence_time FROM opening_lines WHERE game_id = ?
            LIMIT 1
        """
        commence_row = conn.execute(commence_query, [game_id, game_id]).fetchone()

        if not commence_row:
            conn.close()
            return {}

        commence_time = datetime.fromisoformat(commence_row[0].replace('Z', '+00:00'))
        target_time = commence_time - timedelta(hours=hours_before_game)

        # Find snapshot closest to target time (within 1 hour window)
        window_start = target_time - timedelta(hours=0.5)
        window_end = target_time + timedelta(hours=0.5)

        if bookmaker:
            query = """
                SELECT odds, line, implied_prob, no_vig_prob, snapshot_time, bookmaker
                FROM line_snapshots
                WHERE game_id = ? AND bet_type = ? AND side = ? AND bookmaker = ?
                AND snapshot_time BETWEEN ? AND ?
                ORDER BY ABS(JULIANDAY(snapshot_time) - JULIANDAY(?))
                LIMIT 1
            """
            params = [game_id, bet_type, side, bookmaker,
                     window_start.isoformat(), window_end.isoformat(),
                     target_time.isoformat()]
        else:
            # Consensus across books
            query = """
                SELECT
                    AVG(odds) as odds,
                    AVG(line) as line,
                    AVG(implied_prob) as implied_prob,
                    AVG(no_vig_prob) as no_vig_prob,
                    MAX(snapshot_time) as snapshot_time,
                    'consensus' as bookmaker
                FROM line_snapshots
                WHERE game_id = ? AND bet_type = ? AND side = ?
                AND snapshot_time BETWEEN ? AND ?
            """
            params = [game_id, bet_type, side,
                     window_start.isoformat(), window_end.isoformat()]

        result = conn.execute(query, params).fetchone()
        conn.close()

        if result and result['odds'] is not None:
            return dict(result)
        return {}

    def detect_line_reversals(
        self,
        game_id: str,
        bet_type: str = 'spread',
        min_reversal_pts: float = 0.5
    ) -> List[Dict]:
        """
        Detect line reversals (direction changes).

        A reversal occurs when line moves one direction then reverses.

        Args:
            game_id: Game identifier
            bet_type: 'spread', 'totals', or 'moneyline'
            min_reversal_pts: Minimum reversal magnitude to report

        Returns:
            List of reversals with: reversal_time, from_direction,
            peak_line, reversal_magnitude
        """
        df = self.get_line_history(game_id, bet_type)

        if df.empty or len(df) < 3:
            return []

        # Use consensus line (median across books per snapshot)
        consensus = df.groupby('snapshot_time').agg({
            'line': 'median',
            'odds': 'median'
        }).reset_index().sort_values('snapshot_time')

        if len(consensus) < 3:
            return []

        reversals = []
        lines = consensus['line'].values
        times = consensus['snapshot_time'].values

        # Find direction changes
        directions = np.sign(np.diff(lines))
        direction_changes = np.where(np.diff(directions) != 0)[0]

        for idx in direction_changes:
            if idx + 1 < len(lines):
                magnitude = abs(lines[idx + 1] - lines[idx])
                if magnitude >= min_reversal_pts:
                    reversals.append({
                        'reversal_time': str(times[idx + 1]),
                        'from_direction': 'up' if directions[idx] > 0 else 'down',
                        'peak_line': float(lines[idx]),
                        'reversal_magnitude': float(magnitude),
                        'pre_reversal_line': float(lines[max(0, idx-1)]),
                        'post_reversal_line': float(lines[idx + 1])
                    })

        return reversals

    def analyze_movement_pattern(
        self,
        game_id: str,
        bet_type: str = 'spread'
    ) -> MovementPattern:
        """
        Analyze overall movement pattern for a game.

        Args:
            game_id: Game identifier
            bet_type: 'spread', 'totals', or 'moneyline'

        Returns:
            MovementPattern with pattern classification, metrics
        """
        df = self.get_line_history(game_id, bet_type)

        if df.empty or len(df) < 2:
            return MovementPattern(
                game_id=game_id,
                bet_type=bet_type,
                pattern='insufficient_data',
                total_movement=0.0,
                velocity=0.0,
                reversals=0,
                sharp_direction=None,
                confidence=0.0
            )

        # Calculate metrics
        consensus = df.groupby('snapshot_time').agg({
            'line': 'median',
            'odds': 'median'
        }).reset_index().sort_values('snapshot_time')

        first_line = consensus['line'].iloc[0]
        last_line = consensus['line'].iloc[-1]
        total_movement = last_line - first_line

        # Time span in hours
        first_time = pd.to_datetime(consensus['snapshot_time'].iloc[0])
        last_time = pd.to_datetime(consensus['snapshot_time'].iloc[-1])
        hours = (last_time - first_time).total_seconds() / 3600
        velocity = total_movement / hours if hours > 0 else 0

        # Count reversals
        reversals = len(self.detect_line_reversals(game_id, bet_type))

        # Determine pattern
        if abs(total_movement) < 0.5 and reversals <= 1:
            pattern = 'stable'
        elif abs(total_movement) >= 1.5 and reversals == 0:
            pattern = 'trending_up' if total_movement > 0 else 'trending_down'
        elif reversals >= 2:
            pattern = 'volatile'
        elif abs(velocity) > 0.5 and hours < 4:
            pattern = 'late_steam'
        else:
            pattern = 'drifting_' + ('up' if total_movement > 0 else 'down')

        # Sharp direction (based on final movement)
        sharp_direction = 'home' if total_movement < 0 else 'away' if total_movement > 0 else None

        # Confidence based on consistency
        if bet_type in ['spread', 'totals']:
            line_std = consensus['line'].std()
            confidence = max(0, 1 - (line_std / max(abs(total_movement), 1)))
        else:
            confidence = 0.5

        return MovementPattern(
            game_id=game_id,
            bet_type=bet_type,
            pattern=pattern,
            total_movement=total_movement,
            velocity=velocity,
            reversals=reversals,
            sharp_direction=sharp_direction,
            confidence=confidence
        )

    # === BATCH QUERY FUNCTIONS ===

    def get_multi_game_history(
        self,
        game_ids: List[str],
        bet_type: str = 'spread'
    ) -> pd.DataFrame:
        """
        Efficiently fetch line history for multiple games.
        Single query to avoid N+1 problem.

        Args:
            game_ids: List of game identifiers
            bet_type: 'spread', 'totals', or 'moneyline'

        Returns:
            DataFrame with all snapshots for the games
        """
        if not game_ids:
            return pd.DataFrame()

        conn = self._get_connection()
        placeholders = ','.join('?' * len(game_ids))

        query = f"""
            SELECT
                game_id,
                snapshot_time,
                bookmaker,
                odds,
                line,
                implied_prob,
                no_vig_prob,
                side
            FROM line_snapshots
            WHERE game_id IN ({placeholders})
            AND bet_type = ?
            ORDER BY game_id, snapshot_time ASC
        """

        df = pd.read_sql_query(query, conn, params=[*game_ids, bet_type])
        conn.close()

        return df

    def get_closing_line_summary(self, game_id: str) -> Dict:
        """
        Get summary of closing lines (last snapshot before game).

        Args:
            game_id: Game identifier

        Returns:
            Dict with closing_snapshots, closing_time, and market-specific data
        """
        conn = self._get_connection()

        query = """
            SELECT
                bet_type,
                side,
                bookmaker,
                odds,
                line,
                implied_prob,
                snapshot_time
            FROM line_snapshots
            WHERE game_id = ?
            AND snapshot_time = (
                SELECT MAX(snapshot_time)
                FROM line_snapshots
                WHERE game_id = ?
            )
        """

        df = pd.read_sql_query(query, conn, params=[game_id, game_id])
        conn.close()

        if df.empty:
            return {}

        return {
            'closing_snapshots': df.to_dict('records'),
            'closing_time': df['snapshot_time'].max(),
            'spread_closing': df[df['bet_type'] == 'spread'].to_dict('records'),
            'totals_closing': df[df['bet_type'] == 'totals'].to_dict('records'),
            'moneyline_closing': df[df['bet_type'] == 'h2h'].to_dict('records')
        }


# === CONVENIENCE FUNCTIONS (module-level) ===

_manager = None

def _get_manager() -> LineHistoryManager:
    """Get or create singleton LineHistoryManager."""
    global _manager
    if _manager is None:
        _manager = LineHistoryManager()
    return _manager

def get_line_history(
    game_id: str,
    bet_type: str = 'spread',
    side: str = 'home',
    bookmaker: Optional[str] = None
) -> pd.DataFrame:
    """Get full line history timeline for a game."""
    return _get_manager().get_line_history(game_id, bet_type, side, bookmaker)

def get_opening_line(
    game_id: str,
    bet_type: str,
    bookmaker: Optional[str] = None
) -> Dict:
    """Get opening line for a game."""
    return _get_manager().get_opening_line(game_id, bet_type, bookmaker)

def get_line_at_time(
    game_id: str,
    bet_type: str,
    hours_before_game: float,
    side: str = 'home',
    bookmaker: Optional[str] = None
) -> Dict:
    """Get line at specific hours before game."""
    return _get_manager().get_line_at_time(game_id, bet_type, hours_before_game, side, bookmaker)

def detect_line_reversals(
    game_id: str,
    bet_type: str = 'spread'
) -> List[Dict]:
    """Detect line reversals for a game."""
    return _get_manager().detect_line_reversals(game_id, bet_type)

def analyze_movement_pattern(
    game_id: str,
    bet_type: str = 'spread'
) -> MovementPattern:
    """Analyze movement pattern for a game."""
    return _get_manager().analyze_movement_pattern(game_id, bet_type)
