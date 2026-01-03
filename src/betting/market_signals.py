"""
Market Microstructure Signals

Steam moves, RLM, sharp/public money flow detection for NBA betting.
"""

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import pandas as pd
import numpy as np
from loguru import logger


DB_PATH = "data/bets/bets.db"


# ========== Data Classes ==========


@dataclass
class SteamMove:
    """Detected steam move signal."""
    game_id: str
    detected_at: str
    bet_type: str           # 'moneyline', 'spread', 'totals'
    direction: str          # 'home', 'away', 'over', 'under'
    magnitude: float        # Line movement in points
    velocity: float         # Points per hour
    books_moved: int        # How many sportsbooks moved
    total_books: int        # Total books tracked
    confidence: float       # 0-1 confidence score

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'game_id': self.game_id,
            'detected_at': self.detected_at,
            'bet_type': self.bet_type,
            'direction': self.direction,
            'magnitude': self.magnitude,
            'velocity': self.velocity,
            'books_moved': self.books_moved,
            'total_books': self.total_books,
            'confidence': self.confidence
        }


@dataclass
class ReverseLineMove:
    """Detected reverse line movement signal."""
    game_id: str
    detected_at: str
    bet_type: str
    sharp_side: str         # Side sharps are betting (opposite of public)
    public_side: str        # Side getting most public action
    public_pct: float       # e.g., 0.70 = 70% on public side
    line_movement: float    # Movement toward sharp side
    confidence: float       # 0-1

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'game_id': self.game_id,
            'detected_at': self.detected_at,
            'bet_type': self.bet_type,
            'sharp_side': self.sharp_side,
            'public_side': self.public_side,
            'public_pct': self.public_pct,
            'line_movement': self.line_movement,
            'confidence': self.confidence
        }


@dataclass
class MoneyFlowSignal:
    """Sharp/public money flow analysis."""
    game_id: str
    bet_type: str
    side: str
    sharp_indicator: float  # -1 to +1 (positive = sharp on this side)
    public_indicator: float # -1 to +1 (positive = public on this side)
    alignment: str          # 'aligned', 'opposed', 'neutral'
    recommendation: str     # 'follow_sharp', 'fade_public', 'neutral'

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'game_id': self.game_id,
            'bet_type': self.bet_type,
            'side': self.side,
            'sharp_indicator': self.sharp_indicator,
            'public_indicator': self.public_indicator,
            'alignment': self.alignment,
            'recommendation': self.recommendation
        }


# ========== Steam Move Detector ==========


class SteamMoveDetector:
    """
    Detects steam moves from line snapshot history.

    Steam move criteria:
    1. 3+ sportsbooks move in same direction
    2. Movement >= 0.5 points within detection window
    3. Movement velocity > 0.5 pts/hr
    """

    # Thresholds calibrated for NBA
    MIN_BOOKS_MOVED = 3
    MIN_MOVEMENT_PTS = 0.5
    MIN_VELOCITY = 0.5  # pts/hour or implied prob%/hour
    MAX_DETECTION_WINDOW_HOURS = 2

    def __init__(self, db_path: str = DB_PATH):
        """Initialize detector."""
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def detect_steam_moves(
        self,
        game_id: str,
        lookback_hours: int = 6
    ) -> List[SteamMove]:
        """
        Detect steam moves for a specific game.

        Args:
            game_id: Game identifier
            lookback_hours: How far back to look for steam moves

        Returns:
            List of detected steam moves
        """
        conn = self._get_connection()

        # Get snapshots for this game
        cutoff = (datetime.now() - timedelta(hours=lookback_hours)).isoformat()

        snapshots = pd.read_sql_query("""
            SELECT
                snapshot_time,
                bet_type,
                side,
                bookmaker,
                line,
                odds,
                implied_prob
            FROM line_snapshots
            WHERE game_id = ?
            AND snapshot_time >= ?
            ORDER BY snapshot_time ASC
        """, conn, params=(game_id, cutoff))

        conn.close()

        if snapshots.empty:
            return []

        steam_moves = []

        # Check each bet type separately
        for bet_type in snapshots['bet_type'].unique():
            for side in snapshots[snapshots['bet_type'] == bet_type]['side'].unique():
                steam = self._detect_steam_for_side(
                    snapshots=snapshots[
                        (snapshots['bet_type'] == bet_type) &
                        (snapshots['side'] == side)
                    ],
                    game_id=game_id,
                    bet_type=bet_type,
                    side=side
                )
                if steam:
                    steam_moves.append(steam)

        return steam_moves

    def _detect_steam_for_side(
        self,
        snapshots: pd.DataFrame,
        game_id: str,
        bet_type: str,
        side: str
    ) -> Optional[SteamMove]:
        """
        Detect steam move for a specific bet_type and side.
        """
        if snapshots.empty:
            return None

        # Group by bookmaker
        books = snapshots.groupby('bookmaker')
        total_books = len(books)

        if total_books < self.MIN_BOOKS_MOVED:
            return None

        # Track bookmakers that moved significantly
        books_moved = []
        movements = []

        for bookmaker, book_snapshots in books:
            if len(book_snapshots) < 2:
                continue

            # Calculate movement over detection window
            book_snapshots = book_snapshots.sort_values('snapshot_time')
            recent = book_snapshots.iloc[-1]
            window_start = datetime.fromisoformat(recent['snapshot_time']) - \
                          timedelta(hours=self.MAX_DETECTION_WINDOW_HOURS)

            older_snaps = book_snapshots[
                book_snapshots['snapshot_time'] <= window_start.isoformat()
            ]

            if older_snaps.empty:
                continue

            older = older_snaps.iloc[-1]

            # Calculate movement based on bet type
            movement = self._calculate_movement(
                older_value=older.get('line') if bet_type in ['spread', 'totals'] else older.get('implied_prob'),
                recent_value=recent.get('line') if bet_type in ['spread', 'totals'] else recent.get('implied_prob'),
                bet_type=bet_type
            )

            if movement is None:
                continue

            # Check if movement is significant
            if abs(movement) >= self.MIN_MOVEMENT_PTS:
                # Calculate velocity
                time_diff = (
                    datetime.fromisoformat(recent['snapshot_time']) -
                    datetime.fromisoformat(older['snapshot_time'])
                ).total_seconds() / 3600

                if time_diff > 0:
                    velocity = movement / time_diff

                    if abs(velocity) >= self.MIN_VELOCITY:
                        books_moved.append(bookmaker)
                        movements.append(movement)

        # Check if enough books moved in same direction
        if len(books_moved) < self.MIN_BOOKS_MOVED:
            return None

        # Calculate average movement and direction
        avg_movement = np.mean(movements)

        # Calculate velocity with zero protection
        if self.MAX_DETECTION_WINDOW_HOURS > 0:
            avg_velocity = avg_movement / self.MAX_DETECTION_WINDOW_HOURS
        else:
            avg_velocity = 0

        # Confidence based on consensus (with zero protection)
        consensus_pct = len(books_moved) / total_books if total_books > 0 else 0
        if self.MIN_MOVEMENT_PTS > 0:
            confidence = min(1.0, consensus_pct * (abs(avg_movement) / self.MIN_MOVEMENT_PTS))
        else:
            confidence = consensus_pct

        return SteamMove(
            game_id=game_id,
            detected_at=datetime.now().isoformat(),
            bet_type=bet_type,
            direction=side,
            magnitude=abs(avg_movement),
            velocity=abs(avg_velocity),
            books_moved=len(books_moved),
            total_books=total_books,
            confidence=confidence
        )

    def _calculate_movement(
        self,
        older_value: Optional[float],
        recent_value: Optional[float],
        bet_type: str
    ) -> Optional[float]:
        """Calculate line movement."""
        if older_value is None or recent_value is None:
            return None

        if bet_type in ['spread', 'totals']:
            # For spreads/totals, use line value
            return recent_value - older_value
        else:
            # For moneyline, use implied probability (as percentage)
            return (recent_value - older_value) * 100

    def batch_detect_steam_moves(
        self,
        games_df: pd.DataFrame,
        lookback_hours: int = 6
    ) -> pd.DataFrame:
        """
        Batch detect steam moves for all games (optimized - single query).

        Args:
            games_df: DataFrame with game_id column
            lookback_hours: How far back to look for steam moves

        Returns:
            DataFrame with steam move signals
        """
        game_ids = games_df['game_id'].unique().tolist()
        if not game_ids:
            return pd.DataFrame()

        conn = self._get_connection()
        cutoff = (datetime.now() - timedelta(hours=lookback_hours)).isoformat()

        # Fetch all snapshots for all games in one query (no N+1)
        placeholders = ','.join('?' * len(game_ids))
        query = f"""
            SELECT
                snapshot_time,
                game_id,
                bet_type,
                side,
                bookmaker,
                line,
                odds,
                implied_prob
            FROM line_snapshots
            WHERE game_id IN ({placeholders})
            AND snapshot_time >= ?
            ORDER BY game_id, bet_type, side, snapshot_time ASC
        """

        snapshots = pd.read_sql_query(query, conn, params=(*game_ids, cutoff))
        conn.close()

        if snapshots.empty:
            return pd.DataFrame()

        # Process each game's snapshots
        results = []
        for game_id in game_ids:
            game_snapshots = snapshots[snapshots['game_id'] == game_id]
            if game_snapshots.empty:
                continue

            # Check each bet type separately
            for bet_type in game_snapshots['bet_type'].unique():
                for side in game_snapshots[game_snapshots['bet_type'] == bet_type]['side'].unique():
                    steam = self._detect_steam_for_side(
                        snapshots=game_snapshots[
                            (game_snapshots['bet_type'] == bet_type) &
                            (game_snapshots['side'] == side)
                        ],
                        game_id=game_id,
                        bet_type=bet_type,
                        side=side
                    )
                    if steam:
                        results.append(steam.to_dict())

        return pd.DataFrame(results)


# ========== Reverse Line Movement Detector ==========


class RLMDetector:
    """
    Detects reverse line movement.

    RLM criteria:
    1. Public betting heavily one side (>65%)
    2. Line moves toward the less-bet side
    3. Movement is significant (>= 0.5 pts)
    """

    PUBLIC_THRESHOLD = 0.65  # 65%+ on one side = heavy public action
    MIN_LINE_MOVEMENT = 0.5

    def __init__(self, db_path: str = DB_PATH):
        """Initialize detector."""
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def detect_rlm(
        self,
        game_id: str,
        bet_type: str = 'spread'
    ) -> Optional[ReverseLineMove]:
        """
        Detect RLM for a game.

        Since we don't have direct public betting data, we use Pinnacle
        as a sharp proxy. If the line moves opposite to Pinnacle's direction,
        we infer RLM.

        Args:
            game_id: Game identifier
            bet_type: Type of bet to check

        Returns:
            ReverseLineMove if detected, None otherwise
        """
        conn = self._get_connection()

        # Get opening and recent snapshots
        snapshots = pd.read_sql_query("""
            SELECT
                snapshot_time,
                side,
                bookmaker,
                line,
                implied_prob
            FROM line_snapshots
            WHERE game_id = ?
            AND bet_type = ?
            ORDER BY snapshot_time ASC
        """, conn, params=(game_id, bet_type))

        conn.close()

        if snapshots.empty or len(snapshots) < 10:
            return None

        # Separate sharp books from retail books
        sharp_books = ['pinnacle', 'circa', 'betcris']  # If available
        retail_books = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'pointsbet']

        sharp_snaps = snapshots[snapshots['bookmaker'].isin(sharp_books)]
        retail_snaps = snapshots[snapshots['bookmaker'].isin(retail_books)]

        if sharp_snaps.empty or retail_snaps.empty:
            # Fallback: use median movement direction
            return self._detect_rlm_median(snapshots, game_id, bet_type)

        # Calculate movement direction for sharp vs retail
        sharp_movement = self._calculate_aggregate_movement(sharp_snaps)
        retail_movement = self._calculate_aggregate_movement(retail_snaps)

        if sharp_movement is None or retail_movement is None:
            return None

        # RLM: Sharp and retail move in OPPOSITE directions
        if (sharp_movement > 0 > retail_movement) or (sharp_movement < 0 < retail_movement):
            # Sharp side is opposite of retail
            sharp_side = 'home' if sharp_movement > 0 else 'away'
            public_side = 'away' if sharp_side == 'home' else 'home'

            # Estimate public percentage based on retail movement magnitude
            public_pct = min(0.9, 0.65 + abs(retail_movement) * 0.1)

            # Calculate confidence with zero protection
            if self.MIN_LINE_MOVEMENT > 0:
                confidence = min(1.0, abs(sharp_movement - retail_movement) / self.MIN_LINE_MOVEMENT)
            else:
                confidence = 0.5  # Default moderate confidence

            return ReverseLineMove(
                game_id=game_id,
                detected_at=datetime.now().isoformat(),
                bet_type=bet_type,
                sharp_side=sharp_side,
                public_side=public_side,
                public_pct=public_pct,
                line_movement=abs(sharp_movement),
                confidence=confidence
            )

        return None

    def _calculate_aggregate_movement(self, snapshots: pd.DataFrame) -> Optional[float]:
        """Calculate aggregate line movement for a set of bookmakers."""
        if snapshots.empty or len(snapshots) < 2:
            return None

        # Get opening and recent snapshots
        opening = snapshots.iloc[0]
        recent = snapshots.iloc[-1]

        if 'line' in opening and opening['line'] is not None:
            return recent['line'] - opening['line']
        elif 'implied_prob' in opening and opening['implied_prob'] is not None:
            return (recent['implied_prob'] - opening['implied_prob']) * 100

        return None

    def _detect_rlm_median(
        self,
        snapshots: pd.DataFrame,
        game_id: str,
        bet_type: str
    ) -> Optional[ReverseLineMove]:
        """
        Fallback RLM detection using median movement.

        If most books move one direction but overall median moves opposite,
        infer RLM.
        """
        # Calculate movement per bookmaker
        movements = []
        for bookmaker in snapshots['bookmaker'].unique():
            book_snaps = snapshots[snapshots['bookmaker'] == bookmaker]
            if len(book_snaps) < 2:
                continue

            opening = book_snaps.iloc[0]
            recent = book_snaps.iloc[-1]

            if 'line' in opening and opening['line'] is not None:
                movement = recent['line'] - opening['line']
            elif 'implied_prob' in opening:
                movement = (recent['implied_prob'] - opening['implied_prob']) * 100
            else:
                continue

            movements.append(movement)

        if len(movements) < 3:
            return None

        median_movement = np.median(movements)
        mean_movement = np.mean(movements)

        # RLM: Median and mean diverge significantly
        if abs(median_movement - mean_movement) >= self.MIN_LINE_MOVEMENT:
            sharp_side = 'home' if median_movement > mean_movement else 'away'
            public_side = 'away' if sharp_side == 'home' else 'home'

            return ReverseLineMove(
                game_id=game_id,
                detected_at=datetime.now().isoformat(),
                bet_type=bet_type,
                sharp_side=sharp_side,
                public_side=public_side,
                public_pct=0.7,  # Assumed
                line_movement=abs(median_movement - mean_movement),
                confidence=0.6   # Lower confidence for fallback method
            )

        return None


# ========== Money Flow Analyzer ==========


class MoneyFlowAnalyzer:
    """
    Analyzes sharp vs public money flow.

    Without direct handle data, uses proxy signals:
    1. Pinnacle line vs retail book average
    2. Opening line vs current line trajectory
    3. Closing line value at different books
    """

    SHARP_BOOKS = ['pinnacle', 'circa', 'betcris']  # If available
    RETAIL_BOOKS = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'pointsbet']

    def __init__(self, db_path: str = DB_PATH):
        """Initialize analyzer."""
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def analyze_money_flow(
        self,
        game_id: str,
        bet_type: str,
        side: str
    ) -> Optional[MoneyFlowSignal]:
        """
        Analyze current money flow for a game.

        Args:
            game_id: Game identifier
            bet_type: Type of bet
            side: Side to analyze

        Returns:
            MoneyFlowSignal with sharp/public indicators
        """
        conn = self._get_connection()

        snapshots = pd.read_sql_query("""
            SELECT
                snapshot_time,
                bookmaker,
                line,
                implied_prob
            FROM line_snapshots
            WHERE game_id = ?
            AND bet_type = ?
            AND side = ?
            ORDER BY snapshot_time ASC
        """, conn, params=(game_id, bet_type, side))

        conn.close()

        if snapshots.empty:
            return None

        # Separate sharp and retail
        sharp_snaps = snapshots[snapshots['bookmaker'].isin(self.SHARP_BOOKS)]
        retail_snaps = snapshots[snapshots['bookmaker'].isin(self.RETAIL_BOOKS)]

        # Calculate indicators
        sharp_indicator = self._calculate_sharp_indicator(sharp_snaps, retail_snaps)
        public_indicator = -sharp_indicator  # Inverse relationship

        # Determine alignment
        if abs(sharp_indicator) < 0.3:
            alignment = 'neutral'
            recommendation = 'neutral'
        elif sharp_indicator > 0:
            alignment = 'sharp_favors'
            recommendation = 'follow_sharp'
        else:
            alignment = 'public_favors'
            recommendation = 'fade_public'

        return MoneyFlowSignal(
            game_id=game_id,
            bet_type=bet_type,
            side=side,
            sharp_indicator=sharp_indicator,
            public_indicator=public_indicator,
            alignment=alignment,
            recommendation=recommendation
        )

    def _calculate_sharp_indicator(
        self,
        sharp_snaps: pd.DataFrame,
        retail_snaps: pd.DataFrame
    ) -> float:
        """
        Calculate sharp money indicator (-1 to +1).

        Positive = sharp money on this side
        Negative = public money on this side
        """
        if sharp_snaps.empty or retail_snaps.empty:
            return 0.0

        # Compare latest lines
        sharp_recent = sharp_snaps.iloc[-1]
        retail_recent = retail_snaps.iloc[-1]

        # Use implied probability for comparison
        sharp_prob = sharp_recent.get('implied_prob', 0.5)
        retail_prob = retail_recent.get('implied_prob', 0.5)

        # If sharp prob > retail prob, sharp money on this side
        prob_diff = sharp_prob - retail_prob

        # Normalize to [-1, 1]
        indicator = np.clip(prob_diff * 10, -1, 1)

        return indicator


if __name__ == "__main__":
    # Test detection
    print("=== Market Signal Detectors ===\n")

    steam_detector = SteamMoveDetector()
    rlm_detector = RLMDetector()
    flow_analyzer = MoneyFlowAnalyzer()

    print("Detectors initialized successfully")
    print("Ready for signal detection")
