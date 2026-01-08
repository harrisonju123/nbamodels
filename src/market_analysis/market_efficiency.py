"""
Market Efficiency Analysis

Analyzes market efficiency through:
- Sharp vs public bookmaker divergence
- Edge decay and time-to-efficiency tracking
- Sharp signal performance validation
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


# Sharp bookmakers (lower margins, faster to adjust)
SHARP_BOOKS = {'pinnacle', 'circa', 'betcris'}

# Retail bookmakers (higher margins, slower to adjust)
RETAIL_BOOKS = {'draftkings', 'fanduel', 'betmgm', 'caesars', 'pointsbet', 'betrivers'}


@dataclass
class SharpPublicDivergence:
    """Divergence between sharp and public bookmakers."""

    game_id: str
    bet_type: str
    snapshot_time: str
    pinnacle_line: float
    pinnacle_odds: int
    retail_consensus_line: float
    retail_consensus_odds: int
    divergence: float  # In points
    divergence_pct: float
    direction: str  # 'pinnacle_higher', 'pinnacle_lower', 'aligned'
    sharp_side: Optional[str]  # Which side sharps are on
    profitable_to_follow: Optional[bool]  # Historical outcome


@dataclass
class SharpSignalPerformance:
    """Performance of following sharp signals."""

    signal_type: str  # 'pinnacle_divergence', 'rlm', 'steam'
    total_bets: int
    win_rate: float
    roi: float
    avg_clv: float
    profitable_periods: int
    unprofitable_periods: int
    confidence_score: float  # 0-1 score of signal reliability


@dataclass
class EdgeDecayPattern:
    """Pattern of how edges decay over time."""

    game_id: str
    bet_type: str
    initial_edge: float
    time_to_zero_edge: Optional[float]  # Hours
    time_to_half_edge: Optional[float]  # Hours
    decay_velocity: float  # Edge points per hour
    was_arbitraged: bool  # Did edge disappear quickly?
    was_correct_side: bool


class MarketEfficiencyAnalyzer:
    """Analyze sharp vs public markets and efficiency."""

    def __init__(
        self,
        db_path: str = "data/bets/bets.db",
        pinnacle_threshold: float = 0.5,
    ):
        """
        Initialize market efficiency analyzer.

        Args:
            db_path: Path to bets database
            pinnacle_threshold: Minimum divergence to consider significant (points)
        """
        self.db_path = db_path
        self.pinnacle_threshold = pinnacle_threshold

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def calculate_pinnacle_divergence(
        self,
        game_id: str,
        bet_type: str,
        snapshot_time: Optional[str] = None,
    ) -> Optional[SharpPublicDivergence]:
        """
        Calculate divergence between Pinnacle and retail consensus.

        Args:
            game_id: Game ID
            bet_type: Bet type (spread, totals, moneyline)
            snapshot_time: Specific snapshot time (defaults to most recent)

        Returns:
            SharpPublicDivergence object or None if insufficient data
        """
        conn = self._get_connection()

        # Get line snapshots for this game
        query = """
            SELECT bookmaker, line, odds, side, snapshot_time
            FROM line_snapshots
            WHERE game_id = ? AND bet_type = ?
        """
        params = [game_id, bet_type]

        if snapshot_time:
            query += " AND snapshot_time = ?"
            params.append(snapshot_time)
        else:
            # Get most recent snapshot
            query += " ORDER BY snapshot_time DESC LIMIT 100"

        snapshots = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if len(snapshots) == 0:
            return None

        # Normalize bookmaker names
        snapshots['bookmaker_normalized'] = snapshots['bookmaker'].str.lower()

        # Get Pinnacle line
        pinnacle_data = snapshots[snapshots['bookmaker_normalized'] == 'pinnacle']
        if len(pinnacle_data) == 0:
            return None

        # Get retail consensus (median of retail books)
        retail_data = snapshots[
            snapshots['bookmaker_normalized'].isin(RETAIL_BOOKS)
        ]

        if len(retail_data) == 0:
            return None

        # For spread/totals, compare lines. For moneyline, compare implied probabilities
        if bet_type in ['spread', 'totals']:
            # Get Pinnacle line (assume one side, e.g., home/over)
            pinn_line = pinnacle_data['line'].iloc[0] if len(pinnacle_data) > 0 else None
            pinn_odds = pinnacle_data['odds'].iloc[0] if len(pinnacle_data) > 0 else None

            # Get retail consensus line
            retail_line = retail_data['line'].median()
            retail_odds = int(retail_data['odds'].median())

            if pinn_line is None or retail_line is None:
                return None

            divergence = abs(pinn_line - retail_line)
            divergence_pct = (divergence / abs(retail_line) * 100) if retail_line != 0 else 0

            if pinn_line > retail_line:
                direction = 'pinnacle_higher'
                sharp_side = 'under' if bet_type == 'totals' else 'away'
            elif pinn_line < retail_line:
                direction = 'pinnacle_lower'
                sharp_side = 'over' if bet_type == 'totals' else 'home'
            else:
                direction = 'aligned'
                sharp_side = None

            return SharpPublicDivergence(
                game_id=game_id,
                bet_type=bet_type,
                snapshot_time=snapshot_time or pinnacle_data['snapshot_time'].iloc[0],
                pinnacle_line=float(pinn_line),
                pinnacle_odds=int(pinn_odds),
                retail_consensus_line=float(retail_line),
                retail_consensus_odds=retail_odds,
                divergence=divergence,
                divergence_pct=divergence_pct,
                direction=direction,
                sharp_side=sharp_side,
                profitable_to_follow=None,  # Requires game outcome
            )
        else:
            # Moneyline - compare implied probabilities
            # This would require odds-to-probability conversion
            return None  # Simplified for now

    def batch_calculate_divergences(
        self,
        game_ids: List[str],
        bet_type: str = 'spread',
    ) -> pd.DataFrame:
        """
        Calculate divergences for multiple games.

        Args:
            game_ids: List of game IDs
            bet_type: Bet type to analyze

        Returns:
            DataFrame with divergence data
        """
        results = []

        for game_id in game_ids:
            div = self.calculate_pinnacle_divergence(game_id, bet_type)
            if div:
                results.append({
                    'game_id': div.game_id,
                    'bet_type': div.bet_type,
                    'divergence': div.divergence,
                    'direction': div.direction,
                    'sharp_side': div.sharp_side,
                })

        return pd.DataFrame(results)

    def analyze_sharp_signal_performance(
        self,
        signal_type: str = 'sharp_aligned',
        lookback_days: int = 90,
    ) -> SharpSignalPerformance:
        """
        Analyze performance of following sharp signals.

        Args:
            signal_type: Signal to analyze ('sharp_aligned', 'steam_detected', 'rlm_detected')
            lookback_days: Days to look back

        Returns:
            SharpSignalPerformance object
        """
        conn = self._get_connection()

        # Get bets that followed this signal
        query = f"""
            SELECT
                outcome,
                profit,
                bet_amount,
                clv,
                settled_at
            FROM bets
            WHERE {signal_type} = 1
                AND outcome IS NOT NULL
                AND outcome != 'push'
                AND settled_at >= date('now', ?)
        """

        bets = pd.read_sql_query(
            query,
            conn,
            params=(f'-{lookback_days} days',),
        )
        conn.close()

        if len(bets) == 0:
            return SharpSignalPerformance(
                signal_type=signal_type,
                total_bets=0,
                win_rate=0.0,
                roi=0.0,
                avg_clv=0.0,
                profitable_periods=0,
                unprofitable_periods=0,
                confidence_score=0.0,
            )

        # Calculate metrics
        total_bets = len(bets)
        win_rate = (bets['outcome'] == 'win').mean()

        total_wagered = bets['bet_amount'].sum() if 'bet_amount' in bets.columns else len(bets)
        total_profit = bets['profit'].sum()
        roi = total_profit / total_wagered if total_wagered > 0 else 0

        avg_clv = bets['clv'].mean() if 'clv' in bets.columns else 0.0

        # Analyze by period (weekly)
        if 'settled_at' in bets.columns:
            bets['week'] = pd.to_datetime(bets['settled_at']).dt.to_period('W')
            weekly_roi = bets.groupby('week').apply(
                lambda x: x['profit'].sum() / x['bet_amount'].sum() if x['bet_amount'].sum() > 0 else 0
            )
            profitable_periods = (weekly_roi > 0).sum()
            unprofitable_periods = (weekly_roi <= 0).sum()
        else:
            profitable_periods = 0
            unprofitable_periods = 0

        # Calculate confidence score (combination of win rate, ROI, CLV)
        confidence_score = (
            0.4 * min(win_rate / 0.55, 1.0) +  # 55% is good
            0.4 * min(max(roi, 0) / 0.10, 1.0) +  # 10% ROI is excellent
            0.2 * min(max(avg_clv, 0) / 0.02, 1.0)  # 2% CLV is good
        )

        return SharpSignalPerformance(
            signal_type=signal_type,
            total_bets=total_bets,
            win_rate=float(win_rate),
            roi=float(roi),
            avg_clv=float(avg_clv),
            profitable_periods=int(profitable_periods),
            unprofitable_periods=int(unprofitable_periods),
            confidence_score=float(confidence_score),
        )

    def build_sharp_confidence_scores(self) -> Dict[str, float]:
        """
        Build confidence scores for each sharp signal type.

        Returns:
            Dict mapping signal type to confidence score (0-1)
        """
        signals = ['sharp_aligned', 'steam_detected', 'rlm_detected']
        scores = {}

        for signal in signals:
            perf = self.analyze_sharp_signal_performance(signal)
            scores[signal] = perf.confidence_score

        return scores

    def when_does_following_sharp_work(
        self,
        lookback_days: int = 90,
    ) -> Dict:
        """
        Analyze conditions when following sharp money works vs doesn't.

        Args:
            lookback_days: Days to analyze

        Returns:
            Dict with analysis by various conditions
        """
        conn = self._get_connection()

        query = """
            SELECT
                bet_type,
                edge,
                outcome,
                sharp_aligned,
                steam_detected,
                rlm_detected,
                profit,
                bet_amount
            FROM bets
            WHERE outcome IS NOT NULL
                AND outcome != 'push'
                AND settled_at >= date('now', ?)
        """

        bets = pd.read_sql_query(
            query,
            conn,
            params=(f'-{lookback_days} days',),
        )
        conn.close()

        results = {}

        # By edge size
        if 'edge' in bets.columns and 'sharp_aligned' in bets.columns:
            edge_buckets = [
                ('low_edge', 0.0, 0.05),
                ('medium_edge', 0.05, 0.10),
                ('high_edge', 0.10, 1.0),
            ]

            for label, low, high in edge_buckets:
                bucket_bets = bets[
                    (bets['edge'] >= low) & (bets['edge'] < high) & (bets['sharp_aligned'] == 1)
                ]

                if len(bucket_bets) > 0:
                    win_rate = (bucket_bets['outcome'] == 'win').mean()
                    results[f'sharp_with_{label}'] = {
                        'win_rate': float(win_rate),
                        'n_bets': len(bucket_bets),
                    }

        # By bet type
        if 'bet_type' in bets.columns:
            for bet_type in bets['bet_type'].unique():
                type_sharp = bets[
                    (bets['bet_type'] == bet_type) & (bets['sharp_aligned'] == 1)
                ]

                if len(type_sharp) > 0:
                    win_rate = (type_sharp['outcome'] == 'win').mean()
                    results[f'{bet_type}_sharp'] = {
                        'win_rate': float(win_rate),
                        'n_bets': len(type_sharp),
                    }

        return results


class TimeToEfficiencyTracker:
    """Track how quickly markets correct identified edges."""

    def __init__(self, db_path: str = "data/bets/bets.db"):
        """
        Initialize time-to-efficiency tracker.

        Args:
            db_path: Path to bets database
        """
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def track_edge_decay(
        self,
        game_id: str,
        bet_type: str,
        bet_side: str,
        initial_edge: float,
        initial_edge_time: str,
    ) -> EdgeDecayPattern:
        """
        Track how an edge decays over time from initial detection to game start.

        Note: This requires historical edge calculations at multiple timepoints.
        Implementation simplified - full version would recalculate edge at each snapshot.

        Args:
            game_id: Game ID
            bet_type: Bet type
            bet_side: Bet side
            initial_edge: Initial edge when detected
            initial_edge_time: Time when edge was detected

        Returns:
            EdgeDecayPattern object
        """
        conn = self._get_connection()

        # Get line snapshots after initial detection
        query = """
            SELECT snapshot_time, line, odds
            FROM line_snapshots
            WHERE game_id = ? AND bet_type = ? AND side = ?
                AND snapshot_time >= ?
            ORDER BY snapshot_time ASC
        """

        snapshots = pd.read_sql_query(
            query,
            conn,
            params=(game_id, bet_type, bet_side, initial_edge_time),
        )

        # Check bet outcome
        bet_query = """
            SELECT outcome FROM bets
            WHERE game_id = ? AND bet_type = ? AND bet_side = ?
            LIMIT 1
        """
        bet_result = conn.execute(bet_query, (game_id, bet_type, bet_side)).fetchone()
        was_correct = bet_result and bet_result['outcome'] == 'win'

        conn.close()

        # Simplified: assume edge decays linearly with line movement
        # Real implementation would recalculate model edge at each snapshot

        if len(snapshots) == 0:
            return EdgeDecayPattern(
                game_id=game_id,
                bet_type=bet_type,
                initial_edge=initial_edge,
                time_to_zero_edge=None,
                time_to_half_edge=None,
                decay_velocity=0.0,
                was_arbitraged=False,
                was_correct_side=was_correct,
            )

        # Calculate time to half edge (rough estimate)
        initial_time = pd.to_datetime(initial_edge_time)
        final_time = pd.to_datetime(snapshots.iloc[-1]['snapshot_time'])
        hours_elapsed = (final_time - initial_time).total_seconds() / 3600

        # Assume linear decay for simplicity
        decay_velocity = initial_edge / hours_elapsed if hours_elapsed > 0 else 0

        # Fast decay = arbitraged
        was_arbitraged = decay_velocity > 0.01  # >1% per hour

        return EdgeDecayPattern(
            game_id=game_id,
            bet_type=bet_type,
            initial_edge=initial_edge,
            time_to_zero_edge=hours_elapsed if initial_edge > 0.01 else None,
            time_to_half_edge=hours_elapsed / 2 if initial_edge > 0.01 else None,
            decay_velocity=float(decay_velocity),
            was_arbitraged=was_arbitraged,
            was_correct_side=was_correct,
        )

    def get_avg_time_to_efficiency(
        self,
        bet_type: Optional[str] = None,
        edge_bucket: Optional[str] = None,
    ) -> float:
        """
        Get average time for edges to disappear.

        Args:
            bet_type: Filter by bet type
            edge_bucket: Filter by edge bucket ('low', 'medium', 'high')

        Returns:
            Average hours to efficiency
        """
        conn = self._get_connection()

        query = "SELECT AVG(time_to_zero_edge_hrs) as avg_time FROM edge_decay_tracking WHERE 1=1"
        params = []

        if bet_type:
            query += " AND bet_type = ?"
            params.append(bet_type)

        result = conn.execute(query, params).fetchone()
        conn.close()

        return result['avg_time'] if result and result['avg_time'] else 0.0

    def are_edges_being_arbitraged_faster(
        self,
        lookback_periods: List[int] = [30, 60, 90],
    ) -> Dict:
        """
        Check if edges are disappearing faster over time.

        Args:
            lookback_periods: Periods to compare (days)

        Returns:
            Dict with comparison across periods
        """
        conn = self._get_connection()

        results = {}

        for days in lookback_periods:
            query = """
                SELECT AVG(time_to_zero_edge_hrs) as avg_time, AVG(decay_velocity) as avg_velocity
                FROM edge_decay_tracking
                WHERE created_at >= date('now', ?)
            """

            result = conn.execute(query, (f'-{days} days',)).fetchone()

            results[f'last_{days}_days'] = {
                'avg_time_to_zero': result['avg_time'] if result else None,
                'avg_decay_velocity': result['avg_velocity'] if result else None,
            }

        conn.close()

        return results

    def get_decay_pattern_by_market(self) -> pd.DataFrame:
        """
        Get edge decay patterns by market type.

        Returns:
            DataFrame with decay stats by bet type
        """
        conn = self._get_connection()

        query = """
            SELECT
                bet_type,
                AVG(time_to_half_edge_hrs) as avg_half_life,
                AVG(decay_velocity) as avg_velocity,
                SUM(CASE WHEN was_arbitraged THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as arbitrage_rate,
                COUNT(*) as n_samples
            FROM edge_decay_tracking
            GROUP BY bet_type
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df
