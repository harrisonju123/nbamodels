"""
Optimal Bet Timing Analysis

Analyzes historical bet timing patterns to identify optimal windows for placing bets
and provides counterfactual analysis for timing decisions.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class TimingWindow:
    """Optimal timing window for betting."""

    hour_of_day: Optional[int]  # 0-23
    day_of_week: Optional[int]  # 0-6 (Monday=0)
    hours_before_game: Optional[int]
    avg_clv: float
    positive_clv_rate: float
    avg_roi: float
    n_bets: int
    is_optimal: bool


class TimingWindowsAnalyzer:
    """Find optimal betting timing windows."""

    def __init__(self, db_path: str = "data/bets/bets.db"):
        """
        Initialize timing windows analyzer.

        Args:
            db_path: Path to bets database
        """
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_bets_with_timing(
        self,
        lookback_days: int = 90,
        bet_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get bets with timing information.

        Args:
            lookback_days: Days to look back
            bet_type: Filter by bet type

        Returns:
            DataFrame with timing columns
        """
        conn = self._get_connection()

        query = """
            SELECT
                id,
                bet_type,
                clv,
                profit,
                bet_amount,
                outcome,
                logged_at,
                commence_time,
                booked_hours_before
            FROM bets
            WHERE outcome IS NOT NULL
                AND outcome != 'push'
                AND clv IS NOT NULL
                AND logged_at >= date('now', ?)
        """

        params = [f'-{lookback_days} days']

        if bet_type:
            query += " AND bet_type = ?"
            params.append(bet_type)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if len(df) == 0:
            return df

        # Parse datetime columns
        df['logged_at'] = pd.to_datetime(df['logged_at'])
        df['commence_time'] = pd.to_datetime(df['commence_time'])

        # Extract timing features
        df['hour_of_day'] = df['logged_at'].dt.hour
        df['day_of_week'] = df['logged_at'].dt.dayofweek

        # Calculate hours before game if not present
        if 'booked_hours_before' not in df.columns or df['booked_hours_before'].isna().all():
            df['booked_hours_before'] = (
                (df['commence_time'] - df['logged_at']).dt.total_seconds() / 3600
            )

        # Calculate ROI
        df['roi'] = df['profit'] / df['bet_amount']

        return df

    def analyze_by_hour_of_day(
        self,
        bet_type: Optional[str] = None,
        lookback_days: int = 90,
    ) -> pd.DataFrame:
        """
        Analyze performance by hour of day.

        Args:
            bet_type: Filter by bet type
            lookback_days: Days to look back

        Returns:
            DataFrame with stats by hour
        """
        df = self.get_bets_with_timing(lookback_days, bet_type)

        if len(df) == 0:
            return pd.DataFrame()

        stats = df.groupby('hour_of_day').agg({
            'clv': ['mean', 'std', lambda x: (x > 0).mean()],
            'roi': 'mean',
            'id': 'count',
        }).reset_index()

        stats.columns = ['hour', 'avg_clv', 'clv_std', 'positive_clv_rate', 'avg_roi', 'n_bets']

        return stats

    def analyze_by_day_of_week(
        self,
        bet_type: Optional[str] = None,
        lookback_days: int = 90,
    ) -> pd.DataFrame:
        """
        Analyze performance by day of week.

        Args:
            bet_type: Filter by bet type
            lookback_days: Days to look back

        Returns:
            DataFrame with stats by day of week
        """
        df = self.get_bets_with_timing(lookback_days, bet_type)

        if len(df) == 0:
            return pd.DataFrame()

        stats = df.groupby('day_of_week').agg({
            'clv': ['mean', lambda x: (x > 0).mean()],
            'roi': 'mean',
            'id': 'count',
        }).reset_index()

        stats.columns = ['day_of_week', 'avg_clv', 'positive_clv_rate', 'avg_roi', 'n_bets']

        # Add day names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        stats['day_name'] = stats['day_of_week'].apply(lambda x: day_names[int(x)])

        return stats

    def analyze_by_hours_before_game(
        self,
        bet_type: Optional[str] = None,
        lookback_days: int = 90,
        buckets: Optional[List[Tuple[int, int]]] = None,
    ) -> pd.DataFrame:
        """
        Analyze performance by hours before game start.

        Args:
            bet_type: Filter by bet type
            lookback_days: Days to look back
            buckets: Time buckets (default: [(0,1), (1,4), (4,12), (12,24), (24,48)])

        Returns:
            DataFrame with stats by time bucket
        """
        df = self.get_bets_with_timing(lookback_days, bet_type)

        if len(df) == 0:
            return pd.DataFrame()

        if buckets is None:
            buckets = [(0, 1), (1, 4), (4, 12), (12, 24), (24, 48), (48, 999)]

        results = []

        for low, high in buckets:
            bucket_df = df[
                (df['booked_hours_before'] >= low) & (df['booked_hours_before'] < high)
            ]

            if len(bucket_df) == 0:
                continue

            results.append({
                'hours_before_low': low,
                'hours_before_high': high,
                'label': f'{low}-{high}h' if high < 999 else f'{low}h+',
                'avg_clv': bucket_df['clv'].mean(),
                'positive_clv_rate': (bucket_df['clv'] > 0).mean(),
                'avg_roi': bucket_df['roi'].mean(),
                'n_bets': len(bucket_df),
            })

        return pd.DataFrame(results)

    def find_optimal_windows(
        self,
        min_sample_size: int = 20,
        bet_type: Optional[str] = None,
    ) -> List[TimingWindow]:
        """
        Find optimal timing windows across all dimensions.

        Args:
            min_sample_size: Minimum bets required
            bet_type: Filter by bet type

        Returns:
            List of optimal timing windows
        """
        windows = []

        # By hour of day
        hour_stats = self.analyze_by_hour_of_day(bet_type=bet_type)

        if len(hour_stats) > 0:
            hour_stats_filtered = hour_stats[hour_stats['n_bets'] >= min_sample_size]

            if len(hour_stats_filtered) > 0:
                best_hour = hour_stats_filtered.loc[hour_stats_filtered['avg_clv'].idxmax()]

                windows.append(TimingWindow(
                    hour_of_day=int(best_hour['hour']),
                    day_of_week=None,
                    hours_before_game=None,
                    avg_clv=float(best_hour['avg_clv']),
                    positive_clv_rate=float(best_hour['positive_clv_rate']),
                    avg_roi=float(best_hour['avg_roi']),
                    n_bets=int(best_hour['n_bets']),
                    is_optimal=True,
                ))

        # By hours before game
        hours_before_stats = self.analyze_by_hours_before_game(bet_type=bet_type)

        if len(hours_before_stats) > 0:
            filtered = hours_before_stats[hours_before_stats['n_bets'] >= min_sample_size]

            if len(filtered) > 0:
                best_timing = filtered.loc[filtered['avg_clv'].idxmax()]

                windows.append(TimingWindow(
                    hour_of_day=None,
                    day_of_week=None,
                    hours_before_game=int(best_timing['hours_before_low']),
                    avg_clv=float(best_timing['avg_clv']),
                    positive_clv_rate=float(best_timing['positive_clv_rate']),
                    avg_roi=float(best_timing['avg_roi']),
                    n_bets=int(best_timing['n_bets']),
                    is_optimal=True,
                ))

        return windows

    def get_inefficiency_score_by_time(
        self,
        bet_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Calculate market inefficiency score by time.

        Args:
            bet_type: Filter by bet type

        Returns:
            DataFrame with inefficiency scores
        """
        hours_stats = self.analyze_by_hours_before_game(bet_type=bet_type)

        if len(hours_stats) == 0:
            return pd.DataFrame()

        # Inefficiency score = avg_clv * positive_clv_rate
        hours_stats['inefficiency_score'] = (
            hours_stats['avg_clv'] * hours_stats['positive_clv_rate']
        )

        return hours_stats.sort_values('inefficiency_score', ascending=False)


class HistoricalTimingAnalyzer:
    """Deep analysis of historical bet timing patterns."""

    def __init__(self, db_path: str = "data/bets/bets.db"):
        """
        Initialize historical timing analyzer.

        Args:
            db_path: Path to bets database
        """
        self.db_path = db_path
        self.windows_analyzer = TimingWindowsAnalyzer(db_path)

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def analyze_clv_by_booking_time(
        self,
        lookback_days: int = 90,
    ) -> pd.DataFrame:
        """
        Analyze CLV by booking time with statistical significance.

        Args:
            lookback_days: Days to look back

        Returns:
            DataFrame with detailed CLV analysis
        """
        df = self.windows_analyzer.get_bets_with_timing(lookback_days=lookback_days)

        if len(df) == 0:
            return pd.DataFrame()

        # Bucket by hours before
        buckets = [(0, 1), (1, 4), (4, 12), (12, 24), (24, 48)]
        df['timing_bucket'] = pd.cut(
            df['booked_hours_before'],
            bins=[b[0] for b in buckets] + [buckets[-1][1]],
            labels=[f'{b[0]}-{b[1]}h' for b in buckets],
        )

        stats = df.groupby('timing_bucket').agg({
            'clv': ['mean', 'median', 'std', lambda x: (x > 0).mean()],
            'clv_at_1hr': 'mean',
            'clv_at_4hr': 'mean',
            'clv_at_12hr': 'mean',
            'clv_at_24hr': 'mean',
            'roi': 'mean',
            'id': 'count',
        })

        stats.columns = [
            'avg_clv', 'median_clv', 'clv_std', 'positive_clv_rate',
            'avg_clv_1hr', 'avg_clv_4hr', 'avg_clv_12hr', 'avg_clv_24hr',
            'avg_roi', 'n_bets',
        ]

        return stats.reset_index()

    def optimal_timing_by_bet_type(self) -> Dict[str, int]:
        """
        Get optimal hours before game for each bet type.

        Returns:
            Dict mapping bet_type to optimal hours before game
        """
        optimal = {}

        for bet_type in ['spread', 'totals', 'moneyline']:
            windows = self.windows_analyzer.find_optimal_windows(bet_type=bet_type)

            # Find window with hours_before_game specified
            for window in windows:
                if window.hours_before_game is not None:
                    optimal[bet_type] = window.hours_before_game
                    break

        return optimal

    def optimal_timing_by_team(
        self,
        team: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Get optimal timing by team (simplified - would need team filtering in DB).

        Args:
            team: Team name to analyze

        Returns:
            Dict with optimal timing
        """
        # Simplified implementation
        # Full version would query bets filtered by team
        return {}

    def counterfactual_analysis(self, bet_id: str) -> Dict:
        """
        What-if analysis: what if this bet was placed at different time?

        Args:
            bet_id: Bet ID to analyze

        Returns:
            Dict with counterfactual CLV at different times
        """
        conn = self._get_connection()

        query = """
            SELECT
                clv,
                clv_at_1hr,
                clv_at_4hr,
                clv_at_12hr,
                clv_at_24hr,
                booked_hours_before,
                optimal_book_time
            FROM bets
            WHERE id = ?
        """

        bet = conn.execute(query, (bet_id,)).fetchone()
        conn.close()

        if not bet:
            return {}

        actual_clv = bet['clv'] or 0.0
        actual_hours = bet['booked_hours_before'] or 0.0

        counterfactuals = {
            'actual_timing': {
                'hours_before': actual_hours,
                'clv': actual_clv,
            },
            'at_1hr': {
                'hours_before': 1.0,
                'clv': bet['clv_at_1hr'] or 0.0,
                'improvement': (bet['clv_at_1hr'] or 0.0) - actual_clv,
            },
            'at_4hr': {
                'hours_before': 4.0,
                'clv': bet['clv_at_4hr'] or 0.0,
                'improvement': (bet['clv_at_4hr'] or 0.0) - actual_clv,
            },
            'at_12hr': {
                'hours_before': 12.0,
                'clv': bet['clv_at_12hr'] or 0.0,
                'improvement': (bet['clv_at_12hr'] or 0.0) - actual_clv,
            },
            'at_24hr': {
                'hours_before': 24.0,
                'clv': bet['clv_at_24hr'] or 0.0,
                'improvement': (bet['clv_at_24hr'] or 0.0) - actual_clv,
            },
        }

        # Find best timing
        best_time = max(
            counterfactuals.items(),
            key=lambda x: x[1].get('clv', 0.0) if x[0] != 'actual_timing' else -999,
        )

        counterfactuals['optimal'] = best_time[1]

        return counterfactuals

    def timing_value_estimate(self, lookback_days: int = 90) -> float:
        """
        Estimate CLV value gain from perfect timing.

        Args:
            lookback_days: Days to analyze

        Returns:
            Average CLV improvement from optimal timing
        """
        df = self.windows_analyzer.get_bets_with_timing(lookback_days=lookback_days)

        if len(df) == 0:
            return 0.0

        # For bets with multi-snapshot CLV, calculate max CLV - actual CLV
        clv_columns = ['clv_at_1hr', 'clv_at_4hr', 'clv_at_12hr', 'clv_at_24hr']
        available_clv_cols = [col for col in clv_columns if col in df.columns]

        if len(available_clv_cols) == 0:
            return 0.0

        df['max_possible_clv'] = df[available_clv_cols].max(axis=1)
        df['timing_value'] = df['max_possible_clv'] - df['clv']

        avg_value = df['timing_value'].mean()

        return float(avg_value) if not pd.isna(avg_value) else 0.0
