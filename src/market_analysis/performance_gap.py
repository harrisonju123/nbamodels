"""
Performance Gap Analysis

Compares backtested performance to live/paper trading results to detect overfitting
and validate model performance.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


@dataclass
class GapMetrics:
    """Performance metrics for a specific dimension."""

    dimension: str  # 'overall', 'spread', 'totals', etc.
    backtest_roi: float
    live_roi: float
    gap: float  # backtest - live
    gap_pct: float  # gap as percentage of backtest ROI
    backtest_win_rate: float
    live_win_rate: float
    backtest_sharpe: float
    live_sharpe: float
    backtest_n_bets: int
    live_n_bets: int


@dataclass
class PerformanceGapResult:
    """Complete performance gap analysis result."""

    backtest_roi: float
    live_roi: float
    gap: float
    gap_pct: float
    is_overfitting: bool
    confidence_interval: Tuple[float, float]
    by_strategy: Dict[str, GapMetrics]
    by_bet_type: Dict[str, GapMetrics]
    by_edge_bucket: Dict[str, GapMetrics]
    gap_trend: Optional[str]  # 'widening', 'stable', 'narrowing'


class BacktestLiveGapAnalyzer:
    """Analyze performance gap between backtest and live trading."""

    # Thresholds
    OVERFITTING_GAP_THRESHOLD = 0.05  # 5 percentage point gap
    SIGNIFICANT_GAP_THRESHOLD = 0.03  # 3 percentage point gap

    def __init__(self, db_path: str = "data/bets/bets.db"):
        """
        Initialize gap analyzer.

        Args:
            db_path: Path to bets database
        """
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_backtest_results(
        self,
        run_id: Optional[str] = None,
        strategy_type: Optional[str] = None,
        limit: int = 1,
    ) -> pd.DataFrame:
        """
        Get backtest results from database.

        Args:
            run_id: Specific backtest run ID
            strategy_type: Filter by strategy type
            limit: Number of recent runs to return

        Returns:
            DataFrame with backtest results
        """
        conn = self._get_connection()

        query = "SELECT * FROM backtest_results WHERE 1=1"
        params = []

        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)

        if strategy_type:
            query += " AND strategy_type = ?"
            params.append(strategy_type)

        query += f" ORDER BY run_date DESC LIMIT {limit}"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        return df

    def get_live_bets(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        strategy_type: Optional[str] = None,
        days: int = 90,
    ) -> pd.DataFrame:
        """
        Get live/paper trading bets from database.

        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            strategy_type: Filter by strategy type
            days: Number of days to look back (if dates not specified)

        Returns:
            DataFrame with live bets
        """
        conn = self._get_connection()

        query = """
            SELECT
                id,
                game_id,
                bet_type,
                bet_side,
                odds,
                line,
                bet_amount,
                edge,
                outcome,
                profit,
                strategy_type,
                logged_at,
                settled_at
            FROM bets
            WHERE outcome IS NOT NULL
                AND outcome != 'push'
        """

        params = []

        if start_date:
            query += " AND settled_at >= ?"
            params.append(start_date)

        if end_date:
            query += " AND settled_at <= ?"
            params.append(end_date)
        elif not start_date:
            # Default to last N days
            query += " AND settled_at >= date('now', ?)"
            params.append(f'-{days} days')

        if strategy_type:
            query += " AND strategy_type = ?"
            params.append(strategy_type)

        query += " ORDER BY settled_at ASC"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        return df

    def calculate_metrics(self, bets_df: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics for a set of bets.

        Args:
            bets_df: DataFrame with bet data

        Returns:
            Dict with performance metrics
        """
        if len(bets_df) == 0:
            return {
                'roi': 0.0,
                'win_rate': 0.0,
                'sharpe': 0.0,
                'total_profit': 0.0,
                'n_bets': 0,
            }

        # Win rate
        win_rate = (bets_df['outcome'] == 'win').mean()

        # ROI
        total_wagered = bets_df['bet_amount'].sum() if 'bet_amount' in bets_df.columns else len(bets_df)
        total_profit = bets_df['profit'].sum() if 'profit' in bets_df.columns else 0
        roi = total_profit / total_wagered if total_wagered > 0 else 0

        # Sharpe ratio (annualized)
        if 'profit' in bets_df.columns and 'bet_amount' in bets_df.columns:
            returns = bets_df['profit'] / bets_df['bet_amount']
            sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        else:
            sharpe = 0.0

        return {
            'roi': float(roi),
            'win_rate': float(win_rate),
            'sharpe': float(sharpe),
            'total_profit': float(total_profit),
            'n_bets': len(bets_df),
        }

    def calculate_gap(
        self,
        backtest_results: Optional[pd.DataFrame] = None,
        live_bets: Optional[pd.DataFrame] = None,
        run_id: Optional[str] = None,
        strategy_type: Optional[str] = None,
        live_days: int = 90,
    ) -> PerformanceGapResult:
        """
        Calculate performance gap between backtest and live trading.

        Args:
            backtest_results: Backtest results DataFrame (optional if run_id provided)
            live_bets: Live bets DataFrame (optional, will fetch if not provided)
            run_id: Backtest run ID to analyze
            strategy_type: Strategy type to filter
            live_days: Number of days of live data to analyze

        Returns:
            PerformanceGapResult object
        """
        # Get backtest results
        if backtest_results is None:
            backtest_results = self.get_backtest_results(
                run_id=run_id,
                strategy_type=strategy_type,
                limit=1,
            )

        if len(backtest_results) == 0:
            raise ValueError("No backtest results found")

        backtest_row = backtest_results.iloc[0]

        # Get live bets
        if live_bets is None:
            live_bets = self.get_live_bets(
                strategy_type=strategy_type,
                days=live_days,
            )

        # Calculate overall metrics
        live_metrics = self.calculate_metrics(live_bets)

        backtest_roi = backtest_row['roi']
        live_roi = live_metrics['roi']
        gap = backtest_roi - live_roi
        gap_pct = (gap / backtest_roi * 100) if backtest_roi != 0 else 0

        # Check for overfitting
        is_overfitting = gap >= self.OVERFITTING_GAP_THRESHOLD

        # Calculate confidence interval for gap
        # Using bootstrap for robust CI
        ci = self._calculate_gap_confidence_interval(
            backtest_roi,
            live_bets,
        )

        # Analyze by strategy
        by_strategy = {}
        if 'strategy_type' in live_bets.columns:
            for strat in live_bets['strategy_type'].unique():
                if pd.isna(strat):
                    continue
                strat_bets = live_bets[live_bets['strategy_type'] == strat]
                strat_metrics = self.calculate_metrics(strat_bets)

                by_strategy[strat] = GapMetrics(
                    dimension=strat,
                    backtest_roi=backtest_roi,  # May not have per-strategy backtest
                    live_roi=strat_metrics['roi'],
                    gap=backtest_roi - strat_metrics['roi'],
                    gap_pct=((backtest_roi - strat_metrics['roi']) / backtest_roi * 100)
                        if backtest_roi != 0 else 0,
                    backtest_win_rate=backtest_row.get('win_rate', 0),
                    live_win_rate=strat_metrics['win_rate'],
                    backtest_sharpe=backtest_row.get('sharpe_ratio', 0),
                    live_sharpe=strat_metrics['sharpe'],
                    backtest_n_bets=backtest_row.get('total_bets', 0),
                    live_n_bets=strat_metrics['n_bets'],
                )

        # Analyze by bet type
        by_bet_type = {}
        if 'bet_type' in live_bets.columns:
            for bet_type in live_bets['bet_type'].unique():
                if pd.isna(bet_type):
                    continue
                type_bets = live_bets[live_bets['bet_type'] == bet_type]
                type_metrics = self.calculate_metrics(type_bets)

                by_bet_type[bet_type] = GapMetrics(
                    dimension=bet_type,
                    backtest_roi=backtest_roi,
                    live_roi=type_metrics['roi'],
                    gap=backtest_roi - type_metrics['roi'],
                    gap_pct=((backtest_roi - type_metrics['roi']) / backtest_roi * 100)
                        if backtest_roi != 0 else 0,
                    backtest_win_rate=backtest_row.get('win_rate', 0),
                    live_win_rate=type_metrics['win_rate'],
                    backtest_sharpe=backtest_row.get('sharpe_ratio', 0),
                    live_sharpe=type_metrics['sharpe'],
                    backtest_n_bets=backtest_row.get('total_bets', 0),
                    live_n_bets=type_metrics['n_bets'],
                )

        # Analyze by edge bucket
        by_edge_bucket = {}
        if 'edge' in live_bets.columns:
            edge_buckets = [
                ('0-3%', 0.0, 0.03),
                ('3-5%', 0.03, 0.05),
                ('5-7%', 0.05, 0.07),
                ('7-10%', 0.07, 0.10),
                ('10%+', 0.10, 1.0),
            ]

            for label, low, high in edge_buckets:
                bucket_bets = live_bets[
                    (live_bets['edge'] >= low) & (live_bets['edge'] < high)
                ]
                if len(bucket_bets) == 0:
                    continue

                bucket_metrics = self.calculate_metrics(bucket_bets)

                by_edge_bucket[label] = GapMetrics(
                    dimension=label,
                    backtest_roi=backtest_roi,
                    live_roi=bucket_metrics['roi'],
                    gap=backtest_roi - bucket_metrics['roi'],
                    gap_pct=((backtest_roi - bucket_metrics['roi']) / backtest_roi * 100)
                        if backtest_roi != 0 else 0,
                    backtest_win_rate=backtest_row.get('win_rate', 0),
                    live_win_rate=bucket_metrics['win_rate'],
                    backtest_sharpe=backtest_row.get('sharpe_ratio', 0),
                    live_sharpe=bucket_metrics['sharpe'],
                    backtest_n_bets=backtest_row.get('total_bets', 0),
                    live_n_bets=bucket_metrics['n_bets'],
                )

        return PerformanceGapResult(
            backtest_roi=backtest_roi,
            live_roi=live_roi,
            gap=gap,
            gap_pct=gap_pct,
            is_overfitting=is_overfitting,
            confidence_interval=ci,
            by_strategy=by_strategy,
            by_bet_type=by_bet_type,
            by_edge_bucket=by_edge_bucket,
            gap_trend=None,  # Calculated separately
        )

    def _calculate_gap_confidence_interval(
        self,
        backtest_roi: float,
        live_bets: pd.DataFrame,
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for the gap using bootstrap.

        Args:
            backtest_roi: Backtest ROI
            live_bets: Live bets DataFrame
            confidence: Confidence level (default 0.95)
            n_bootstrap: Number of bootstrap samples

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(live_bets) == 0:
            return (0.0, 0.0)

        gaps = []

        for _ in range(n_bootstrap):
            # Resample live bets
            sample = live_bets.sample(n=len(live_bets), replace=True)
            sample_metrics = self.calculate_metrics(sample)
            gap = backtest_roi - sample_metrics['roi']
            gaps.append(gap)

        gaps = np.array(gaps)
        alpha = 1 - confidence
        lower = np.percentile(gaps, alpha / 2 * 100)
        upper = np.percentile(gaps, (1 - alpha / 2) * 100)

        return (float(lower), float(upper))

    def detect_overfitting(
        self,
        gap_threshold: float = 0.05,
        **kwargs,
    ) -> bool:
        """
        Detect if model is overfitting based on backtest vs live gap.

        Args:
            gap_threshold: Gap threshold for overfitting (default 5%)
            **kwargs: Arguments to pass to calculate_gap()

        Returns:
            True if overfitting detected
        """
        result = self.calculate_gap(**kwargs)
        return result.gap >= gap_threshold

    def analyze_gap_by_factor(
        self,
        factor: str = 'bet_type',
        **kwargs,
    ) -> pd.DataFrame:
        """
        Analyze gap breakdown by a specific factor.

        Args:
            factor: Factor to analyze by ('bet_type', 'strategy', 'edge_bucket')
            **kwargs: Arguments to pass to calculate_gap()

        Returns:
            DataFrame with gap analysis by factor
        """
        result = self.calculate_gap(**kwargs)

        if factor == 'bet_type':
            data = result.by_bet_type
        elif factor == 'strategy':
            data = result.by_strategy
        elif factor == 'edge_bucket':
            data = result.by_edge_bucket
        else:
            raise ValueError(f"Unknown factor: {factor}")

        rows = []
        for key, metrics in data.items():
            rows.append({
                factor: key,
                'backtest_roi': metrics.backtest_roi,
                'live_roi': metrics.live_roi,
                'gap': metrics.gap,
                'gap_pct': metrics.gap_pct,
                'backtest_win_rate': metrics.backtest_win_rate,
                'live_win_rate': metrics.live_win_rate,
                'live_n_bets': metrics.live_n_bets,
            })

        return pd.DataFrame(rows)

    def get_gap_trend(
        self,
        window: int = 30,
        strategy_type: Optional[str] = None,
    ) -> str:
        """
        Determine if gap is widening, stable, or narrowing over time.

        Args:
            window: Days per window to analyze
            strategy_type: Strategy type to filter

        Returns:
            'widening', 'stable', or 'narrowing'
        """
        # Get live bets for multiple time windows
        windows_data = []

        for i in range(3):  # 3 windows
            start_days = (i + 1) * window
            end_days = i * window if i > 0 else 0

            bets = self.get_live_bets(
                days=start_days,
                strategy_type=strategy_type,
            )

            # Filter to window
            if end_days > 0:
                end_date = (datetime.now() - timedelta(days=end_days)).isoformat()
                bets = bets[bets['settled_at'] < end_date]

            if len(bets) > 0:
                metrics = self.calculate_metrics(bets)
                windows_data.append({
                    'window': i,
                    'roi': metrics['roi'],
                    'n_bets': metrics['n_bets'],
                })

        if len(windows_data) < 2:
            return 'stable'  # Not enough data

        # Get most recent backtest
        backtest = self.get_backtest_results(strategy_type=strategy_type, limit=1)
        if len(backtest) == 0:
            return 'stable'

        backtest_roi = backtest.iloc[0]['roi']

        # Calculate gaps for each window
        gaps = [backtest_roi - w['roi'] for w in windows_data]

        # Check if gaps are increasing (widening) or decreasing (narrowing)
        if len(gaps) >= 2:
            # Simple trend: compare most recent to oldest
            if gaps[0] > gaps[-1] + 0.02:  # Recent gap larger
                return 'widening'
            elif gaps[-1] > gaps[0] + 0.02:  # Older gap larger
                return 'narrowing'

        return 'stable'
