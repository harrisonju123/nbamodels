"""
Alpha Monitoring System

Central monitoring for tracking betting performance, detecting edge decay,
and providing early warning of model degradation.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


@dataclass
class PerformanceMetrics:
    """Container for performance metrics at a point in time."""
    timestamp: str
    window_size: int
    n_bets: int
    win_rate: float
    roi: float
    clv: float
    clv_std: float
    sharpe_ratio: float
    max_drawdown: float
    avg_edge: float
    positive_clv_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


class AlphaMonitor:
    """
    Central monitoring system for betting edge health.

    Tracks rolling performance metrics, detects decay patterns,
    and provides early warning of edge degradation.
    """

    # Break-even win rate at -110 odds (accounting for vig)
    BREAK_EVEN_WIN_RATE = 0.524

    # Thresholds for alerts
    CLV_DECAY_THRESHOLD = -0.005  # -0.5% CLV triggers warning
    WIN_RATE_DECAY_THRESHOLD = 0.50  # Below 50% is concerning
    SHARPE_MIN_THRESHOLD = 0.0  # Negative Sharpe is bad

    def __init__(
        self,
        lookback_windows: List[int] = None,
        baseline_path: str = None,
    ):
        """
        Initialize AlphaMonitor.

        Args:
            lookback_windows: Windows for rolling calculations (default: [20, 50, 100])
            baseline_path: Path to saved baseline metrics for comparison
        """
        self.windows = lookback_windows or [20, 50, 100]
        self.baseline_path = baseline_path or "data/monitoring/baseline_metrics.json"
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        self._load_baseline()

    def _load_baseline(self):
        """Load baseline metrics if available."""
        if os.path.exists(self.baseline_path):
            try:
                with open(self.baseline_path, 'r') as f:
                    data = json.load(f)
                self.baseline_metrics = PerformanceMetrics(**data)
                logger.info(f"Loaded baseline metrics from {self.baseline_path}")
            except Exception as e:
                logger.warning(f"Could not load baseline: {e}")

    def save_baseline(self, metrics: PerformanceMetrics):
        """Save current metrics as baseline for future comparison."""
        os.makedirs(os.path.dirname(self.baseline_path), exist_ok=True)
        with open(self.baseline_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        self.baseline_metrics = metrics
        logger.info(f"Saved baseline metrics to {self.baseline_path}")

    def get_rolling_metrics(
        self,
        bets_df: pd.DataFrame,
        window: int = 50,
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.

        Args:
            bets_df: DataFrame with bet history (from bet_tracker)
            window: Rolling window size

        Returns:
            DataFrame with rolling metrics over time
        """
        if bets_df.empty or len(bets_df) < window:
            return pd.DataFrame()

        # Ensure sorted by time
        df = bets_df.sort_values('logged_at').reset_index(drop=True)

        # Only settled bets
        settled = df[df['outcome'].notna()].copy()
        if len(settled) < window:
            return pd.DataFrame()

        # Calculate rolling metrics
        settled['is_win'] = (settled['outcome'] == 'win').astype(int)
        # Safe division: replace 0 bet_amount with 100 to avoid division by zero
        bet_amounts = settled['bet_amount'].fillna(100).replace(0, 100)
        settled['profit_pct'] = settled['profit'] / bet_amounts

        results = []
        for i in range(window, len(settled) + 1):
            window_df = settled.iloc[i-window:i]

            metrics = {
                'date': window_df['settled_at'].iloc[-1],
                'bet_index': i,
                'window': window,
                'n_bets': len(window_df),
                'win_rate': window_df['is_win'].mean(),
                'roi': window_df['profit_pct'].mean(),
                'clv': window_df['clv'].mean() if 'clv' in window_df.columns else 0,
                'clv_std': window_df['clv'].std() if 'clv' in window_df.columns else 0,
                'avg_edge': window_df['edge'].mean() if 'edge' in window_df.columns else 0,
                'total_profit': window_df['profit'].sum(),
            }

            # Calculate Sharpe ratio (annualized)
            if metrics['roi'] != 0 and window_df['profit_pct'].std() > 0:
                # Assume ~3 bets per day on average
                bets_per_year = 3 * 365
                daily_return = metrics['roi']
                daily_vol = window_df['profit_pct'].std()
                metrics['sharpe'] = (daily_return / daily_vol) * np.sqrt(bets_per_year / window)
            else:
                metrics['sharpe'] = 0

            results.append(metrics)

        return pd.DataFrame(results)

    def get_current_metrics(
        self,
        bets_df: pd.DataFrame,
        window: int = 50,
    ) -> PerformanceMetrics:
        """
        Get current performance metrics for a specific window.

        Args:
            bets_df: DataFrame with bet history
            window: Lookback window

        Returns:
            PerformanceMetrics object
        """
        settled = bets_df[bets_df['outcome'].notna()].copy()

        if len(settled) < window:
            window = len(settled)

        if window == 0:
            return PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                window_size=0,
                n_bets=0,
                win_rate=0.0,
                roi=0.0,
                clv=0.0,
                clv_std=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                avg_edge=0.0,
                positive_clv_rate=0.0,
            )

        recent = settled.sort_values('logged_at').tail(window)

        # Win rate
        wins = (recent['outcome'] == 'win').sum()
        losses = (recent['outcome'] == 'loss').sum()
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

        # ROI
        total_wagered = recent['bet_amount'].fillna(100).sum()
        total_profit = recent['profit'].sum()
        roi = total_profit / total_wagered if total_wagered > 0 else 0

        # CLV metrics
        clv = recent['clv'].mean() if 'clv' in recent.columns and recent['clv'].notna().any() else 0
        clv_std = recent['clv'].std() if 'clv' in recent.columns and recent['clv'].notna().any() else 0
        positive_clv_rate = (recent['clv'] > 0).mean() if 'clv' in recent.columns else 0

        # Sharpe ratio
        profit_pct = recent['profit'] / recent['bet_amount'].fillna(100)
        if profit_pct.std() > 0:
            sharpe = (profit_pct.mean() / profit_pct.std()) * np.sqrt(len(recent))
        else:
            sharpe = 0

        # Max drawdown
        cumulative = recent['profit'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max)
        max_dd = drawdown.min() / total_wagered if total_wagered > 0 else 0

        # Average edge
        avg_edge = recent['edge'].mean() if 'edge' in recent.columns else 0

        return PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            window_size=window,
            n_bets=len(recent),
            win_rate=win_rate,
            roi=roi,
            clv=clv,
            clv_std=clv_std,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            avg_edge=avg_edge,
            positive_clv_rate=positive_clv_rate,
        )

    def detect_performance_decay(
        self,
        bets_df: pd.DataFrame,
        window: int = 50,
    ) -> List[Dict]:
        """
        Detect signs of performance decay.

        Returns list of detected issues with severity levels.
        """
        alerts = []
        metrics = self.get_current_metrics(bets_df, window)

        if metrics.n_bets < 20:
            return [{'severity': 'info', 'message': f'Insufficient bets ({metrics.n_bets}) for decay detection'}]

        # Check CLV
        if metrics.clv < self.CLV_DECAY_THRESHOLD:
            alerts.append({
                'severity': 'warning' if metrics.clv > -0.01 else 'critical',
                'category': 'clv',
                'message': f'CLV below threshold: {metrics.clv:.2%} (threshold: {self.CLV_DECAY_THRESHOLD:.2%})',
                'value': metrics.clv,
                'threshold': self.CLV_DECAY_THRESHOLD,
            })

        # Check win rate
        if metrics.win_rate < self.WIN_RATE_DECAY_THRESHOLD:
            alerts.append({
                'severity': 'warning',
                'category': 'win_rate',
                'message': f'Win rate below 50%: {metrics.win_rate:.1%}',
                'value': metrics.win_rate,
                'threshold': self.WIN_RATE_DECAY_THRESHOLD,
            })

        if metrics.win_rate < self.BREAK_EVEN_WIN_RATE:
            alerts.append({
                'severity': 'critical' if metrics.win_rate < 0.48 else 'warning',
                'category': 'win_rate',
                'message': f'Win rate below break-even ({self.BREAK_EVEN_WIN_RATE:.1%}): {metrics.win_rate:.1%}',
                'value': metrics.win_rate,
                'threshold': self.BREAK_EVEN_WIN_RATE,
            })

        # Check Sharpe ratio
        if metrics.sharpe_ratio < self.SHARPE_MIN_THRESHOLD:
            alerts.append({
                'severity': 'warning',
                'category': 'sharpe',
                'message': f'Negative Sharpe ratio: {metrics.sharpe_ratio:.2f}',
                'value': metrics.sharpe_ratio,
                'threshold': self.SHARPE_MIN_THRESHOLD,
            })

        # Check positive CLV rate
        if metrics.positive_clv_rate < 0.45:
            alerts.append({
                'severity': 'warning',
                'category': 'clv_rate',
                'message': f'Low positive CLV rate: {metrics.positive_clv_rate:.1%}',
                'value': metrics.positive_clv_rate,
                'threshold': 0.45,
            })

        # Compare to baseline if available
        if self.baseline_metrics and self.baseline_metrics.n_bets >= 50:
            baseline_comparison = self._compare_to_baseline(metrics)
            alerts.extend(baseline_comparison)

        if not alerts:
            alerts.append({
                'severity': 'info',
                'category': 'health',
                'message': 'All metrics within acceptable ranges',
                'metrics': metrics.to_dict(),
            })

        return alerts

    def _compare_to_baseline(self, current: PerformanceMetrics) -> List[Dict]:
        """Compare current metrics to baseline and flag significant drops."""
        alerts = []
        baseline = self.baseline_metrics

        # CLV drop
        if baseline.clv > 0 and current.clv < baseline.clv * 0.5:
            alerts.append({
                'severity': 'warning',
                'category': 'baseline_comparison',
                'message': f'CLV dropped >50% from baseline: {current.clv:.2%} vs {baseline.clv:.2%}',
                'current': current.clv,
                'baseline': baseline.clv,
            })

        # Win rate drop
        if current.win_rate < baseline.win_rate - 0.05:
            alerts.append({
                'severity': 'warning',
                'category': 'baseline_comparison',
                'message': f'Win rate dropped >5% from baseline: {current.win_rate:.1%} vs {baseline.win_rate:.1%}',
                'current': current.win_rate,
                'baseline': baseline.win_rate,
            })

        return alerts

    def get_clv_trend(
        self,
        bets_df: pd.DataFrame,
        window: int = 100,
    ) -> Dict:
        """
        Analyze CLV trend over time.

        Returns trend direction, slope, and prediction.
        """
        settled = bets_df[bets_df['outcome'].notna()].copy()

        if 'clv' not in settled.columns or len(settled) < window:
            return {
                'direction': 'insufficient_data',
                'slope': 0,
                'r_squared': 0,
                'days_to_zero': None,
            }

        recent = settled.sort_values('logged_at').tail(window)
        clv_values = recent['clv'].dropna()

        if len(clv_values) < 20:
            return {
                'direction': 'insufficient_data',
                'slope': 0,
                'r_squared': 0,
                'days_to_zero': None,
            }

        # Linear regression on CLV over bet index
        x = np.arange(len(clv_values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, clv_values)

        # Determine trend direction
        if abs(slope) < 0.0001:  # Essentially flat
            direction = 'stable'
        elif slope > 0:
            direction = 'improving'
        else:
            direction = 'declining'

        # Calculate days to zero CLV if declining
        days_to_zero = None
        if slope < 0 and clv_values.iloc[-1] > 0:
            # Assume 3 bets per day
            bets_to_zero = -clv_values.iloc[-1] / slope
            days_to_zero = bets_to_zero / 3

        return {
            'direction': direction,
            'slope': slope,
            'slope_per_100_bets': slope * 100,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'current_clv': clv_values.iloc[-1],
            'days_to_zero': days_to_zero,
            'significant': p_value < 0.05,
        }

    def get_performance_by_edge_bucket(
        self,
        bets_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Break down performance by edge level.

        Helps identify which edge levels are actually profitable.
        """
        settled = bets_df[bets_df['outcome'].notna()].copy()

        if 'edge' not in settled.columns or settled.empty:
            return pd.DataFrame()

        # Create edge buckets
        bins = [0, 0.03, 0.05, 0.07, 0.10, 1.0]
        labels = ['0-3%', '3-5%', '5-7%', '7-10%', '10%+']

        settled['edge_bucket'] = pd.cut(
            settled['edge'].abs(),
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        # Calculate metrics by bucket
        results = []
        for bucket in labels:
            bucket_df = settled[settled['edge_bucket'] == bucket]

            if len(bucket_df) == 0:
                continue

            wins = (bucket_df['outcome'] == 'win').sum()
            losses = (bucket_df['outcome'] == 'loss').sum()
            total = wins + losses

            if total == 0:
                continue

            win_rate = wins / total
            total_wagered = bucket_df['bet_amount'].fillna(100).sum()
            roi = bucket_df['profit'].sum() / total_wagered if total_wagered > 0 else 0

            # Statistical significance (requires n >= 30 for normal approximation)
            if total >= 30:
                z_score = (win_rate - 0.524) / np.sqrt(0.524 * 0.476 / total)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            else:
                z_score = 0
                p_value = 1.0

            results.append({
                'edge_bucket': bucket,
                'n_bets': total,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'roi': roi,
                'total_profit': bucket_df['profit'].sum(),
                'avg_clv': bucket_df['clv'].mean() if 'clv' in bucket_df.columns else 0,
                'z_score': z_score,
                'p_value': p_value,
                'significant': p_value < 0.05,
            })

        return pd.DataFrame(results)

    def get_signal_health(
        self,
        bets_df: pd.DataFrame,
        feature_values: pd.Series,
        feature_name: str,
        window: int = 100,
    ) -> Dict:
        """
        Track predictive power of a feature over time.

        Args:
            bets_df: Bet history
            feature_values: Series of feature values aligned with bets
            feature_name: Name of the feature
            window: Lookback window

        Returns:
            Dictionary with signal health metrics
        """
        settled = bets_df[bets_df['outcome'].notna()].copy()

        if len(settled) < window or len(feature_values) != len(settled):
            return {
                'feature': feature_name,
                'status': 'insufficient_data',
            }

        # Align data
        settled['feature'] = feature_values.values
        settled['is_win'] = (settled['outcome'] == 'win').astype(int)

        recent = settled.tail(window)

        # Current correlation
        current_corr = recent['feature'].corr(recent['is_win'])

        # Rolling correlation to detect trend
        rolling_corrs = []
        for i in range(20, len(settled) + 1, 10):
            window_df = settled.iloc[max(0, i-50):i]
            if len(window_df) >= 20:
                corr = window_df['feature'].corr(window_df['is_win'])
                rolling_corrs.append({'index': i, 'correlation': corr})

        rolling_df = pd.DataFrame(rolling_corrs)

        # Trend in correlation
        if len(rolling_df) >= 5:
            slope, _, r_value, p_value, _ = stats.linregress(
                rolling_df['index'],
                rolling_df['correlation']
            )
            trend = 'declining' if slope < -0.0001 else 'stable' if abs(slope) < 0.0001 else 'improving'
        else:
            slope = 0
            trend = 'unknown'
            p_value = 1

        return {
            'feature': feature_name,
            'current_correlation': current_corr,
            'correlation_trend': trend,
            'trend_slope': slope,
            'trend_p_value': p_value,
            'historical_correlations': rolling_corrs[-10:] if rolling_corrs else [],
            'status': 'healthy' if abs(current_corr) > 0.05 and trend != 'declining' else 'warning',
        }

    def generate_health_report(
        self,
        bets_df: pd.DataFrame,
    ) -> Dict:
        """
        Generate comprehensive health report.

        Returns dictionary with all monitoring metrics and alerts.
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {},
            'metrics_by_window': {},
            'alerts': [],
            'trends': {},
            'edge_analysis': None,
        }

        # Metrics for each window
        for window in self.windows:
            metrics = self.get_current_metrics(bets_df, window)
            report['metrics_by_window'][f'last_{window}'] = metrics.to_dict()

        # Primary metrics summary (50-bet window)
        primary_metrics = self.get_current_metrics(bets_df, 50)
        report['summary'] = {
            'n_bets': primary_metrics.n_bets,
            'win_rate': f"{primary_metrics.win_rate:.1%}",
            'roi': f"{primary_metrics.roi:.2%}",
            'clv': f"{primary_metrics.clv:.2%}" if primary_metrics.clv else "N/A",
            'sharpe': f"{primary_metrics.sharpe_ratio:.2f}",
            'status': 'healthy' if primary_metrics.clv > 0 and primary_metrics.win_rate > 0.52 else 'warning',
        }

        # Detect decay
        report['alerts'] = self.detect_performance_decay(bets_df, 50)

        # CLV trend
        report['trends']['clv'] = self.get_clv_trend(bets_df, 100)

        # Edge bucket analysis
        edge_analysis = self.get_performance_by_edge_bucket(bets_df)
        if not edge_analysis.empty:
            report['edge_analysis'] = edge_analysis.to_dict('records')

        return report


    # ========== Market Signal Health Tracking ==========

    def get_signal_performance(
        self,
        bets_df: pd.DataFrame,
        signal_type: str,
        window: int = 100
    ) -> Dict:
        """
        Track performance of market microstructure signals.

        Args:
            bets_df: DataFrame with bet history
            signal_type: 'steam', 'rlm', or 'money_flow'
            window: Lookback window for analysis

        Returns:
            Dict with signal performance metrics
        """
        if bets_df.empty:
            return {'error': 'No bets available'}

        # Filter to recent bets with signal
        recent_bets = bets_df.tail(window).copy()

        # Check filters_passed column for signal presence
        if 'filters_passed' not in recent_bets.columns:
            return {'error': 'No filters_passed column'}

        # Find bets where signal was used
        signal_patterns = {
            'steam': 'steam_aligned',
            'rlm': 'rlm_aligned',
            'money_flow': 'sharp_aligned'
        }

        pattern = signal_patterns.get(signal_type)
        if not pattern:
            return {'error': f'Unknown signal type: {signal_type}'}

        # Filter bets where signal was applied
        signal_bets = recent_bets[
            recent_bets['filters_passed'].astype(str).str.contains(pattern, na=False)
        ]

        if signal_bets.empty:
            return {
                'signal_type': signal_type,
                'bets_with_signal': 0,
                'note': f'No bets with {signal_type} signal in last {window} bets'
            }

        # Calculate performance with signal
        wins = (signal_bets['outcome'] == 'win').sum()
        total = len(signal_bets)
        win_rate = wins / total if total > 0 else 0

        if 'clv' in signal_bets.columns:
            avg_clv = signal_bets['clv'].mean()
            positive_clv_rate = (signal_bets['clv'] > 0).sum() / total
        else:
            avg_clv = None
            positive_clv_rate = None

        if 'profit' in signal_bets.columns:
            roi = signal_bets['profit'].sum() / total
        else:
            roi = None

        # Compare to baseline (bets without signal)
        no_signal_bets = recent_bets[
            ~recent_bets['filters_passed'].astype(str).str.contains(pattern, na=False)
        ]

        if not no_signal_bets.empty:
            baseline_win_rate = (no_signal_bets['outcome'] == 'win').sum() / len(no_signal_bets)
            win_rate_lift = win_rate - baseline_win_rate
        else:
            baseline_win_rate = None
            win_rate_lift = None

        return {
            'signal_type': signal_type,
            'window': window,
            'bets_with_signal': total,
            'bets_without_signal': len(no_signal_bets),
            'win_rate': win_rate,
            'baseline_win_rate': baseline_win_rate,
            'win_rate_lift': win_rate_lift,
            'avg_clv': avg_clv,
            'positive_clv_rate': positive_clv_rate,
            'roi': roi,
            'recommendation': self._interpret_signal_performance(
                win_rate, avg_clv, win_rate_lift, total
            )
        }

    def _interpret_signal_performance(
        self,
        win_rate: float,
        avg_clv: Optional[float],
        win_rate_lift: Optional[float],
        sample_size: int
    ) -> str:
        """Interpret signal performance and provide recommendation."""
        if sample_size < 20:
            return "INSUFFICIENT_DATA - Need more bets to evaluate"

        if win_rate > 0.55 and (avg_clv is None or avg_clv > 0.01):
            return "STRONG - Signal performing well, continue using"
        elif win_rate > 0.52 and (avg_clv is None or avg_clv > 0):
            return "MODERATE - Signal showing positive results"
        elif win_rate_lift and win_rate_lift > 0.03:
            return "BENEFICIAL - Signal provides lift vs baseline"
        elif win_rate < 0.50:
            return "WEAK - Consider removing signal filter"
        elif avg_clv and avg_clv < -0.01:
            return "NEGATIVE_CLV - Signal may be harmful"
        else:
            return "NEUTRAL - Continue monitoring"

    def get_all_signals_health(self, bets_df: pd.DataFrame, window: int = 100) -> Dict:
        """
        Get health report for all market signals.

        Args:
            bets_df: DataFrame with bet history
            window: Lookback window

        Returns:
            Dict with health status for each signal type
        """
        signals = ['steam', 'rlm', 'money_flow']
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'window': window,
            'signals': {}
        }

        for signal_type in signals:
            health_report['signals'][signal_type] = self.get_signal_performance(
                bets_df=bets_df,
                signal_type=signal_type,
                window=window
            )

        return health_report

    def detect_signal_decay(
        self,
        bets_df: pd.DataFrame,
        signal_type: str,
        recent_window: int = 50,
        baseline_window: int = 100
    ) -> Dict:
        """
        Detect if a signal's performance is decaying.

        Compares recent performance to baseline performance.

        Args:
            bets_df: DataFrame with bet history
            signal_type: Type of signal to check
            recent_window: Recent window for comparison
            baseline_window: Baseline window

        Returns:
            Dict with decay analysis
        """
        if len(bets_df) < baseline_window:
            return {'error': 'Insufficient data for decay detection'}

        # Get baseline performance (full window)
        baseline_perf = self.get_signal_performance(
            bets_df=bets_df.tail(baseline_window),
            signal_type=signal_type,
            window=baseline_window
        )

        # Get recent performance
        recent_perf = self.get_signal_performance(
            bets_df=bets_df.tail(recent_window),
            signal_type=signal_type,
            window=recent_window
        )

        if 'error' in baseline_perf or 'error' in recent_perf:
            return {'error': 'Could not calculate signal performance'}

        # Check for decay
        win_rate_decay = (recent_perf.get('win_rate', 0) -
                         baseline_perf.get('win_rate', 0))

        clv_decay = None
        if recent_perf.get('avg_clv') and baseline_perf.get('avg_clv'):
            clv_decay = recent_perf['avg_clv'] - baseline_perf['avg_clv']

        # Determine if signal is decaying
        is_decaying = (
            win_rate_decay < -0.05 or  # Win rate dropped >5%
            (clv_decay is not None and clv_decay < -0.02)  # CLV dropped >2%
        )

        return {
            'signal_type': signal_type,
            'is_decaying': is_decaying,
            'win_rate_decay': win_rate_decay,
            'clv_decay': clv_decay,
            'recent_performance': recent_perf,
            'baseline_performance': baseline_perf,
            'recommendation': 'DISABLE_SIGNAL' if is_decaying else 'CONTINUE'
        }


def create_monitor_from_tracker(tracker_module) -> AlphaMonitor:
    """
    Factory function to create AlphaMonitor from existing bet tracker.

    Usage:
        from src import bet_tracker
        from src.monitoring import create_monitor_from_tracker
        monitor = create_monitor_from_tracker(bet_tracker)
    """
    monitor = AlphaMonitor()
    return monitor
