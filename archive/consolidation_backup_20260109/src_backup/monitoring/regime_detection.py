"""
Regime Detection for Betting Model Performance

Detects structural changes in model performance to avoid betting on stale signals.
Uses change point detection to identify when the model's edge is degrading.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from loguru import logger


class Regime(Enum):
    """Market regime classifications."""
    NORMAL = "normal"           # Model performing as expected
    VOLATILE = "volatile"       # High variance in performance
    EDGE_DECAY = "edge_decay"   # Systematic underperformance
    HOT_STREAK = "hot_streak"   # Unusually strong performance


@dataclass
class ChangePoint:
    """Detected change point in a metric series."""
    index: int
    timestamp: Optional[pd.Timestamp]
    metric_before: float
    metric_after: float
    change_magnitude: float
    confidence: float


@dataclass
class RegimeAlert:
    """Alert about regime change or performance issue."""
    regime: Regime
    severity: str  # 'info', 'warning', 'critical'
    message: str
    metrics: Dict[str, float]
    recommended_action: str


class RegimeDetector:
    """
    Detects regime changes in betting performance.

    Uses statistical methods to identify:
    - Change points in CLV, win rate, ROI
    - Volatility regime shifts
    - Systematic edge decay
    """

    def __init__(
        self,
        clv_threshold: float = 0.0,
        win_rate_threshold: float = 0.524,  # ~-110 breakeven
        lookback_window: int = 50,
        min_samples_for_detection: int = 30,
    ):
        """
        Initialize regime detector.

        Args:
            clv_threshold: CLV below this triggers edge decay alert
            win_rate_threshold: Win rate below this triggers alert
            lookback_window: Window for rolling calculations
            min_samples_for_detection: Minimum bets before detecting regimes
        """
        self.clv_threshold = clv_threshold
        self.win_rate_threshold = win_rate_threshold
        self.lookback_window = lookback_window
        self.min_samples_for_detection = min_samples_for_detection

        self._baseline_metrics: Optional[Dict[str, float]] = None
        self._performance_history: List[Dict] = []

    def set_baseline(self, metrics: Dict[str, float]) -> None:
        """
        Set baseline performance metrics for comparison.

        Args:
            metrics: Dict with 'clv', 'win_rate', 'roi' baseline values
        """
        self._baseline_metrics = metrics
        logger.info(f"Baseline set: {metrics}")

    def add_performance_sample(
        self,
        clv: float,
        win: bool,
        roi: float,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> None:
        """Add a single bet result to performance history."""
        self._performance_history.append({
            'clv': clv,
            'win': 1 if win else 0,
            'roi': roi,
            'timestamp': timestamp or pd.Timestamp.now(),
        })

    def detect_changepoints_cusum(
        self,
        series: np.ndarray,
        threshold: float = 4.0,
        drift: float = 0.0,
    ) -> List[int]:
        """
        Detect change points using CUSUM algorithm.

        CUSUM (Cumulative Sum) is simple and effective for detecting
        mean shifts in a time series.

        Args:
            series: Time series to analyze
            threshold: Detection threshold (higher = fewer false positives)
            drift: Expected drift in the series

        Returns:
            List of indices where change points detected
        """
        if len(series) < self.min_samples_for_detection:
            return []

        # Normalize series
        mean = np.mean(series)
        std = np.std(series)
        if std < 1e-6:
            return []

        normalized = (series - mean) / std

        # CUSUM for positive and negative shifts
        cusum_pos = np.zeros(len(series))
        cusum_neg = np.zeros(len(series))

        changepoints = []
        last_detection = -1
        min_gap = max(10, len(series) // 20)  # Minimum samples between detections

        for i in range(1, len(series)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + normalized[i] - drift)
            cusum_neg[i] = max(0, cusum_neg[i-1] - normalized[i] - drift)

            # Detect change point only if sufficient gap from last detection
            if (cusum_pos[i] > threshold or cusum_neg[i] > threshold) and (i - last_detection) >= min_gap:
                changepoints.append(i)
                last_detection = i
                # Reset after detection
                cusum_pos[i] = 0
                cusum_neg[i] = 0

        return changepoints

    def detect_changepoints_binseg(
        self,
        series: np.ndarray,
        max_changepoints: int = 5,
        min_segment_length: int = 10,
    ) -> List[int]:
        """
        Detect change points using Binary Segmentation.

        Binary segmentation recursively splits the series at the point
        of maximum statistical change.

        Args:
            series: Time series to analyze
            max_changepoints: Maximum number of change points to find
            min_segment_length: Minimum length of a segment

        Returns:
            List of change point indices
        """
        if len(series) < 2 * min_segment_length:
            return []

        def find_best_split(start: int, end: int) -> Tuple[int, float]:
            """Find the best split point in a segment."""
            if end - start < 2 * min_segment_length:
                return -1, 0.0

            best_idx = -1
            best_cost_reduction = 0.0

            segment = series[start:end]
            total_var = np.var(segment) * len(segment)

            # Skip if segment has near-zero variance (numerically unstable)
            if total_var < 1e-8:
                return -1, 0.0

            for i in range(min_segment_length, len(segment) - min_segment_length):
                left = segment[:i]
                right = segment[i:]

                # Safe variance calculation
                var_left = np.var(left) if len(left) > 1 else 0.0
                var_right = np.var(right) if len(right) > 1 else 0.0

                split_var = var_left * len(left) + var_right * len(right)
                cost_reduction = total_var - split_var

                if cost_reduction > best_cost_reduction:
                    best_cost_reduction = cost_reduction
                    best_idx = start + i

            return best_idx, best_cost_reduction

        changepoints = []
        segments = [(0, len(series))]

        while len(changepoints) < max_changepoints and segments:
            # Find segment with best potential split
            best_segment_idx = -1
            best_split = -1
            best_reduction = 0.0

            for seg_idx, (start, end) in enumerate(segments):
                split_idx, reduction = find_best_split(start, end)
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_split = split_idx
                    best_segment_idx = seg_idx

            # Check if split is significant
            if best_split == -1 or best_reduction < np.var(series) * 0.1:
                break

            # Add change point and update segments
            changepoints.append(best_split)
            start, end = segments.pop(best_segment_idx)
            segments.append((start, best_split))
            segments.append((best_split, end))

        return sorted(changepoints)

    def detect_changepoints(
        self,
        metric_series: np.ndarray,
        method: str = "cusum",
        timestamps: Optional[np.ndarray] = None,
    ) -> List[ChangePoint]:
        """
        Detect structural breaks in a metric series.

        Args:
            metric_series: Array of metric values over time
            method: Detection method ('cusum' or 'binseg')
            timestamps: Optional array of timestamps (must match metric_series length)

        Returns:
            List of ChangePoint objects
        """
        if len(metric_series) < self.min_samples_for_detection:
            logger.debug(f"Insufficient samples ({len(metric_series)}) for change detection")
            return []

        # Validate timestamps array if provided
        if timestamps is not None and len(timestamps) != len(metric_series):
            raise ValueError(
                f"timestamps length ({len(timestamps)}) must match "
                f"metric_series length ({len(metric_series)})"
            )

        # Detect raw change points
        if method == "cusum":
            raw_points = self.detect_changepoints_cusum(metric_series)
        elif method == "binseg":
            raw_points = self.detect_changepoints_binseg(metric_series)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Convert to ChangePoint objects with metadata
        changepoints = []
        series_std = np.std(metric_series)

        for idx in raw_points:
            # Validate index is within bounds
            if idx <= 0 or idx >= len(metric_series):
                continue

            # Calculate window size
            window = min(10, idx, len(metric_series) - idx)
            if window < 3:
                continue

            # Safe slicing with non-empty checks
            before_slice = metric_series[max(0, idx-window):idx]
            after_slice = metric_series[idx:min(len(metric_series), idx+window)]

            if len(before_slice) == 0 or len(after_slice) == 0:
                continue

            before = np.mean(before_slice)
            after = np.mean(after_slice)

            cp = ChangePoint(
                index=idx,
                timestamp=timestamps[idx] if timestamps is not None else None,
                metric_before=before,
                metric_after=after,
                change_magnitude=after - before,
                confidence=min(1.0, abs(after - before) / (series_std + 1e-6)),
            )
            changepoints.append(cp)

        return changepoints

    def calculate_rolling_volatility(
        self,
        series: np.ndarray,
        window: int = None,
    ) -> np.ndarray:
        """Calculate rolling volatility of a series."""
        window = window or self.lookback_window

        if len(series) < window:
            return np.full(len(series), np.nan)

        result = np.full(len(series), np.nan)
        for i in range(window - 1, len(series)):
            result[i] = np.std(series[i-window+1:i+1])

        return result

    def get_current_regime(self) -> Regime:
        """
        Classify the current market regime based on recent performance.

        Returns:
            Current Regime classification
        """
        if len(self._performance_history) < self.min_samples_for_detection:
            return Regime.NORMAL

        recent = self._performance_history[-self.lookback_window:]

        # Calculate recent metrics
        clvs = np.array([p['clv'] for p in recent])
        wins = np.array([p['win'] for p in recent])

        avg_clv = np.mean(clvs)
        win_rate = np.mean(wins)
        clv_volatility = np.std(clvs)

        # Check for edge decay
        if avg_clv < self.clv_threshold and win_rate < self.win_rate_threshold:
            return Regime.EDGE_DECAY

        # Check for high volatility
        baseline_vol = np.std([p['clv'] for p in self._performance_history])
        if baseline_vol < 1e-6:
            baseline_vol = 0.01  # Use small default to avoid division issues
        if clv_volatility > 2 * baseline_vol:
            return Regime.VOLATILE

        # Check for hot streak (unsustainable positive performance)
        if win_rate > 0.65 and avg_clv > 0.03:
            return Regime.HOT_STREAK

        return Regime.NORMAL

    def get_rolling_metrics(
        self,
        window: int = None,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate rolling performance metrics.

        Args:
            window: Rolling window size

        Returns:
            Dict with rolling CLV, win rate, ROI arrays
        """
        window = window or self.lookback_window

        if len(self._performance_history) < window:
            return {}

        df = pd.DataFrame(self._performance_history)

        return {
            'rolling_clv': df['clv'].rolling(window).mean().values,
            'rolling_win_rate': df['win'].rolling(window).mean().values,
            'rolling_roi': df['roi'].rolling(window).mean().values,
            'rolling_clv_std': df['clv'].rolling(window).std().values,
        }

    def detect_performance_decay(self) -> List[RegimeAlert]:
        """
        Check for systematic performance degradation.

        Returns:
            List of RegimeAlert objects if decay detected
        """
        alerts = []

        if len(self._performance_history) < self.min_samples_for_detection:
            return alerts

        # Get rolling metrics
        metrics = self.get_rolling_metrics()
        if not metrics:
            return alerts

        recent_clv = metrics['rolling_clv'][-1]
        recent_win_rate = metrics['rolling_win_rate'][-1]

        # Check CLV decay
        if recent_clv < self.clv_threshold:
            severity = "critical" if recent_clv < -0.02 else "warning"
            alerts.append(RegimeAlert(
                regime=Regime.EDGE_DECAY,
                severity=severity,
                message=f"CLV has dropped to {recent_clv:.3f} (threshold: {self.clv_threshold})",
                metrics={'rolling_clv': recent_clv, 'rolling_win_rate': recent_win_rate},
                recommended_action="Consider reducing bet sizes or pausing until edge is re-established",
            ))

        # Check win rate decay
        if recent_win_rate < self.win_rate_threshold:
            alerts.append(RegimeAlert(
                regime=Regime.EDGE_DECAY,
                severity="warning",
                message=f"Win rate at {recent_win_rate:.1%} below breakeven {self.win_rate_threshold:.1%}",
                metrics={'rolling_clv': recent_clv, 'rolling_win_rate': recent_win_rate},
                recommended_action="Review recent predictions for systematic errors",
            ))

        # Check for change points in CLV
        clv_series = np.array([p['clv'] for p in self._performance_history])
        changepoints = self.detect_changepoints(clv_series)

        for cp in changepoints[-3:]:  # Only recent change points
            if cp.change_magnitude < -0.01:  # Negative shift in CLV
                alerts.append(RegimeAlert(
                    regime=Regime.EDGE_DECAY,
                    severity="info",
                    message=f"Change point detected: CLV shifted from {cp.metric_before:.3f} to {cp.metric_after:.3f}",
                    metrics={'change_magnitude': cp.change_magnitude, 'confidence': cp.confidence},
                    recommended_action="Investigate what changed in the market or model",
                ))

        return alerts

    def get_signal_health(
        self,
        feature_predictions: np.ndarray,
        feature_actuals: np.ndarray,
        feature_name: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Track individual feature/signal predictive power over time.

        Args:
            feature_predictions: Predictions based on this feature
            feature_actuals: Actual outcomes
            feature_name: Name of the feature

        Returns:
            Dict with signal health metrics
        """
        if len(feature_predictions) < self.min_samples_for_detection:
            return {'status': 'insufficient_data', 'feature': feature_name}

        # Calculate correlation with variance check to avoid division by zero
        if np.std(feature_predictions) < 1e-6 or np.std(feature_actuals) < 1e-6:
            correlation = 0.0
        else:
            corr_matrix = np.corrcoef(feature_predictions, feature_actuals)
            correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0

        # Calculate directional accuracy
        if len(feature_predictions) > 1:
            pred_direction = np.sign(np.diff(feature_predictions))
            actual_direction = np.sign(np.diff(feature_actuals))
            directional_accuracy = np.mean(pred_direction == actual_direction)
        else:
            directional_accuracy = np.nan

        # Calculate information coefficient (IC)
        rank_corr = _spearman_correlation(feature_predictions, feature_actuals)

        # Determine health status
        if abs(correlation) < 0.05:
            status = 'weak'
        elif correlation < 0:
            status = 'inverted'
        elif correlation < 0.15:
            status = 'degrading'
        else:
            status = 'healthy'

        return {
            'feature': feature_name,
            'status': status,
            'correlation': correlation,
            'rank_correlation': rank_corr,
            'directional_accuracy': directional_accuracy,
            'n_samples': len(feature_predictions),
        }

    def should_pause_betting(self) -> Tuple[bool, str]:
        """
        Determine if betting should be paused based on regime.

        Returns:
            Tuple of (should_pause, reason)
        """
        regime = self.get_current_regime()
        alerts = self.detect_performance_decay()

        # Critical alerts = pause
        critical_alerts = [a for a in alerts if a.severity == 'critical']
        if critical_alerts:
            return True, critical_alerts[0].message

        # Edge decay regime = pause
        if regime == Regime.EDGE_DECAY:
            return True, "Model in edge decay regime"

        # Multiple warnings = consider pausing
        if len([a for a in alerts if a.severity == 'warning']) >= 2:
            return True, "Multiple warning signs detected"

        return False, ""

    def get_regime_summary(self) -> Dict[str, Any]:
        """Get comprehensive regime analysis summary."""
        regime = self.get_current_regime()
        alerts = self.detect_performance_decay()
        should_pause, pause_reason = self.should_pause_betting()

        metrics = self.get_rolling_metrics()

        return {
            'regime': regime.value,
            'should_pause': should_pause,
            'pause_reason': pause_reason,
            'n_alerts': len(alerts),
            'alert_severities': [a.severity for a in alerts],
            'rolling_clv': metrics.get('rolling_clv', [np.nan])[-1] if metrics else np.nan,
            'rolling_win_rate': metrics.get('rolling_win_rate', [np.nan])[-1] if metrics else np.nan,
            'n_samples': len(self._performance_history),
        }


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Spearman rank correlation."""
    if len(x) < 3:
        return np.nan

    x_ranks = np.argsort(np.argsort(x))
    y_ranks = np.argsort(np.argsort(y))

    return np.corrcoef(x_ranks, y_ranks)[0, 1]


class SeasonalRegimeDetector(RegimeDetector):
    """
    Extended regime detector with NBA season awareness.

    NBA seasons have distinct phases with different market efficiency:
    - Early season: More variance, less efficient pricing
    - All-Star break: Disruption point
    - Post All-Star: Playoff push begins
    - Playoffs: Highest efficiency, hardest to beat
    """

    # Season phase definitions (approximate game numbers)
    SEASON_PHASES = {
        'early_season': (0, 20),      # First ~6 weeks
        'pre_allstar': (20, 55),      # December through All-Star
        'post_allstar': (55, 72),     # All-Star through end of regular season
        'playoff_push': (72, 82),     # Final stretch
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._phase_performance: Dict[str, List[Dict]] = {
            phase: [] for phase in self.SEASON_PHASES
        }

    def get_season_phase(self, games_played: int) -> str:
        """Determine current season phase based on games played."""
        for phase, (start, end) in self.SEASON_PHASES.items():
            if start <= games_played < end:
                return phase
        return 'playoff_push'  # Default to late season

    def add_performance_sample_with_phase(
        self,
        clv: float,
        win: bool,
        roi: float,
        games_played: int,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> None:
        """Add performance sample with season phase tracking."""
        self.add_performance_sample(clv, win, roi, timestamp)

        phase = self.get_season_phase(games_played)
        self._phase_performance[phase].append({
            'clv': clv,
            'win': 1 if win else 0,
            'roi': roi,
        })

    def get_phase_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance breakdown by season phase."""
        results = {}

        for phase, samples in self._phase_performance.items():
            if not samples:
                continue

            results[phase] = {
                'n_bets': len(samples),
                'avg_clv': np.mean([s['clv'] for s in samples]),
                'win_rate': np.mean([s['win'] for s in samples]),
                'avg_roi': np.mean([s['roi'] for s in samples]),
            }

        return results

    def get_current_phase_regime(self, games_played: int) -> Regime:
        """Get regime for current season phase specifically."""
        phase = self.get_season_phase(games_played)
        phase_samples = self._phase_performance.get(phase, [])

        if len(phase_samples) < 10:
            return Regime.NORMAL

        avg_clv = np.mean([s['clv'] for s in phase_samples[-20:]])
        win_rate = np.mean([s['win'] for s in phase_samples[-20:]])

        if avg_clv < -0.01 and win_rate < 0.5:
            return Regime.EDGE_DECAY
        elif win_rate > 0.6:
            return Regime.HOT_STREAK

        return Regime.NORMAL
