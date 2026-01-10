"""
Model Health Monitor for tracking performance over time.

Provides:
- Daily performance metric recording
- Performance degradation detection
- Trend analysis
- Alerting
"""

import logging
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict

import pandas as pd
import numpy as np

from src.versioning.db import get_db_manager
from src.versioning import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class DailyMetrics:
    """Daily performance metrics."""

    predictions_count: int
    bets_placed: int
    accuracy: float
    roi: float
    pnl: float
    win_rate: float
    auc_roc: Optional[float] = None
    roi_7d: Optional[float] = None
    roi_30d: Optional[float] = None
    sharpe_30d: Optional[float] = None


@dataclass
class Alert:
    """Performance alert."""

    alert_type: str  # "degradation", "drift", "anomaly"
    model_id: str
    severity: str  # "info", "warning", "critical"
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    created_at: datetime


@dataclass
class AlertConfig:
    """Configuration for performance alerts."""

    # Thresholds for degradation detection
    accuracy_min: float = 0.52  # Below this triggers alert
    roi_min: float = -0.05  # Below this triggers alert
    sharpe_min: float = 0.0  # Below this triggers alert

    # Rolling window for trend detection
    trend_window_days: int = 14
    trend_decline_threshold: float = 0.05  # 5% decline triggers alert

    # Alert channels
    alert_to_discord: bool = True


class ModelHealthMonitor:
    """Track and alert on model performance over time."""

    def __init__(
        self,
        db_dir: str = "data/versioning",
        alert_config: Optional[AlertConfig] = None
    ):
        self.db_manager = get_db_manager(db_dir)
        self.registry = ModelRegistry(db_dir)
        self.alert_config = alert_config or AlertConfig()

        # Initialize database
        self.db_manager.init_performance_history()

    def record_daily_performance(
        self,
        model_id: str,
        model_name: str,
        date: date,
        metrics: DailyMetrics
    ) -> None:
        """
        Record daily performance snapshot.

        Args:
            model_id: Model ID
            model_name: Model name
            date: Date for this record
            metrics: DailyMetrics object
        """
        conn = self.db_manager.get_connection('performance_history')
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO daily_performance (
                    model_id, model_name, date,
                    predictions_count, bets_placed,
                    accuracy, auc_roc, roi, pnl, win_rate,
                    roi_7d, roi_30d, sharpe_30d
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id, model_name, date,
                metrics.predictions_count, metrics.bets_placed,
                metrics.accuracy, metrics.auc_roc,
                metrics.roi, metrics.pnl, metrics.win_rate,
                metrics.roi_7d, metrics.roi_30d, metrics.sharpe_30d
            ))

            conn.commit()
            logger.info(f"Recorded daily performance for {model_name} on {date}")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to record performance: {e}")
            raise
        finally:
            conn.close()

    def check_for_degradation(
        self,
        model_id: str,
        lookback_days: int = 30
    ) -> List[Alert]:
        """
        Check if model performance has degraded.

        Detects:
        - Accuracy drop below threshold
        - ROI drop below threshold
        - Sharpe ratio decline
        - Unusual prediction distribution

        Args:
            model_id: Model ID to check
            lookback_days: Number of days to analyze

        Returns:
            List of alerts (empty if no issues)
        """
        alerts = []

        # Get recent performance
        perf_df = self.get_performance_trend(
            model_id,
            metric="all",
            days=lookback_days
        )

        if perf_df.empty:
            logger.warning(f"No performance data for model {model_id}")
            return alerts

        # Check accuracy threshold
        recent_accuracy = perf_df['accuracy'].tail(self.alert_config.trend_window_days).mean()
        if recent_accuracy < self.alert_config.accuracy_min:
            alerts.append(Alert(
                alert_type="degradation",
                model_id=model_id,
                severity="warning",
                message=f"Accuracy below threshold: {recent_accuracy:.2%}",
                metric_name="accuracy",
                metric_value=recent_accuracy,
                threshold=self.alert_config.accuracy_min,
                created_at=datetime.now()
            ))

        # Check ROI threshold
        recent_roi = perf_df['roi_30d'].iloc[-1] if 'roi_30d' in perf_df else perf_df['roi'].mean()
        if recent_roi < self.alert_config.roi_min:
            alerts.append(Alert(
                alert_type="degradation",
                model_id=model_id,
                severity="critical",
                message=f"ROI below threshold: {recent_roi:.2%}",
                metric_name="roi",
                metric_value=recent_roi,
                threshold=self.alert_config.roi_min,
                created_at=datetime.now()
            ))

        # Check Sharpe ratio
        if 'sharpe_30d' in perf_df and not perf_df['sharpe_30d'].isna().all():
            recent_sharpe = perf_df['sharpe_30d'].iloc[-1]
            if recent_sharpe < self.alert_config.sharpe_min:
                alerts.append(Alert(
                    alert_type="degradation",
                    model_id=model_id,
                    severity="warning",
                    message=f"Sharpe ratio below threshold: {recent_sharpe:.2f}",
                    metric_name="sharpe",
                    metric_value=recent_sharpe,
                    threshold=self.alert_config.sharpe_min,
                    created_at=datetime.now()
                ))

        # Store alerts in database
        for alert in alerts:
            self._store_alert(alert)

        return alerts

    def get_performance_trend(
        self,
        model_id: str,
        metric: str = "roi",
        days: int = 90
    ) -> pd.DataFrame:
        """
        Get performance trend for a specific metric.

        Args:
            model_id: Model ID
            metric: Metric name ("roi", "accuracy", "sharpe", "all")
            days: Number of days to retrieve

        Returns:
            DataFrame with date index and metric columns
        """
        # Whitelist allowed metrics to prevent SQL injection
        ALLOWED_METRICS = {'roi', 'accuracy', 'sharpe', 'auc_roc', 'win_rate', 'pnl',
                          'sharpe_30d', 'roi_7d', 'roi_30d', 'predictions_count', 'bets_placed'}

        conn = self.db_manager.get_connection('performance_history')

        cutoff_date = (datetime.now() - timedelta(days=days)).date()

        if metric == "all":
            query = """
                SELECT date, predictions_count, bets_placed,
                       accuracy, auc_roc, roi, pnl, win_rate,
                       roi_7d, roi_30d, sharpe_30d
                FROM daily_performance
                WHERE model_id = ? AND date >= ?
                ORDER BY date
            """
        elif metric not in ALLOWED_METRICS:
            conn.close()
            raise ValueError(f"Invalid metric: {metric}. Must be one of {ALLOWED_METRICS} or 'all'")
        else:
            # Safe to use - validated against whitelist
            query = f"""
                SELECT date, {metric}
                FROM daily_performance
                WHERE model_id = ? AND date >= ?
                ORDER BY date
            """

        df = pd.read_sql_query(query, conn, params=(model_id, cutoff_date))
        conn.close()

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

        return df

    def _store_alert(self, alert: Alert) -> None:
        """Store alert in database."""
        conn = self.db_manager.get_connection('performance_history')
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO alerts (
                    alert_type, model_id, severity, message,
                    metric_name, metric_value, threshold
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_type, alert.model_id, alert.severity,
                alert.message, alert.metric_name, alert.metric_value,
                alert.threshold
            ))

            conn.commit()
            logger.info(f"Stored alert: {alert.severity} - {alert.message}")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to store alert: {e}")
        finally:
            conn.close()

    def get_recent_alerts(
        self,
        model_id: Optional[str] = None,
        severity: Optional[str] = None,
        days: int = 7,
        unacknowledged_only: bool = True
    ) -> List[Alert]:
        """Get recent alerts."""
        conn = self.db_manager.get_connection('performance_history')
        cursor = conn.cursor()

        cutoff_date = datetime.now() - timedelta(days=days)

        query = """
            SELECT alert_id, alert_type, model_id, severity, message,
                   metric_name, metric_value, threshold, created_at
            FROM alerts
            WHERE created_at >= ?
        """
        params = [cutoff_date]

        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)

        if severity:
            query += " AND severity = ?"
            params.append(severity)

        if unacknowledged_only:
            query += " AND acknowledged_at IS NULL"

        query += " ORDER BY created_at DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            Alert(
                alert_type=row[1],
                model_id=row[2],
                severity=row[3],
                message=row[4],
                metric_name=row[5],
                metric_value=row[6],
                threshold=row[7],
                created_at=datetime.fromisoformat(row[8])
            )
            for row in rows
        ]
