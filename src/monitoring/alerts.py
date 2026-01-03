"""
Alert System for Betting Model Monitoring

Centralized alert management for performance issues, drift detection,
and system health monitoring.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable

import pandas as pd
from loguru import logger


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertCategory(Enum):
    """Alert categories."""
    PERFORMANCE = "performance"
    CLV = "clv"
    WIN_RATE = "win_rate"
    DRIFT = "drift"
    REGIME = "regime"
    SYSTEM = "system"


@dataclass
class Alert:
    """
    Represents a single alert.

    Attributes:
        severity: Alert severity (info, warning, critical)
        category: Alert category (performance, drift, etc.)
        message: Human-readable alert message
        metrics: Dictionary of relevant metrics
        timestamp: When the alert was created
        recommended_action: Suggested response to the alert
        acknowledged: Whether the alert has been acknowledged
    """
    severity: str
    category: str
    message: str
    metrics: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    recommended_action: str = ""
    acknowledged: bool = False
    alert_id: str = ""

    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = f"{self.category}_{self.timestamp}"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Alert':
        return cls(**data)


class AlertSystem:
    """
    Centralized alert management system.

    Collects alerts from various monitoring components,
    manages alert history, and provides notification hooks.
    """

    # Default thresholds for alerts
    DEFAULT_THRESHOLDS = {
        'clv_warning': -0.005,      # -0.5% CLV
        'clv_critical': -0.01,      # -1.0% CLV
        'win_rate_warning': 0.50,   # 50% win rate
        'win_rate_critical': 0.48,  # 48% win rate
        'sharpe_warning': 0.0,      # Negative Sharpe
        'psi_warning': 0.1,         # Moderate drift
        'psi_critical': 0.2,        # Significant drift
        'drawdown_warning': 0.20,   # 20% drawdown
        'drawdown_critical': 0.40,  # 40% drawdown
    }

    def __init__(
        self,
        thresholds: Dict = None,
        history_path: str = None,
        max_history: int = 1000,
    ):
        """
        Initialize AlertSystem.

        Args:
            thresholds: Custom thresholds for alerts
            history_path: Path to persist alert history
            max_history: Maximum alerts to keep in history
        """
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.history_path = history_path or "data/monitoring/alert_history.json"
        self.max_history = max_history
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable[[Alert], None]] = []
        self._load_history()

    def _load_history(self):
        """Load alert history from disk."""
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    data = json.load(f)
                self.alert_history = [Alert.from_dict(a) for a in data]
                logger.info(f"Loaded {len(self.alert_history)} alerts from history")
            except Exception as e:
                logger.warning(f"Could not load alert history: {e}")

    def _save_history(self):
        """Save alert history to disk."""
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)

        # Trim to max history
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]

        with open(self.history_path, 'w') as f:
            json.dump([a.to_dict() for a in self.alert_history], f, indent=2)

    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """
        Add a notification handler.

        Handler will be called for each new alert.
        Can be used to send emails, Slack messages, etc.
        """
        self.notification_handlers.append(handler)

    # Valid severity levels
    VALID_SEVERITIES = {'info', 'warning', 'critical'}

    # Recursion guard for notification handlers
    _notifying: bool = False

    def create_alert(
        self,
        severity: str,
        category: str,
        message: str,
        metrics: Dict = None,
        recommended_action: str = "",
    ) -> Alert:
        """
        Create and register a new alert.

        Args:
            severity: 'info', 'warning', or 'critical'
            category: Alert category
            message: Alert message
            metrics: Related metrics
            recommended_action: Suggested action

        Returns:
            Created Alert object
        """
        # Validate severity
        if severity not in self.VALID_SEVERITIES:
            logger.warning(f"Invalid severity '{severity}', defaulting to 'info'")
            severity = 'info'

        alert = Alert(
            severity=severity,
            category=category,
            message=message,
            metrics=metrics or {},
            recommended_action=recommended_action,
        )

        self.active_alerts.append(alert)
        self.alert_history.append(alert)

        # Trim history immediately to prevent memory growth
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]

        self._save_history()

        # Notify handlers with recursion guard to prevent infinite loops
        if not self._notifying:
            self._notifying = True
            try:
                for handler in self.notification_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Notification handler failed: {e}")
            finally:
                self._notifying = False

        if severity == 'critical':
            logger.critical(f"ALERT: {message}")
        elif severity == 'warning':
            logger.warning(f"ALERT: {message}")
        else:
            logger.info(f"ALERT: {message}")

        return alert

    def check_performance_alerts(
        self,
        metrics: Dict,
    ) -> List[Alert]:
        """
        Check performance metrics and generate alerts.

        Args:
            metrics: Dictionary with performance metrics
                Expected keys: clv, win_rate, sharpe, drawdown

        Returns:
            List of generated alerts
        """
        alerts = []

        # CLV alerts
        clv = metrics.get('clv', 0)
        if clv < self.thresholds['clv_critical']:
            alerts.append(self.create_alert(
                severity='critical',
                category='clv',
                message=f"CLV critically low: {clv:.2%}",
                metrics={'clv': clv},
                recommended_action="Consider pausing betting until CLV recovers. Review recent bets for patterns.",
            ))
        elif clv < self.thresholds['clv_warning']:
            alerts.append(self.create_alert(
                severity='warning',
                category='clv',
                message=f"CLV below warning threshold: {clv:.2%}",
                metrics={'clv': clv},
                recommended_action="Monitor closely. Reduce position sizes if trend continues.",
            ))

        # Win rate alerts
        win_rate = metrics.get('win_rate', 0.5)
        if win_rate < self.thresholds['win_rate_critical']:
            alerts.append(self.create_alert(
                severity='critical',
                category='win_rate',
                message=f"Win rate critically low: {win_rate:.1%}",
                metrics={'win_rate': win_rate},
                recommended_action="Stop betting. Review model and strategy immediately.",
            ))
        elif win_rate < self.thresholds['win_rate_warning']:
            alerts.append(self.create_alert(
                severity='warning',
                category='win_rate',
                message=f"Win rate below 50%: {win_rate:.1%}",
                metrics={'win_rate': win_rate},
                recommended_action="Review recent bet selection criteria.",
            ))

        # Sharpe ratio alerts
        sharpe = metrics.get('sharpe', 0)
        if sharpe < self.thresholds['sharpe_warning']:
            alerts.append(self.create_alert(
                severity='warning',
                category='performance',
                message=f"Negative Sharpe ratio: {sharpe:.2f}",
                metrics={'sharpe': sharpe},
                recommended_action="Risk-adjusted returns are negative. Review position sizing.",
            ))

        # Drawdown alerts
        drawdown = abs(metrics.get('drawdown', 0))
        if drawdown > self.thresholds['drawdown_critical']:
            alerts.append(self.create_alert(
                severity='critical',
                category='performance',
                message=f"Maximum drawdown exceeded: {drawdown:.1%}",
                metrics={'drawdown': drawdown},
                recommended_action="Consider pausing until market conditions improve.",
            ))
        elif drawdown > self.thresholds['drawdown_warning']:
            alerts.append(self.create_alert(
                severity='warning',
                category='performance',
                message=f"Drawdown warning: {drawdown:.1%}",
                metrics={'drawdown': drawdown},
                recommended_action="Reduce position sizes to manage risk.",
            ))

        return alerts

    def check_drift_alerts(
        self,
        drift_results: Dict,
    ) -> List[Alert]:
        """
        Check feature drift results and generate alerts.

        Args:
            drift_results: Dictionary from FeatureDriftMonitor

        Returns:
            List of generated alerts
        """
        alerts = []

        significant_drift = drift_results.get('distribution_summary', {}).get('significant_drift', 0)

        if significant_drift > 5:
            alerts.append(self.create_alert(
                severity='critical',
                category='drift',
                message=f"{significant_drift} features with significant drift",
                metrics={'features_drifted': significant_drift},
                recommended_action="Investigate feature distributions. Model may need retraining.",
            ))
        elif significant_drift > 0:
            alerts.append(self.create_alert(
                severity='warning',
                category='drift',
                message=f"{significant_drift} features with significant drift",
                metrics={'features_drifted': significant_drift},
                recommended_action="Monitor closely. Check if model predictions are affected.",
            ))

        return alerts

    def acknowledge_alert(self, alert_id: str):
        """Mark an alert as acknowledged."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                self._save_history()
                return True
        return False

    def clear_acknowledged(self):
        """Remove acknowledged alerts from active list."""
        self.active_alerts = [a for a in self.active_alerts if not a.acknowledged]

    def get_active_alerts(
        self,
        severity: str = None,
        category: str = None,
    ) -> List[Alert]:
        """
        Get active (unacknowledged) alerts.

        Args:
            severity: Filter by severity
            category: Filter by category

        Returns:
            List of matching alerts
        """
        alerts = [a for a in self.active_alerts if not a.acknowledged]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if category:
            alerts = [a for a in alerts if a.category == category]

        return alerts

    def get_alert_summary(self) -> Dict:
        """Get summary of current alert status."""
        active = self.get_active_alerts()

        return {
            'total_active': len(active),
            'critical': len([a for a in active if a.severity == 'critical']),
            'warning': len([a for a in active if a.severity == 'warning']),
            'info': len([a for a in active if a.severity == 'info']),
            'by_category': self._count_by_category(active),
            'most_recent': active[-1].to_dict() if active else None,
        }

    def _count_by_category(self, alerts: List[Alert]) -> Dict[str, int]:
        """Count alerts by category."""
        counts = {}
        for alert in alerts:
            counts[alert.category] = counts.get(alert.category, 0) + 1
        return counts

    def get_daily_summary(self) -> str:
        """
        Generate daily health report as text.

        Returns:
            Formatted string with daily summary
        """
        summary = self.get_alert_summary()

        lines = [
            "=" * 50,
            "DAILY ALERT SUMMARY",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 50,
            "",
            f"Active Alerts: {summary['total_active']}",
            f"  - Critical: {summary['critical']}",
            f"  - Warning: {summary['warning']}",
            f"  - Info: {summary['info']}",
            "",
        ]

        if summary['by_category']:
            lines.append("By Category:")
            for cat, count in summary['by_category'].items():
                lines.append(f"  - {cat}: {count}")
            lines.append("")

        if summary['critical'] > 0:
            lines.append("CRITICAL ALERTS:")
            for alert in self.get_active_alerts(severity='critical'):
                lines.append(f"  ! {alert.message}")
                if alert.recommended_action:
                    lines.append(f"    Action: {alert.recommended_action}")
            lines.append("")

        if summary['warning'] > 0:
            lines.append("WARNINGS:")
            for alert in self.get_active_alerts(severity='warning'):
                lines.append(f"  * {alert.message}")
            lines.append("")

        lines.append("=" * 50)

        return "\n".join(lines)

    def get_alerts_df(self) -> pd.DataFrame:
        """Get alert history as DataFrame."""
        return pd.DataFrame([a.to_dict() for a in self.alert_history])


# Convenience functions for quick alerts
def log_console_handler(alert: Alert):
    """Simple console logging handler."""
    icon = "🔴" if alert.severity == 'critical' else "🟡" if alert.severity == 'warning' else "🔵"
    print(f"\n{icon} [{alert.severity.upper()}] {alert.message}")
    if alert.recommended_action:
        print(f"   Recommended: {alert.recommended_action}")


def create_default_alert_system() -> AlertSystem:
    """Create AlertSystem with default configuration."""
    system = AlertSystem()
    system.add_notification_handler(log_console_handler)
    return system
