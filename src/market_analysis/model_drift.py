"""
Model Drift Monitoring

Tracks model calibration and prediction quality over time to detect performance degradation.
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from ..models.calibration import evaluate_calibration


@dataclass
class CalibrationSnapshot:
    """Snapshot of model calibration metrics at a point in time."""

    timestamp: str
    brier_score: float
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    log_loss: float
    accuracy: float
    n_samples: int
    by_prob_bucket: Dict[str, Dict[str, float]]


@dataclass
class DriftAlert:
    """Alert when model performance has drifted."""

    metric: str  # 'brier', 'ece', 'accuracy', 'log_loss'
    severity: str  # 'warning', 'critical'
    current_value: float
    baseline_value: float
    change_pct: float
    recommendation: str


class ModelDriftMonitor:
    """Track calibration and prediction quality over time."""

    # Thresholds for drift detection
    BRIER_INCREASE_WARNING = 0.10  # 10% increase
    BRIER_INCREASE_CRITICAL = 0.20  # 20% increase
    ACCURACY_DROP_WARNING = 0.05  # 5 percentage points
    ACCURACY_DROP_CRITICAL = 0.10  # 10 percentage points
    ECE_INCREASE_WARNING = 0.02  # 2 percentage point increase
    ECE_INCREASE_CRITICAL = 0.05  # 5 percentage point increase

    def __init__(
        self,
        db_path: str = "data/bets/bets.db",
        baseline_window: int = 200,
        alert_window: int = 50,
    ):
        """
        Initialize drift monitor.

        Args:
            db_path: Path to bets database
            baseline_window: Number of bets to use for baseline metrics
            alert_window: Number of recent bets to check for drift
        """
        self.db_path = db_path
        self.baseline_window = baseline_window
        self.alert_window = alert_window

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_settled_bets(
        self,
        limit: Optional[int] = None,
        model_type: Optional[str] = None,
        strategy_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get settled bets with predictions and outcomes.

        Args:
            limit: Maximum number of bets to return (most recent)
            model_type: Filter by model type (spread, totals, moneyline)
            strategy_type: Filter by strategy type

        Returns:
            DataFrame with columns: id, model_prob, outcome, bet_type, strategy_type
        """
        conn = self._get_connection()

        query = """
            SELECT
                id,
                model_prob,
                outcome,
                bet_type,
                strategy_type,
                settled_at
            FROM bets
            WHERE outcome IS NOT NULL
                AND outcome != 'push'
                AND model_prob IS NOT NULL
        """

        params = []

        if model_type:
            query += " AND bet_type = ?"
            params.append(model_type)

        if strategy_type:
            query += " AND strategy_type = ?"
            params.append(strategy_type)

        query += " ORDER BY settled_at DESC"

        if limit:
            query += f" LIMIT {limit}"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        # Convert outcome to binary (win=1, loss=0)
        df['y_true'] = (df['outcome'] == 'win').astype(int)
        df['y_prob'] = df['model_prob']

        return df

    def calculate_rolling_brier(
        self,
        bets_df: pd.DataFrame,
        window: int = 50,
    ) -> pd.DataFrame:
        """
        Calculate rolling Brier score over time.

        Args:
            bets_df: DataFrame with y_true and y_prob columns
            window: Rolling window size

        Returns:
            DataFrame with rolling Brier scores
        """
        from sklearn.metrics import brier_score_loss

        brier_scores = []

        for i in range(len(bets_df)):
            if i < window - 1:
                continue

            window_data = bets_df.iloc[i - window + 1 : i + 1]
            brier = brier_score_loss(
                window_data['y_true'],
                window_data['y_prob'],
            )
            brier_scores.append({
                'index': i,
                'brier_score': brier,
                'n_samples': len(window_data),
            })

        return pd.DataFrame(brier_scores)

    def calculate_rolling_calibration(
        self,
        bets_df: pd.DataFrame,
        window: int = 50,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Calculate rolling calibration metrics over time.

        Args:
            bets_df: DataFrame with y_true and y_prob columns
            window: Rolling window size
            n_bins: Number of bins for calibration

        Returns:
            DataFrame with rolling ECE and MCE
        """
        results = []

        for i in range(len(bets_df)):
            if i < window - 1:
                continue

            window_data = bets_df.iloc[i - window + 1 : i + 1]

            metrics = evaluate_calibration(
                window_data['y_true'].values,
                window_data['y_prob'].values,
                n_bins=n_bins,
            )

            results.append({
                'index': i,
                'ece': metrics['ece'],
                'mce': metrics['mce'],
                'brier_score': metrics['brier_score'],
                'log_loss': metrics['log_loss'],
                'n_samples': len(window_data),
            })

        return pd.DataFrame(results)

    def calculate_accuracy_by_prob_bucket(
        self,
        bets_df: pd.DataFrame,
        n_buckets: int = 10,
    ) -> pd.DataFrame:
        """
        Calculate accuracy within predicted probability buckets.

        Args:
            bets_df: DataFrame with y_true and y_prob columns
            n_buckets: Number of probability buckets

        Returns:
            DataFrame with bucket stats
        """
        bucket_edges = np.linspace(0, 1, n_buckets + 1)
        bets_df['bucket'] = pd.cut(
            bets_df['y_prob'],
            bins=bucket_edges,
            include_lowest=True,
        )

        bucket_stats = bets_df.groupby('bucket').agg({
            'y_true': ['mean', 'count'],
            'y_prob': 'mean',
        }).reset_index()

        bucket_stats.columns = ['bucket', 'actual_win_rate', 'n_bets', 'avg_predicted_prob']
        bucket_stats['calibration_error'] = abs(
            bucket_stats['actual_win_rate'] - bucket_stats['avg_predicted_prob']
        )

        return bucket_stats

    def detect_calibration_drift(
        self,
        model_type: Optional[str] = None,
        strategy_type: Optional[str] = None,
    ) -> List[DriftAlert]:
        """
        Detect if model calibration has drifted.

        Compares recent performance (last alert_window bets) to
        baseline performance (older baseline_window bets).

        Args:
            model_type: Filter by model type
            strategy_type: Filter by strategy type

        Returns:
            List of drift alerts
        """
        alerts = []

        # Get bets: need baseline + alert windows
        total_needed = self.baseline_window + self.alert_window
        bets_df = self.get_settled_bets(
            limit=total_needed,
            model_type=model_type,
            strategy_type=strategy_type,
        )

        if len(bets_df) < total_needed:
            logger.warning(
                f"Not enough data for drift detection. "
                f"Need {total_needed}, have {len(bets_df)}"
            )
            return alerts

        # Split into baseline and recent
        # Since data is ordered DESC by settled_at, reverse it for chronological order
        bets_df = bets_df.iloc[::-1].reset_index(drop=True)

        baseline_df = bets_df.iloc[:self.baseline_window]
        recent_df = bets_df.iloc[-self.alert_window:]

        # Calculate baseline metrics
        baseline_metrics = evaluate_calibration(
            baseline_df['y_true'].values,
            baseline_df['y_prob'].values,
        )
        baseline_accuracy = baseline_df['y_true'].mean()

        # Calculate recent metrics
        recent_metrics = evaluate_calibration(
            recent_df['y_true'].values,
            recent_df['y_prob'].values,
        )
        recent_accuracy = recent_df['y_true'].mean()

        # Check Brier score drift
        brier_change = (
            (recent_metrics['brier_score'] - baseline_metrics['brier_score']) /
            baseline_metrics['brier_score']
        )

        if brier_change >= self.BRIER_INCREASE_CRITICAL:
            alerts.append(DriftAlert(
                metric='brier_score',
                severity='critical',
                current_value=recent_metrics['brier_score'],
                baseline_value=baseline_metrics['brier_score'],
                change_pct=brier_change * 100,
                recommendation=(
                    "Brier score has increased significantly. "
                    "Model predictions are becoming less reliable. "
                    "Consider retraining model or reducing bet sizes."
                ),
            ))
        elif brier_change >= self.BRIER_INCREASE_WARNING:
            alerts.append(DriftAlert(
                metric='brier_score',
                severity='warning',
                current_value=recent_metrics['brier_score'],
                baseline_value=baseline_metrics['brier_score'],
                change_pct=brier_change * 100,
                recommendation=(
                    "Brier score has increased. "
                    "Monitor model performance closely."
                ),
            ))

        # Check accuracy drift
        accuracy_drop = baseline_accuracy - recent_accuracy

        if accuracy_drop >= self.ACCURACY_DROP_CRITICAL:
            alerts.append(DriftAlert(
                metric='accuracy',
                severity='critical',
                current_value=recent_accuracy,
                baseline_value=baseline_accuracy,
                change_pct=accuracy_drop * 100,
                recommendation=(
                    "Win rate has dropped significantly. "
                    "Model performance has degraded. "
                    "Stop betting and investigate immediately."
                ),
            ))
        elif accuracy_drop >= self.ACCURACY_DROP_WARNING:
            alerts.append(DriftAlert(
                metric='accuracy',
                severity='warning',
                current_value=recent_accuracy,
                baseline_value=baseline_accuracy,
                change_pct=accuracy_drop * 100,
                recommendation=(
                    "Win rate has declined. "
                    "Monitor performance and consider reducing exposure."
                ),
            ))

        # Check ECE drift (calibration quality)
        ece_increase = recent_metrics['ece'] - baseline_metrics['ece']

        if ece_increase >= self.ECE_INCREASE_CRITICAL:
            alerts.append(DriftAlert(
                metric='ece',
                severity='critical',
                current_value=recent_metrics['ece'],
                baseline_value=baseline_metrics['ece'],
                change_pct=ece_increase * 100,
                recommendation=(
                    "Model calibration has degraded significantly. "
                    "Predicted probabilities are unreliable. "
                    "Recalibrate model or reduce position sizes."
                ),
            ))
        elif ece_increase >= self.ECE_INCREASE_WARNING:
            alerts.append(DriftAlert(
                metric='ece',
                severity='warning',
                current_value=recent_metrics['ece'],
                baseline_value=baseline_metrics['ece'],
                change_pct=ece_increase * 100,
                recommendation=(
                    "Model calibration has declined. "
                    "Verify probability estimates before betting."
                ),
            ))

        return alerts

    def save_calibration_snapshot(
        self,
        model_type: str,
        strategy_type: str,
        snapshot_date: Optional[str] = None,
    ) -> CalibrationSnapshot:
        """
        Calculate and save calibration snapshot to database.

        Args:
            model_type: Model type (spread, totals, moneyline)
            strategy_type: Strategy type
            snapshot_date: Date for snapshot (defaults to today)

        Returns:
            CalibrationSnapshot object
        """
        if snapshot_date is None:
            snapshot_date = datetime.now().date().isoformat()

        # Get recent settled bets
        bets_df = self.get_settled_bets(
            limit=self.baseline_window,
            model_type=model_type,
            strategy_type=strategy_type,
        )

        if len(bets_df) == 0:
            raise ValueError(f"No settled bets found for {model_type}/{strategy_type}")

        # Calculate calibration metrics
        metrics = evaluate_calibration(
            bets_df['y_true'].values,
            bets_df['y_prob'].values,
        )

        accuracy = bets_df['y_true'].mean()

        # Calculate accuracy by probability bucket
        bucket_stats = self.calculate_accuracy_by_prob_bucket(bets_df)
        by_prob_bucket = {}
        for _, row in bucket_stats.iterrows():
            bucket_key = str(row['bucket'])
            by_prob_bucket[bucket_key] = {
                'actual_win_rate': float(row['actual_win_rate']),
                'avg_predicted_prob': float(row['avg_predicted_prob']),
                'n_bets': int(row['n_bets']),
                'calibration_error': float(row['calibration_error']),
            }

        # Create snapshot
        snapshot = CalibrationSnapshot(
            timestamp=snapshot_date,
            brier_score=float(metrics['brier_score']),
            ece=float(metrics['ece']),
            mce=float(metrics['mce']),
            log_loss=float(metrics['log_loss']),
            accuracy=float(accuracy),
            n_samples=len(bets_df),
            by_prob_bucket=by_prob_bucket,
        )

        # Save to database
        conn = self._get_connection()

        conn.execute(
            """
            INSERT OR REPLACE INTO calibration_snapshots
                (snapshot_date, model_type, strategy_type, brier_score, log_loss,
                 ece, mce, accuracy, n_samples, prob_bucket_stats)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_date,
                model_type,
                strategy_type,
                snapshot.brier_score,
                snapshot.log_loss,
                snapshot.ece,
                snapshot.mce,
                snapshot.accuracy,
                snapshot.n_samples,
                json.dumps(snapshot.by_prob_bucket),
            ),
        )

        conn.commit()
        conn.close()

        logger.info(
            f"Saved calibration snapshot for {model_type}/{strategy_type}: "
            f"Brier={snapshot.brier_score:.4f}, ECE={snapshot.ece:.4f}, "
            f"Accuracy={snapshot.accuracy:.3f}"
        )

        return snapshot

    def get_calibration_trend(
        self,
        model_type: str,
        strategy_type: str,
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get calibration metrics trend over time.

        Args:
            model_type: Model type
            strategy_type: Strategy type
            days: Number of days to look back

        Returns:
            DataFrame with calibration metrics over time
        """
        conn = self._get_connection()

        query = """
            SELECT
                snapshot_date,
                brier_score,
                ece,
                mce,
                log_loss,
                accuracy,
                n_samples
            FROM calibration_snapshots
            WHERE model_type = ?
                AND strategy_type = ?
                AND snapshot_date >= date('now', ?)
            ORDER BY snapshot_date DESC
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=(model_type, strategy_type, f'-{days} days'),
        )

        conn.close()

        return df

    def compare_to_baseline(
        self,
        model_type: str,
        strategy_type: str,
        baseline_date: Optional[str] = None,
    ) -> Dict:
        """
        Compare current performance to a baseline snapshot.

        Args:
            model_type: Model type
            strategy_type: Strategy type
            baseline_date: Date of baseline snapshot (defaults to earliest)

        Returns:
            Dict with comparison metrics
        """
        conn = self._get_connection()

        # Get baseline snapshot
        if baseline_date:
            baseline_query = """
                SELECT * FROM calibration_snapshots
                WHERE model_type = ? AND strategy_type = ? AND snapshot_date = ?
            """
            baseline_row = conn.execute(
                baseline_query,
                (model_type, strategy_type, baseline_date),
            ).fetchone()
        else:
            baseline_query = """
                SELECT * FROM calibration_snapshots
                WHERE model_type = ? AND strategy_type = ?
                ORDER BY snapshot_date ASC
                LIMIT 1
            """
            baseline_row = conn.execute(
                baseline_query,
                (model_type, strategy_type),
            ).fetchone()

        # Get latest snapshot
        latest_query = """
            SELECT * FROM calibration_snapshots
            WHERE model_type = ? AND strategy_type = ?
            ORDER BY snapshot_date DESC
            LIMIT 1
        """
        latest_row = conn.execute(
            latest_query,
            (model_type, strategy_type),
        ).fetchone()

        conn.close()

        if not baseline_row or not latest_row:
            raise ValueError("Missing calibration snapshots for comparison")

        # Calculate changes
        return {
            'baseline_date': baseline_row['snapshot_date'],
            'latest_date': latest_row['snapshot_date'],
            'brier_change': latest_row['brier_score'] - baseline_row['brier_score'],
            'brier_change_pct': (
                (latest_row['brier_score'] - baseline_row['brier_score']) /
                baseline_row['brier_score'] * 100
            ),
            'ece_change': latest_row['ece'] - baseline_row['ece'],
            'accuracy_change': latest_row['accuracy'] - baseline_row['accuracy'],
            'baseline_accuracy': baseline_row['accuracy'],
            'latest_accuracy': latest_row['accuracy'],
        }
