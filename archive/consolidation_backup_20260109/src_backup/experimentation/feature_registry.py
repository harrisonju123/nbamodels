"""
Feature Registry for tracking feature candidates and experiments.

Provides:
- Feature candidate registration
- Experiment result tracking
- Feature lifecycle management (propose → test → accept/reject)
"""

import uuid
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any

from src.versioning.db import get_db_manager

logger = logging.getLogger(__name__)


@dataclass
class FeatureCandidate:
    """Feature candidate record."""

    feature_id: str
    feature_name: str
    status: str  # "proposed", "testing", "accepted", "rejected"
    created_at: datetime

    category: Optional[str] = None
    description: Optional[str] = None
    hypothesis: Optional[str] = None
    builder_class: Optional[str] = None
    dependencies: Optional[List[str]] = None

    tested_at: Optional[datetime] = None
    decided_at: Optional[datetime] = None


@dataclass
class FeatureExperimentResult:
    """Results from a feature experiment."""

    experiment_id: int
    feature_id: str
    feature_name: str
    experiment_date: datetime

    # Performance comparison
    baseline_accuracy: float
    experiment_accuracy: float
    accuracy_lift: float

    baseline_roi: float
    experiment_roi: float
    roi_lift: float

    # Statistical significance
    p_value: float
    is_significant: bool
    effect_size: float

    # Feature importance
    feature_importance_rank: Optional[int] = None
    feature_importance_gain: Optional[float] = None

    # Decision
    decision: str = "pending"  # "accept", "reject", "needs_more_testing"
    decision_reason: Optional[str] = None


class FeatureRegistry:
    """Registry for tracking feature candidates and experiments."""

    def __init__(self, db_dir: str = "data/versioning"):
        self.db_manager = get_db_manager(db_dir)
        # Initialize database if it doesn't exist
        self.db_manager.init_feature_registry()

    def propose_feature(
        self,
        name: str,
        category: str = "custom",
        description: str = "",
        hypothesis: str = "",
        builder_class: Optional[str] = None,
        dependencies: Optional[List[str]] = None
    ) -> str:
        """
        Propose a new feature for testing.

        Args:
            name: Feature name (must be unique)
            category: Feature category (e.g., 'team_stats', 'matchup', 'alternative')
            description: What this feature measures
            hypothesis: Why this feature might help
            builder_class: Which FeatureBuilder implements it
            dependencies: List of required data sources

        Returns:
            feature_id (UUID)
        """
        feature_id = str(uuid.uuid4())

        conn = self.db_manager.get_connection('feature_registry')
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO feature_candidates (
                    feature_id, feature_name, category, description,
                    hypothesis, builder_class, dependencies, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'proposed')
            """, (
                feature_id, name, category, description,
                hypothesis, builder_class,
                json.dumps(dependencies) if dependencies else None
            ))

            conn.commit()
            logger.info(f"Proposed feature '{name}' (ID: {feature_id})")
            return feature_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to propose feature: {e}")
            raise
        finally:
            conn.close()

    def get_feature_by_id(self, feature_id: str) -> Optional[FeatureCandidate]:
        """Get feature candidate by ID."""
        conn = self.db_manager.get_connection('feature_registry')
        cursor = conn.cursor()

        cursor.execute("""
            SELECT feature_id, feature_name, category, description, hypothesis,
                   builder_class, dependencies, status, created_at, tested_at, decided_at
            FROM feature_candidates
            WHERE feature_id = ?
        """, (feature_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return FeatureCandidate(
                feature_id=row[0],
                feature_name=row[1],
                category=row[2],
                description=row[3],
                hypothesis=row[4],
                builder_class=row[5],
                dependencies=json.loads(row[6]) if row[6] else None,
                status=row[7],
                created_at=datetime.fromisoformat(row[8]),
                tested_at=datetime.fromisoformat(row[9]) if row[9] else None,
                decided_at=datetime.fromisoformat(row[10]) if row[10] else None
            )
        return None

    def get_feature_by_name(self, name: str) -> Optional[FeatureCandidate]:
        """Get feature candidate by name."""
        conn = self.db_manager.get_connection('feature_registry')
        cursor = conn.cursor()

        cursor.execute("""
            SELECT feature_id, feature_name, category, description, hypothesis,
                   builder_class, dependencies, status, created_at, tested_at, decided_at
            FROM feature_candidates
            WHERE feature_name = ?
        """, (name,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return FeatureCandidate(
                feature_id=row[0],
                feature_name=row[1],
                category=row[2],
                description=row[3],
                hypothesis=row[4],
                builder_class=row[5],
                dependencies=json.loads(row[6]) if row[6] else None,
                status=row[7],
                created_at=datetime.fromisoformat(row[8]),
                tested_at=datetime.fromisoformat(row[9]) if row[9] else None,
                decided_at=datetime.fromisoformat(row[10]) if row[10] else None
            )
        return None

    def get_pending_features(self) -> List[FeatureCandidate]:
        """Get features with 'proposed' status awaiting testing."""
        return self.get_features_by_status('proposed')

    def get_features_by_status(self, status: str) -> List[FeatureCandidate]:
        """Get features by status."""
        conn = self.db_manager.get_connection('feature_registry')
        cursor = conn.cursor()

        cursor.execute("""
            SELECT feature_id, feature_name, category, description, hypothesis,
                   builder_class, dependencies, status, created_at, tested_at, decided_at
            FROM feature_candidates
            WHERE status = ?
            ORDER BY created_at DESC
        """, (status,))

        rows = cursor.fetchall()
        conn.close()

        return [
            FeatureCandidate(
                feature_id=row[0],
                feature_name=row[1],
                category=row[2],
                description=row[3],
                hypothesis=row[4],
                builder_class=row[5],
                dependencies=json.loads(row[6]) if row[6] else None,
                status=row[7],
                created_at=datetime.fromisoformat(row[8]),
                tested_at=datetime.fromisoformat(row[9]) if row[9] else None,
                decided_at=datetime.fromisoformat(row[10]) if row[10] else None
            )
            for row in rows
        ]

    def record_experiment(
        self,
        feature_id: str,
        baseline_model_id: Optional[str],
        experiment_model_id: Optional[str],
        baseline_accuracy: float,
        experiment_accuracy: float,
        baseline_roi: float,
        experiment_roi: float,
        p_value: float,
        is_significant: bool,
        effect_size: float = 0.0,
        feature_importance_rank: Optional[int] = None,
        feature_importance_gain: Optional[float] = None,
        full_results: Optional[Dict] = None
    ) -> int:
        """
        Record experiment results for a feature.

        Returns:
            experiment_id
        """
        conn = self.db_manager.get_connection('feature_registry')
        cursor = conn.cursor()

        try:
            accuracy_lift = experiment_accuracy - baseline_accuracy
            roi_lift = experiment_roi - baseline_roi

            cursor.execute("""
                INSERT INTO feature_experiments (
                    feature_id, baseline_model_id, experiment_model_id,
                    baseline_accuracy, experiment_accuracy, accuracy_lift,
                    baseline_roi, experiment_roi, roi_lift,
                    p_value, is_significant, effect_size,
                    feature_importance_rank, feature_importance_gain,
                    full_results
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feature_id, baseline_model_id, experiment_model_id,
                baseline_accuracy, experiment_accuracy, accuracy_lift,
                baseline_roi, experiment_roi, roi_lift,
                p_value, is_significant, effect_size,
                feature_importance_rank, feature_importance_gain,
                json.dumps(full_results) if full_results else None
            ))

            experiment_id = cursor.lastrowid

            # Update feature status to 'testing'
            cursor.execute("""
                UPDATE feature_candidates
                SET status = 'testing', tested_at = CURRENT_TIMESTAMP
                WHERE feature_id = ?
            """, (feature_id,))

            conn.commit()
            logger.info(f"Recorded experiment for feature {feature_id} (exp_id: {experiment_id})")
            return experiment_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to record experiment: {e}")
            raise
        finally:
            conn.close()

    def accept_feature(self, feature_id: str, reason: str = "") -> bool:
        """Mark feature as accepted."""
        return self._update_feature_decision(feature_id, "accepted", reason)

    def reject_feature(self, feature_id: str, reason: str = "") -> bool:
        """Mark feature as rejected."""
        return self._update_feature_decision(feature_id, "rejected", reason)

    def _update_feature_decision(
        self,
        feature_id: str,
        status: str,
        reason: str
    ) -> bool:
        """Update feature decision status."""
        conn = self.db_manager.get_connection('feature_registry')
        cursor = conn.cursor()

        try:
            # Update feature status
            cursor.execute("""
                UPDATE feature_candidates
                SET status = ?, decided_at = CURRENT_TIMESTAMP
                WHERE feature_id = ?
            """, (status, feature_id))

            # Update latest experiment decision
            cursor.execute("""
                UPDATE feature_experiments
                SET decision = ?, decision_reason = ?
                WHERE feature_id = ?
                  AND experiment_id = (
                      SELECT MAX(experiment_id)
                      FROM feature_experiments
                      WHERE feature_id = ?
                  )
            """, (status, reason, feature_id, feature_id))

            conn.commit()
            logger.info(f"Feature {feature_id} marked as {status}: {reason}")
            return True

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update feature decision: {e}")
            return False
        finally:
            conn.close()

    def get_experiment_results(
        self,
        feature_id: str
    ) -> List[FeatureExperimentResult]:
        """Get all experiment results for a feature."""
        conn = self.db_manager.get_connection('feature_registry')
        cursor = conn.cursor()

        cursor.execute("""
            SELECT e.experiment_id, e.feature_id, f.feature_name, e.experiment_date,
                   e.baseline_accuracy, e.experiment_accuracy, e.accuracy_lift,
                   e.baseline_roi, e.experiment_roi, e.roi_lift,
                   e.p_value, e.is_significant, e.effect_size,
                   e.feature_importance_rank, e.feature_importance_gain,
                   e.decision, e.decision_reason
            FROM feature_experiments e
            JOIN feature_candidates f ON e.feature_id = f.feature_id
            WHERE e.feature_id = ?
            ORDER BY e.experiment_date DESC
        """, (feature_id,))

        rows = cursor.fetchall()
        conn.close()

        return [
            FeatureExperimentResult(
                experiment_id=row[0],
                feature_id=row[1],
                feature_name=row[2],
                experiment_date=datetime.fromisoformat(row[3]),
                baseline_accuracy=row[4],
                experiment_accuracy=row[5],
                accuracy_lift=row[6],
                baseline_roi=row[7],
                experiment_roi=row[8],
                roi_lift=row[9],
                p_value=row[10],
                is_significant=bool(row[11]),
                effect_size=row[12],
                feature_importance_rank=row[13],
                feature_importance_gain=row[14],
                decision=row[15] or "pending",
                decision_reason=row[16]
            )
            for row in rows
        ]
