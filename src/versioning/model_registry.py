"""
Model Registry for tracking model versions and their metadata.

Provides central repository for all model versions with:
- Semantic versioning
- Champion/challenger status tracking
- Performance metrics storage
- Model comparison results
- Promotion history
"""

import uuid
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pathlib import Path

from .db import get_db_manager
from .version import Version, parse_version

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Model version record."""

    model_id: str
    model_name: str
    version: str
    status: str  # "champion", "challenger", "archived", "failed"
    created_at: datetime
    model_path: str

    created_by: Optional[str] = None
    description: Optional[str] = None
    parent_version: Optional[str] = None
    metadata_path: Optional[str] = None

    train_start_date: Optional[date] = None
    train_end_date: Optional[date] = None
    games_count: Optional[int] = None
    feature_count: Optional[int] = None

    @property
    def version_obj(self) -> Version:
        """Get Version object from version string."""
        return parse_version(self.version)


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    # Classification metrics
    accuracy: Optional[float] = None
    auc_roc: Optional[float] = None
    log_loss: Optional[float] = None
    brier_score: Optional[float] = None
    calibration_error: Optional[float] = None

    # Betting metrics
    roi: Optional[float] = None
    win_rate: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    num_bets: Optional[int] = None
    total_wagered: Optional[float] = None

    # Confidence intervals (stored as JSON)
    confidence_intervals: Optional[Dict[str, tuple]] = None

    # Test period
    test_period_start: Optional[date] = None
    test_period_end: Optional[date] = None


class ModelRegistry:
    """Central registry for model versions."""

    def __init__(self, db_dir: str = "data/versioning"):
        self.db_manager = get_db_manager(db_dir)
        # Initialize databases if they don't exist
        self.db_manager.init_model_registry()

    def _validate_model_path(self, model_path: str) -> str:
        """
        Validate and normalize model path.

        Args:
            model_path: Path to model file

        Returns:
            Normalized absolute path as string

        Raises:
            ValueError: If path is invalid or outside allowed directory
            FileNotFoundError: If file doesn't exist
        """
        # Convert to Path object and resolve
        path = Path(model_path).resolve()

        # Ensure path is within models directory
        models_dir = Path("models").resolve()

        try:
            # Check if path is relative to models directory
            path.relative_to(models_dir)
        except ValueError:
            raise ValueError(
                f"Model path must be within models directory. "
                f"Got: {model_path}, Expected prefix: {models_dir}"
            )

        # Check file exists
        if not path.exists():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")

        # Check it's a file, not directory
        if not path.is_file():
            raise ValueError(f"Model path must be a file: {model_path}")

        # Check extension
        if path.suffix not in {'.pkl', '.pickle'}:
            raise ValueError(f"Invalid model file extension: {path.suffix}")

        # Return normalized absolute path as string
        return str(path)

    def register_model(
        self,
        model_name: str,
        version: str,
        model_path: str,
        description: str = "",
        parent_version: Optional[str] = None,
        metrics: Optional[ModelMetrics] = None,
        status: str = "challenger",
        created_by: str = "system",
        metadata_path: Optional[str] = None,
        train_start_date: Optional[date] = None,
        train_end_date: Optional[date] = None,
        games_count: Optional[int] = None,
        feature_count: Optional[int] = None,
    ) -> str:
        """
        Register a new model version.

        Args:
            model_name: Model name (e.g., 'spread_model')
            version: Semantic version (e.g., '1.2.0')
            model_path: Path to model pickle file
            description: Description of changes
            parent_version: Version this was based on
            metrics: Optional initial metrics
            status: Model status ('champion', 'challenger', 'archived')
            created_by: Who created this version
            metadata_path: Path to metadata JSON file
            train_start_date: Training data start date
            train_end_date: Training data end date
            games_count: Number of games in training data
            feature_count: Number of features

        Returns:
            model_id (UUID)
        """
        # Validate paths before storing
        validated_model_path = self._validate_model_path(model_path)
        validated_metadata_path = None
        if metadata_path:
            # Metadata can be JSON files
            meta_path = Path(metadata_path).resolve()
            models_dir = Path("models").resolve()
            try:
                meta_path.relative_to(models_dir)
                if meta_path.exists() and meta_path.is_file():
                    validated_metadata_path = str(meta_path)
            except ValueError:
                logger.warning(f"Metadata path outside models directory, ignoring: {metadata_path}")

        model_id = str(uuid.uuid4())

        with self.db_manager.get_connection('model_registry') as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO models (
                        model_id, model_name, version, status, created_by,
                        description, parent_version, model_path, metadata_path,
                        train_start_date, train_end_date, games_count, feature_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_id, model_name, version, status, created_by,
                    description, parent_version, validated_model_path, validated_metadata_path,
                    train_start_date, train_end_date, games_count, feature_count
                ))

                conn.commit()
                logger.info(f"Registered model {model_name} v{version} (ID: {model_id})")

                # Record metrics if provided
                if metrics:
                    self.record_metrics(model_id, metrics, metric_type="training")

                return model_id

            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to register model: {e}")
                raise

    def get_champion(self, model_name: str) -> Optional[ModelVersion]:
        """
        Get current champion for a model type.

        Args:
            model_name: Model name

        Returns:
            ModelVersion if exists, None otherwise
        """
        with self.db_manager.get_connection('model_registry') as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model_id, model_name, version, status, created_at, created_by,
                       description, parent_version, model_path, metadata_path,
                       train_start_date, train_end_date, games_count, feature_count
                FROM models
                WHERE model_name = ? AND status = 'champion'
                ORDER BY created_at DESC
                LIMIT 1
            """, (model_name,))

            row = cursor.fetchone()

            if row:
                return ModelVersion(
                    model_id=row['model_id'],
                    model_name=row['model_name'],
                    version=row['version'],
                    status=row['status'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    created_by=row['created_by'],
                    description=row['description'],
                    parent_version=row['parent_version'],
                    model_path=row['model_path'],
                    metadata_path=row['metadata_path'],
                    train_start_date=date.fromisoformat(row['train_start_date']) if row['train_start_date'] else None,
                    train_end_date=date.fromisoformat(row['train_end_date']) if row['train_end_date'] else None,
                    games_count=row['games_count'],
                    feature_count=row['feature_count']
                )
            return None

    def get_challengers(self, model_name: str) -> List[ModelVersion]:
        """
        Get all challenger versions for a model type.

        Args:
            model_name: Model name

        Returns:
            List of ModelVersion objects
        """
        with self.db_manager.get_connection('model_registry') as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model_id, model_name, version, status, created_at, created_by,
                       description, parent_version, model_path, metadata_path,
                       train_start_date, train_end_date, games_count, feature_count
                FROM models
                WHERE model_name = ? AND status = 'challenger'
                ORDER BY created_at DESC
            """, (model_name,))

            rows = cursor.fetchall()

            return [
                ModelVersion(
                    model_id=row['model_id'],
                    model_name=row['model_name'],
                    version=row['version'],
                    status=row['status'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    created_by=row['created_by'],
                    description=row['description'],
                    parent_version=row['parent_version'],
                    model_path=row['model_path'],
                    metadata_path=row['metadata_path'],
                    train_start_date=date.fromisoformat(row['train_start_date']) if row['train_start_date'] else None,
                    train_end_date=date.fromisoformat(row['train_end_date']) if row['train_end_date'] else None,
                    games_count=row['games_count'],
                    feature_count=row['feature_count']
                )
                for row in rows
            ]

    def get_model_by_id(self, model_id: str) -> Optional[ModelVersion]:
        """Get model by ID."""
        with self.db_manager.get_connection('model_registry') as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model_id, model_name, version, status, created_at, created_by,
                       description, parent_version, model_path, metadata_path,
                       train_start_date, train_end_date, games_count, feature_count
                FROM models
                WHERE model_id = ?
            """, (model_id,))

            row = cursor.fetchone()

            if row:
                return ModelVersion(
                    model_id=row['model_id'],
                    model_name=row['model_name'],
                    version=row['version'],
                    status=row['status'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    created_by=row['created_by'],
                    description=row['description'],
                    parent_version=row['parent_version'],
                    model_path=row['model_path'],
                    metadata_path=row['metadata_path'],
                    train_start_date=date.fromisoformat(row['train_start_date']) if row['train_start_date'] else None,
                    train_end_date=date.fromisoformat(row['train_end_date']) if row['train_end_date'] else None,
                    games_count=row['games_count'],
                    feature_count=row['feature_count']
                )
            return None

    def promote_to_champion(
        self,
        model_id: str,
        reason: str,
        comparison_id: Optional[int] = None,
        promoted_by: str = "system"
    ) -> bool:
        """
        Promote a challenger to champion with atomic transaction.

        Uses BEGIN IMMEDIATE to prevent race conditions between
        concurrent promotion attempts.

        Args:
            model_id: Model ID to promote
            reason: Reason for promotion
            comparison_id: Optional comparison ID that justified promotion
            promoted_by: Who initiated the promotion

        Returns:
            True if successful
        """
        with self.db_manager.get_connection('model_registry') as conn:
            # Enable immediate transaction to lock database
            conn.execute("BEGIN IMMEDIATE")

            try:
                cursor = conn.cursor()

                # Get the model to promote (within transaction)
                cursor.execute("""
                    SELECT model_id, model_name, version, status
                    FROM models
                    WHERE model_id = ?
                """, (model_id,))
                row = cursor.fetchone()

                if not row:
                    raise ValueError(f"Model not found: {model_id}")

                model_name = row['model_name']
                model_version = row['version']
                model_status = row['status']

                if model_status == "champion":
                    logger.info(f"Model {model_id} is already champion")
                    conn.commit()
                    return True

                # Atomically archive all current champions for this model
                cursor.execute("""
                    UPDATE models
                    SET status = 'archived'
                    WHERE model_name = ? AND status = 'champion'
                """, (model_name,))

                old_champion_count = cursor.rowcount
                old_champion_id = None

                # Get the old champion ID if any was archived
                if old_champion_count > 0:
                    cursor.execute("""
                        SELECT model_id, version FROM models
                        WHERE model_name = ? AND status = 'archived'
                        ORDER BY created_at DESC LIMIT 1
                    """, (model_name,))
                    old_row = cursor.fetchone()
                    if old_row:
                        old_champion_id = old_row['model_id']
                        logger.info(f"Archived old champion: {old_row['version']}")

                # Promote new champion
                cursor.execute("""
                    UPDATE models
                    SET status = 'champion'
                    WHERE model_id = ?
                """, (model_id,))

                # Record promotion history
                cursor.execute("""
                    INSERT INTO promotion_history (
                        model_name, old_champion_id, new_champion_id,
                        promoted_by, reason, comparison_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (model_name, old_champion_id, model_id,
                      promoted_by, reason, comparison_id))

                conn.commit()
                logger.info(f"Promoted {model_name} v{model_version} to champion")
                return True

            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to promote model: {e}")
                raise

    def archive_model(self, model_id: str, reason: str = "") -> bool:
        """Archive a model version."""
        with self.db_manager.get_connection('model_registry') as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE models
                    SET status = 'archived'
                    WHERE model_id = ?
                """, (model_id,))

                conn.commit()
                logger.info(f"Archived model {model_id}: {reason}")
                return True

            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to archive model: {e}")
                return False

    def record_metrics(
        self,
        model_id: str,
        metrics: ModelMetrics,
        metric_type: str = "backtest"
    ) -> int:
        """
        Record performance metrics for a model.

        Args:
            model_id: Model ID
            metrics: ModelMetrics object
            metric_type: Type of metrics ('training', 'validation', 'backtest', 'live')

        Returns:
            metric_id
        """
        with self.db_manager.get_connection('model_registry') as conn:
            try:
                cursor = conn.cursor()
                # Serialize confidence intervals as JSON
                ci_json = json.dumps(metrics.confidence_intervals) if metrics.confidence_intervals else None

                cursor.execute("""
                    INSERT INTO model_metrics (
                        model_id, metric_type,
                        accuracy, auc_roc, log_loss, brier_score, calibration_error,
                        roi, win_rate, sharpe_ratio, max_drawdown, num_bets, total_wagered,
                        confidence_intervals, test_period_start, test_period_end
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_id, metric_type,
                    metrics.accuracy, metrics.auc_roc, metrics.log_loss,
                    metrics.brier_score, metrics.calibration_error,
                    metrics.roi, metrics.win_rate, metrics.sharpe_ratio,
                    metrics.max_drawdown, metrics.num_bets, metrics.total_wagered,
                    ci_json, metrics.test_period_start, metrics.test_period_end
                ))

                metric_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Recorded {metric_type} metrics for model {model_id}")
                return metric_id

            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to record metrics: {e}")
                raise

    def get_model_history(
        self,
        model_name: str,
        limit: int = 10
    ) -> List[ModelVersion]:
        """Get version history for a model type."""
        with self.db_manager.get_connection('model_registry') as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model_id, model_name, version, status, created_at, created_by,
                       description, parent_version, model_path, metadata_path,
                       train_start_date, train_end_date, games_count, feature_count
                FROM models
                WHERE model_name = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (model_name, limit))

            rows = cursor.fetchall()

            return [
                ModelVersion(
                    model_id=row['model_id'],
                    model_name=row['model_name'],
                    version=row['version'],
                    status=row['status'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    created_by=row['created_by'],
                    description=row['description'],
                    parent_version=row['parent_version'],
                    model_path=row['model_path'],
                    metadata_path=row['metadata_path'],
                    train_start_date=date.fromisoformat(row['train_start_date']) if row['train_start_date'] else None,
                    train_end_date=date.fromisoformat(row['train_end_date']) if row['train_end_date'] else None,
                    games_count=row['games_count'],
                    feature_count=row['feature_count']
                )
                for row in rows
            ]
