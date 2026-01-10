"""
Database management for model versioning system.

Provides SQLite database initialization and schema management
for model registry, feature registry, and performance history.
"""

import sqlite3
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite databases for the versioning system."""

    def __init__(self, db_dir: str = "data/versioning"):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        self.model_registry_path = self.db_dir / "model_registry.db"
        self.feature_registry_path = self.db_dir / "feature_registry.db"
        self.performance_history_path = self.db_dir / "performance_history.db"

    def init_model_registry(self) -> None:
        """Initialize model registry database with schema."""
        conn = sqlite3.connect(self.model_registry_path)
        cursor = conn.cursor()

        # Models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                version TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                description TEXT,
                parent_version TEXT,
                model_path TEXT NOT NULL,
                metadata_path TEXT,

                train_start_date DATE,
                train_end_date DATE,
                games_count INTEGER,
                feature_count INTEGER,

                UNIQUE(model_name, version)
            )
        """)

        # Model metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metric_type TEXT NOT NULL,

                accuracy REAL,
                auc_roc REAL,
                log_loss REAL,
                brier_score REAL,
                calibration_error REAL,

                roi REAL,
                win_rate REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                num_bets INTEGER,
                total_wagered REAL,

                confidence_intervals TEXT,

                test_period_start DATE,
                test_period_end DATE,
                backtest_config TEXT,

                FOREIGN KEY (model_id) REFERENCES models(model_id)
            )
        """)

        # Model comparisons table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_comparisons (
                comparison_id INTEGER PRIMARY KEY AUTOINCREMENT,
                champion_model_id TEXT NOT NULL,
                challenger_model_id TEXT NOT NULL,
                comparison_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                champion_roi REAL,
                challenger_roi REAL,
                roi_difference REAL,
                p_value REAL,
                is_significant BOOLEAN,

                winner TEXT,
                recommendation TEXT,

                full_results TEXT,

                FOREIGN KEY (champion_model_id) REFERENCES models(model_id),
                FOREIGN KEY (challenger_model_id) REFERENCES models(model_id)
            )
        """)

        # Promotion history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS promotion_history (
                promotion_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                old_champion_id TEXT,
                new_champion_id TEXT NOT NULL,
                promoted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                promoted_by TEXT,
                reason TEXT,
                comparison_id INTEGER,

                FOREIGN KEY (old_champion_id) REFERENCES models(model_id),
                FOREIGN KEY (new_champion_id) REFERENCES models(model_id),
                FOREIGN KEY (comparison_id) REFERENCES model_comparisons(comparison_id)
            )
        """)

        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_name_status ON models(model_name, status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_model_id ON model_metrics(model_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_comparisons_date ON model_comparisons(comparison_date)")

        conn.commit()
        conn.close()
        logger.info(f"Initialized model registry database at {self.model_registry_path}")

    def init_feature_registry(self) -> None:
        """Initialize feature registry database with schema."""
        conn = sqlite3.connect(self.feature_registry_path)
        cursor = conn.cursor()

        # Feature candidates table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_candidates (
                feature_id TEXT PRIMARY KEY,
                feature_name TEXT NOT NULL UNIQUE,
                category TEXT,
                description TEXT,
                hypothesis TEXT,

                builder_class TEXT,
                dependencies TEXT,

                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tested_at TIMESTAMP,
                decided_at TIMESTAMP
            )
        """)

        # Feature experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_experiments (
                experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_id TEXT NOT NULL,
                experiment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                baseline_model_id TEXT,
                experiment_model_id TEXT,

                baseline_accuracy REAL,
                experiment_accuracy REAL,
                accuracy_lift REAL,

                baseline_roi REAL,
                experiment_roi REAL,
                roi_lift REAL,

                p_value REAL,
                is_significant BOOLEAN,
                effect_size REAL,

                feature_importance_rank INTEGER,
                feature_importance_gain REAL,

                full_results TEXT,

                decision TEXT,
                decision_reason TEXT,

                FOREIGN KEY (feature_id) REFERENCES feature_candidates(feature_id)
            )
        """)

        # Feature correlations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_correlations (
                correlation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_id TEXT NOT NULL,
                correlated_feature TEXT NOT NULL,
                correlation_coefficient REAL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (feature_id) REFERENCES feature_candidates(feature_id)
            )
        """)

        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_features_status ON feature_candidates(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiments_feature ON feature_experiments(feature_id)")

        conn.commit()
        conn.close()
        logger.info(f"Initialized feature registry database at {self.feature_registry_path}")

    def init_performance_history(self) -> None:
        """Initialize performance history database with schema."""
        conn = sqlite3.connect(self.performance_history_path)
        cursor = conn.cursor()

        # Daily performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_performance (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                date DATE NOT NULL,

                predictions_count INTEGER,
                bets_placed INTEGER,

                accuracy REAL,
                auc_roc REAL,

                roi REAL,
                pnl REAL,
                win_rate REAL,

                roi_7d REAL,
                roi_30d REAL,
                sharpe_30d REAL,

                UNIQUE(model_id, date)
            )
        """)

        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT NOT NULL,
                model_id TEXT,
                severity TEXT,
                message TEXT,
                metric_name TEXT,
                metric_value REAL,
                threshold REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                acknowledged_at TIMESTAMP,
                resolved_at TIMESTAMP
            )
        """)

        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_perf_model_date ON daily_performance(model_id, date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)")

        conn.commit()
        conn.close()
        logger.info(f"Initialized performance history database at {self.performance_history_path}")

    def init_all_databases(self) -> None:
        """Initialize all databases."""
        self.init_model_registry()
        self.init_feature_registry()
        self.init_performance_history()
        logger.info("All databases initialized successfully")

    def get_connection(self, db_name: str) -> sqlite3.Connection:
        """
        Get database connection with context manager support.

        Args:
            db_name: One of 'model_registry', 'feature_registry', 'performance_history'

        Returns:
            SQLite connection with row_factory configured

        Example:
            with db_manager.get_connection('model_registry') as conn:
                cursor = conn.cursor()
                cursor.execute(...)
                conn.commit()
        """
        db_paths = {
            'model_registry': self.model_registry_path,
            'feature_registry': self.feature_registry_path,
            'performance_history': self.performance_history_path
        }

        if db_name not in db_paths:
            raise ValueError(f"Unknown database: {db_name}")

        conn = sqlite3.connect(db_paths[db_name])
        # Enable row factory for named column access
        conn.row_factory = sqlite3.Row
        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager(db_dir: str = "data/versioning") -> DatabaseManager:
    """Get or create the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(db_dir)
    return _db_manager
