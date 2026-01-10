"""
Risk Management Database Schema

Defines schema migrations for risk tracking tables:
- risk_snapshots: Daily risk state snapshots
- bet_risk_metadata: Per-bet risk factors
- team_attribution: Cached team P&L rollups
"""

import sqlite3
import json
from typing import Dict, Optional
from pathlib import Path
from loguru import logger


DB_PATH = "data/bets/bets.db"


class RiskDatabaseManager:
    """
    Manages risk management database schema and migrations.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._ensure_db_exists()

    def _get_connection(self):
        """Get database connection with foreign key enforcement."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_db_exists(self):
        """Ensure database file exists."""
        db_file = Path(self.db_path)
        if not db_file.exists():
            logger.warning(f"Database file not found: {self.db_path}")
            db_file.parent.mkdir(parents=True, exist_ok=True)

    def init_risk_tables(self):
        """
        Initialize all risk management tables.

        Creates:
        - risk_snapshots
        - bet_risk_metadata
        - team_attribution
        """
        conn = self._get_connection()

        try:
            # 1. Risk snapshots table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_date DATE NOT NULL UNIQUE,
                    bankroll REAL NOT NULL,
                    peak_bankroll REAL NOT NULL,
                    drawdown_pct REAL NOT NULL,
                    daily_wagered REAL DEFAULT 0,
                    weekly_wagered REAL DEFAULT 0,
                    pending_exposure REAL DEFAULT 0,
                    exposure_by_team TEXT,
                    exposure_by_conference TEXT,
                    exposure_by_division TEXT,
                    scale_factor REAL DEFAULT 1.0,
                    regime TEXT DEFAULT 'normal',
                    created_at TEXT NOT NULL
                )
            """)

            # 2. Bet risk metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bet_risk_metadata (
                    bet_id TEXT PRIMARY KEY,
                    correlation_factor REAL,
                    drawdown_factor REAL,
                    base_kelly REAL,
                    adjusted_kelly REAL,
                    same_team_exposure REAL,
                    same_game_exposure REAL,
                    same_conference_exposure REAL,
                    same_division_exposure REAL,
                    limiting_factor TEXT,
                    adjustments TEXT,
                    warnings TEXT,
                    FOREIGN KEY (bet_id) REFERENCES bets(id)
                )
            """)

            # 3. Team attribution cache
            conn.execute("""
                CREATE TABLE IF NOT EXISTS team_attribution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team TEXT NOT NULL,
                    period_start DATE NOT NULL,
                    period_end DATE NOT NULL,
                    bets_for INTEGER DEFAULT 0,
                    bets_against INTEGER DEFAULT 0,
                    profit_for REAL DEFAULT 0,
                    profit_against REAL DEFAULT 0,
                    wagered_for REAL DEFAULT 0,
                    wagered_against REAL DEFAULT 0,
                    win_rate_for REAL DEFAULT 0,
                    win_rate_against REAL DEFAULT 0,
                    roi_for REAL DEFAULT 0,
                    roi_against REAL DEFAULT 0,
                    UNIQUE(team, period_start, period_end)
                )
            """)

            # Create indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_risk_snapshots_date
                ON risk_snapshots(snapshot_date)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_team_attribution_team
                ON team_attribution(team)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_team_attribution_period
                ON team_attribution(period_start, period_end)
            """)

            conn.commit()
            logger.info("✓ Risk management tables initialized")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to initialize risk tables: {e}")
            raise
        finally:
            conn.close()

    def add_bets_table_columns(self):
        """
        Add risk-related columns to existing bets table.

        Adds:
        - home_b2b, away_b2b: Back-to-back flags
        - rest_advantage: Rest days difference
        - edge_bucket: Edge category (5-7%, 7-10%, 10%+)
        - conference, division: Team conference/division
        """
        conn = self._get_connection()

        columns_to_add = [
            ("home_b2b", "BOOLEAN DEFAULT 0"),
            ("away_b2b", "BOOLEAN DEFAULT 0"),
            ("rest_advantage", "INTEGER DEFAULT 0"),
            ("edge_bucket", "TEXT"),
            ("conference", "TEXT"),
            ("division", "TEXT"),
        ]

        for column_name, column_def in columns_to_add:
            try:
                conn.execute(f"ALTER TABLE bets ADD COLUMN {column_name} {column_def}")
                logger.info(f"✓ Added column {column_name} to bets table")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    logger.debug(f"Column {column_name} already exists")
                else:
                    logger.error(f"Failed to add column {column_name}: {e}")
                    raise

        conn.commit()
        conn.close()

    def record_risk_snapshot(self, snapshot_date: str, data: Dict):
        """
        Record daily risk snapshot.

        Args:
            snapshot_date: Date in YYYY-MM-DD format
            data: Dict with keys: bankroll, peak_bankroll, drawdown_pct,
                  daily_wagered, weekly_wagered, pending_exposure,
                  exposure_by_team, exposure_by_conference,
                  exposure_by_division, scale_factor, regime
        """
        conn = self._get_connection()

        try:
            # Convert dict fields to JSON
            exposure_by_team = json.dumps(data.get('exposure_by_team', {}))
            exposure_by_conference = json.dumps(data.get('exposure_by_conference', {}))
            exposure_by_division = json.dumps(data.get('exposure_by_division', {}))

            conn.execute("""
                INSERT OR REPLACE INTO risk_snapshots (
                    snapshot_date, bankroll, peak_bankroll, drawdown_pct,
                    daily_wagered, weekly_wagered, pending_exposure,
                    exposure_by_team, exposure_by_conference, exposure_by_division,
                    scale_factor, regime, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                snapshot_date,
                data['bankroll'],
                data['peak_bankroll'],
                data['drawdown_pct'],
                data.get('daily_wagered', 0),
                data.get('weekly_wagered', 0),
                data.get('pending_exposure', 0),
                exposure_by_team,
                exposure_by_conference,
                exposure_by_division,
                data.get('scale_factor', 1.0),
                data.get('regime', 'normal')
            ))

            conn.commit()
            logger.debug(f"Recorded risk snapshot for {snapshot_date}")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to record risk snapshot: {e}")
            raise
        finally:
            conn.close()

    def record_bet_risk_metadata(self, bet_id: str, metadata: Dict):
        """
        Record risk metadata for a bet.

        Args:
            bet_id: Bet identifier
            metadata: Dict with risk factors and adjustments
        """
        conn = self._get_connection()

        try:
            # Convert list fields to JSON
            adjustments = json.dumps(metadata.get('adjustments', []))
            warnings = json.dumps(metadata.get('warnings', []))

            conn.execute("""
                INSERT OR REPLACE INTO bet_risk_metadata (
                    bet_id, correlation_factor, drawdown_factor,
                    base_kelly, adjusted_kelly,
                    same_team_exposure, same_game_exposure,
                    same_conference_exposure, same_division_exposure,
                    limiting_factor, adjustments, warnings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                bet_id,
                metadata.get('correlation_factor'),
                metadata.get('drawdown_factor'),
                metadata.get('base_kelly'),
                metadata.get('adjusted_kelly'),
                metadata.get('same_team_exposure'),
                metadata.get('same_game_exposure'),
                metadata.get('same_conference_exposure'),
                metadata.get('same_division_exposure'),
                metadata.get('limiting_factor'),
                adjustments,
                warnings
            ))

            conn.commit()
            logger.debug(f"Recorded risk metadata for bet {bet_id}")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to record bet risk metadata: {e}")
            raise
        finally:
            conn.close()

    def get_latest_snapshot(self) -> Optional[Dict]:
        """Get the most recent risk snapshot."""
        conn = self._get_connection()

        result = conn.execute("""
            SELECT * FROM risk_snapshots
            ORDER BY snapshot_date DESC
            LIMIT 1
        """).fetchone()

        conn.close()

        if result:
            data = dict(result)
            # Parse JSON fields
            data['exposure_by_team'] = json.loads(data['exposure_by_team'] or '{}')
            data['exposure_by_conference'] = json.loads(data['exposure_by_conference'] or '{}')
            data['exposure_by_division'] = json.loads(data['exposure_by_division'] or '{}')
            return data

        return None

    def migrate_all(self):
        """
        Run all migrations.

        Safe to call multiple times - uses IF NOT EXISTS.
        """
        logger.info("Running risk management database migrations...")
        self.init_risk_tables()
        self.add_bets_table_columns()
        logger.info("✓ All risk management migrations complete")


if __name__ == "__main__":
    # Test migrations
    print("Risk Database Manager - Test Mode")
    print("=" * 70)

    manager = RiskDatabaseManager()
    manager.migrate_all()

    print("\n✓ All migrations completed successfully")
