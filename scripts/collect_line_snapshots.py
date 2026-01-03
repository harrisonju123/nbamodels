#!/usr/bin/env python3
"""
Hourly Line Snapshot Collection Script

Run via cron for automated collection:
    0 * * * * cd /path/to/nbamodels && python scripts/collect_line_snapshots.py

For testing/manual execution:
    python scripts/collect_line_snapshots.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.line_snapshot_collector import LineSnapshotCollector
from loguru import logger
from datetime import datetime


def main():
    """Main execution function."""
    logger.info(f"=== Line Snapshot Collection started at {datetime.now()} ===")

    try:
        # Initialize collector
        collector = LineSnapshotCollector(
            collection_window_hours=24,  # Games within next 24 hours
            snapshot_interval_hours=1    # Hourly snapshots
        )

        # Collect snapshots
        count = collector.collect_snapshot()

        logger.info(f"Successfully collected {count} snapshot records")

        # Log stats
        stats = collector.get_snapshot_stats()
        logger.info(f"Total snapshots in DB: {stats['total_snapshots']}")
        logger.info(f"Unique games tracked: {stats['unique_games']}")

        return 0

    except Exception as e:
        logger.error(f"Error during snapshot collection: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
