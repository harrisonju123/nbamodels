"""
Initialize all versioning databases.

Run this once to set up the model versioning system.

Usage:
    python scripts/versioning/init_databases.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.versioning.db import get_db_manager


def main():
    print("Initializing model versioning databases...")

    db_manager = get_db_manager()
    db_manager.init_all_databases()

    print("\n✓ All databases initialized successfully!")
    print(f"\nDatabases created at:")
    print(f"  - {db_manager.model_registry_path}")
    print(f"  - {db_manager.feature_registry_path}")
    print(f"  - {db_manager.performance_history_path}")
    print("\nYou can now use the model versioning system.")
    print("\nNext steps:")
    print("  1. Run model retraining with versioning: python scripts/retrain_models.py")
    print("  2. List model versions: python scripts/versioning/list_models.py")
    print("  3. Test a new feature: python scripts/experimentation/test_feature.py --feature-name X")


if __name__ == "__main__":
    main()
