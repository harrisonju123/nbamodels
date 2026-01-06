"""
List all model versions in the registry.

Usage:
    python scripts/versioning/list_models.py
    python scripts/versioning/list_models.py --model-name spread_model
    python scripts/versioning/list_models.py --status champion
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from src.versioning import ModelRegistry


def main():
    parser = argparse.ArgumentParser(description="List model versions")
    parser.add_argument("--model-name", help="Filter by model name")
    parser.add_argument("--status", help="Filter by status (champion, challenger, archived)")
    args = parser.parse_args()

    registry = ModelRegistry()

    if args.model_name:
        print(f"\n=== {args.model_name} Versions ===\n")
        history = registry.get_model_history(args.model_name, limit=20)

        if not history:
            print(f"No versions found for {args.model_name}")
            return

        for model in history:
            print(f"v{model.version} - {model.status.upper()}")
            print(f"  ID: {model.model_id}")
            print(f"  Created: {model.created_at}")
            if model.description:
                print(f"  Description: {model.description}")
            if model.games_count:
                print(f"  Training data: {model.games_count} games, {model.feature_count} features")
            print()

    else:
        # List all model types
        model_types = ["dual_model", "spread_model", "point_spread_model", "totals_model"]

        for model_name in model_types:
            champion = registry.get_champion(model_name)
            challengers = registry.get_challengers(model_name)

            print(f"\n=== {model_name} ===")

            if champion:
                print(f"  Champion: v{champion.version}")
                print(f"    Created: {champion.created_at}")
                if champion.description:
                    print(f"    {champion.description}")
            else:
                print("  Champion: None")

            if challengers:
                print(f"  Challengers: {len(challengers)}")
                for c in challengers[:3]:  # Show top 3
                    print(f"    - v{c.version} ({c.created_at.strftime('%Y-%m-%d')})")
            else:
                print("  Challengers: None")

        print()


if __name__ == "__main__":
    main()
