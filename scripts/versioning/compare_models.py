"""
Compare champion vs challenger models.

Usage:
    python scripts/versioning/compare_models.py --model-name spread_model
    python scripts/versioning/compare_models.py --champion-id abc123 --challenger-id def456
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from src.versioning import ModelRegistry, ChampionChallengerFramework


def main():
    parser = argparse.ArgumentParser(description="Compare models")
    parser.add_argument("--model-name", help="Model name (will compare champion vs latest challenger)")
    parser.add_argument("--champion-id", help="Specific champion model ID")
    parser.add_argument("--challenger-id", help="Specific challenger model ID")
    args = parser.parse_args()

    registry = ModelRegistry()
    framework = ChampionChallengerFramework(registry)

    if args.model_name:
        # Compare champion vs latest challenger for this model type
        champion = registry.get_champion(args.model_name)
        if not champion:
            print(f"No champion found for {args.model_name}")
            return

        challengers = registry.get_challengers(args.model_name)
        if not challengers:
            print(f"No challengers found for {args.model_name}")
            return

        challenger = challengers[0]  # Most recent
        champion_id = champion.model_id
        challenger_id = challenger.model_id

    elif args.champion_id and args.challenger_id:
        champion_id = args.champion_id
        challenger_id = args.challenger_id
    else:
        print("Error: Must specify either --model-name or both --champion-id and --challenger-id")
        return

    print("\n=== Model Comparison ===\n")
    print(f"Champion ID: {champion_id}")
    print(f"Challenger ID: {challenger_id}")
    print("\nRunning comparison...\n")

    comparison = framework.compare_models(champion_id, challenger_id, test_data=None)

    print(f"Champion ROI: {comparison.champion_roi:.2%}")
    print(f"Challenger ROI: {comparison.challenger_roi:.2%}")
    print(f"ROI Difference: {comparison.roi_difference:+.2%}")
    print(f"\nP-value: {comparison.p_value:.4f}")
    print(f"Significant: {comparison.is_significant}")
    print(f"\nWinner: {comparison.winner}")
    print(f"Recommendation: {comparison.recommendation}")
    print(f"Reason: {comparison.reason}")
    print()


if __name__ == "__main__":
    main()
