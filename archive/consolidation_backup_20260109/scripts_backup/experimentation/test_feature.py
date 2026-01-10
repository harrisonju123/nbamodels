"""
Test a new feature's impact on model performance.

Usage:
    python scripts/experimentation/test_feature.py --feature-name ref_home_bias --category referee --hypothesis "Referee home bias affects game outcomes"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from src.experimentation import FeatureExperiment


def main():
    parser = argparse.ArgumentParser(description="Test a feature")
    parser.add_argument("--feature-name", required=True, help="Feature name")
    parser.add_argument("--category", default="custom", help="Feature category")
    parser.add_argument("--hypothesis", default="", help="Why this feature might help")
    args = parser.parse_args()

    experiment = FeatureExperiment()

    print(f"\n=== Testing Feature: {args.feature_name} ===\n")
    print(f"Category: {args.category}")
    print(f"Hypothesis: {args.hypothesis}")
    print("\nRunning experiment...\n")

    result = experiment.test_feature(
        feature_name=args.feature_name,
        category=args.category,
        hypothesis=args.hypothesis
    )

    print("=== Results ===\n")
    print(f"Baseline Accuracy: {result.baseline_accuracy:.2%}")
    print(f"Experiment Accuracy: {result.experiment_accuracy:.2%}")
    print(f"Accuracy Lift: {result.accuracy_lift:+.2%}")
    print()
    print(f"Baseline ROI: {result.baseline_roi:.2%}")
    print(f"Experiment ROI: {result.experiment_roi:.2%}")
    print(f"ROI Lift: {result.roi_lift:+.2%}")
    print()
    print(f"P-value: {result.p_value:.4f}")
    print(f"Significant: {result.is_significant}")
    print(f"Effect Size: {result.effect_size:.3f}")
    print()
    print(f"Decision: {result.decision.upper()}")
    print(f"Reason: {result.decision_reason}")
    print()


if __name__ == "__main__":
    main()
