#!/usr/bin/env python3
"""
Investigate AST Data Leakage

Analyzes which features are causing near-perfect AST predictions (R² = 0.999).
Identifies features that leak information about the target variable.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_correlations(df: pd.DataFrame, target: str = 'ast') -> pd.DataFrame:
    """
    Find features highly correlated with target.

    Args:
        df: Feature dataframe
        target: Target variable name

    Returns:
        DataFrame of correlations sorted by absolute value
    """
    logger.info(f"Analyzing correlations with {target.upper()}...")

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Calculate correlations with target
    correlations = numeric_df.corr()[target].sort_values(ascending=False)

    # Remove self-correlation
    correlations = correlations[correlations.index != target]

    return correlations


def identify_leakage_candidates(
    correlations: pd.Series,
    threshold: float = 0.95
) -> list:
    """
    Identify features that might be leaking data.

    Features with correlation > threshold are suspects.
    """
    suspects = correlations[abs(correlations) > threshold]
    return suspects.index.tolist()


def main():
    logger.info("=" * 80)
    logger.info("INVESTIGATING AST DATA LEAKAGE")
    logger.info("=" * 80)
    logger.info("")

    # Load features
    features_path = "data/features/player_game_features_advanced.parquet"
    logger.info(f"Loading features from {features_path}...")
    df = pd.read_parquet(features_path)
    logger.info(f"  Loaded {len(df):,} records, {len(df.columns)} columns")

    # Analyze correlations
    correlations = analyze_correlations(df, 'ast')

    # Display top correlations
    logger.info("")
    logger.info("TOP 30 FEATURES CORRELATED WITH AST:")
    logger.info("=" * 80)
    logger.info(f"{'Feature':<50} {'Correlation':>10}")
    logger.info("-" * 80)

    for feature, corr in correlations.head(30).items():
        logger.info(f"{feature:<50} {corr:>10.6f}")

    # Identify leakage candidates
    logger.info("")
    logger.info("=" * 80)
    logger.info("LEAKAGE CANDIDATES (|correlation| > 0.95):")
    logger.info("=" * 80)

    suspects = identify_leakage_candidates(correlations, threshold=0.95)

    if suspects:
        for suspect in suspects:
            corr = correlations[suspect]
            logger.warning(f"  🚨 {suspect:<50} {corr:>10.6f}")
    else:
        logger.success("  ✓ No obvious leakage candidates found (threshold: 0.95)")

    # Check for AST-related features
    logger.info("")
    logger.info("=" * 80)
    logger.info("AST-RELATED FEATURES:")
    logger.info("=" * 80)

    ast_features = [col for col in df.columns if 'ast' in col.lower()]
    logger.info(f"Found {len(ast_features)} features containing 'ast':")
    for feat in ast_features:
        if feat in correlations.index:
            logger.info(f"  {feat:<50} {correlations[feat]:>10.6f}")

    # Recommendations
    logger.info("")
    logger.info("=" * 80)
    logger.info("RECOMMENDATIONS:")
    logger.info("=" * 80)

    if len(suspects) > 0:
        logger.warning(f"Found {len(suspects)} features with |correlation| > 0.95")
        logger.warning("These features likely leak target information")
        logger.info("")
        logger.info("Recommended actions:")
        logger.info("  1. Remove these features from AST model")
        logger.info("  2. Re-run backtest to verify fix")
        logger.info("  3. Check if these features are lagged properly")
    else:
        logger.info("No obvious data leakage detected")
        logger.info("Possible other causes of perfect R²:")
        logger.info("  1. Overfitting (model memorized training data)")
        logger.info("  2. Indirect leakage through combined features")
        logger.info("  3. Target variable used in feature engineering")

    # Save results
    output_path = "data/backtest/player_props/ast_correlation_analysis.csv"
    correlations.to_csv(output_path)
    logger.success(f"\n✓ Saved correlation analysis to {output_path}")

    logger.info("")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
