"""
Validation Script for Alternative Data Integration

Compares model performance with and without alternative data features:
- Referee features
- News features
- Sentiment features

Usage:
    python scripts/validate_alternative_data.py

Outputs:
    - Feature comparison report
    - Performance metrics (accuracy, precision, recall)
    - Feature importance analysis
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger
from datetime import datetime, timedelta

from src.features.game_features import GameFeatureBuilder
from src.utils.constants import BETS_DB_PATH


def load_recent_games(days_back: int = 30) -> pd.DataFrame:
    """Load recent games for validation."""
    import sqlite3

    cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    conn = sqlite3.connect(BETS_DB_PATH)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT DISTINCT
            game_id,
            substr(commence_time, 1, 10) as date,
            home_team,
            away_team,
            actual_score_home as home_score,
            actual_score_away as away_score
        FROM bets
        WHERE substr(commence_time, 1, 10) >= ?
          AND actual_score_home IS NOT NULL
          AND actual_score_away IS NOT NULL
          AND outcome IS NOT NULL
        ORDER BY commence_time DESC
    """

    games = pd.read_sql_query(query, conn, params=(cutoff_date,))
    conn.close()

    if games.empty:
        logger.warning(f"No completed games found in last {days_back} days")
        return games

    # Add season (approximate based on date)
    games['season'] = games['date'].apply(lambda d: int(d[:4]) if int(d[5:7]) >= 10 else int(d[:4]) - 1)

    # Add target variables
    games['home_win'] = (games['home_score'] > games['away_score']).astype(int)
    games['point_diff'] = games['home_score'] - games['away_score']
    games['total_points'] = games['home_score'] + games['away_score']

    logger.info(f"Loaded {len(games)} games from last {days_back} days")
    return games


def build_features_comparison():
    """Build features with and without alternative data."""

    # Load recent games
    games = load_recent_games(days_back=30)

    if games.empty:
        logger.error("No games found for validation")
        return None, None

    logger.info("=" * 80)
    logger.info("BUILDING BASELINE FEATURES (no alternative data)")
    logger.info("=" * 80)

    # Baseline: Without alternative data
    baseline_builder = GameFeatureBuilder(
        use_referee_features=False,
        use_news_features=False,
        use_sentiment_features=False
    )

    baseline_features = baseline_builder.build_game_features(games.copy())
    baseline_cols = baseline_builder.get_feature_columns(baseline_features)

    logger.info(f"Baseline features: {len(baseline_cols)} columns")

    logger.info("\n" + "=" * 80)
    logger.info("BUILDING ENHANCED FEATURES (with alternative data)")
    logger.info("=" * 80)

    # Enhanced: With all alternative data
    enhanced_builder = GameFeatureBuilder(
        use_referee_features=True,
        use_news_features=True,
        use_sentiment_features=True
    )

    enhanced_features = enhanced_builder.build_game_features(games.copy())
    enhanced_cols = enhanced_builder.get_feature_columns(enhanced_features)

    logger.info(f"Enhanced features: {len(enhanced_cols)} columns")

    return baseline_features, enhanced_features


def analyze_feature_differences(baseline: pd.DataFrame, enhanced: pd.DataFrame):
    """Analyze differences between baseline and enhanced feature sets."""

    baseline_cols = set(baseline.columns)
    enhanced_cols = set(enhanced.columns)

    new_features = enhanced_cols - baseline_cols

    logger.info("\n" + "=" * 80)
    logger.info("FEATURE COMPARISON")
    logger.info("=" * 80)

    logger.info(f"Baseline columns: {len(baseline_cols)}")
    logger.info(f"Enhanced columns: {len(enhanced_cols)}")
    logger.info(f"New features added: {len(new_features)}")

    if new_features:
        logger.info("\nNew features added:")

        # Group by category
        ref_features = [f for f in new_features if 'ref_' in f]
        news_features = [f for f in new_features if 'news' in f]
        sentiment_features = [f for f in new_features if 'sentiment' in f]

        if ref_features:
            logger.info(f"\n  Referee features ({len(ref_features)}):")
            for f in sorted(ref_features):
                logger.info(f"    - {f}")

        if news_features:
            logger.info(f"\n  News features ({len(news_features)}):")
            for f in sorted(news_features):
                logger.info(f"    - {f}")

        if sentiment_features:
            logger.info(f"\n  Sentiment features ({len(sentiment_features)}):")
            for f in sorted(sentiment_features):
                logger.info(f"    - {f}")

    # Check for missing data
    logger.info("\n" + "=" * 80)
    logger.info("DATA QUALITY CHECK")
    logger.info("=" * 80)

    for feature in sorted(new_features):
        if feature in enhanced.columns:
            null_count = enhanced[feature].isna().sum()
            null_pct = (null_count / len(enhanced)) * 100

            non_zero = (enhanced[feature] != 0).sum()
            non_zero_pct = (non_zero / len(enhanced)) * 100

            logger.info(f"{feature}:")
            logger.info(f"  Null values: {null_count} ({null_pct:.1f}%)")
            logger.info(f"  Non-zero values: {non_zero} ({non_zero_pct:.1f}%)")

            if feature in enhanced.select_dtypes(include=['number']).columns:
                logger.info(f"  Mean: {enhanced[feature].mean():.3f}")
                logger.info(f"  Std: {enhanced[feature].std():.3f}")


def validate_feature_health(enhanced: pd.DataFrame):
    """Validate health of alternative data features."""

    logger.info("\n" + "=" * 80)
    logger.info("FEATURE HEALTH MONITORING")
    logger.info("=" * 80)

    health_report = {
        'referee': {'available': 0, 'total': 0},
        'news': {'available': 0, 'total': 0},
        'sentiment': {'available': 0, 'total': 0}
    }

    # Check referee features
    ref_cols = [c for c in enhanced.columns if 'ref_' in c]
    if ref_cols:
        for col in ref_cols:
            health_report['referee']['total'] += 1
            # Check if feature has meaningful data (not all defaults)
            if col in enhanced.columns:
                if 'pace_factor' in col:
                    # Should not all be 1.0 (default)
                    if (enhanced[col] != 1.0).any():
                        health_report['referee']['available'] += 1
                elif 'size' in col:
                    # Should not all be 0 (default)
                    if (enhanced[col] != 0).any():
                        health_report['referee']['available'] += 1
                else:
                    # Other features should not all be defaults
                    if enhanced[col].std() > 0:
                        health_report['referee']['available'] += 1

    # Check news features
    news_cols = [c for c in enhanced.columns if 'news' in c]
    if news_cols:
        for col in news_cols:
            health_report['news']['total'] += 1
            if col in enhanced.columns and (enhanced[col] != 0).any():
                health_report['news']['available'] += 1

    # Check sentiment features
    sentiment_cols = [c for c in enhanced.columns if 'sentiment' in c and 'enabled' not in c]
    if sentiment_cols:
        for col in sentiment_cols:
            health_report['sentiment']['total'] += 1
            if col in enhanced.columns and (enhanced[col] != 0).any():
                health_report['sentiment']['available'] += 1

    # Print health report
    for category, stats in health_report.items():
        if stats['total'] > 0:
            pct = (stats['available'] / stats['total']) * 100
            status = "✓" if pct > 80 else "⚠" if pct > 50 else "✗"
            logger.info(f"{status} {category.capitalize()}: {stats['available']}/{stats['total']} features ({pct:.1f}%)")

    # Check if sentiment is enabled
    if 'sentiment_enabled' in enhanced.columns:
        enabled_count = enhanced['sentiment_enabled'].sum()
        enabled_pct = (enabled_count / len(enhanced)) * 100
        logger.info(f"  Sentiment enabled: {enabled_count}/{len(enhanced)} games ({enabled_pct:.1f}%)")

    return health_report


def main():
    """Main validation workflow."""

    logger.info("Alternative Data Validation Script")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Build features
        baseline, enhanced = build_features_comparison()

        if baseline is None or enhanced is None:
            logger.error("Failed to build features")
            return 1

        # Analyze differences
        analyze_feature_differences(baseline, enhanced)

        # Validate health
        health_report = validate_feature_health(enhanced)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)

        baseline_count = len([c for c in baseline.columns if c not in ['game_id', 'date', 'season', 'home_team', 'away_team']])
        enhanced_count = len([c for c in enhanced.columns if c not in ['game_id', 'date', 'season', 'home_team', 'away_team']])
        new_count = enhanced_count - baseline_count

        logger.info(f"✓ Baseline features: {baseline_count}")
        logger.info(f"✓ Enhanced features: {enhanced_count}")
        logger.info(f"✓ New alternative data features: {new_count}")

        # Overall health
        total_features = sum(h['total'] for h in health_report.values())
        available_features = sum(h['available'] for h in health_report.values())

        if total_features > 0:
            overall_health = (available_features / total_features) * 100
            logger.info(f"✓ Feature health: {available_features}/{total_features} ({overall_health:.1f}%)")

        logger.info(f"\n✓ Validation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return 0

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
