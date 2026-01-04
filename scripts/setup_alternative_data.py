#!/usr/bin/env python
"""
Setup Alternative Data - One-Command Integration

This script helps you get started with alternative data features:
1. Collects initial data from all sources
2. Validates feature integration
3. Guides you through model retraining

Usage:
    python scripts/setup_alternative_data.py              # Collect data and validate
    python scripts/setup_alternative_data.py --retrain    # Also retrain models
    python scripts/setup_alternative_data.py --check-only # Just check current status
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import sqlite3
from loguru import logger


def print_section(title):
    """Print a section header."""
    logger.info("=" * 80)
    logger.info(title)
    logger.info("=" * 80)


def check_data_status():
    """Check current status of alternative data sources."""
    print_section("ALTERNATIVE DATA STATUS CHECK")

    db_path = "data/bets/bets.db"

    if not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}")
        return False

    conn = sqlite3.connect(db_path)

    try:
        # Check sentiment data
        cursor = conn.execute("SELECT COUNT(*) as count, MAX(collected_at) as latest FROM sentiment_scores")
        row = cursor.fetchone()
        sentiment_count = row[0] if row else 0
        sentiment_latest = row[1] if row and row[1] else "Never"

        # Check news data
        cursor = conn.execute("SELECT COUNT(*) as count, MAX(collected_at) as latest FROM news_articles")
        row = cursor.fetchone()
        news_count = row[0] if row else 0
        news_latest = row[1] if row and row[1] else "Never"

        # Check referee data
        cursor = conn.execute("SELECT COUNT(*) as count, MAX(collected_at) as latest FROM referee_assignments")
        row = cursor.fetchone()
        referee_count = row[0] if row else 0
        referee_latest = row[1] if row and row[1] else "Never"

        conn.close()

        # Print status
        logger.info(f"Sentiment data: {sentiment_count} records (last updated: {sentiment_latest})")
        logger.info(f"News data: {news_count} articles (last updated: {news_latest})")
        logger.info(f"Referee data: {referee_count} assignments (last updated: {referee_latest})")

        # Determine if we have any data
        has_data = sentiment_count > 0 or news_count > 0 or referee_count > 0

        if has_data:
            logger.info("✓ Some alternative data is available")
        else:
            logger.warning("⚠ No alternative data found - need to run collection scripts")

        return has_data

    except sqlite3.OperationalError as e:
        logger.error(f"Database error: {e}")
        logger.warning("Alternative data tables may not exist yet")
        return False


def collect_sentiment_data():
    """Collect sentiment data from Reddit."""
    print_section("COLLECTING SENTIMENT DATA")

    try:
        logger.info("Running sentiment collection script...")
        result = subprocess.run(
            ["python", "scripts/collect_sentiment.py"],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            logger.info("✓ Sentiment collection completed")
            return True
        else:
            logger.error(f"✗ Sentiment collection failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("✗ Sentiment collection timed out")
        return False
    except Exception as e:
        logger.error(f"✗ Sentiment collection error: {e}")
        return False


def collect_news_data():
    """Collect news data from RSS feeds."""
    print_section("COLLECTING NEWS DATA")

    try:
        logger.info("Running news collection script...")
        result = subprocess.run(
            ["python", "scripts/collect_news.py"],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            logger.info("✓ News collection completed")
            return True
        else:
            logger.error(f"✗ News collection failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("✗ News collection timed out")
        return False
    except Exception as e:
        logger.error(f"✗ News collection error: {e}")
        return False


def collect_referee_data():
    """Collect referee data from NBA API."""
    print_section("COLLECTING REFEREE DATA")

    try:
        logger.info("Running referee collection script...")
        result = subprocess.run(
            ["python", "scripts/collect_referees.py"],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            logger.info("✓ Referee collection completed")
            return True
        else:
            logger.warning(f"⚠ Referee collection had issues (may be no games today): {result.stderr[:200]}")
            return True  # Don't fail if no games today

    except subprocess.TimeoutExpired:
        logger.error("✗ Referee collection timed out")
        return False
    except Exception as e:
        logger.error(f"✗ Referee collection error: {e}")
        return False


def run_integration_test():
    """Run integration test to verify features work."""
    print_section("RUNNING INTEGRATION TEST")

    try:
        logger.info("Running alternative data integration test...")
        result = subprocess.run(
            ["python", "scripts/test_alternative_data_integration.py"],
            capture_output=True,
            text=True,
            timeout=120
        )

        # Print output
        logger.info(result.stdout)

        if result.returncode == 0:
            logger.info("✓ All integration tests passed")
            return True
        else:
            logger.error(f"✗ Some integration tests failed")
            logger.error(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        logger.error("✗ Integration test timed out")
        return False
    except Exception as e:
        logger.error(f"✗ Integration test error: {e}")
        return False


def check_model_features():
    """Check what features are in the current trained model."""
    print_section("CHECKING CURRENT MODEL FEATURES")

    model_path = Path("models/spread_model.pkl")

    if not model_path.exists():
        logger.warning("⚠ No trained model found at models/spread_model.pkl")
        logger.info("You'll need to train models first")
        return False

    try:
        import pickle

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
            features = model_data.get("feature_columns", [])

        # Check for alternative data features
        ref_features = [f for f in features if 'ref_' in f]
        news_features = [f for f in features if 'news' in f]
        sentiment_features = [f for f in features if 'sentiment' in f]

        logger.info(f"Total features in model: {len(features)}")
        logger.info(f"  Referee features: {len(ref_features)}")
        logger.info(f"  News features: {len(news_features)}")
        logger.info(f"  Sentiment features: {len(sentiment_features)}")

        has_alt_data = len(ref_features) + len(news_features) + len(sentiment_features) > 0

        if has_alt_data:
            logger.info("✓ Model already includes alternative data features")
        else:
            logger.warning("⚠ Model does NOT include alternative data features")
            logger.info("You need to retrain models to include the new features")

        return has_alt_data

    except Exception as e:
        logger.error(f"Failed to check model features: {e}")
        return False


def retrain_models():
    """Retrain all models with new features."""
    print_section("RETRAINING MODELS")

    logger.info("This will retrain all models with alternative data features")
    logger.info("This may take 5-10 minutes...")

    try:
        result = subprocess.run(
            ["python", "scripts/retrain_models.py", "--force"],
            timeout=600
        )

        if result.returncode == 0:
            logger.info("✓ Models retrained successfully")
            return True
        else:
            logger.error("✗ Model retraining failed")
            return False

    except subprocess.TimeoutExpired:
        logger.error("✗ Model retraining timed out (>10 minutes)")
        return False
    except Exception as e:
        logger.error(f"✗ Model retraining error: {e}")
        return False


def print_next_steps(has_data, has_features, retrained):
    """Print next steps based on current status."""
    print_section("NEXT STEPS")

    if not has_data:
        logger.info("1. Alternative data collection failed or returned no data")
        logger.info("   → Check that data sources are accessible (Reddit, RSS feeds)")
        logger.info("   → Review logs in logs/ directory")
        logger.info("   → Try running collection scripts manually")

    if has_data and not has_features:
        logger.info("1. Alternative data collected successfully!")
        logger.info("2. Models need to be retrained to use the new features")
        logger.info("   → Run: python scripts/retrain_models.py --force")

    if has_data and has_features and not retrained:
        logger.info("1. Alternative data is available")
        logger.info("2. Models already include alternative data features")
        logger.info("3. ✓ You're all set!")
        logger.info("")
        logger.info("To keep data fresh, set up cron jobs:")
        logger.info("   → See docs/ALTERNATIVE_DATA_INTEGRATION.md")

    if retrained:
        logger.info("1. ✓ Alternative data collected")
        logger.info("2. ✓ Models retrained with new features")
        logger.info("3. ✓ You're ready to use alternative data in predictions!")
        logger.info("")
        logger.info("Set up automated data collection:")
        logger.info("   → See docs/ALTERNATIVE_DATA_INTEGRATION.md")


def main():
    """Main execution flow."""
    parser = argparse.ArgumentParser(description="Setup alternative data features")
    parser.add_argument("--retrain", action="store_true", help="Retrain models after collecting data")
    parser.add_argument("--check-only", action="store_true", help="Only check current status")
    args = parser.parse_args()

    logger.info("Alternative Data Setup Script")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # Step 1: Check current status
    has_data = check_data_status()
    logger.info("")

    # Step 2: Check if models already have features
    has_features = check_model_features()
    logger.info("")

    # If check-only mode, stop here
    if args.check_only:
        print_next_steps(has_data, has_features, False)
        return 0

    # Step 3: Collect data if not in check-only mode
    if not has_data:
        logger.info("No alternative data found. Collecting now...")
        logger.info("")

        sentiment_ok = collect_sentiment_data()
        logger.info("")

        news_ok = collect_news_data()
        logger.info("")

        referee_ok = collect_referee_data()
        logger.info("")

        # Re-check status
        has_data = check_data_status()
        logger.info("")

    # Step 4: Run integration test
    test_ok = run_integration_test()
    logger.info("")

    # Step 5: Retrain models if requested
    retrained = False
    if args.retrain:
        retrained = retrain_models()
        logger.info("")

        # Re-check model features
        has_features = check_model_features()
        logger.info("")

    # Final summary
    print_next_steps(has_data, has_features, retrained)

    logger.info("")
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
