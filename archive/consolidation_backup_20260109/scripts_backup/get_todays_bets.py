#!/usr/bin/env python3
"""
Get Today's Bet Recommendations (Simple One-Stop Tool)

Shows today's NBA bets with risk-adjusted bet amounts in a clean format.

Usage:
    python scripts/get_todays_bets.py              # Show today's bets
    python scripts/get_todays_bets.py --help       # Show options
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_pipeline_and_capture():
    """Run the daily betting pipeline in dry-run mode and capture output."""
    project_root = Path(__file__).parent.parent
    pipeline_script = project_root / "scripts" / "daily_betting_pipeline.py"

    # Run pipeline with --dry-run flag
    result = subprocess.run(
        [sys.executable, str(pipeline_script), "--dry-run"],
        capture_output=True,
        text=True,
        cwd=str(project_root)
    )

    return result.stdout, result.stderr, result.returncode


def print_header():
    """Print header."""
    print("\n" + "="*80)
    print("🎯 TODAY'S BET RECOMMENDATIONS")
    print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Get today's NBA bet recommendations with risk-adjusted amounts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool runs the daily betting pipeline in dry-run mode to show you today's
recommended bets with risk-adjusted sizing.

The risk management system automatically applies:
  - Correlation-aware position sizing (reduces bets on correlated positions)
  - Drawdown-based bet scaling (reduces sizing during losing streaks)
  - Daily/weekly exposure limits (caps total risk)
  - Circuit breaker (stops betting at 30%% drawdown)

Examples:
  python scripts/get_todays_bets.py              # Show today's bets

To actually log these bets to your database:
  python scripts/daily_betting_pipeline.py       # Log as paper trades
  python scripts/daily_betting_pipeline.py --live  # Place LIVE bets (⚠️ REAL MONEY)
        """
    )

    args = parser.parse_args()

    print_header()
    print("\n🔄 Running prediction and risk analysis...\n")

    # Run the pipeline
    stdout, stderr, returncode = run_pipeline_and_capture()

    if returncode != 0:
        print("❌ Error running pipeline:")
        print(stderr)
        return 1

    # Print the output
    print(stdout)

    if stderr:
        # Print warnings/info from logger
        print("\n" + "="*80)
        print("📋 ADDITIONAL INFO:")
        print("="*80)
        for line in stderr.split('\n'):
            if line.strip():
                print(line)

    print("\n" + "="*80)
    print("💡 NEXT STEPS:")
    print("="*80)
    print("  To log bets to database:  python scripts/daily_betting_pipeline.py")
    print("  To place LIVE bets:       python scripts/daily_betting_pipeline.py --live")
    print("  View bet history:         python analytics_dashboard.py")
    print("="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
