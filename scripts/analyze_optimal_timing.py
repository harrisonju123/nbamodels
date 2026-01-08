#!/usr/bin/env python3
"""
Optimal Timing Analysis

Weekly script to analyze optimal bet timing patterns and estimate value
of timing improvements.

Usage:
    python scripts/analyze_optimal_timing.py
    python scripts/analyze_optimal_timing.py --days 90
    python scripts/analyze_optimal_timing.py --bet-type spread
    python scripts/analyze_optimal_timing.py --export timing_report.json
"""

import argparse
import json
from datetime import datetime
from loguru import logger

from src.market_analysis.timing_analysis import (
    TimingWindowsAnalyzer,
    HistoricalTimingAnalyzer,
)


def main():
    parser = argparse.ArgumentParser(description="Analyze optimal bet timing")
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days to analyze (default: 90)",
    )
    parser.add_argument(
        "--bet-type",
        type=str,
        help="Filter by bet type (spread, totals, moneyline)",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export results to JSON file",
    )

    args = parser.parse_args()

    logger.info("Optimal Timing Analysis")
    logger.info(f"Analyzing last {args.days} days")
    if args.bet_type:
        logger.info(f"Filtering by bet type: {args.bet_type}")

    windows_analyzer = TimingWindowsAnalyzer()
    historical_analyzer = HistoricalTimingAnalyzer()

    # Analyze by hour of day
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE BY HOUR OF DAY")
    logger.info("=" * 60)

    hour_stats = windows_analyzer.analyze_by_hour_of_day(
        bet_type=args.bet_type,
        lookback_days=args.days,
    )

    if len(hour_stats) > 0:
        # Find best hours
        best_hours = hour_stats.nlargest(3, 'avg_clv')

        logger.info("\nTop 3 Hours for CLV:")
        for _, row in best_hours.iterrows():
            logger.info(
                f"  {int(row['hour']):02d}:00 - "
                f"CLV: {row['avg_clv']:+.3%}, "
                f"Positive Rate: {row['positive_clv_rate']:.1%}, "
                f"N: {int(row['n_bets'])}"
            )

        # Worst hours
        worst_hours = hour_stats.nsmallest(3, 'avg_clv')
        logger.info("\nWorst 3 Hours for CLV:")
        for _, row in worst_hours.iterrows():
            logger.info(
                f"  {int(row['hour']):02d}:00 - "
                f"CLV: {row['avg_clv']:+.3%}, "
                f"N: {int(row['n_bets'])}"
            )

    # Analyze by day of week
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE BY DAY OF WEEK")
    logger.info("=" * 60)

    day_stats = windows_analyzer.analyze_by_day_of_week(
        bet_type=args.bet_type,
        lookback_days=args.days,
    )

    if len(day_stats) > 0:
        for _, row in day_stats.iterrows():
            logger.info(
                f"{row['day_name']:10s}: "
                f"CLV {row['avg_clv']:+.3%}, "
                f"Positive Rate {row['positive_clv_rate']:.1%}, "
                f"N={int(row['n_bets'])}"
            )

    # Analyze by hours before game
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE BY HOURS BEFORE GAME")
    logger.info("=" * 60)

    hours_before_stats = windows_analyzer.analyze_by_hours_before_game(
        bet_type=args.bet_type,
        lookback_days=args.days,
    )

    if len(hours_before_stats) > 0:
        for _, row in hours_before_stats.iterrows():
            logger.info(
                f"{row['label']:8s}: "
                f"CLV {row['avg_clv']:+.3%}, "
                f"Positive Rate {row['positive_clv_rate']:.1%}, "
                f"ROI {row['avg_roi']:+.2%}, "
                f"N={int(row['n_bets'])}"
            )

        # Highlight optimal window
        best_window = hours_before_stats.loc[hours_before_stats['avg_clv'].idxmax()]
        logger.success(
            f"\n✓ Optimal timing window: {best_window['label']} "
            f"(CLV: {best_window['avg_clv']:+.3%})"
        )

    # Find optimal windows
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMAL TIMING WINDOWS")
    logger.info("=" * 60)

    windows = windows_analyzer.find_optimal_windows(bet_type=args.bet_type)

    for window in windows:
        if window.hours_before_game:
            logger.info(
                f"\n{window.hours_before_game}h before game:"
            )
        elif window.hour_of_day is not None:
            logger.info(
                f"\n{window.hour_of_day:02d}:00 hour of day:"
            )

        logger.info(f"  Avg CLV:         {window.avg_clv:+.3%}")
        logger.info(f"  Positive Rate:   {window.positive_clv_rate:.1%}")
        logger.info(f"  Avg ROI:         {window.avg_roi:+.2%}")
        logger.info(f"  N Bets:          {window.n_bets}")

    # Timing value estimate
    logger.info("\n" + "=" * 60)
    logger.info("TIMING VALUE ANALYSIS")
    logger.info("=" * 60)

    timing_value = historical_analyzer.timing_value_estimate(lookback_days=args.days)

    logger.info(
        f"\nAverage CLV improvement from perfect timing: {timing_value:+.3%}"
    )

    if timing_value > 0.01:
        logger.success(
            f"✓ Timing optimization could improve CLV by {timing_value:.2%}"
        )
        logger.info("  Recommendation: Use timing advisor to optimize bet placement")
    elif timing_value > 0.005:
        logger.info(
            f"⚠ Moderate timing value ({timing_value:.2%})"
        )
        logger.info("  Recommendation: Monitor timing but not critical")
    else:
        logger.info(
            f"✗ Low timing value ({timing_value:.2%})"
        )
        logger.info("  Your current timing is already near-optimal")

    # Detailed CLV by booking time
    logger.info("\n" + "=" * 60)
    logger.info("CLV BY BOOKING TIME (DETAILED)")
    logger.info("=" * 60)

    clv_detail = historical_analyzer.analyze_clv_by_booking_time(lookback_days=args.days)

    if len(clv_detail) > 0:
        logger.info("\nWindow      | Avg CLV | Med CLV | Pos% | ROI    | N")
        logger.info("-" * 60)
        for _, row in clv_detail.iterrows():
            logger.info(
                f"{str(row['timing_bucket']):11s} | "
                f"{row['avg_clv']:+.3%} | "
                f"{row['median_clv']:+.3%} | "
                f"{row['positive_clv_rate']:.0%} | "
                f"{row['avg_roi']:+.2%} | "
                f"{int(row['n_bets']):3d}"
            )

    # Export if requested
    if args.export:
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'analysis_period_days': args.days,
            'bet_type': args.bet_type,
            'timing_value': timing_value,
            'optimal_windows': [
                {
                    'hour_of_day': w.hour_of_day,
                    'hours_before_game': w.hours_before_game,
                    'avg_clv': w.avg_clv,
                    'positive_clv_rate': w.positive_clv_rate,
                    'avg_roi': w.avg_roi,
                    'n_bets': w.n_bets,
                }
                for w in windows
            ],
            'by_hours_before': hours_before_stats.to_dict('records') if len(hours_before_stats) > 0 else [],
            'by_hour_of_day': hour_stats.to_dict('records') if len(hour_stats) > 0 else [],
        }

        with open(args.export, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.success(f"\nResults exported to {args.export}")

    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    main()
