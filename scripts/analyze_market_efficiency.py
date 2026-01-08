#!/usr/bin/env python3
"""
Market Efficiency Analysis

Weekly script to analyze market efficiency through sharp vs public analysis,
edge decay patterns, and signal performance validation.

Usage:
    python scripts/analyze_market_efficiency.py
    python scripts/analyze_market_efficiency.py --days 90
    python scripts/analyze_market_efficiency.py --export efficiency_report.json
"""

import argparse
import json
from datetime import datetime
from loguru import logger

from src.market_analysis.market_efficiency import (
    MarketEfficiencyAnalyzer,
    TimeToEfficiencyTracker,
)


def main():
    parser = argparse.ArgumentParser(description="Analyze market efficiency")
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days to analyze (default: 90)",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export results to JSON file",
    )

    args = parser.parse_args()

    logger.info("Market Efficiency Analysis")
    logger.info(f"Analyzing last {args.days} days")

    analyzer = MarketEfficiencyAnalyzer()
    tracker = TimeToEfficiencyTracker()

    # Analyze sharp signal performance
    logger.info("\n" + "=" * 60)
    logger.info("SHARP SIGNAL PERFORMANCE")
    logger.info("=" * 60)

    signals = ['sharp_aligned', 'steam_detected', 'rlm_detected']
    signal_results = {}

    for signal in signals:
        try:
            perf = analyzer.analyze_sharp_signal_performance(
                signal_type=signal,
                lookback_days=args.days,
            )

            signal_results[signal] = perf

            logger.info(f"\n{signal.upper().replace('_', ' ')}:")
            logger.info(f"  Total Bets:      {perf.total_bets}")
            logger.info(f"  Win Rate:        {perf.win_rate:.1%}")
            logger.info(f"  ROI:             {perf.roi:+.2%}")
            logger.info(f"  Avg CLV:         {perf.avg_clv:+.2%}")
            logger.info(f"  Profitable Periods: {perf.profitable_periods}/{perf.profitable_periods + perf.unprofitable_periods}")
            logger.info(f"  Confidence Score: {perf.confidence_score:.2f}/1.0")

            if perf.confidence_score >= 0.7:
                logger.success(f"  ✓ {signal} is a reliable signal")
            elif perf.confidence_score >= 0.5:
                logger.warning(f"  ⚠ {signal} has moderate reliability")
            else:
                logger.error(f"  ✗ {signal} is not reliable")

        except Exception as e:
            logger.error(f"Failed to analyze {signal}: {e}")

    # Build confidence scores
    try:
        confidence_scores = analyzer.build_sharp_confidence_scores()

        logger.info("\n" + "=" * 60)
        logger.info("SHARP SIGNAL CONFIDENCE SCORES")
        logger.info("=" * 60)

        for signal, score in sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"{signal:20s}: {score:.2f}/1.0")

    except Exception as e:
        logger.error(f"Failed to build confidence scores: {e}")

    # Analyze when following sharp works
    logger.info("\n" + "=" * 60)
    logger.info("WHEN FOLLOWING SHARP WORKS")
    logger.info("=" * 60)

    try:
        conditions = analyzer.when_does_following_sharp_work(lookback_days=args.days)

        for condition, stats in conditions.items():
            logger.info(f"\n{condition}:")
            logger.info(f"  Win Rate: {stats['win_rate']:.1%}")
            logger.info(f"  N Bets:   {stats['n_bets']}")

    except Exception as e:
        logger.error(f"Failed to analyze conditions: {e}")

    # Edge decay analysis
    logger.info("\n" + "=" * 60)
    logger.info("EDGE DECAY ANALYSIS")
    logger.info("=" * 60)

    try:
        avg_time = tracker.get_avg_time_to_efficiency()
        logger.info(f"Average time to efficiency: {avg_time:.1f} hours")

        # Check if edges being arbitraged faster
        trend = tracker.are_edges_being_arbitraged_faster()

        logger.info("\nEdge decay velocity by period:")
        for period, stats in trend.items():
            if stats['avg_time_to_zero']:
                logger.info(f"  {period:15s}: {stats['avg_time_to_zero']:.1f} hrs")

        # Check if getting faster
        periods = sorted(trend.keys())
        if len(periods) >= 2:
            recent = trend[periods[0]]
            older = trend[periods[-1]]

            if recent['avg_time_to_zero'] and older['avg_time_to_zero']:
                if recent['avg_time_to_zero'] < older['avg_time_to_zero']:
                    logger.warning(
                        "\n⚠️  Edges are being arbitraged FASTER over time "
                        f"({recent['avg_time_to_zero']:.1f}h vs {older['avg_time_to_zero']:.1f}h)"
                    )
                    logger.warning("Markets may be becoming more efficient to your edges")
                else:
                    logger.success(
                        "\n✓ Edge decay rate is stable or improving "
                        f"({recent['avg_time_to_zero']:.1f}h vs {older['avg_time_to_zero']:.1f}h)"
                    )

        # By market type
        decay_by_market = tracker.get_decay_pattern_by_market()

        if len(decay_by_market) > 0:
            logger.info("\nEdge decay by market type:")
            for _, row in decay_by_market.iterrows():
                logger.info(
                    f"  {row['bet_type']:10s}: "
                    f"Half-life {row['avg_half_life']:.1f}h, "
                    f"Arbitrage rate {row['arbitrage_rate']:.1%}, "
                    f"N={int(row['n_samples'])}"
                )

    except Exception as e:
        logger.error(f"Failed edge decay analysis: {e}")

    # Export if requested
    if args.export:
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'analysis_period_days': args.days,
            'signal_performance': {
                k: {
                    'total_bets': v.total_bets,
                    'win_rate': v.win_rate,
                    'roi': v.roi,
                    'avg_clv': v.avg_clv,
                    'confidence_score': v.confidence_score,
                }
                for k, v in signal_results.items()
            },
            'edge_decay': {
                'avg_time_to_efficiency': avg_time,
                'by_period': trend,
            },
        }

        with open(args.export, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.success(f"\nResults exported to {args.export}")

    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    main()
