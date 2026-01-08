#!/usr/bin/env python3
"""
Backtest vs Live Performance Gap Analysis

Weekly script to compare backtested performance to live/paper trading results.
Detects overfitting and validates model performance.

Usage:
    python scripts/analyze_backtest_gap.py
    python scripts/analyze_backtest_gap.py --run-id bt_2024_01_15
    python scripts/analyze_backtest_gap.py --strategy-type primary --days 60
    python scripts/analyze_backtest_gap.py --export gap_report.json
"""

import argparse
import json
import sys
from datetime import datetime
from loguru import logger

from src.market_analysis.performance_gap import BacktestLiveGapAnalyzer


def format_pct(value: float) -> str:
    """Format percentage with + sign."""
    return f"{value:+.2f}%"


def main():
    parser = argparse.ArgumentParser(description="Analyze backtest vs live performance gap")
    parser.add_argument(
        "--run-id",
        type=str,
        help="Specific backtest run ID to analyze",
    )
    parser.add_argument(
        "--strategy-type",
        type=str,
        help="Strategy type to analyze",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days of live data to analyze (default: 90)",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export results to JSON file",
    )

    args = parser.parse_args()

    logger.info("Backtest vs Live Performance Gap Analysis")
    logger.info(f"Analyzing last {args.days} days of live trading")

    analyzer = BacktestLiveGapAnalyzer()

    try:
        # Calculate gap
        result = analyzer.calculate_gap(
            run_id=args.run_id,
            strategy_type=args.strategy_type,
            live_days=args.days,
        )

        # Get gap trend
        trend = analyzer.get_gap_trend(
            window=30,
            strategy_type=args.strategy_type,
        )
        result.gap_trend = trend

        # Print overall results
        logger.info("\n" + "=" * 60)
        logger.info("OVERALL PERFORMANCE COMPARISON")
        logger.info("=" * 60)

        logger.info(f"Backtest ROI:     {result.backtest_roi:+.2%}")
        logger.info(f"Live ROI:         {result.live_roi:+.2%}")
        logger.info(f"Gap:              {result.gap:+.2%} ({format_pct(result.gap_pct)})")
        logger.info(f"95% CI:           [{result.confidence_interval[0]:.2%}, {result.confidence_interval[1]:.2%}]")
        logger.info(f"Gap Trend:        {result.gap_trend}")

        if result.is_overfitting:
            logger.warning("\n⚠️  OVERFITTING DETECTED")
            logger.warning(
                f"Gap of {result.gap:.2%} exceeds threshold of {analyzer.OVERFITTING_GAP_THRESHOLD:.2%}"
            )
            logger.warning("Model performance in live trading is significantly worse than backtest")
            logger.warning("Recommendations:")
            logger.warning("  - Review model features for look-ahead bias")
            logger.warning("  - Check for data leakage in training")
            logger.warning("  - Consider using more robust validation methods")
            logger.warning("  - Reduce bet sizes until model is re-validated")
        else:
            logger.success(f"\n✓ No significant overfitting detected (gap: {result.gap:.2%})")

        # Print by bet type
        if result.by_bet_type:
            logger.info("\n" + "=" * 60)
            logger.info("PERFORMANCE BY BET TYPE")
            logger.info("=" * 60)

            for bet_type, metrics in result.by_bet_type.items():
                logger.info(f"\n{bet_type.upper()}:")
                logger.info(f"  Live ROI:       {metrics.live_roi:+.2%}")
                logger.info(f"  Gap:            {metrics.gap:+.2%} ({format_pct(metrics.gap_pct)})")
                logger.info(f"  Win Rate:       {metrics.live_win_rate:.1%} (backtest: {metrics.backtest_win_rate:.1%})")
                logger.info(f"  N Bets:         {metrics.live_n_bets}")

        # Print by edge bucket
        if result.by_edge_bucket:
            logger.info("\n" + "=" * 60)
            logger.info("PERFORMANCE BY EDGE BUCKET")
            logger.info("=" * 60)

            for bucket, metrics in result.by_edge_bucket.items():
                logger.info(f"\n{bucket}:")
                logger.info(f"  Live ROI:       {metrics.live_roi:+.2%}")
                logger.info(f"  Gap:            {metrics.gap:+.2%}")
                logger.info(f"  Win Rate:       {metrics.live_win_rate:.1%}")
                logger.info(f"  N Bets:         {metrics.live_n_bets}")

        # Print by strategy
        if result.by_strategy:
            logger.info("\n" + "=" * 60)
            logger.info("PERFORMANCE BY STRATEGY")
            logger.info("=" * 60)

            for strategy, metrics in result.by_strategy.items():
                logger.info(f"\n{strategy}:")
                logger.info(f"  Live ROI:       {metrics.live_roi:+.2%}")
                logger.info(f"  Gap:            {metrics.gap:+.2%} ({format_pct(metrics.gap_pct)})")
                logger.info(f"  Sharpe:         {metrics.live_sharpe:.2f} (backtest: {metrics.backtest_sharpe:.2f})")
                logger.info(f"  N Bets:         {metrics.live_n_bets}")

        # Export if requested
        if args.export:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'analysis_period_days': args.days,
                'overall': {
                    'backtest_roi': result.backtest_roi,
                    'live_roi': result.live_roi,
                    'gap': result.gap,
                    'gap_pct': result.gap_pct,
                    'is_overfitting': result.is_overfitting,
                    'confidence_interval': result.confidence_interval,
                    'gap_trend': result.gap_trend,
                },
                'by_bet_type': {
                    k: {
                        'live_roi': v.live_roi,
                        'gap': v.gap,
                        'gap_pct': v.gap_pct,
                        'live_win_rate': v.live_win_rate,
                        'live_n_bets': v.live_n_bets,
                    }
                    for k, v in result.by_bet_type.items()
                },
                'by_edge_bucket': {
                    k: {
                        'live_roi': v.live_roi,
                        'gap': v.gap,
                        'live_win_rate': v.live_win_rate,
                        'live_n_bets': v.live_n_bets,
                    }
                    for k, v in result.by_edge_bucket.items()
                },
            }

            with open(args.export, 'w') as f:
                json.dump(export_data, f, indent=2)

            logger.success(f"\nResults exported to {args.export}")

        logger.info("\n" + "=" * 60)

        # Exit code based on overfitting
        if result.is_overfitting:
            logger.warning("⚠️  Overfitting detected - review model before continuing")
            sys.exit(1)
        else:
            logger.success("✓ Model performance validated")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
