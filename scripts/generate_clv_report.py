#!/usr/bin/env python3
"""
CLV Report Generator

CLI tool for comprehensive CLV analytics and reporting.
Generates detailed reports on CLV performance, optimal timing,
line velocity, and snapshot coverage.

Usage:
    # Generate full report
    python scripts/generate_clv_report.py

    # Report for specific date range
    python scripts/generate_clv_report.py --start-date 2024-01-01 --end-date 2024-12-31

    # Report for specific bet type
    python scripts/generate_clv_report.py --bet-type spread

    # Export to JSON
    python scripts/generate_clv_report.py --export clv_report.json

    # Minimal output
    python scripts/generate_clv_report.py --minimal
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import pandas as pd
from loguru import logger

from src.bet_tracker import DB_PATH, get_enhanced_clv_summary
from src.data.line_history import LineHistoryManager
import sqlite3


def _get_connection() -> sqlite3.Connection:
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_bets(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    bet_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Load bet history with filters.

    Args:
        start_date: Start date filter (ISO format)
        end_date: End date filter (ISO format)
        bet_type: Bet type filter

    Returns:
        DataFrame with bet history
    """
    conn = _get_connection()

    query = """
        SELECT
            id, game_id, home_team, away_team, commence_time,
            bet_type, bet_side, odds, line,
            edge, kelly, outcome, profit, settled_at,
            closing_odds, closing_line, clv,
            clv_at_1hr, clv_at_4hr, clv_at_12hr, clv_at_24hr,
            line_velocity, max_clv_achieved, snapshot_coverage,
            closing_line_source, booked_hours_before
        FROM bets
        WHERE outcome IS NOT NULL
    """

    params = []

    if start_date:
        query += " AND settled_at >= ?"
        params.append(start_date)

    if end_date:
        query += " AND settled_at <= ?"
        params.append(end_date)

    if bet_type:
        query += " AND bet_type = ?"
        params.append(bet_type)

    query += " ORDER BY settled_at DESC"

    df = pd.read_sql_query(query, conn, params=params if params else None)
    conn.close()

    return df


def get_clv_by_time_window() -> Dict:
    """Get CLV breakdown by time window."""
    summary = get_enhanced_clv_summary()

    if not summary or 'clv_by_time_window' not in summary:
        return {}

    return summary['clv_by_time_window']


def get_optimal_booking_times() -> Dict:
    """Get optimal booking times for each bet type and side."""
    manager = LineHistoryManager()
    results = {}

    for bet_type in ['spread', 'moneyline', 'totals']:
        for bet_side in ['home', 'away', 'over', 'under']:
            # Skip invalid combinations
            if bet_type in ['spread', 'moneyline'] and bet_side in ['over', 'under']:
                continue
            if bet_type == 'totals' and bet_side in ['home', 'away']:
                continue

            try:
                optimal = manager.analyze_optimal_booking_time(
                    bet_type=bet_type,
                    bet_side=bet_side
                )

                if optimal:
                    key = f"{bet_type}_{bet_side}"
                    results[key] = optimal

            except Exception as e:
                logger.debug(f"Could not get optimal timing for {bet_type}/{bet_side}: {e}")

    return results


def get_velocity_analysis(bets_df: pd.DataFrame) -> Dict:
    """Analyze line velocity and its correlation with CLV."""
    # Filter to bets with both velocity and CLV data
    velocity_bets = bets_df[
        bets_df['line_velocity'].notna() &
        bets_df['clv'].notna()
    ].copy()

    if len(velocity_bets) < 10:
        return {'error': 'Insufficient data (need 10+ bets with velocity and CLV)'}

    from scipy.stats import pearsonr

    # Calculate correlation
    corr, p_value = pearsonr(
        velocity_bets['line_velocity'],
        velocity_bets['clv']
    )

    # Categorize bets by velocity
    velocity_bets['velocity_category'] = pd.cut(
        velocity_bets['line_velocity'],
        bins=[-float('inf'), -0.5, 0.5, float('inf')],
        labels=['favorable', 'neutral', 'unfavorable']
    )

    # CLV by velocity category
    clv_by_velocity = velocity_bets.groupby('velocity_category')['clv'].agg([
        'mean', 'std', 'count'
    ]).to_dict('index')

    return {
        'correlation': float(corr),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'n_samples': len(velocity_bets),
        'clv_by_velocity': clv_by_velocity,
        'interpretation': (
            'Positive correlation: Favorable line movement predicts positive CLV'
            if corr > 0 and p_value < 0.05 else
            'Negative correlation: Unfavorable line movement predicts positive CLV'
            if corr < 0 and p_value < 0.05 else
            'No significant correlation between line velocity and CLV'
        )
    }


def get_snapshot_coverage_report(bets_df: pd.DataFrame) -> Dict:
    """Analyze snapshot coverage quality."""
    coverage_bets = bets_df[bets_df['snapshot_coverage'].notna()].copy()

    if len(coverage_bets) == 0:
        return {'error': 'No bets with snapshot coverage data'}

    # Coverage statistics
    stats = {
        'average': float(coverage_bets['snapshot_coverage'].mean()),
        'median': float(coverage_bets['snapshot_coverage'].median()),
        'min': float(coverage_bets['snapshot_coverage'].min()),
        'max': float(coverage_bets['snapshot_coverage'].max()),
        'total_bets': len(coverage_bets),
    }

    # Coverage buckets
    coverage_bets['coverage_bucket'] = pd.cut(
        coverage_bets['snapshot_coverage'],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['poor (0-25%)', 'fair (25-50%)', 'good (50-75%)', 'excellent (75-100%)']
    )

    bucket_counts = coverage_bets['coverage_bucket'].value_counts().to_dict()
    stats['by_bucket'] = {str(k): int(v) for k, v in bucket_counts.items()}

    # CLV by coverage quality
    clv_by_coverage = coverage_bets.groupby('coverage_bucket')['clv'].agg([
        'mean', 'count'
    ]).to_dict('index')

    stats['clv_by_coverage'] = {
        str(k): {'avg_clv': float(v['mean']), 'count': int(v['count'])}
        for k, v in clv_by_coverage.items()
    }

    return stats


def get_closing_line_source_report(bets_df: pd.DataFrame) -> Dict:
    """Analyze closing line source quality."""
    source_bets = bets_df[bets_df['closing_line_source'].notna()].copy()

    if len(source_bets) == 0:
        return {'error': 'No bets with closing line source data'}

    # Source distribution
    source_counts = source_bets['closing_line_source'].value_counts().to_dict()
    total = len(source_bets)

    distribution = {
        source: {'count': int(count), 'percentage': float(count / total * 100)}
        for source, count in source_counts.items()
    }

    # CLV by source
    clv_by_source = source_bets.groupby('closing_line_source')['clv'].agg([
        'mean', 'std', 'count'
    ]).to_dict('index')

    return {
        'total_bets': total,
        'distribution': distribution,
        'clv_by_source': {
            source: {
                'avg_clv': float(stats['mean']),
                'std_clv': float(stats['std']) if pd.notna(stats['std']) else 0.0,
                'count': int(stats['count'])
            }
            for source, stats in clv_by_source.items()
        }
    }


def print_report(report: Dict, minimal: bool = False):
    """
    Print formatted CLV report.

    Args:
        report: Report dictionary
        minimal: If True, print minimal output
    """
    logger.info("\n" + "=" * 80)
    logger.info("CLV ANALYTICS REPORT")
    logger.info("=" * 80)
    logger.info(f"Generated: {report['generated_at']}")
    logger.info(f"Total Bets: {report['total_bets']}")

    # CLV by Time Window
    if report['clv_by_time_window']:
        logger.info(f"\n{'=' * 80}")
        logger.info("CLV BY TIME WINDOW")
        logger.info(f"{'=' * 80}")
        for window, clv in sorted(report['clv_by_time_window'].items()):
            if clv is not None:
                logger.info(f"  {window}: {clv:+.3f} ({clv*100:+.2f}%)")
            else:
                logger.info(f"  {window}: N/A")

    if minimal:
        return

    # Optimal Booking Times
    if report['optimal_booking']:
        logger.info(f"\n{'=' * 80}")
        logger.info("OPTIMAL BOOKING TIMES")
        logger.info(f"{'=' * 80}")
        for key, data in sorted(report['optimal_booking'].items()):
            bet_type, side = key.split('_')
            optimal_hrs = data.get('optimal_hours_before')
            avg_clv = data.get('avg_clv_at_optimal')

            if optimal_hrs and avg_clv:
                logger.info(f"  {bet_type.upper()} {side.upper()}:")
                logger.info(f"    Optimal: {optimal_hrs:.1f} hours before game")
                logger.info(f"    Avg CLV at optimal: {avg_clv:+.3f}")

    # Velocity Analysis
    if 'velocity_analysis' in report and 'error' not in report['velocity_analysis']:
        logger.info(f"\n{'=' * 80}")
        logger.info("LINE VELOCITY ANALYSIS")
        logger.info(f"{'=' * 80}")
        va = report['velocity_analysis']
        logger.info(f"Correlation with CLV: {va['correlation']:+.3f}")
        logger.info(f"P-value: {va['p_value']:.4f}")
        logger.info(f"Statistically significant: {'YES' if va['significant'] else 'NO'}")
        logger.info(f"Interpretation: {va['interpretation']}")

    # Snapshot Coverage
    if 'snapshot_coverage' in report and 'error' not in report['snapshot_coverage']:
        logger.info(f"\n{'=' * 80}")
        logger.info("SNAPSHOT COVERAGE QUALITY")
        logger.info(f"{'=' * 80}")
        sc = report['snapshot_coverage']
        logger.info(f"Average Coverage: {sc['average']:.1%}")
        logger.info(f"Median Coverage: {sc['median']:.1%}")
        logger.info(f"\nCoverage Distribution:")
        for bucket, count in sorted(sc['by_bucket'].items()):
            pct = (count / sc['total_bets']) * 100
            logger.info(f"  {bucket}: {count} ({pct:.1f}%)")

    # Closing Line Sources
    if 'closing_line_sources' in report and 'error' not in report['closing_line_sources']:
        logger.info(f"\n{'=' * 80}")
        logger.info("CLOSING LINE SOURCE QUALITY")
        logger.info(f"{'=' * 80}")
        cls = report['closing_line_sources']
        logger.info(f"\nSource Distribution:")
        for source, data in sorted(cls['distribution'].items(), key=lambda x: x[1]['count'], reverse=True):
            logger.info(f"  {source.upper()}: {data['count']} ({data['percentage']:.1f}%)")

        logger.info(f"\nAverage CLV by Source:")
        for source, data in sorted(cls['clv_by_source'].items()):
            logger.info(f"  {source.upper()}: {data['avg_clv']:+.3f} (n={data['count']})")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Generate comprehensive CLV analytics report')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--bet-type', type=str, choices=['spread', 'moneyline', 'totals'],
                       help='Filter by bet type')
    parser.add_argument('--export', type=str, help='Export report to JSON file')
    parser.add_argument('--minimal', action='store_true', help='Minimal output (CLV by window only)')
    args = parser.parse_args()

    logger.info("=== Generating CLV Report ===")

    # Load data
    bets_df = load_bets(
        start_date=args.start_date,
        end_date=args.end_date,
        bet_type=args.bet_type
    )

    if bets_df.empty:
        logger.error("No bets found matching criteria")
        return 1

    logger.info(f"Loaded {len(bets_df)} bets")

    # Generate report
    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'filters': {
            'start_date': args.start_date,
            'end_date': args.end_date,
            'bet_type': args.bet_type,
        },
        'total_bets': len(bets_df),
        'clv_by_time_window': get_clv_by_time_window(),
    }

    if not args.minimal:
        report['optimal_booking'] = get_optimal_booking_times()
        report['velocity_analysis'] = get_velocity_analysis(bets_df)
        report['snapshot_coverage'] = get_snapshot_coverage_report(bets_df)
        report['closing_line_sources'] = get_closing_line_source_report(bets_df)

    # Print report
    print_report(report, minimal=args.minimal)

    # Export if requested
    if args.export:
        with open(args.export, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"\nReport exported to {args.export}")

    logger.info("\n=== Report Complete ===")

    return 0


if __name__ == "__main__":
    sys.exit(main())
