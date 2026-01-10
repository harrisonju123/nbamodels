#!/usr/bin/env python3
"""
Advanced Performance Analytics

Comprehensive analysis of betting performance across multiple dimensions:
- Win rate by confidence level
- ROI by bet type, Kelly fraction, and edge size
- Model calibration over time
- Feature contribution analysis

Usage:
    python scripts/analyze_performance.py
    python scripts/analyze_performance.py --min-bets 20  # Require 20+ bets per category
"""

import sys
sys.path.insert(0, '.')

import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

from src.utils.constants import BETS_DB_PATH


def load_settled_bets(min_date=None):
    """Load all settled bets from database."""
    conn = sqlite3.connect(BETS_DB_PATH)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            id,
            game_id,
            home_team,
            away_team,
            bet_type,
            bet_side,
            bet_amount,
            odds,
            line,
            model_prob,
            edge,
            kelly,
            outcome,
            profit,
            logged_at,
            settled_at,
            closing_line,
            clv
        FROM bets
        WHERE outcome IS NOT NULL
    """

    if min_date:
        query += f" AND DATE(logged_at) >= '{min_date}'"

    query += " ORDER BY logged_at DESC"

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        logger.warning("No settled bets found")
        return df

    # Convert types
    df['logged_at'] = pd.to_datetime(df['logged_at'])
    df['settled_at'] = pd.to_datetime(df['settled_at'])
    df['bet_amount'] = df['bet_amount'].astype(float)
    df['profit'] = df['profit'].astype(float)
    df['edge'] = df['edge'].fillna(0).astype(float)
    df['kelly'] = df['kelly'].fillna(0).astype(float)
    df['model_prob'] = df['model_prob'].fillna(0.5).astype(float)

    # Add derived columns
    df['won'] = (df['outcome'] == 'win').astype(int)
    df['roi'] = (df['profit'] / df['bet_amount']) * 100

    # Derive confidence from edge (since confidence column doesn't exist)
    # Map edge to confidence levels
    df['confidence'] = pd.cut(
        df['edge'],
        bins=[-np.inf, 4, 6, 8, np.inf],
        labels=['LOW', 'MEDIUM', 'HIGH', 'VERY HIGH']
    )

    # Add edge buckets
    df['edge_bucket'] = pd.cut(
        df['edge'],
        bins=[-np.inf, 5, 7, 10, np.inf],
        labels=['<5%', '5-7%', '7-10%', '>10%']
    )

    # Add Kelly fraction buckets (convert kelly from decimal to percentage)
    df['kelly_pct'] = df['kelly'] * 100
    df['kelly_bucket'] = pd.cut(
        df['kelly_pct'],
        bins=[-np.inf, 10, 15, 20, np.inf],
        labels=['<10%', '10-15%', '15-20%', '>20%']
    )

    logger.info(f"Loaded {len(df)} settled bets")

    return df


def analyze_win_rate_by_confidence(df, min_bets=10):
    """
    Analyze win rate by confidence level.

    Returns DataFrame with columns:
    - confidence: Confidence level
    - bets: Number of bets
    - wins: Number of wins
    - win_rate: Win rate percentage
    - expected_win_rate: Expected based on odds
    - roi: Average ROI
    """
    if df.empty or 'confidence' not in df.columns:
        logger.warning("No confidence data available")
        return pd.DataFrame()

    # Filter out null confidence
    df_conf = df[df['confidence'].notna()].copy()

    if df_conf.empty:
        logger.warning("No bets with confidence levels")
        return pd.DataFrame()

    # Define confidence order
    confidence_order = ['LOW', 'MEDIUM', 'HIGH', 'VERY HIGH']

    results = []

    for conf_level in confidence_order:
        conf_bets = df_conf[df_conf['confidence'] == conf_level]

        if len(conf_bets) < min_bets:
            continue

        # Calculate expected win rate from odds
        expected_win_rate = conf_bets['model_prob'].mean() * 100

        results.append({
            'confidence': conf_level,
            'bets': len(conf_bets),
            'wins': conf_bets['won'].sum(),
            'win_rate': (conf_bets['won'].mean() * 100),
            'expected_win_rate': expected_win_rate,
            'avg_edge': conf_bets['edge'].mean(),
            'roi': conf_bets['roi'].mean(),
            'total_profit': conf_bets['profit'].sum(),
        })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        logger.warning(f"Not enough bets per confidence level (min: {min_bets})")
        return result_df

    # Add calibration error
    result_df['calibration_error'] = abs(result_df['win_rate'] - result_df['expected_win_rate'])

    logger.info(f"Analyzed {len(result_df)} confidence levels")

    return result_df


def analyze_roi_by_bet_type(df, min_bets=10):
    """Analyze ROI by bet type (spread, moneyline, totals)."""
    if df.empty:
        return pd.DataFrame()

    results = []

    for bet_type in ['spread', 'moneyline', 'totals']:
        type_bets = df[df['bet_type'] == bet_type]

        if len(type_bets) < min_bets:
            continue

        results.append({
            'bet_type': bet_type,
            'bets': len(type_bets),
            'wins': type_bets['won'].sum(),
            'win_rate': (type_bets['won'].mean() * 100),
            'avg_edge': type_bets['edge'].mean(),
            'avg_kelly': type_bets['kelly_pct'].mean(),
            'roi': type_bets['roi'].mean(),
            'total_profit': type_bets['profit'].sum(),
            'total_wagered': type_bets['bet_amount'].sum(),
        })

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        result_df['profit_per_bet'] = result_df['total_profit'] / result_df['bets']

    logger.info(f"Analyzed {len(result_df)} bet types")

    return result_df


def analyze_roi_by_edge_bucket(df, min_bets=10):
    """Analyze ROI by edge size buckets."""
    if df.empty or 'edge_bucket' not in df.columns:
        return pd.DataFrame()

    results = []

    for bucket in ['<5%', '5-7%', '7-10%', '>10%']:
        bucket_bets = df[df['edge_bucket'] == bucket]

        if len(bucket_bets) < min_bets:
            continue

        results.append({
            'edge_bucket': bucket,
            'bets': len(bucket_bets),
            'wins': bucket_bets['won'].sum(),
            'win_rate': (bucket_bets['won'].mean() * 100),
            'avg_edge': bucket_bets['edge'].mean(),
            'roi': bucket_bets['roi'].mean(),
            'total_profit': bucket_bets['profit'].sum(),
        })

    result_df = pd.DataFrame(results)

    logger.info(f"Analyzed {len(result_df)} edge buckets")

    return result_df


def analyze_roi_by_kelly_fraction(df, min_bets=10):
    """Analyze ROI by Kelly fraction buckets."""
    if df.empty or 'kelly_bucket' not in df.columns:
        return pd.DataFrame()

    results = []

    for bucket in ['<10%', '10-15%', '15-20%', '>20%']:
        bucket_bets = df[df['kelly_bucket'] == bucket]

        if len(bucket_bets) < min_bets:
            continue

        results.append({
            'kelly_bucket': bucket,
            'bets': len(bucket_bets),
            'wins': bucket_bets['won'].sum(),
            'win_rate': (bucket_bets['won'].mean() * 100),
            'avg_kelly': bucket_bets['kelly_pct'].mean(),
            'roi': bucket_bets['roi'].mean(),
            'total_profit': bucket_bets['profit'].sum(),
            'avg_bet_size': bucket_bets['bet_amount'].mean(),
        })

    result_df = pd.DataFrame(results)

    logger.info(f"Analyzed {len(result_df)} Kelly buckets")

    return result_df


def analyze_model_calibration(df):
    """
    Analyze model calibration - do predicted probabilities match outcomes?

    Returns calibration stats and binned analysis.
    """
    if df.empty or 'model_prob' not in df.columns:
        return {}, pd.DataFrame()

    # Overall calibration
    predicted_prob = df['model_prob'].mean()
    actual_win_rate = df['won'].mean()
    calibration_error = abs(predicted_prob - actual_win_rate)

    stats = {
        'total_bets': len(df),
        'predicted_win_rate': predicted_prob * 100,
        'actual_win_rate': actual_win_rate * 100,
        'calibration_error': calibration_error * 100,
        'brier_score': ((df['model_prob'] - df['won']) ** 2).mean(),
    }

    # Binned calibration (10 bins)
    df_cal = df.copy()
    df_cal['prob_bin'] = pd.cut(df_cal['model_prob'], bins=10)

    binned = []
    for prob_bin in df_cal['prob_bin'].unique():
        if pd.isna(prob_bin):
            continue

        bin_bets = df_cal[df_cal['prob_bin'] == prob_bin]

        if len(bin_bets) < 5:
            continue

        binned.append({
            'prob_bin': f"{prob_bin.left:.2f}-{prob_bin.right:.2f}",
            'bets': len(bin_bets),
            'predicted': bin_bets['model_prob'].mean() * 100,
            'actual': bin_bets['won'].mean() * 100,
            'error': abs(bin_bets['model_prob'].mean() - bin_bets['won'].mean()) * 100,
        })

    binned_df = pd.DataFrame(binned).sort_values('predicted')

    logger.info("Calculated model calibration metrics")

    return stats, binned_df


def analyze_clv_impact(df, min_bets=10):
    """Analyze impact of CLV on profitability."""
    if df.empty or 'clv' not in df.columns:
        return pd.DataFrame()

    df_clv = df[df['clv'].notna()].copy()

    if df_clv.empty:
        logger.warning("No CLV data available")
        return pd.DataFrame()

    # CLV buckets
    df_clv['clv_bucket'] = pd.cut(
        df_clv['clv'],
        bins=[-np.inf, -2, 0, 2, 5, np.inf],
        labels=['Negative (<-2%)', 'Slightly Negative (-2-0%)', 'Neutral (0-2%)', 'Positive (2-5%)', 'Very Positive (>5%)']
    )

    results = []

    for bucket in df_clv['clv_bucket'].unique():
        if pd.isna(bucket):
            continue

        bucket_bets = df_clv[df_clv['clv_bucket'] == bucket]

        if len(bucket_bets) < min_bets:
            continue

        results.append({
            'clv_bucket': bucket,
            'bets': len(bucket_bets),
            'wins': bucket_bets['won'].sum(),
            'win_rate': (bucket_bets['won'].mean() * 100),
            'avg_clv': bucket_bets['clv'].mean(),
            'roi': bucket_bets['roi'].mean(),
            'total_profit': bucket_bets['profit'].sum(),
        })

    result_df = pd.DataFrame(results)

    logger.info(f"Analyzed {len(result_df)} CLV buckets")

    return result_df


def print_report(df, min_bets=10):
    """Generate and print comprehensive performance report."""
    print("\n" + "=" * 80)
    print("ADVANCED PERFORMANCE ANALYTICS REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Settled Bets: {len(df)}")
    print(f"Date Range: {df['logged_at'].min().date()} to {df['logged_at'].max().date()}")
    print("=" * 80)

    # Overall summary
    print("\n📊 OVERALL PERFORMANCE")
    print("-" * 80)
    print(f"Total Bets:        {len(df)}")
    print(f"Wins:              {df['won'].sum()} ({df['won'].mean()*100:.1f}%)")
    print(f"Total Wagered:     ${df['bet_amount'].sum():.2f}")
    print(f"Total Profit:      ${df['profit'].sum():.2f}")
    print(f"ROI:               {df['roi'].mean():.2f}%")
    print(f"Avg Bet Size:      ${df['bet_amount'].mean():.2f}")
    print(f"Avg Edge:          {df['edge'].mean():.2f}%")
    print(f"Avg Kelly:         {df['kelly_pct'].mean():.2f}%")

    # Win rate by confidence
    print("\n\n🎯 WIN RATE BY CONFIDENCE LEVEL")
    print("-" * 80)
    conf_df = analyze_win_rate_by_confidence(df, min_bets=min_bets)

    if not conf_df.empty:
        print(f"\n{'Confidence':<15} {'Bets':<8} {'Wins':<8} {'Win %':<10} {'Expected %':<12} {'ROI %':<10} {'Profit':<12}")
        print("-" * 80)
        for _, row in conf_df.iterrows():
            print(f"{row['confidence']:<15} {row['bets']:<8} {row['wins']:<8} "
                  f"{row['win_rate']:<10.1f} {row['expected_win_rate']:<12.1f} "
                  f"{row['roi']:<10.1f} ${row['total_profit']:<11.2f}")

        print(f"\nCalibration Quality: ", end="")
        avg_calib_error = conf_df['calibration_error'].mean()
        if avg_calib_error < 3:
            print(f"✅ EXCELLENT ({avg_calib_error:.1f}% error)")
        elif avg_calib_error < 5:
            print(f"✅ GOOD ({avg_calib_error:.1f}% error)")
        elif avg_calib_error < 10:
            print(f"⚠️  FAIR ({avg_calib_error:.1f}% error)")
        else:
            print(f"❌ POOR ({avg_calib_error:.1f}% error)")
    else:
        print(f"⚠️  Insufficient data (min {min_bets} bets per level required)")

    # ROI by bet type
    print("\n\n💰 ROI BY BET TYPE")
    print("-" * 80)
    type_df = analyze_roi_by_bet_type(df, min_bets=min_bets)

    if not type_df.empty:
        print(f"\n{'Bet Type':<12} {'Bets':<8} {'Win %':<10} {'ROI %':<10} {'Profit':<12} {'$/Bet':<10}")
        print("-" * 80)
        for _, row in type_df.iterrows():
            print(f"{row['bet_type']:<12} {row['bets']:<8} {row['win_rate']:<10.1f} "
                  f"{row['roi']:<10.1f} ${row['total_profit']:<11.2f} ${row['profit_per_bet']:<9.2f}")

        # Identify best bet type
        best_type = type_df.loc[type_df['roi'].idxmax()]
        print(f"\n🏆 Best Bet Type: {best_type['bet_type']} ({best_type['roi']:.1f}% ROI)")
    else:
        print(f"⚠️  Insufficient data (min {min_bets} bets per type required)")

    # ROI by edge bucket
    print("\n\n📈 ROI BY EDGE SIZE")
    print("-" * 80)
    edge_df = analyze_roi_by_edge_bucket(df, min_bets=min_bets)

    if not edge_df.empty:
        print(f"\n{'Edge Bucket':<12} {'Bets':<8} {'Win %':<10} {'Avg Edge':<12} {'ROI %':<10} {'Profit':<12}")
        print("-" * 80)
        for _, row in edge_df.iterrows():
            print(f"{row['edge_bucket']:<12} {row['bets']:<8} {row['win_rate']:<10.1f} "
                  f"{row['avg_edge']:<12.1f} {row['roi']:<10.1f} ${row['total_profit']:<11.2f}")

        # Check if higher edge = better ROI
        if edge_df['avg_edge'].corr(edge_df['roi']) > 0.5:
            print("\n✅ Higher edge correlates with better ROI (as expected)")
        else:
            print("\n⚠️  Edge does not strongly correlate with ROI (investigate)")
    else:
        print(f"⚠️  Insufficient data (min {min_bets} bets per bucket required)")

    # ROI by Kelly fraction
    print("\n\n📊 ROI BY KELLY FRACTION")
    print("-" * 80)
    kelly_df = analyze_roi_by_kelly_fraction(df, min_bets=min_bets)

    if not kelly_df.empty:
        print(f"\n{'Kelly %':<12} {'Bets':<8} {'Win %':<10} {'ROI %':<10} {'Profit':<12} {'Avg Size':<10}")
        print("-" * 80)
        for _, row in kelly_df.iterrows():
            print(f"{row['kelly_bucket']:<12} {row['bets']:<8} {row['win_rate']:<10.1f} "
                  f"{row['roi']:<10.1f} ${row['total_profit']:<11.2f} ${row['avg_bet_size']:<9.2f}")
    else:
        print(f"⚠️  Insufficient data (min {min_bets} bets per bucket required)")

    # Model calibration
    print("\n\n🎲 MODEL CALIBRATION")
    print("-" * 80)
    cal_stats, cal_binned = analyze_model_calibration(df)

    if cal_stats:
        print(f"\nPredicted Win Rate:  {cal_stats['predicted_win_rate']:.2f}%")
        print(f"Actual Win Rate:     {cal_stats['actual_win_rate']:.2f}%")
        print(f"Calibration Error:   {cal_stats['calibration_error']:.2f}%")
        print(f"Brier Score:         {cal_stats['brier_score']:.4f} (lower is better)")

        if cal_stats['calibration_error'] < 2:
            print("\n✅ Model is VERY WELL calibrated")
        elif cal_stats['calibration_error'] < 5:
            print("\n✅ Model is WELL calibrated")
        elif cal_stats['calibration_error'] < 10:
            print("\n⚠️  Model calibration is FAIR")
        else:
            print("\n❌ Model needs RECALIBRATION")

        if not cal_binned.empty and len(cal_binned) >= 3:
            print("\nCalibration by Probability Bin:")
            print(f"\n{'Probability':<20} {'Bets':<8} {'Predicted':<12} {'Actual':<12} {'Error':<10}")
            print("-" * 80)
            for _, row in cal_binned.iterrows():
                print(f"{row['prob_bin']:<20} {row['bets']:<8} {row['predicted']:<12.1f} "
                      f"{row['actual']:<12.1f} {row['error']:<10.2f}")

    # CLV impact
    print("\n\n💎 CLOSING LINE VALUE (CLV) IMPACT")
    print("-" * 80)
    clv_df = analyze_clv_impact(df, min_bets=min_bets)

    if not clv_df.empty:
        print(f"\n{'CLV Bucket':<25} {'Bets':<8} {'Win %':<10} {'Avg CLV':<12} {'ROI %':<10} {'Profit':<12}")
        print("-" * 80)
        for _, row in clv_df.iterrows():
            print(f"{row['clv_bucket']:<25} {row['bets']:<8} {row['win_rate']:<10.1f} "
                  f"{row['avg_clv']:<12.1f} {row['roi']:<10.1f} ${row['total_profit']:<11.2f}")

        # Check if positive CLV = better outcomes
        positive_clv = clv_df[clv_df['avg_clv'] > 0]
        negative_clv = clv_df[clv_df['avg_clv'] < 0]

        if not positive_clv.empty and not negative_clv.empty:
            pos_roi = positive_clv['roi'].mean()
            neg_roi = negative_clv['roi'].mean()

            print(f"\nPositive CLV ROI:  {pos_roi:.1f}%")
            print(f"Negative CLV ROI:  {neg_roi:.1f}%")

            if pos_roi > neg_roi:
                print("✅ Positive CLV leads to better outcomes (validates line shopping)")
            else:
                print("⚠️  CLV not predictive of outcomes (investigate)")
    else:
        print("⚠️  Insufficient CLV data")

    # Key insights
    print("\n\n💡 KEY INSIGHTS")
    print("-" * 80)

    insights = []

    # Best confidence level
    if not conf_df.empty:
        best_conf = conf_df.loc[conf_df['roi'].idxmax()]
        insights.append(f"✅ {best_conf['confidence']} confidence bets have best ROI ({best_conf['roi']:.1f}%)")

        # Check if higher confidence = better results
        if conf_df['roi'].is_monotonic_increasing:
            insights.append("✅ Confidence levels are well-ordered (higher confidence = better ROI)")

    # Edge threshold recommendation
    if not edge_df.empty:
        profitable_edges = edge_df[edge_df['roi'] > 0]
        if not profitable_edges.empty:
            min_profitable_edge = profitable_edges['edge_bucket'].iloc[0]
            insights.append(f"📊 Only bet edges {min_profitable_edge} or higher for profitability")

    # Kelly recommendation
    if not kelly_df.empty:
        best_kelly = kelly_df.loc[kelly_df['roi'].idxmax()]
        insights.append(f"📈 Optimal Kelly range: {best_kelly['kelly_bucket']} ({best_kelly['roi']:.1f}% ROI)")

    # Sample size check
    if len(df) < 100:
        insights.append("⚠️  Sample size is small (<100 bets) - results may not be representative")
    elif len(df) < 300:
        insights.append("⏳ Sample size is moderate (100-300 bets) - continue tracking for confidence")
    else:
        insights.append(f"✅ Sample size is good ({len(df)} bets) - results are statistically meaningful")

    for insight in insights:
        print(f"  {insight}")

    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Advanced performance analytics')
    parser.add_argument('--min-bets', type=int, default=10, help='Minimum bets per category (default: 10)')
    parser.add_argument('--days', type=int, help='Only analyze last N days')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("ADVANCED PERFORMANCE ANALYTICS")
    logger.info("=" * 80)

    # Calculate min date if days specified
    min_date = None
    if args.days:
        min_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
        logger.info(f"Analyzing last {args.days} days (since {min_date})")

    # Load data
    df = load_settled_bets(min_date=min_date)

    if df.empty:
        logger.error("No settled bets found - cannot generate analytics")
        return 1

    # Generate report
    print_report(df, min_bets=args.min_bets)

    return 0


if __name__ == "__main__":
    sys.exit(main())
