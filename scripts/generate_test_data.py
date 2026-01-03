#!/usr/bin/env python3
"""
Generate Realistic Test Data for CLV Backtest

Creates synthetic historical bets with realistic CLV patterns
to test and demonstrate the backtest functionality.

Usage:
    python scripts/generate_test_data.py --num-bets 100
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import sqlite3
from datetime import datetime, timedelta, timezone
import numpy as np
from loguru import logger

from src.bet_tracker import DB_PATH


def generate_realistic_bet_data(num_bets: int = 100) -> list:
    """
    Generate realistic bet data with CLV patterns.

    Patterns simulated:
    - Positive CLV bets have ~55-58% win rate
    - Negative CLV bets have ~48-52% win rate
    - CLV correlates with edge
    - Line velocity affects CLV
    """
    np.random.seed(42)  # Reproducible results

    bets = []
    now = datetime.now(timezone.utc)

    teams = ['LAL', 'GSW', 'BOS', 'MIL', 'DEN', 'PHX', 'MIA', 'DAL',
             'PHI', 'NYK', 'CLE', 'MEM', 'SAC', 'ATL', 'TOR', 'MIN']

    for i in range(num_bets):
        # Random game 1-60 days ago
        days_ago = np.random.uniform(1, 60)
        settle_time = now - timedelta(days=days_ago)
        commence_time = settle_time - timedelta(hours=2)

        # Random teams
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])

        # Edge: normally distributed around 5.5 with some variance
        edge = np.random.normal(5.5, 2.0)
        edge = np.clip(edge, 3.0, 10.0)

        # Bet side based on edge
        if edge > 0:
            bet_side = 'home' if np.random.random() > 0.3 else 'away'
        else:
            bet_side = 'away' if np.random.random() > 0.3 else 'home'

        # CLV patterns:
        # Higher edge → higher CLV (correlation ~0.3)
        # Add noise to make it realistic
        base_clv = (edge - 5.0) * 0.004  # Maps edge 5-10 to CLV 0-2%
        clv_noise = np.random.normal(0, 0.015)  # ±1.5% noise
        clv = base_clv + clv_noise

        # Multi-snapshot CLV (decay over time)
        clv_at_1hr = clv + np.random.normal(0, 0.005)
        clv_at_4hr = clv + np.random.normal(0, 0.003)
        clv_at_12hr = clv * 0.8 + np.random.normal(0, 0.002)
        clv_at_24hr = clv * 0.6 + np.random.normal(0, 0.002)
        max_clv = max(clv_at_1hr, clv_at_4hr, clv_at_12hr, clv_at_24hr)

        # Line velocity (random, with some correlation to CLV)
        line_velocity = np.random.normal(clv * 10, 2.0)  # Weak correlation

        # Outcome: positive CLV → higher win probability
        # Base win rate: 52.4% (break-even at -110)
        # Adjust by CLV: +1% CLV → +2% win rate
        win_prob = 0.524 + (clv * 2.0)
        win_prob = np.clip(win_prob, 0.35, 0.70)  # Realistic bounds

        outcome = 'win' if np.random.random() < win_prob else 'loss'

        # Profit calculation (assume $100 bet at -110 odds)
        if outcome == 'win':
            profit = 90.91  # Win $90.91 on $100 bet at -110
        else:
            profit = -100.0

        # Odds and line
        odds = -110.0
        line = np.random.uniform(-7.5, 7.5)

        # Closing odds (slightly different from booking odds based on CLV)
        closing_odds = odds + (clv * 1000)  # CLV affects closing odds
        closing_line = line + np.random.normal(0, 0.5)

        # Snapshot coverage (mostly good data)
        coverage = np.random.choice([1.0, 0.75, 0.5, 0.25], p=[0.5, 0.3, 0.15, 0.05])

        # Closing line source (mostly snapshot, some API)
        source = np.random.choice(
            ['snapshot', 'api', 'opening'],
            p=[0.70, 0.25, 0.05]
        )

        # Booked hours before
        booked_hours = np.random.uniform(1, 24)

        bet = {
            'id': f"test_bet_{i:04d}",
            'game_id': f"test_game_{i:04d}",
            'home_team': home_team,
            'away_team': away_team,
            'commence_time': commence_time.isoformat(),
            'bet_type': 'spread',
            'bet_side': bet_side,
            'odds': odds,
            'line': line,
            'edge': edge,
            'kelly': edge * 0.01,  # Simple Kelly
            'model_prob': 0.50 + (edge * 0.01),
            'market_prob': 0.50,
            'logged_at': (commence_time - timedelta(hours=booked_hours)).isoformat(),
            'outcome': outcome,
            'profit': profit,
            'settled_at': settle_time.isoformat(),
            'closing_odds': closing_odds,
            'closing_line': closing_line,
            'clv': clv,
            'clv_at_1hr': clv_at_1hr,
            'clv_at_4hr': clv_at_4hr,
            'clv_at_12hr': clv_at_12hr,
            'clv_at_24hr': clv_at_24hr,
            'line_velocity': line_velocity,
            'max_clv_achieved': max_clv,
            'snapshot_coverage': coverage,
            'closing_line_source': source,
            'booked_hours_before': booked_hours,
            'bookmaker': 'draftkings',
            'clv_updated_at': settle_time.isoformat(),
        }

        bets.append(bet)

    return bets


def insert_test_data(bets: list):
    """Insert test bets into database."""
    conn = sqlite3.connect(DB_PATH)

    # Clear existing test data
    conn.execute("DELETE FROM bets WHERE id LIKE 'test_bet_%'")
    conn.commit()

    # Insert new test data
    for bet in bets:
        conn.execute("""
            INSERT INTO bets (
                id, game_id, home_team, away_team, commence_time,
                bet_type, bet_side, odds, line,
                edge, kelly, model_prob, market_prob,
                logged_at, outcome, profit, settled_at,
                closing_odds, closing_line, clv,
                clv_at_1hr, clv_at_4hr, clv_at_12hr, clv_at_24hr,
                line_velocity, max_clv_achieved, snapshot_coverage,
                closing_line_source, booked_hours_before, bookmaker,
                clv_updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            bet['id'], bet['game_id'], bet['home_team'], bet['away_team'],
            bet['commence_time'], bet['bet_type'], bet['bet_side'],
            bet['odds'], bet['line'], bet['edge'], bet['kelly'],
            bet['model_prob'], bet['market_prob'], bet['logged_at'],
            bet['outcome'], bet['profit'], bet['settled_at'],
            bet['closing_odds'], bet['closing_line'], bet['clv'],
            bet['clv_at_1hr'], bet['clv_at_4hr'], bet['clv_at_12hr'],
            bet['clv_at_24hr'], bet['line_velocity'], bet['max_clv_achieved'],
            bet['snapshot_coverage'], bet['closing_line_source'],
            bet['booked_hours_before'], bet['bookmaker'], bet['clv_updated_at']
        ))

    conn.commit()
    conn.close()


def print_summary(bets: list):
    """Print summary statistics of generated data."""
    import pandas as pd

    df = pd.DataFrame(bets)

    logger.info("\n" + "=" * 60)
    logger.info("TEST DATA SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Bets Generated: {len(bets)}")
    logger.info(f"\nWin Rate: {(df['outcome'] == 'win').mean():.1%}")
    logger.info(f"Average CLV: {df['clv'].mean():+.3f} ({df['clv'].mean()*100:+.2f}%)")
    logger.info(f"Average Edge: {df['edge'].mean():.2f} points")
    logger.info(f"Total Profit: ${df['profit'].sum():+,.2f}")
    logger.info(f"ROI: {(df['profit'].sum() / (len(df) * 100)) * 100:+.2f}%")

    logger.info(f"\nCLV Distribution:")
    logger.info(f"  Positive CLV: {(df['clv'] > 0).sum()} ({(df['clv'] > 0).mean():.1%})")
    logger.info(f"  Negative CLV: {(df['clv'] < 0).sum()} ({(df['clv'] < 0).mean():.1%})")

    # Win rate by CLV sign
    pos_clv = df[df['clv'] > 0]
    neg_clv = df[df['clv'] < 0]

    if len(pos_clv) > 0:
        logger.info(f"\nWin Rate (Positive CLV): {(pos_clv['outcome'] == 'win').mean():.1%}")
    if len(neg_clv) > 0:
        logger.info(f"Win Rate (Negative CLV): {(neg_clv['outcome'] == 'win').mean():.1%}")

    logger.info(f"\nSnapshot Coverage:")
    logger.info(f"  Average: {df['snapshot_coverage'].mean():.1%}")
    logger.info(f"  Full coverage (100%): {(df['snapshot_coverage'] == 1.0).sum()}")

    logger.info(f"\nClosing Line Sources:")
    for source in df['closing_line_source'].value_counts().items():
        logger.info(f"  {source[0]}: {source[1]}")


def main():
    parser = argparse.ArgumentParser(description='Generate test data for CLV backtest')
    parser.add_argument('--num-bets', type=int, default=100,
                       help='Number of test bets to generate (default: 100)')
    args = parser.parse_args()

    logger.info("=== Generating Test Data for CLV Backtest ===")

    # Generate data
    bets = generate_realistic_bet_data(args.num_bets)

    # Print summary
    print_summary(bets)

    # Insert into database
    logger.info("\nInserting test data into database...")
    insert_test_data(bets)
    logger.info(f"✓ Inserted {len(bets)} test bets")

    logger.info("\n=== Test Data Generation Complete ===")
    logger.info("Ready to run backtest: python scripts/backtest_clv_strategy.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
