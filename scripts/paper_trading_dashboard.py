"""
Paper Trading Performance Dashboard

Shows current performance metrics for the paper trading system.
"""

import sqlite3
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd


DB_PATH = "data/bets/bets.db"


def get_performance_summary():
    """Get overall performance summary."""
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT
        COUNT(*) as total_bets,
        SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
        SUM(CASE WHEN outcome = 'push' THEN 1 ELSE 0 END) as pushes,
        ROUND(100.0 * SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) /
              NULLIF(COUNT(CASE WHEN outcome IS NOT NULL THEN 1 END), 0), 2) as win_rate,
        ROUND(SUM(COALESCE(bet_amount, 100)), 2) as total_wagered,
        ROUND(SUM(COALESCE(profit, 0)), 2) as total_profit,
        ROUND(100.0 * SUM(COALESCE(profit, 0)) /
              NULLIF(SUM(COALESCE(bet_amount, 100)), 0), 2) as roi
    FROM bets
    WHERE outcome IS NOT NULL
    """

    result = conn.execute(query).fetchone()
    conn.close()

    return {
        'total_bets': result[0],
        'wins': result[1],
        'losses': result[2],
        'pushes': result[3],
        'win_rate': result[4],
        'total_wagered': result[5],
        'total_profit': result[6],
        'roi': result[7],
    }


def get_performance_by_side():
    """Get performance broken down by home/away."""
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT
        bet_side,
        COUNT(*) as bets,
        SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) /
              NULLIF(COUNT(CASE WHEN outcome IS NOT NULL THEN 1 END), 0), 2) as win_rate,
        ROUND(SUM(COALESCE(profit, 0)), 2) as profit,
        ROUND(100.0 * SUM(COALESCE(profit, 0)) /
              NULLIF(SUM(COALESCE(bet_amount, 100)), 0), 2) as roi
    FROM bets
    WHERE outcome IS NOT NULL
    GROUP BY bet_side
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def get_recent_performance(days=30):
    """Get performance for the last N days."""
    conn = sqlite3.connect(DB_PATH)

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()

    query = f"""
    SELECT
        DATE(logged_at) as date,
        COUNT(*) as bets,
        SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
        ROUND(SUM(COALESCE(profit, 0)), 2) as daily_profit,
        ROUND(SUM(SUM(COALESCE(profit, 0))) OVER (ORDER BY DATE(logged_at)), 2) as cumulative_profit
    FROM bets
    WHERE outcome IS NOT NULL AND logged_at >= '{cutoff}'
    GROUP BY DATE(logged_at)
    ORDER BY date DESC
    LIMIT 15
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def get_pending_bets():
    """Get bets awaiting settlement."""
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT
        id,
        home_team,
        away_team,
        bet_side,
        line,
        odds,
        bet_amount,
        edge,
        logged_at
    FROM bets
    WHERE outcome IS NULL
    ORDER BY logged_at DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def print_dashboard():
    """Print formatted dashboard."""

    print("=" * 80)
    print("📊 PAPER TRADING PERFORMANCE DASHBOARD")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Overall Summary
    summary = get_performance_summary()

    print("📈 OVERALL PERFORMANCE")
    print("-" * 80)
    print(f"Total Bets:      {summary['total_bets']}")
    print(f"Wins:            {summary['wins']}")
    print(f"Losses:          {summary['losses']}")
    print(f"Pushes:          {summary['pushes']}")
    print(f"Win Rate:        {summary['win_rate']:.1f}%")
    print(f"Total Wagered:   ${summary['total_wagered']:,.2f}")
    print(f"Total Profit:    ${summary['total_profit']:+,.2f}")
    print(f"ROI:             {summary['roi']:.2f}%")
    print()

    # Performance by Side
    by_side = get_performance_by_side()

    print("🏠 PERFORMANCE BY SIDE")
    print("-" * 80)
    for _, row in by_side.iterrows():
        side = row['bet_side'].upper()
        bets = row['bets']
        wins = row['wins']
        win_rate = row['win_rate']
        profit = row['profit']
        roi = row['roi']

        print(f"{side:6} | {bets:3} bets | {wins:3}W-{bets-wins:3}L | {win_rate:5.1f}% WR | ${profit:+8.2f} | {roi:+6.2f}% ROI")
    print()

    # Recent Performance
    recent = get_recent_performance(days=30)

    print("📅 LAST 15 DAYS")
    print("-" * 80)
    print(f"{'Date':<12} {'Bets':>5} {'Wins':>5} {'Daily P&L':>12} {'Cumulative':>12}")
    print("-" * 80)

    for _, row in recent.iterrows():
        date = row['date']
        bets = row['bets']
        wins = row['wins']
        daily = row['daily_profit']
        cumulative = row['cumulative_profit']

        print(f"{date:<12} {bets:5} {wins:5} {daily:+12.2f} {cumulative:+12.2f}")
    print()

    # Pending Bets
    pending = get_pending_bets()

    print(f"⏳ PENDING BETS ({len(pending)} awaiting settlement)")
    print("-" * 80)

    if len(pending) > 0:
        for _, row in pending.head(5).iterrows():
            try:
                side = str(row['bet_side']).upper()
                team = str(row['home_team'] if side == 'HOME' else row['away_team'])
                opponent = str(row['away_team'] if side == 'HOME' else row['home_team'])

                # Handle potentially corrupt numeric fields
                try:
                    line = float(row['line']) if row['line'] is not None else 0.0
                except (ValueError, TypeError):
                    line = 0.0

                try:
                    odds = float(row['odds']) if row['odds'] is not None else -110.0
                except (ValueError, TypeError):
                    odds = -110.0

                try:
                    bet_amt = float(row['bet_amount']) if row['bet_amount'] is not None else 0.0
                except (ValueError, TypeError):
                    bet_amt = 0.0

                try:
                    edge = float(row['edge']) if row['edge'] is not None else 0.0
                except (ValueError, TypeError):
                    edge = 0.0

                logged = pd.to_datetime(row['logged_at']).strftime('%m/%d %H:%M')

                print(f"{logged} | {side:4} {team:20} vs {opponent:20} | {line:+5.1f} @ {odds:+4.0f} | ${bet_amt:.2f} | Edge: {edge:.3f}")
            except Exception as e:
                logger.debug(f"Skipping corrupt pending bet row: {e}")
                continue

        if len(pending) > 5:
            print(f"... and {len(pending) - 5} more")
    else:
        print("No pending bets")

    print()
    print("=" * 80)
    print("💡 TIP: Run 'python scripts/daily_betting_pipeline.py' to get new bets")
    print("=" * 80)


if __name__ == "__main__":
    try:
        print_dashboard()
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        import traceback
        traceback.print_exc()
