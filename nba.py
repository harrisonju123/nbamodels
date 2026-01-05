#!/usr/bin/env python3
"""
NBA Betting System - Central Orchestrator

Single command-line tool to manage all operations in your NBA betting system.

Usage:
    python nba.py --help                    # Show all commands
    python nba.py bets                      # Get today's bet recommendations
    python nba.py pipeline                  # Run full daily pipeline
    python nba.py update                    # Update all data (lines, odds, etc.)
    python nba.py status                    # Show system status
"""

import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import sqlite3

PROJECT_ROOT = Path(__file__).parent


def run_script(script_path: str, args: list = None, description: str = None):
    """Run a Python script and return result."""
    if description:
        print(f"\n{'='*80}")
        print(f"🔄 {description}")
        print(f"{'='*80}\n")

    cmd = [sys.executable, str(PROJECT_ROOT / script_path)]
    if args:
        cmd.extend(args)

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def get_bankroll_status():
    """Get current bankroll and drawdown."""
    try:
        conn = sqlite3.connect(PROJECT_ROOT / "data" / "bets" / "bets.db")
        cursor = conn.cursor()

        # Get current bankroll
        cursor.execute("SELECT current_bankroll FROM bankroll ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        current_bankroll = row[0] if row else 1000.0

        # Get settled bets count
        cursor.execute("SELECT COUNT(*) FROM bets WHERE outcome IS NOT NULL")
        settled_bets = cursor.fetchone()[0]

        # Get pending bets count
        cursor.execute("SELECT COUNT(*) FROM bets WHERE outcome IS NULL")
        pending_bets = cursor.fetchone()[0]

        # Calculate win rate
        cursor.execute("SELECT COUNT(*) FROM bets WHERE outcome = 'win'")
        wins = cursor.fetchone()[0]
        win_rate = (wins / settled_bets * 100) if settled_bets > 0 else 0

        # Get total profit
        cursor.execute("""
            SELECT SUM(CASE
                WHEN outcome = 'win' THEN bet_amount * (odds - 1)
                WHEN outcome = 'loss' THEN -bet_amount
                ELSE 0
            END) as total_profit
            FROM bets WHERE outcome IS NOT NULL
        """)
        total_profit = cursor.fetchone()[0] or 0

        conn.close()

        return {
            'bankroll': current_bankroll,
            'settled_bets': settled_bets,
            'pending_bets': pending_bets,
            'win_rate': win_rate,
            'total_profit': total_profit
        }
    except Exception as e:
        return None


def cmd_status(args):
    """Show system status."""
    print("\n" + "="*80)
    print("📊 NBA BETTING SYSTEM STATUS")
    print("="*80)

    print(f"\n⏰ Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Bankroll status
    status = get_bankroll_status()
    if status:
        print(f"\n💰 BANKROLL")
        print(f"   Current: ${status['bankroll']:,.2f}")
        print(f"   Total P&L: ${status['total_profit']:+,.2f}")

        print(f"\n📈 BETTING HISTORY")
        print(f"   Settled Bets: {status['settled_bets']}")
        print(f"   Pending Bets: {status['pending_bets']}")
        print(f"   Win Rate: {status['win_rate']:.1f}%")
    else:
        print("\n⚠️  Could not retrieve bankroll data")

    # Check model file
    model_path = PROJECT_ROOT / "models" / "spread_model.pkl"
    print(f"\n🤖 MODEL")
    if model_path.exists():
        mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
        age = datetime.now() - mtime
        print(f"   Status: ✅ Trained")
        print(f"   Last Updated: {mtime.strftime('%Y-%m-%d %H:%M:%S')} ({age.days} days ago)")
    else:
        print(f"   Status: ❌ Not Found")
        print(f"   Action: Run 'python nba.py train' to train model")

    # Check database
    db_path = PROJECT_ROOT / "data" / "bets" / "bets.db"
    print(f"\n💾 DATABASE")
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"   Status: ✅ Connected")
        print(f"   Size: {size_mb:.2f} MB")
    else:
        print(f"   Status: ❌ Not Found")

    print("\n" + "="*80)


def cmd_bets(args):
    """Get today's bet recommendations."""
    return run_script(
        "scripts/get_todays_bets.py",
        description="Getting Today's Bet Recommendations"
    )


def cmd_pipeline(args):
    """Run full daily betting pipeline."""
    pipeline_args = []
    if args.live:
        pipeline_args.append("--live")
    if args.dry_run:
        pipeline_args.append("--dry-run")
    if args.strategy:
        pipeline_args.extend(["--strategy", args.strategy])

    return run_script(
        "scripts/daily_betting_pipeline.py",
        args=pipeline_args,
        description="Running Daily Betting Pipeline"
    )


def cmd_update(args):
    """Update all data (lines, odds, settlement)."""
    print("\n" + "="*80)
    print("🔄 UPDATING ALL DATA")
    print("="*80)

    tasks = [
        ("scripts/fetch_opening_lines.py", "Fetching Opening Lines"),
        ("scripts/fetch_closing_lines.py", "Fetching Closing Lines"),
        ("scripts/settle_bets.py", "Settling Bets"),
        ("scripts/calculate_clv.py", "Calculating CLV"),
    ]

    for script, desc in tasks:
        result = run_script(script, description=desc)
        if result != 0:
            print(f"\n⚠️  Warning: {desc} failed with exit code {result}")

    print("\n" + "="*80)
    print("✅ Data update complete")
    print("="*80)


def cmd_train(args):
    """Train the prediction model."""
    return run_script(
        "scripts/train_model.py",
        description="Training Prediction Model"
    )


def cmd_backtest(args):
    """Run backtest."""
    return run_script(
        "scripts/run_proper_backtest.py",
        description="Running Backtest"
    )


def cmd_dashboard(args):
    """Launch analytics dashboard."""
    print("\n" + "="*80)
    print("📊 LAUNCHING ANALYTICS DASHBOARD")
    print("="*80)
    print("\n🌐 Dashboard will open at: http://localhost:8501")
    print("   Press Ctrl+C to stop\n")

    return run_script("analytics_dashboard.py")


def cmd_settle(args):
    """Settle pending bets."""
    return run_script(
        "scripts/settle_bets.py",
        description="Settling Pending Bets"
    )


def cmd_daily(args):
    """Run complete daily workflow."""
    print("\n" + "="*80)
    print("🌅 DAILY WORKFLOW")
    print("="*80)
    print("\nThis will:")
    print("  1. Update all data (lines, odds, settlement)")
    print("  2. Get today's bet recommendations")
    print("  3. Show current system status")

    # Step 1: Update data
    cmd_update(args)

    # Step 2: Get bets
    cmd_bets(args)

    # Step 3: Show status
    cmd_status(args)

    print("\n" + "="*80)
    print("✅ Daily workflow complete")
    print("="*80)


def cmd_init(args):
    """Initialize the system (first-time setup)."""
    print("\n" + "="*80)
    print("🚀 INITIALIZING NBA BETTING SYSTEM")
    print("="*80)

    print("\n📋 This will:")
    print("  1. Create necessary directories")
    print("  2. Initialize database")
    print("  3. Fetch historical data")
    print("  4. Train initial model")

    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Aborted")
        return 1

    # Create directories
    print("\n1️⃣  Creating directories...")
    dirs = ["data/bets", "data/lines", "data/backtest", "logs", "models"]
    for dir_path in dirs:
        (PROJECT_ROOT / dir_path).mkdir(parents=True, exist_ok=True)
    print("   ✓ Directories created")

    # Initialize database
    print("\n2️⃣  Initializing database...")
    # Database is auto-created by scripts on first use
    print("   ✓ Database will be created on first bet")

    # Fetch data
    print("\n3️⃣  Fetching historical data...")
    run_script("scripts/fetch_opening_lines.py")
    run_script("scripts/fetch_closing_lines.py")

    # Train model
    print("\n4️⃣  Training initial model...")
    run_script("scripts/train_model.py")

    print("\n" + "="*80)
    print("✅ Initialization complete!")
    print("="*80)
    print("\n💡 Next steps:")
    print("   - Run 'python nba.py bets' to get today's recommendations")
    print("   - Run 'python nba.py dashboard' to view analytics")
    print("   - Run 'python nba.py --help' to see all commands")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NBA Betting System - Central Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nba.py status              # Check system status
  python nba.py bets                # Get today's bet recommendations
  python nba.py pipeline            # Run full betting pipeline (paper mode)
  python nba.py pipeline --live     # Run pipeline with LIVE bets
  python nba.py update              # Update all data
  python nba.py daily               # Complete daily workflow
  python nba.py dashboard           # Launch analytics dashboard
  python nba.py train               # Retrain prediction model
  python nba.py backtest            # Run backtest
  python nba.py init                # First-time setup

Common Workflows:
  - Morning routine:   python nba.py update && python nba.py status
  - Before games:      python nba.py bets
  - After games:       python nba.py settle
  - Weekly review:     python nba.py dashboard
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Status command
    subparsers.add_parser('status', help='Show system status')

    # Bets command
    subparsers.add_parser('bets', help='Get today\'s bet recommendations')

    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run daily betting pipeline')
    pipeline_parser.add_argument('--live', action='store_true', help='Live mode (real bets)')
    pipeline_parser.add_argument('--dry-run', action='store_true', help='Dry run (no logging)')
    pipeline_parser.add_argument('--strategy', type=str, choices=['baseline', 'clv_filtered', 'optimal_timing', 'team_filtered'],
                                help='Betting strategy to use')

    # Update command
    subparsers.add_parser('update', help='Update all data (lines, odds, settlement)')

    # Train command
    subparsers.add_parser('train', help='Train prediction model')

    # Backtest command
    subparsers.add_parser('backtest', help='Run backtest')

    # Dashboard command
    subparsers.add_parser('dashboard', help='Launch analytics dashboard')

    # Settle command
    subparsers.add_parser('settle', help='Settle pending bets')

    # Daily command
    subparsers.add_parser('daily', help='Run complete daily workflow')

    # Init command
    subparsers.add_parser('init', help='Initialize system (first-time setup)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Dispatch to command handler
    commands = {
        'status': cmd_status,
        'bets': cmd_bets,
        'pipeline': cmd_pipeline,
        'update': cmd_update,
        'train': cmd_train,
        'backtest': cmd_backtest,
        'dashboard': cmd_dashboard,
        'settle': cmd_settle,
        'daily': cmd_daily,
        'init': cmd_init,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
