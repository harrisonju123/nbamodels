"""
Risk Management Backtest Comparison

Compares backtest performance with and without advanced risk management.

Measures impact of:
- Correlation-aware position sizing
- Drawdown-based bet scaling
- Enhanced exposure limits
- Risk attribution

Usage:
    python scripts/backtest_risk_comparison.py --season 2023-24 --mode both
"""

import sys
sys.path.insert(0, '.')

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from loguru import logger

from src.betting.rigorous_backtest.core import BacktestConfig, RigorousBet
from src.betting.rigorous_backtest.constraints import ConstraintManager
from src.betting.rigorous_backtest.risk_integration import RiskAwareBacktester
from src.risk import RiskConfig, RiskAttributionEngine
from src.betting.kelly import KellyBetSizer
import sqlite3


def load_bets_from_database(
    db_path: str = "data/bets/bets.db",
    min_bets: int = 20,
) -> list:
    """
    Load actual betting history from database.

    Args:
        db_path: Path to bets database
        min_bets: Minimum number of settled bets required

    Returns:
        List of RigorousBet objects sorted chronologically
    """
    logger.info(f"Loading bets from {db_path}...")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Query settled bets ordered by game time
    query = """
    SELECT
        game_id,
        home_team,
        away_team,
        commence_time,
        bet_type,
        bet_side,
        odds,
        line,
        model_prob,
        market_prob,
        edge,
        bet_amount,
        outcome,
        profit
    FROM bets
    WHERE outcome IS NOT NULL
        AND bet_amount IS NOT NULL
        AND edge IS NOT NULL
        AND model_prob IS NOT NULL
    ORDER BY commence_time ASC
    """

    cursor = conn.execute(query)
    rows = cursor.fetchall()
    conn.close()

    if len(rows) < min_bets:
        logger.warning(
            f"Only found {len(rows)} settled bets (minimum {min_bets} recommended). "
            f"Results may not be statistically significant."
        )

    logger.info(f"Loaded {len(rows)} settled bets")

    # Convert to RigorousBet objects
    bets = []
    for row in rows:
        # Parse bet side to uppercase
        bet_side = row['bet_side'].upper() if row['bet_side'] else 'HOME'

        # Convert commence_time to datetime
        try:
            if 'T' in row['commence_time']:
                # ISO format with T
                date_str = row['commence_time'].replace('+0000', '').replace('+00:00', '')
                if 'T' in date_str:
                    date = datetime.fromisoformat(date_str.split('+')[0])
                else:
                    date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            else:
                # Space-separated format
                date_str = row['commence_time'].split('+')[0].strip()
                date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        except Exception as e:
            logger.warning(f"Could not parse date {row['commence_time']}: {e}")
            date = datetime.now()

        bet = RigorousBet(
            date=date,
            game_id=row['game_id'],
            bet_side=bet_side,
            bet_type=row['bet_type'] or 'spread',
            bet_size=row['bet_amount'],
            odds=row['odds'] or -110,
            line=row['line'] if row['line'] else 0.0,
            model_prob=row['model_prob'],
            market_prob=row['market_prob'] if row['market_prob'] else 0.5,
            edge=row['edge']
        )
        bets.append(bet)

    return bets


def run_baseline_backtest(
    bets: list,
    config: BacktestConfig
) -> dict:
    """
    Run baseline backtest with standard constraints only.

    Args:
        bets: List of RigorousBet objects (chronologically sorted)
        config: Backtest configuration

    Returns:
        Dict with results
    """
    logger.info("=" * 80)
    logger.info("BASELINE BACKTEST (Standard Constraints Only)")
    logger.info("=" * 80)

    manager = ConstraintManager(config)
    bankroll = config.initial_bankroll
    peak_bankroll = bankroll

    placed_bets = []
    rejected_bets = []
    current_date = None

    for i, bet in enumerate(bets):
        # Reset daily tracking
        if current_date != bet.date.date():
            current_date = bet.date.date()
            manager.reset_daily()

        # Evaluate bet
        constraints = manager.check_and_adjust_bet(bet, bankroll)

        if constraints.is_allowed and constraints.adjusted_size >= 1.0:
            # Record bet
            manager.record_bet(bet)

            # Simulate outcome (using bet.won if available)
            if bet.won is not None:
                if bet.won:
                    profit = constraints.adjusted_size * (bet.odds - 1)
                else:
                    profit = -constraints.adjusted_size

                bankroll += profit
                peak_bankroll = max(peak_bankroll, bankroll)

                # Update bet record
                bet_copy = bet
                bet_copy.bet_size = constraints.adjusted_size
                bet_copy.pnl = profit
                bet_copy.bankroll_before = bankroll - profit
                bet_copy.bankroll_after = bankroll
                placed_bets.append(bet_copy)

        else:
            rejected_bets.append({
                'bet': bet,
                'reason': constraints.constraint_violations
            })

        # Log progress
        if (i + 1) % 100 == 0:
            logger.info(
                f"Processed {i+1}/{len(bets)} bets | "
                f"Placed: {len(placed_bets)} | "
                f"Bankroll: ${bankroll:.2f}"
            )

    # Calculate metrics
    drawdown = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
    roi = (bankroll - config.initial_bankroll) / config.initial_bankroll

    wins = sum(1 for b in placed_bets if b.won)
    losses = sum(1 for b in placed_bets if not b.won and b.won is not None)
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    total_wagered = sum(b.bet_size for b in placed_bets)
    total_profit = sum(b.pnl for b in placed_bets)

    # Calculate Sharpe ratio (daily returns)
    daily_returns = pd.Series([b.pnl / b.bankroll_before for b in placed_bets])
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0

    results = {
        'mode': 'baseline',
        'initial_bankroll': config.initial_bankroll,
        'final_bankroll': bankroll,
        'peak_bankroll': peak_bankroll,
        'total_profit': total_profit,
        'roi': roi,
        'roi_pct': roi * 100,
        'max_drawdown': drawdown,
        'max_drawdown_pct': drawdown * 100,
        'total_bets': len(placed_bets),
        'rejected_bets': len(rejected_bets),
        'win_rate': win_rate,
        'win_rate_pct': win_rate * 100,
        'total_wagered': total_wagered,
        'sharpe_ratio': sharpe,
        'placed_bets': placed_bets,
        'rejected_bets': rejected_bets
    }

    logger.info("\n" + "=" * 80)
    logger.info("BASELINE RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total Bets Placed: {results['total_bets']}")
    logger.info(f"Total Rejected: {results['rejected_bets']}")
    logger.info(f"Win Rate: {results['win_rate_pct']:.2f}%")
    logger.info(f"Final Bankroll: ${results['final_bankroll']:.2f}")
    logger.info(f"Total Profit: ${results['total_profit']:.2f}")
    logger.info(f"ROI: {results['roi_pct']:.2f}%")
    logger.info(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")

    return results


def run_risk_aware_backtest(
    bets: list,
    config: BacktestConfig,
    risk_config: RiskConfig
) -> dict:
    """
    Run risk-aware backtest with full risk management.

    Args:
        bets: List of RigorousBet objects (chronologically sorted)
        config: Backtest configuration
        risk_config: Risk configuration

    Returns:
        Dict with results
    """
    logger.info("\n" + "=" * 80)
    logger.info("RISK-AWARE BACKTEST (Full Risk Management)")
    logger.info("=" * 80)

    manager = RiskAwareBacktester(config, risk_config)
    bankroll = config.initial_bankroll
    peak_bankroll = bankroll

    placed_bets = []
    rejected_bets = []
    risk_adjustments = []
    current_date = None

    for i, bet in enumerate(bets):
        # Reset daily tracking
        if current_date != bet.date.date():
            current_date = bet.date.date()
            manager.reset_daily()

        # Evaluate bet with risk management
        constraints = manager.evaluate_bet(bet, bankroll, bet.date)

        if constraints.is_allowed and constraints.adjusted_size >= 1.0:
            # Record adjustment details
            adjustment_factor = constraints.adjusted_size / bet.bet_size if bet.bet_size > 0 else 0
            risk_adjustments.append({
                'original_size': bet.bet_size,
                'adjusted_size': constraints.adjusted_size,
                'adjustment_factor': adjustment_factor,
                'violations': constraints.constraint_violations
            })

            # Record bet
            manager.record_bet(bet, constraints.adjusted_size)

            # Simulate outcome
            if bet.won is not None:
                if bet.won:
                    profit = constraints.adjusted_size * (bet.odds - 1)
                else:
                    profit = -constraints.adjusted_size

                bankroll += profit
                peak_bankroll = max(peak_bankroll, bankroll)

                # Update bet record
                bet_copy = bet
                bet_copy.bet_size = constraints.adjusted_size
                bet_copy.pnl = profit
                bet_copy.bankroll_before = bankroll - profit
                bet_copy.bankroll_after = bankroll
                placed_bets.append(bet_copy)

                # Settle game for correlation tracking
                manager.settle_game(bet.game_id)
        else:
            rejected_bets.append({
                'bet': bet,
                'reason': constraints.constraint_violations
            })

        # Log progress
        if (i + 1) % 100 == 0:
            risk_summary = manager.get_risk_summary(bankroll)
            logger.info(
                f"Processed {i+1}/{len(bets)} bets | "
                f"Placed: {len(placed_bets)} | "
                f"Bankroll: ${bankroll:.2f} | "
                f"DD: {risk_summary['drawdown']:.1%} | "
                f"Scale: {risk_summary['drawdown_scale_factor']:.0%}"
            )

    # Calculate metrics
    drawdown = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
    roi = (bankroll - config.initial_bankroll) / config.initial_bankroll

    wins = sum(1 for b in placed_bets if b.won)
    losses = sum(1 for b in placed_bets if not b.won and b.won is not None)
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    total_wagered = sum(b.bet_size for b in placed_bets)
    total_profit = sum(b.pnl for b in placed_bets)

    # Sharpe ratio
    daily_returns = pd.Series([b.pnl / b.bankroll_before for b in placed_bets])
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0

    # Risk-specific metrics
    avg_adjustment = np.mean([a['adjustment_factor'] for a in risk_adjustments]) if risk_adjustments else 1.0
    correlation_reductions = sum(1 for a in risk_adjustments if any('correlation' in v for v in a['violations']))
    drawdown_reductions = sum(1 for a in risk_adjustments if any('drawdown' in v for v in a['violations']))

    final_risk_summary = manager.get_risk_summary(bankroll)

    results = {
        'mode': 'risk_aware',
        'initial_bankroll': config.initial_bankroll,
        'final_bankroll': bankroll,
        'peak_bankroll': peak_bankroll,
        'total_profit': total_profit,
        'roi': roi,
        'roi_pct': roi * 100,
        'max_drawdown': drawdown,
        'max_drawdown_pct': drawdown * 100,
        'total_bets': len(placed_bets),
        'rejected_bets': len(rejected_bets),
        'win_rate': win_rate,
        'win_rate_pct': win_rate * 100,
        'total_wagered': total_wagered,
        'sharpe_ratio': sharpe,
        'avg_adjustment_factor': avg_adjustment,
        'correlation_reductions': correlation_reductions,
        'drawdown_reductions': drawdown_reductions,
        'final_risk_status': final_risk_summary,
        'placed_bets': placed_bets,
        'rejected_bets': rejected_bets,
        'risk_adjustments': risk_adjustments
    }

    logger.info("\n" + "=" * 80)
    logger.info("RISK-AWARE RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total Bets Placed: {results['total_bets']}")
    logger.info(f"Total Rejected: {results['rejected_bets']}")
    logger.info(f"  - Correlation filters: {results['correlation_reductions']}")
    logger.info(f"  - Drawdown reductions: {results['drawdown_reductions']}")
    logger.info(f"Avg Adjustment Factor: {results['avg_adjustment_factor']:.2%}")
    logger.info(f"Win Rate: {results['win_rate_pct']:.2f}%")
    logger.info(f"Final Bankroll: ${results['final_bankroll']:.2f}")
    logger.info(f"Total Profit: ${results['total_profit']:.2f}")
    logger.info(f"ROI: {results['roi_pct']:.2f}%")
    logger.info(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")

    return results


def compare_results(baseline: dict, risk_aware: dict) -> dict:
    """
    Compare baseline vs risk-aware results.

    Args:
        baseline: Baseline results dict
        risk_aware: Risk-aware results dict

    Returns:
        Dict with comparison metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON: Risk Management Impact")
    logger.info("=" * 80)

    comparison = {
        'roi_diff': risk_aware['roi_pct'] - baseline['roi_pct'],
        'roi_improvement_pct': ((risk_aware['roi'] / baseline['roi']) - 1) * 100 if baseline['roi'] != 0 else 0,
        'drawdown_diff': risk_aware['max_drawdown_pct'] - baseline['max_drawdown_pct'],
        'drawdown_reduction_pct': ((baseline['max_drawdown'] - risk_aware['max_drawdown']) / baseline['max_drawdown']) * 100 if baseline['max_drawdown'] > 0 else 0,
        'sharpe_diff': risk_aware['sharpe_ratio'] - baseline['sharpe_ratio'],
        'sharpe_improvement_pct': ((risk_aware['sharpe_ratio'] / baseline['sharpe_ratio']) - 1) * 100 if baseline['sharpe_ratio'] != 0 else 0,
        'bets_diff': risk_aware['total_bets'] - baseline['total_bets'],
        'bets_reduction_pct': ((baseline['total_bets'] - risk_aware['total_bets']) / baseline['total_bets']) * 100 if baseline['total_bets'] > 0 else 0,
        'profit_diff': risk_aware['total_profit'] - baseline['total_profit']
    }

    logger.info("\nROI:")
    logger.info(f"  Baseline: {baseline['roi_pct']:.2f}%")
    logger.info(f"  Risk-Aware: {risk_aware['roi_pct']:.2f}%")
    logger.info(f"  Difference: {comparison['roi_diff']:+.2f}pp")

    logger.info("\nMax Drawdown:")
    logger.info(f"  Baseline: {baseline['max_drawdown_pct']:.2f}%")
    logger.info(f"  Risk-Aware: {risk_aware['max_drawdown_pct']:.2f}%")
    logger.info(f"  Reduction: {comparison['drawdown_reduction_pct']:.2f}%")

    logger.info("\nSharpe Ratio:")
    logger.info(f"  Baseline: {baseline['sharpe_ratio']:.2f}")
    logger.info(f"  Risk-Aware: {risk_aware['sharpe_ratio']:.2f}")
    logger.info(f"  Improvement: {comparison['sharpe_improvement_pct']:+.2f}%")

    logger.info("\nBet Volume:")
    logger.info(f"  Baseline: {baseline['total_bets']} bets")
    logger.info(f"  Risk-Aware: {risk_aware['total_bets']} bets")
    logger.info(f"  Reduction: {comparison['bets_reduction_pct']:.2f}%")

    logger.info("\nProfit:")
    logger.info(f"  Baseline: ${baseline['total_profit']:.2f}")
    logger.info(f"  Risk-Aware: ${risk_aware['total_profit']:.2f}")
    logger.info(f"  Difference: ${comparison['profit_diff']:+.2f}")

    return comparison


def generate_sample_bets(n: int = 500) -> list:
    """
    Generate sample bets for testing.

    Args:
        n: Number of bets to generate

    Returns:
        List of RigorousBet objects
    """
    logger.info(f"Generating {n} sample bets for testing...")

    bets = []
    start_date = datetime(2023, 10, 1)

    for i in range(n):
        # Random outcome with slight edge
        won = np.random.random() < 0.53  # 53% win rate

        bet = RigorousBet(
            date=start_date + pd.Timedelta(days=i // 3),  # ~3 bets per day
            game_id=f"2023{10 + i // 30:02d}{(i % 30) + 1:02d}_LAL_BOS",
            bet_side="HOME" if i % 2 == 0 else "AWAY",
            bet_type="SPREAD",
            bet_size=50.0,  # Base size, will be adjusted
            odds=1.909,  # -110 American
            line=-3.5,
            model_prob=0.55,
            market_prob=0.52,
            edge=0.03,
            won=won,
            actual_score_home=110 if won else 105,
            actual_score_away=105 if won else 110,
        )
        bets.append(bet)

    return bets


def main():
    parser = argparse.ArgumentParser(description='Compare backtest with/without risk management')
    parser.add_argument('--mode', choices=['baseline', 'risk', 'both'], default='both',
                       help='Which mode to run')
    parser.add_argument('--sample', action='store_true',
                       help='Use sample data for testing')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use realistic synthetic data generated from real patterns')
    parser.add_argument('--database', type=str, default='data/bets/bets.db',
                       help='Path to bets database (default: data/bets/bets.db)')
    parser.add_argument('--initial-bankroll', type=float, default=10000.0,
                       help='Initial bankroll')
    parser.add_argument('--output', type=str, default='data/backtest/risk_comparison.json',
                       help='Output file for results')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("RISK MANAGEMENT BACKTEST COMPARISON")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Initial Bankroll: ${args.initial_bankroll:.2f}")

    # Configuration
    # Using very small Kelly fraction (0.01 = 1% Kelly) to match $100 flat-stake betting
    # With 7.7% edges and $10k bankroll, full Kelly = ~$1500, so 1% Kelly ≈ $100-150
    backtest_config = BacktestConfig(
        initial_bankroll=args.initial_bankroll,
        kelly_fraction=0.01,  # 1% Kelly to match $100 stakes
        max_bet_fraction=0.03,  # Max 3% per bet
        min_edge_threshold=0.02,  # Lower threshold to 2%
        max_daily_exposure=0.20,  # 20% daily limit
        max_per_game_exposure=0.03
    )

    risk_config = RiskConfig(
        max_same_team_exposure=0.15,  # 15% max per team
        max_same_game_exposure=0.08,  # 8% max per game
        max_same_conference_exposure=0.25,  # 25% per conference
        max_same_division_exposure=0.15,  # 15% per division
        max_daily_exposure=0.20,  # 20% daily
        max_weekly_exposure=0.50,  # 50% weekly
        drawdown_hard_stop=0.30
    )

    # Load or generate bets
    if args.sample:
        bets = generate_sample_bets(500)
    elif args.synthetic:
        import pickle
        synthetic_path = Path("data/backtest/synthetic_bets_realistic.pkl")
        if not synthetic_path.exists():
            logger.error(f"Synthetic data not found at {synthetic_path}")
            logger.error("Run: python scripts/generate_realistic_backtest_data.py")
            return
        with open(synthetic_path, 'rb') as f:
            bets = pickle.load(f)
        logger.info(f"Loaded {len(bets)} synthetic bets from {synthetic_path}")
    else:
        bets = load_bets_from_database(db_path=args.database, min_bets=20)
        if not bets:
            logger.error("No bets found in database")
            return

    # Run backtests
    results = {}

    if args.mode in ['baseline', 'both']:
        results['baseline'] = run_baseline_backtest(bets, backtest_config)

    if args.mode in ['risk', 'both']:
        results['risk_aware'] = run_risk_aware_backtest(bets, backtest_config, risk_config)

    # Compare if both modes run
    if args.mode == 'both':
        comparison = compare_results(results['baseline'], results['risk_aware'])
        results['comparison'] = comparison

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    import json
    json_results = {
        k: {
            sk: sv for sk, sv in v.items()
            if sk not in ['placed_bets', 'rejected_bets', 'risk_adjustments']
        }
        for k, v in results.items()
    }

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    logger.info(f"\n✓ Results saved to: {output_path}")
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
