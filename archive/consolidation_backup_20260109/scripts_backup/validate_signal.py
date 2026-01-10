#!/usr/bin/env python3
"""
Signal Validation Script

Validates market microstructure signals through backtesting.

Usage:
    python scripts/validate_signal.py --signal steam_move
    python scripts/validate_signal.py --signal rlm
    python scripts/validate_signal.py --signal money_flow --min-samples 50
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.betting.signal_validator import SignalValidator, backtest_steam_follow_strategy, backtest_rlm_strategy
from src.betting.market_signals import SteamMoveDetector, RLMDetector, MoneyFlowAnalyzer
from src.bet_tracker import get_bet_history


def load_historical_data() -> pd.DataFrame:
    """Load historical bet data for backtesting."""
    logger.info("Loading historical bet data...")

    # Load from bet tracker
    bets_df = get_bet_history()

    # Filter to settled bets with CLV
    settled = bets_df[
        (bets_df['outcome'].notna()) &
        (bets_df['clv'].notna())
    ].copy()

    logger.info(f"Loaded {len(settled)} settled bets for backtesting")

    return settled


def validate_steam_signal(
    historical_data: pd.DataFrame,
    validator: SignalValidator
) -> dict:
    """Validate steam move signal."""
    logger.info("=== Validating Steam Move Signal ===")

    # Detect steam moves on historical data
    steam_detector = SteamMoveDetector()
    steam_signals = steam_detector.batch_detect_steam_moves(historical_data)

    if steam_signals.empty:
        logger.warning("No steam moves detected in historical data")
        return {'error': 'No steam moves detected'}

    logger.info(f"Detected {len(steam_signals)} steam moves")

    # Apply steam filter to bets
    steam_bets = backtest_steam_follow_strategy(
        historical_games=historical_data,
        steam_signals=steam_signals,
        baseline_bets=historical_data,
        min_steam_confidence=0.7
    )

    # Validate against baseline
    result = validator.validate_signal(
        signal_name='steam_move',
        signal_bets_df=steam_bets,
        baseline_bets_df=historical_data
    )

    return result.to_dict()


def validate_rlm_signal(
    historical_data: pd.DataFrame,
    validator: SignalValidator
) -> dict:
    """Validate reverse line movement signal."""
    logger.info("=== Validating RLM Signal ===")

    # Detect RLM on historical data (optimized batch processing)
    rlm_detector = RLMDetector()
    game_ids = historical_data['game_id'].unique()

    rlm_signals = []

    # Process in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(game_ids), batch_size):
        batch = game_ids[i:i+batch_size]
        for game_id in batch:
            rlm = rlm_detector.detect_rlm(game_id, bet_type='spread')
            if rlm:
                rlm_signals.append(rlm.to_dict())

    rlm_df = pd.DataFrame(rlm_signals)

    if rlm_df.empty:
        logger.warning("No RLM signals detected in historical data")
        return {'error': 'No RLM signals detected'}

    logger.info(f"Detected {len(rlm_df)} RLM signals")

    # Apply RLM filter to bets
    rlm_bets = backtest_rlm_strategy(
        historical_games=historical_data,
        rlm_signals=rlm_df,
        baseline_bets=historical_data,
        require_model_alignment=True
    )

    # Validate against baseline
    result = validator.validate_signal(
        signal_name='rlm',
        signal_bets_df=rlm_bets,
        baseline_bets_df=historical_data
    )

    return result.to_dict()


def validate_money_flow_signal(
    historical_data: pd.DataFrame,
    validator: SignalValidator
) -> dict:
    """Validate money flow signal."""
    logger.info("=== Validating Money Flow Signal ===")

    # Analyze money flow for historical bets
    flow_analyzer = MoneyFlowAnalyzer()

    flow_signals = []
    for _, bet in historical_data.iterrows():
        flow = flow_analyzer.analyze_money_flow(
            game_id=bet['game_id'],
            bet_type=bet['bet_type'],
            side=bet['bet_side']
        )
        if flow:
            flow_signals.append({**flow.to_dict(), **bet.to_dict()})

    flow_df = pd.DataFrame(flow_signals)

    if flow_df.empty:
        logger.warning("No money flow signals detected")
        return {'error': 'No money flow signals detected'}

    logger.info(f"Analyzed {len(flow_df)} bets with money flow")

    # Filter to bets where signal recommends action
    actionable_flow = flow_df[flow_df['recommendation'] == 'follow_sharp']

    # Validate against baseline
    result = validator.validate_signal(
        signal_name='money_flow',
        signal_bets_df=actionable_flow,
        baseline_bets_df=historical_data
    )

    return result.to_dict()


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description='Validate market microstructure signals')
    parser.add_argument('--signal', required=True, choices=['steam_move', 'rlm', 'money_flow', 'all'],
                       help='Signal to validate')
    parser.add_argument('--min-samples', type=int, default=100,
                       help='Minimum sample size for validation')
    parser.add_argument('--min-roi', type=float, default=0.01,
                       help='Minimum ROI threshold (default: 1%)')
    args = parser.parse_args()

    # Initialize validator
    validator = SignalValidator(
        min_sample_size=args.min_samples,
        min_roi=args.min_roi,
        p_value_threshold=0.05
    )

    # Load historical data
    historical_data = load_historical_data()

    if historical_data.empty:
        logger.error("No historical data available for validation")
        return 1

    # Run validation
    results = {}

    if args.signal == 'steam_move' or args.signal == 'all':
        results['steam_move'] = validate_steam_signal(historical_data, validator)

    if args.signal == 'rlm' or args.signal == 'all':
        results['rlm'] = validate_rlm_signal(historical_data, validator)

    if args.signal == 'money_flow' or args.signal == 'all':
        results['money_flow'] = validate_money_flow_signal(historical_data, validator)

    # Print results
    print("\n" + "="*60)
    print("SIGNAL VALIDATION RESULTS")
    print("="*60 + "\n")

    for signal_name, result in results.items():
        if 'error' in result:
            print(f"❌ {signal_name.upper()}: {result['error']}\n")
            continue

        is_valid = result.get('is_valid', False)
        status = "✅ APPROVED" if is_valid else "❌ REJECTED"

        print(f"{status}: {signal_name.upper()}")
        print(f"  Recommendation: {result.get('recommendation', 'N/A')}")
        print(f"  In-sample ROI: {result.get('in_sample_roi', 0):.2%}")
        print(f"  Out-sample ROI: {result.get('out_sample_roi', 0):.2%}")
        print(f"  Win Rate (out): {result.get('out_sample_win_rate', 0):.2%}")
        print(f"  Avg CLV: {result.get('avg_clv', 0):.3f}")
        print(f"  P-value: {result.get('p_value', 1.0):.4f}")
        print(f"  Sample size: {result.get('num_bets', 0)}")
        print(f"  Notes: {result.get('notes', 'N/A')}")
        print()

    print("="*60 + "\n")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
