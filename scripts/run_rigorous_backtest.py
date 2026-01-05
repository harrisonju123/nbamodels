"""
Run rigorous backtest with walk-forward validation and Monte Carlo simulation.

This script demonstrates how to use the rigorous backtesting framework
to properly validate betting strategies with statistical rigor.

Usage:
    python scripts/run_rigorous_backtest.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import rigorous backtest framework
from src.betting.rigorous_backtest import (
    BacktestConfig,
    WalkForwardValidator,
    StatisticalAnalyzer,
    MonteCarloSimulator,
    TransactionCostModel,
    ConstraintManager,
    BacktestVisualizer,
    RigorousBet,
    FoldResult,
    RigorousBacktestResult,
)

# Import existing model infrastructure
from src.betting.kelly import KellyBetSizer
from src.betting.edge_strategy import EdgeStrategy


def load_historical_data():
    """Load historical game data with features and odds."""
    print("Loading historical data...")

    # Load game features
    features_path = Path("data/features/game_features.parquet")
    if features_path.exists():
        df = pd.read_parquet(features_path)
        print(f"Loaded {len(df)} games from features")
    else:
        print("Warning: game_features.parquet not found, using raw games")
        df = pd.read_parquet("data/raw/games.parquet")

    # Load historical odds
    odds_path = Path("data/raw/historical_odds.csv")
    if odds_path.exists():
        odds_df = pd.read_csv(odds_path)
        print(f"Loaded {len(odds_df)} historical odds records")

        # Convert date columns to same type
        odds_df["date"] = pd.to_datetime(odds_df["date"])

        # Normalize team names to match features format (uppercase 3-letter codes)
        team_mapping = {
            'gs': 'GSW',     # Golden State Warriors
            'sa': 'SAS',     # San Antonio Spurs
            'no': 'NOP',     # New Orleans Pelicans
            'ny': 'NYK',     # New York Knicks
            'utah': 'UTA',   # Utah Jazz
        }

        # Apply mapping and uppercase
        odds_df['home'] = odds_df['home'].str.upper().replace(team_mapping)
        odds_df['away'] = odds_df['away'].str.upper().replace(team_mapping)

        # Merge odds with features (include moneyline odds for market probability)
        df = df.merge(
            odds_df[["date", "home", "away", "spread", "total", "moneyline_home", "moneyline_away"]],
            left_on=["date", "home_team", "away_team"],
            right_on=["date", "home", "away"],
            how="left",
        )
        df = df.drop(["home", "away"], axis=1, errors="ignore")
        print(f"Merged spread data - games with spreads: {df['spread'].notna().sum()} ({df['spread'].notna().mean():.1%})")

    # Filter to test period (2020-2024)
    if "season" in df.columns:
        df = df[df["season"].between(2020, 2024)]
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"].dt.year.between(2020, 2024)]

    print(f"Filtered to {len(df)} games (2020-2024)")

    return df


def run_rigorous_backtest():
    """Run complete rigorous backtest pipeline."""

    print("=" * 60)
    print("RIGOROUS BACKTESTING FRAMEWORK")
    print("=" * 60)
    print()

    # Configuration
    config = BacktestConfig(
        # Walk-forward settings
        initial_train_size=500,
        retrain_frequency="monthly",
        use_expanding_window=True,

        # Betting parameters (EdgeStrategy validated thresholds)
        kelly_fraction=0.2,
        max_bet_fraction=0.05,
        min_edge_threshold=4.0,  # 4 points edge (≈10% prob advantage, realistic ATS threshold)

        # Initial bankroll
        initial_bankroll=10000.0,

        # Monte Carlo
        n_simulations=1000,
        bootstrap_samples=5000,
        random_seed=42,

        # Transaction costs
        base_vig=0.045,
        slippage_model="sqrt",
        execution_probability=0.95,

        # Constraints
        max_bet_per_book=500.0,
        max_daily_exposure=0.15,
        max_per_game_exposure=0.05,

        # Statistics
        confidence_level=0.95,
        multiple_testing_method="fdr_bh",
    )

    # Market efficiency parameter (not in config, pass separately)
    market_efficiency = 0.85  # Market knows 85% of what our model knows

    print("Configuration:")
    print(f"  Test Period: 2020-2024")
    print(f"  Initial Bankroll: ${config.initial_bankroll:,.0f}")
    print(f"  Retrain Frequency: {config.retrain_frequency}")
    print(f"  Kelly Fraction: {config.kelly_fraction:.0%}")
    print(f"  Min Edge: {config.min_edge_threshold:.1%}")
    print(f"  Market Efficiency: {market_efficiency:.0%}")
    print(f"  Monte Carlo Sims: {config.n_simulations:,}")
    print()

    # Load data
    df = load_historical_data()

    if len(df) < config.initial_train_size:
        print(f"Error: Need at least {config.initial_train_size} games, found {len(df)}")
        return

    # Initialize components
    walk_forward = WalkForwardValidator(config)
    statistics = StatisticalAnalyzer(config)
    monte_carlo = MonteCarloSimulator(config)
    cost_model = TransactionCostModel(config)
    constraints = ConstraintManager(config)

    # Collect all bets across folds
    all_bets = []
    fold_results = []

    print("Running walk-forward backtesting...")
    print()

    # Generate and process folds
    for train_df, test_df, fold_info in walk_forward.generate_folds(df, date_col="date"):
        print(f"Fold {fold_info.fold_id}:")
        print(f"  Train: {fold_info.train_start.date()} to {fold_info.train_end.date()} ({fold_info.train_size} games)")
        print(f"  Test:  {fold_info.test_start.date()} to {fold_info.test_end.date()} ({fold_info.test_size} games)")

        # Train model on train data and generate predictions on test data
        fold_bets = generate_model_bets(
            train_df, test_df, config, cost_model, constraints,
            market_efficiency=market_efficiency
        )

        if len(fold_bets) > 0:
            # Calculate fold statistics
            fold_roi, fold_roi_ci = statistics.calculate_roi_ci(fold_bets)
            fold_win_rate, fold_wr_ci = statistics.calculate_win_rate_ci(fold_bets)
            fold_sharpe, fold_sharpe_ci = statistics.calculate_sharpe_ci(fold_bets)

            fold_result = FoldResult(
                fold_id=fold_info.fold_id,
                train_start=fold_info.train_start,
                train_end=fold_info.train_end,
                test_start=fold_info.test_start,
                test_end=fold_info.test_end,
                roi=fold_roi,
                win_rate=fold_win_rate,
                sharpe_ratio=fold_sharpe,
                max_drawdown=calculate_max_drawdown(fold_bets),
                roi_ci=fold_roi_ci,
                win_rate_ci=fold_wr_ci,
                num_bets=len(fold_bets),
                total_wagered=sum(b.bet_size for b in fold_bets),
                total_pnl=sum(b.pnl for b in fold_bets),
                bets=fold_bets,
            )

            fold_results.append(fold_result)
            all_bets.extend(fold_bets)

            print(f"  Bets: {len(fold_bets)}, ROI: {fold_roi:.1%}, Win Rate: {fold_win_rate:.1%}")
        else:
            print(f"  No bets generated")

        print()

    if not all_bets:
        print("No bets generated - check edge threshold and data")
        return

    print(f"Total bets across all folds: {len(all_bets)}")
    print()

    # Calculate overall statistics
    print("Calculating statistics...")
    roi, roi_ci = statistics.calculate_roi_ci(all_bets)
    win_rate, wr_ci = statistics.calculate_win_rate_ci(all_bets)
    sharpe, sharpe_ci = statistics.calculate_sharpe_ci(all_bets)
    p_value = statistics.calculate_p_value(all_bets, null_roi=0.0)

    # Run Monte Carlo simulation
    print("Running Monte Carlo simulation...")
    mc_result = monte_carlo.run_bootstrap_simulation(all_bets)

    # Calculate transaction cost impact
    gross_roi = roi
    net_roi = gross_roi - config.base_vig
    total_slippage = sum(b.slippage for b in all_bets if hasattr(b, 'slippage'))
    total_vig = sum(b.vig_cost for b in all_bets if hasattr(b, 'vig_cost'))
    bets_not_executed = sum(1 for b in all_bets if hasattr(b, 'execution_failed') and b.execution_failed)

    # Create result object
    result = RigorousBacktestResult(
        config=config,
        roi=roi,
        win_rate=win_rate,
        sharpe_ratio=sharpe,
        max_drawdown=calculate_max_drawdown(all_bets),
        num_bets=len(all_bets),
        total_wagered=sum(b.bet_size for b in all_bets),
        total_pnl=sum(b.pnl for b in all_bets),
        roi_ci=roi_ci,
        win_rate_ci=wr_ci,
        sharpe_ci=sharpe_ci,
        max_drawdown_ci=(mc_result.percentiles["max_drawdown"][5] / 100, mc_result.percentiles["max_drawdown"][95] / 100),
        roi_distribution=mc_result.roi_distribution,
        sharpe_distribution=mc_result.sharpe_distribution,
        drawdown_distribution=mc_result.drawdown_distribution,
        roi_percentiles=mc_result.percentiles["roi"],
        sharpe_percentiles=mc_result.percentiles["sharpe"],
        drawdown_percentiles=mc_result.percentiles["max_drawdown"],
        p_value_vs_breakeven=p_value,
        risk_of_ruin=mc_result.risk_of_ruin,
        probability_profitable=mc_result.probability_profitable,
        fold_results=fold_results,
        gross_roi=gross_roi,
        net_roi=net_roi,
        total_slippage=total_slippage,
        total_vig_paid=total_vig,
        bets_not_executed=bets_not_executed,
        bets=all_bets,
        bankroll_history=calculate_bankroll_history(all_bets, config.initial_bankroll),
        test_seasons=[2020, 2021, 2022, 2023, 2024],
    )

    # Print summary
    print()
    print(result.summary())

    # Create visualizations
    print("Creating visualizations...")
    viz = BacktestVisualizer()

    # Create output directory
    output_dir = Path("data/backtest/rigorous")
    output_dir.mkdir(parents=True, exist_ok=True)

    viz.create_summary_dashboard(result, save_dir=output_dir)

    print(f"\nResults saved to {output_dir}")


def american_to_implied_prob(odds):
    """Convert American odds to implied probability."""
    if pd.isna(odds):
        return None
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def remove_vig(home_implied, away_implied):
    """
    Remove vig from implied probabilities to get fair market probabilities.

    Uses the multiplicative method (Shin's method approximation).
    """
    if home_implied is None or away_implied is None:
        return 0.5, 0.5

    total_implied = home_implied + away_implied
    if total_implied <= 0:
        return 0.5, 0.5

    # Normalize to remove vig
    home_fair = home_implied / total_implied
    away_fair = away_implied / total_implied

    return home_fair, away_fair


def apply_market_efficiency(model_prob, market_prob, efficiency=0.85):
    """
    Model market efficiency: market knows efficiency% of what our model knows.

    This simulates the fact that professional betting markets are highly efficient
    and incorporate most public information.

    Args:
        model_prob: Our model's probability
        market_prob: Market's fair probability (vig-removed)
        efficiency: How much market knows (0.85 = market knows 85% of model's info)

    Returns:
        Adjusted market probability that incorporates model's info
    """
    # Market moves toward model's prediction based on efficiency
    adjusted_market = (1 - efficiency) * market_prob + efficiency * model_prob
    return np.clip(adjusted_market, 0.25, 0.75)


def generate_model_bets(train_df, test_df, config, cost_model, constraints, market_efficiency=0.85):
    """
    Generate bets using EdgeStrategy approach with rigorous backtesting.

    1. Train XGBoost REGRESSOR to predict point differential (ATS)
    2. Calculate edge = |pred_diff - market_spread|
    3. Apply EdgeStrategy filters (6% edge, no B2B, team exclusions)
    4. Apply Kelly sizing
    5. Apply constraints
    6. Simulate outcomes based on actual game results

    This tests the ACTUAL live strategy (EdgeStrategy) not a hypothetical ML approach.

    Args:
        market_efficiency: Not used for EdgeStrategy (edge calculated from spread directly)
    """
    from xgboost import XGBRegressor

    bets = []
    bankroll = config.initial_bankroll

    # EdgeStrategy team exclusions (validated from live performance)
    TEAMS_TO_EXCLUDE = {"CHA", "IND", "MIA", "NOP", "PHX"}

    # Define feature columns (exclude metadata and targets)
    exclude_cols = {
        'game_id', 'date', 'season', 'home_team', 'away_team',
        'home_win', 'away_win', 'home_score', 'away_score',
        'point_diff', 'total_points', 'spread', 'total',
        'home_is_home', 'away_is_home',  # Redundant features
        'home_point_diff', 'away_point_diff',  # Target leakage
        'moneyline_home', 'moneyline_away',  # Not used for ATS prediction
    }

    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    target_col = 'point_diff'  # Predict point differential (home - away)

    # Check if we have enough data and required columns
    if len(train_df) < 100 or target_col not in train_df.columns:
        print(f"  Insufficient training data or missing target column")
        return []

    # Prepare training data
    X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
    y_train = train_df[target_col]

    # Train REGRESSION model to predict point differential (not win probability)
    try:
        model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"  Model training failed: {e}")
        return []

    # Initialize Kelly sizer
    kelly = KellyBetSizer(
        fraction=config.kelly_fraction,
        max_bet_pct=config.max_bet_fraction,
    )

    # Generate predictions and bets using EdgeStrategy approach
    for _, row in test_df.iterrows():
        # Get model prediction for point differential
        X_test = pd.DataFrame([row[feature_cols]]).fillna(train_df[feature_cols].median())
        try:
            pred_diff = model.predict(X_test)[0]  # Predicted margin (home - away)
        except Exception:
            continue

        # Get market spread (negative means home favored)
        market_spread = row.get('spread')
        if pd.isna(market_spread):
            continue  # Skip if no spread available

        # EdgeStrategy: Calculate edge as difference between prediction and market
        # If pred_diff = +5 and spread = -3, we think home wins by 5 but market has -3
        # Edge for home = pred_diff - spread = 5 - (-3) = 8 points
        # If pred_diff = -5 and spread = +3, we think away wins by 5 but market has +3
        # Edge for away = abs(pred_diff - spread) when pred_diff < spread

        edge_points = pred_diff - market_spread

        # Determine bet side based on edge
        if edge_points >= config.min_edge_threshold:
            # Model predicts home to beat the spread
            bet_side = "HOME"
            edge = edge_points
        elif edge_points <= -config.min_edge_threshold:
            # Model predicts away to beat the spread
            bet_side = "AWAY"
            edge = abs(edge_points)
        else:
            continue  # No sufficient edge

        # Apply EdgeStrategy filters
        home_team = row.get('home_team', '')
        away_team = row.get('away_team', '')
        home_b2b = row.get('home_is_b2b', False) or row.get('home_b2b', False)
        away_b2b = row.get('away_is_b2b', False) or row.get('away_b2b', False)

        # Filter 1: No back-to-back games
        if bet_side == "HOME" and home_b2b:
            continue
        if bet_side == "AWAY" and away_b2b:
            continue

        # Filter 2: Team exclusions (poor ATS performers)
        if bet_side == "HOME" and home_team in TEAMS_TO_EXCLUDE:
            continue
        if bet_side == "AWAY" and away_team in TEAMS_TO_EXCLUDE:
            continue

        # Calculate Kelly bet size
        # Convert point edge to probability (rough: 1 point ≈ 2.5% probability)
        # Base win rate for ATS at -110 is 52.4% (break-even)
        # Add probability advantage from our edge
        prob_advantage = edge * 0.025  # 2.5% per point of edge
        win_prob = 0.524 + prob_advantage  # Break-even + advantage
        win_prob = np.clip(win_prob, 0.3, 0.95)  # Reasonable bounds

        bet_size = kelly.calculate_bet_size(
            bankroll=bankroll,
            win_prob=win_prob,
            odds=1.909,  # -110 American odds in decimal
            odds_format="decimal",
        )

        if bet_size < 1.0:
            continue

        # Create bet (EdgeStrategy ATS bet)
        bet = RigorousBet(
            date=row.get("date", datetime.now()),
            game_id=row.get("game_id", f"game_{len(bets)}"),
            bet_side=bet_side,
            bet_type="SPREAD",
            bet_size=bet_size,
            odds=1.909,  # -110 odds standard
            line=float(market_spread),  # The actual spread we're betting against
            model_prob=win_prob,  # Estimated win probability from edge
            market_prob=0.524,  # Break-even probability at -110
            edge=edge,  # Edge in points
            bankroll_before=bankroll,
        )

        # Apply transaction costs
        bet, costs = cost_model.apply_transaction_costs(bet)

        if costs.execution_failed:
            continue

        # Determine outcome from actual ATS result
        if 'point_diff' in row and pd.isna(row['point_diff']):
            # Skip if no actual result
            continue

        actual_point_diff = row['point_diff']  # Actual margin (home - away)

        # Check if bet covered the spread
        # For HOME bet: need actual_point_diff > spread (home beats spread)
        # For AWAY bet: need actual_point_diff < spread (away beats spread)
        if bet_side == "HOME":
            bet.won = actual_point_diff > market_spread
        else:  # AWAY
            bet.won = actual_point_diff < market_spread

        # Calculate P&L
        bet.pnl = bet.bet_size * (bet.odds - 1) if bet.won else -bet.bet_size
        bankroll += bet.pnl
        bet.bankroll_after = bankroll

        bets.append(bet)

    return bets


def calculate_max_drawdown(bets):
    """Calculate maximum drawdown from bets."""
    if not bets:
        return 0.0

    peak = bets[0].bankroll_before
    max_dd = 0.0

    for bet in bets:
        current = bet.bankroll_after if hasattr(bet, 'bankroll_after') else bet.bankroll_before
        if current > peak:
            peak = current
        dd = (peak - current) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return max_dd


def calculate_bankroll_history(bets, initial_bankroll):
    """Calculate bankroll evolution."""
    if not bets:
        return np.array([initial_bankroll])

    history = [initial_bankroll]
    for bet in bets:
        if hasattr(bet, 'bankroll_after'):
            history.append(bet.bankroll_after)
        else:
            history.append(history[-1] + bet.pnl)

    return np.array(history)


if __name__ == "__main__":
    run_rigorous_backtest()
