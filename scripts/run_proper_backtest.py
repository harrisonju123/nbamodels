"""
Proper Walk-Forward Backtest with Fresh Feature Calculation

This script fixes the data leakage issue by calculating features FRESH
for each fold, using only data available at that point in time.

Key differences from run_rigorous_backtest.py:
- Loads raw games (not pre-computed features)
- Calculates features fresh for each fold
- Mirrors the live trading pipeline approach

Usage:
    python scripts/run_proper_backtest.py
"""

import sys
from pathlib import Path
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

# Import feature builders
from src.features import GameFeatureBuilder, TeamFeatureBuilder

# Import existing model infrastructure
from src.betting.kelly import KellyBetSizer


def load_raw_data():
    """Load raw games and odds data."""
    print("Loading raw game data...")

    # Load raw games (basic stats only, no pre-computed features)
    games_path = Path("data/raw/games.parquet")
    games_df = pd.read_parquet(games_path)
    print(f"Loaded {len(games_df)} raw games")

    # Load historical odds
    odds_path = Path("data/raw/historical_odds.csv")
    odds_df = pd.read_csv(odds_path)
    print(f"Loaded {len(odds_df)} historical odds records")

    # Convert dates
    games_df["date"] = pd.to_datetime(games_df["date"])
    odds_df["date"] = pd.to_datetime(odds_df["date"])

    # Normalize team names in odds to match games format
    team_mapping = {
        'gs': 'GSW',
        'sa': 'SAS',
        'no': 'NOP',
        'ny': 'NYK',
        'utah': 'UTA',
    }
    odds_df['home'] = odds_df['home'].str.upper().replace(team_mapping)
    odds_df['away'] = odds_df['away'].str.upper().replace(team_mapping)

    return games_df, odds_df


def build_features_for_fold(train_games, test_games, odds_df):
    """
    Build features fresh for a single fold.

    This mirrors the live trading pipeline approach:
    1. Calculate team features from training games only
    2. Get latest features for each team
    3. Build feature rows for test games

    Args:
        train_games: Historical games for training period
        test_games: Games in test period
        odds_df: All historical odds

    Returns:
        train_features_df, test_features_df with full feature sets
    """
    # Initialize feature builder
    team_builder = TeamFeatureBuilder()

    # Calculate team features from training games ONLY
    # This ensures we only use information available at training time
    team_features = team_builder.build_all_features(train_games)

    # Get latest features for each team (as of end of training period)
    latest_features = (
        team_features
        .sort_values("date")
        .groupby("team")
        .last()
        .reset_index()
    )

    # Build feature rows for training games
    train_feature_rows = []
    for _, game in train_games.iterrows():
        home_stats = latest_features[latest_features["team"] == game["home_team"]]
        away_stats = latest_features[latest_features["team"] == game["away_team"]]

        if home_stats.empty or away_stats.empty:
            continue

        # Get odds for this game
        game_odds = odds_df[
            (odds_df["date"] == game["date"]) &
            (odds_df["home"] == game["home_team"]) &
            (odds_df["away"] == game["away_team"])
        ]

        # Build feature row
        row = {
            "game_id": game["game_id"],
            "date": game["date"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "point_diff": game["point_diff"],  # Target variable
        }

        # Add home team features (prefix with home_)
        for col in home_stats.columns:
            if col not in ["team", "game_id", "date", "season", "opponent"]:
                row[f"home_{col}"] = home_stats[col].values[0]

        # Add away team features (prefix with away_)
        for col in away_stats.columns:
            if col not in ["team", "game_id", "date", "season", "opponent"]:
                row[f"away_{col}"] = away_stats[col].values[0]

        # Add odds if available
        if not game_odds.empty:
            row["spread"] = game_odds["spread"].values[0]
            row["total"] = game_odds["total"].values[0]
            row["moneyline_home"] = game_odds.get("moneyline_home", pd.Series([np.nan])).values[0]
            row["moneyline_away"] = game_odds.get("moneyline_away", pd.Series([np.nan])).values[0]

        train_feature_rows.append(row)

    # Build feature rows for test games (same process)
    test_feature_rows = []
    for _, game in test_games.iterrows():
        home_stats = latest_features[latest_features["team"] == game["home_team"]]
        away_stats = latest_features[latest_features["team"] == game["away_team"]]

        if home_stats.empty or away_stats.empty:
            continue

        # Get odds for this game
        game_odds = odds_df[
            (odds_df["date"] == game["date"]) &
            (odds_df["home"] == game["home_team"]) &
            (odds_df["away"] == game["away_team"])
        ]

        # Build feature row
        row = {
            "game_id": game["game_id"],
            "date": game["date"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "point_diff": game["point_diff"],  # Actual result (for evaluation)
        }

        # Add home team features
        for col in home_stats.columns:
            if col not in ["team", "game_id", "date", "season", "opponent"]:
                row[f"home_{col}"] = home_stats[col].values[0]

        # Add away team features
        for col in away_stats.columns:
            if col not in ["team", "game_id", "date", "season", "opponent"]:
                row[f"away_{col}"] = away_stats[col].values[0]

        # Add odds if available
        if not game_odds.empty:
            row["spread"] = game_odds["spread"].values[0]
            row["total"] = game_odds["total"].values[0]
            row["moneyline_home"] = game_odds.get("moneyline_home", pd.Series([np.nan])).values[0]
            row["moneyline_away"] = game_odds.get("moneyline_away", pd.Series([np.nan])).values[0]

        test_feature_rows.append(row)

    train_df = pd.DataFrame(train_feature_rows)
    test_df = pd.DataFrame(test_feature_rows)

    return train_df, test_df


def generate_model_bets(train_df, test_df, config, cost_model, constraints):
    """
    Generate bets using EdgeStrategy approach.

    Identical to run_rigorous_backtest.py but works with freshly calculated features.
    """
    from xgboost import XGBRegressor

    bets = []
    bankroll = config.initial_bankroll

    # EdgeStrategy team exclusions
    TEAMS_TO_EXCLUDE = {"CHA", "IND", "MIA", "NOP", "PHX"}

    # Define feature columns (exclude metadata and targets)
    exclude_cols = {
        'game_id', 'date', 'season', 'home_team', 'away_team',
        'home_win', 'away_win', 'home_score', 'away_score',
        'point_diff', 'total_points', 'spread', 'total',
        'home_is_home', 'away_is_home',
        'home_point_diff', 'away_point_diff',
        'moneyline_home', 'moneyline_away',
    }

    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    target_col = 'point_diff'

    # Check if we have enough data
    if len(train_df) < 100 or target_col not in train_df.columns:
        return []

    # Prepare training data
    X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
    y_train = train_df[target_col]

    # Train regression model
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

    # Generate predictions and bets
    for _, row in test_df.iterrows():
        # Get model prediction
        X_test = pd.DataFrame([row[feature_cols]]).fillna(train_df[feature_cols].median())
        try:
            pred_diff = model.predict(X_test)[0]
        except Exception:
            continue

        # Get market spread
        market_spread = row.get('spread')
        if pd.isna(market_spread):
            continue

        # Calculate edge
        edge_points = pred_diff - market_spread

        # Determine bet side
        if edge_points >= config.min_edge_threshold:
            bet_side = "HOME"
            edge = edge_points
            team = row["home_team"]
        elif edge_points <= -config.min_edge_threshold:
            bet_side = "AWAY"
            edge = abs(edge_points)
            team = row["away_team"]
        else:
            continue

        # Apply team exclusions
        if team in TEAMS_TO_EXCLUDE:
            continue

        # Calculate win probability from edge
        # Rough approximation: 1 point edge ≈ 4% probability advantage
        prob_advantage = min(edge / 25.0, 0.25)  # Cap at 25% advantage
        win_prob = 0.525 + prob_advantage  # Base 52.5% for a legitimate edge
        win_prob = np.clip(win_prob, 0.3, 0.9)

        # Calculate bet size using Kelly
        odds = -110  # Standard ATS odds
        bet_size = kelly.calculate_bet_size(
            win_prob=win_prob,
            odds=odds,
            bankroll=bankroll,
        )

        # Apply constraints - simple minimum bet filter
        if bet_size < 1.0:
            continue

        # Cap at maximum bet
        bet_size = min(bet_size, config.max_bet_per_book)

        # Simulate outcome
        actual_diff = row["point_diff"]
        if bet_side == "HOME":
            covered = actual_diff > market_spread
        else:  # AWAY
            covered = actual_diff < market_spread

        # Calculate P&L
        if covered:
            pnl = bet_size * (100 / 110)  # Win at -110 odds
            outcome = "win"
        else:
            pnl = -bet_size
            outcome = "loss"

        # Record bet
        bet = RigorousBet(
            date=row["date"],
            game_id=row["game_id"],
            bet_side=bet_side,
            bet_type="SPREAD",
            bet_size=bet_size,
            odds=1.909,  # -110 in decimal
            line=float(market_spread),
            model_prob=win_prob,
            market_prob=0.525,  # Implied prob at -110
            edge=edge,
            won=covered,
            pnl=pnl,
            bankroll_before=bankroll,
        )

        bets.append(bet)
        bankroll += pnl
        bet.bankroll_after = bankroll

    return bets


def calculate_max_drawdown(bets):
    """Calculate maximum drawdown from bet sequence."""
    if not bets:
        return 0.0

    cumulative = np.cumsum([b.pnl for b in bets])
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (running_max - cumulative) / (running_max + 1e-10)

    return drawdown.max()


def run_proper_backtest():
    """Run walk-forward backtest with fresh feature calculation."""

    print("=" * 60)
    print("PROPER WALK-FORWARD BACKTEST (NO DATA LEAKAGE)")
    print("=" * 60)
    print()

    # Configuration
    config = BacktestConfig(
        initial_train_size=500,
        retrain_frequency="monthly",
        use_expanding_window=True,
        kelly_fraction=0.2,
        max_bet_fraction=0.05,
        min_edge_threshold=4.0,  # 4 points edge
        initial_bankroll=10000.0,
        n_simulations=1000,
        bootstrap_samples=5000,
        random_seed=42,
        base_vig=0.045,
        slippage_model="sqrt",
        execution_probability=0.95,
        max_bet_per_book=500.0,
        max_daily_exposure=0.15,
        max_per_game_exposure=0.05,
        confidence_level=0.95,
        multiple_testing_method="fdr_bh",
    )

    print("Configuration:")
    print(f"  Test Period: 2020-2024")
    print(f"  Initial Bankroll: ${config.initial_bankroll:,.0f}")
    print(f"  Retrain Frequency: {config.retrain_frequency}")
    print(f"  Kelly Fraction: {config.kelly_fraction:.0%}")
    print(f"  Min Edge: {config.min_edge_threshold:.1f} points")
    print(f"  Fresh Features: YES (calculated per fold)")
    print()

    # Load raw data
    games_df, odds_df = load_raw_data()

    # Filter to 2020-2024
    games_df = games_df[games_df["date"].dt.year.between(2020, 2024)]
    print(f"Filtered to {len(games_df)} games (2020-2024)")
    print()

    # Initialize components
    walk_forward = WalkForwardValidator(config)
    statistics = StatisticalAnalyzer(config)
    monte_carlo = MonteCarloSimulator(config)
    cost_model = TransactionCostModel(config)
    constraints = ConstraintManager(config)

    # Collect all bets
    all_bets = []
    fold_results = []

    print("Running walk-forward backtesting with fresh feature calculation...")
    print()

    # Generate folds
    for train_games, test_games, fold_info in walk_forward.generate_folds(games_df, date_col="date"):
        print(f"Fold {fold_info.fold_id}:")
        print(f"  Train: {fold_info.train_start.date()} to {fold_info.train_end.date()} ({fold_info.train_size} games)")
        print(f"  Test:  {fold_info.test_start.date()} to {fold_info.test_end.date()} ({fold_info.test_size} games)")

        # Calculate features FRESH for this fold
        print(f"  Calculating features fresh from {fold_info.train_size} training games...")
        train_df, test_df = build_features_for_fold(train_games, test_games, odds_df)

        print(f"  Train features: {len(train_df)} games")
        print(f"  Test features: {len(test_df)} games with spreads")

        # Generate bets
        fold_bets = generate_model_bets(train_df, test_df, config, cost_model, constraints)

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

    # Create result object
    result = RigorousBacktestResult(
        config=config,
        roi=roi,
        win_rate=win_rate,
        sharpe_ratio=sharpe,
        max_drawdown=calculate_max_drawdown(all_bets),
        roi_ci=roi_ci,
        win_rate_ci=wr_ci,
        sharpe_ci=sharpe_ci,
        num_bets=len(all_bets),
        total_wagered=sum(b.bet_size for b in all_bets),
        total_pnl=sum(b.pnl for b in all_bets),
        p_value=p_value,
        probability_profitable=mc_result.probability_profitable,
        risk_of_ruin=mc_result.risk_of_ruin,
        gross_roi=gross_roi,
        net_roi=net_roi,
        total_slippage=total_slippage,
        total_vig=total_vig,
        fold_results=fold_results,
        all_bets=all_bets,
    )

    # Print results
    print("\n")
    print("=" * 60)
    print("=== PROPER BACKTEST RESULTS (NO DATA LEAKAGE) ===")
    print("=" * 60)
    print()
    print(f"ROI: {result.roi:.2%} (95% CI: {result.roi_ci[0]:.2%}, {result.roi_ci[1]:.2%})")
    print(f"Win Rate: {result.win_rate:.1%} (95% CI: {result.win_rate_ci[0]:.1%}, {result.win_rate_ci[1]:.1%})")
    print(f"Sharpe: {result.sharpe_ratio:.2f} (95% CI: {result.sharpe_ci[0]:.2f}, {result.sharpe_ci[1]:.2f})")
    print(f"Max Drawdown: {result.max_drawdown:.1%}")
    print()
    print(f"Number of Bets: {result.num_bets}")
    print(f"Total Wagered: ${result.total_wagered:,.2f}")
    print(f"Total P&L: ${result.total_pnl:,.2f}")
    print()
    print(f"Statistical Significance:")
    print(f"  P-value vs break-even: {result.p_value:.4f}")
    print()
    print(f"Risk Metrics:")
    print(f"  Risk of Ruin (50%+ loss): {result.risk_of_ruin:.1%}")
    print(f"  Probability Profitable: {result.probability_profitable:.1%}")
    print()
    print(f"Transaction Costs:")
    print(f"  Gross ROI: {result.gross_roi:.2%}")
    print(f"  Net ROI: {result.net_roi:.2%}")
    print()
    print(f"Walk-Forward Folds: {len(result.fold_results)}")
    print()

    # Save results
    print("Creating visualizations...")
    viz = BacktestVisualizer()
    output_dir = Path("data/backtest/proper")
    output_dir.mkdir(parents=True, exist_ok=True)
    viz.create_summary_dashboard(result, save_dir=output_dir)
    print(f"Dashboard saved to {output_dir}")


if __name__ == "__main__":
    run_proper_backtest()
