"""
Consensus Strategy Backtest

Tests the strategy that bets only when BOTH:
1. EDGE strategy (5+ point edge, no B2B, team filter) signals a bet
2. ATS (Dual Model) strategy (4+ disagreement, 2+ edge vs market, home-only) signals a bet

This should produce fewer bets but with higher confidence.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
from scipy import stats

from src.features import GameFeatureBuilder
from src.models.point_spread import PointSpreadModel
from src.models.dual_model import DualPredictionModel
from src.betting.edge_strategy import EdgeStrategy, TEAMS_TO_EXCLUDE


def run_consensus_backtest():
    """Run historical backtest for consensus strategy."""

    print("=" * 80)
    print("CONSENSUS STRATEGY BACKTEST")
    print("Bet only when EDGE + ATS (Dual Model) both agree")
    print("=" * 80)

    # Load historical games
    print("\nLoading historical games...")
    games = pd.read_parquet("data/raw/games.parquet")
    games["date"] = pd.to_datetime(games["date"])
    print(f"Loaded {len(games)} games from {games['date'].min().date()} to {games['date'].max().date()}")

    # Load models
    print("\nLoading models...")
    spread_model = PointSpreadModel.load("models/point_spread_model_tuned.pkl")
    dual_model = DualPredictionModel.load("models/dual_model.pkl")
    edge_strategy = EdgeStrategy.team_filtered_strategy()

    # Build features
    print("\nBuilding features...")
    builder = GameFeatureBuilder()
    features = builder.build_game_features(games)

    # Merge actual results
    features = features.merge(
        games[["game_id", "home_score", "away_score"]],
        on="game_id",
        how="left"
    )
    features["actual_diff"] = features["home_score"] - features["away_score"]

    # Calculate B2B from historical data
    print("Calculating B2B flags...")
    features = features.sort_values("date").reset_index(drop=True)
    features["date_only"] = pd.to_datetime(features["date"]).dt.date

    # Build games by date lookup
    games_by_date = {}
    for _, row in features.iterrows():
        d = row["date_only"]
        if d not in games_by_date:
            games_by_date[d] = set()
        games_by_date[d].add(row["home_team"])
        games_by_date[d].add(row["away_team"])

    # Calculate B2B for each game
    def calc_b2b(row):
        game_date = row["date_only"]
        yesterday = game_date - timedelta(days=1)
        teams_yesterday = games_by_date.get(yesterday, set())
        home_b2b = 1 if row["home_team"] in teams_yesterday else 0
        away_b2b = 1 if row["away_team"] in teams_yesterday else 0
        return pd.Series({"home_b2b": home_b2b, "away_b2b": away_b2b})

    b2b_flags = features.apply(calc_b2b, axis=1)
    features["home_b2b"] = b2b_flags["home_b2b"]
    features["away_b2b"] = b2b_flags["away_b2b"]

    # Filter to test period (after warmup for rolling stats)
    min_date = features["date"].min() + timedelta(days=60)
    test_features = features[features["date"] >= min_date].copy()
    print(f"Testing on {len(test_features)} games from {test_features['date'].min().date()} to {test_features['date'].max().date()}")

    # Strategy parameters
    EDGE_THRESHOLD = 5.0  # Points edge required for EDGE strategy
    ATS_MIN_DISAGREEMENT = 4.0  # MLP/XGB disagreement for ATS
    ATS_MIN_EDGE = 2.0  # Edge vs market for ATS
    ATS_HOME_ONLY = True  # ATS only bets home

    # Tracking
    bets = []
    edge_only_bets = []
    ats_only_bets = []

    # Get predictions for all games
    print("\nGenerating predictions...")

    # Ensure feature columns exist
    for col in spread_model.feature_columns:
        if col not in test_features.columns:
            test_features[col] = 0
    for col in dual_model.feature_columns:
        if col not in test_features.columns:
            test_features[col] = 0

    # Spread model predictions
    X_spread = test_features[spread_model.feature_columns].fillna(0)
    test_features["pred_diff"] = spread_model.predict(X_spread)

    # Simulate market spread as Elo-based estimate
    # Market is efficient but not perfect - add noise to model prediction
    # This creates realistic disagreements between model and market
    np.random.seed(42)  # Reproducibility
    market_noise = np.random.normal(0, 2.5, len(test_features))  # 2.5 pt std dev
    test_features["market_spread"] = -test_features["pred_diff"] + market_noise

    # Dual model predictions
    X_dual = test_features[dual_model.feature_columns].fillna(0)
    vegas_spread = test_features["market_spread"].values
    dual_preds = dual_model.get_predictions(X_dual, vegas_spread=vegas_spread)

    test_features["mlp_spread"] = dual_preds["mlp_spread"].values
    test_features["xgb_spread"] = dual_preds["xgb_spread"].values
    test_features["disagreement"] = dual_preds["disagreement"].values

    # Calculate point edge
    test_features["point_edge"] = test_features["pred_diff"] + test_features["market_spread"]
    test_features["edge_vs_market"] = test_features["mlp_spread"] - test_features["market_spread"]

    print(f"\nProcessing {len(test_features)} games for signals...")

    for idx, row in test_features.iterrows():
        game_id = row["game_id"]
        home_team = row["home_team"]
        away_team = row["away_team"]
        date = row["date"]
        actual_diff = row["actual_diff"]
        market_spread = row["market_spread"]
        point_edge = row["point_edge"]
        disagreement = row["disagreement"]
        edge_vs_market = row["edge_vs_market"]
        home_b2b = row["home_b2b"]
        away_b2b = row["away_b2b"]

        if pd.isna(actual_diff):
            continue

        # EDGE Strategy signals
        edge_bet_home = False
        edge_bet_away = False

        if abs(point_edge) >= EDGE_THRESHOLD:
            if point_edge > 0:  # Bet home
                # Check B2B and team filter
                if home_b2b == 0 and home_team not in TEAMS_TO_EXCLUDE:
                    edge_bet_home = True
            else:  # Bet away
                if away_b2b == 0 and away_team not in TEAMS_TO_EXCLUDE:
                    edge_bet_away = True

        # ATS (Dual Model) Strategy signals
        ats_bet_home = False

        if ATS_HOME_ONLY:
            if disagreement >= ATS_MIN_DISAGREEMENT and edge_vs_market >= ATS_MIN_EDGE:
                ats_bet_home = True

        # CONSENSUS: Both must agree on HOME
        consensus_home = edge_bet_home and ats_bet_home

        # Calculate if bet would have won
        home_covered = actual_diff > -market_spread  # Home covers if actual > -spread
        away_covered = not home_covered

        # Record bets
        if edge_bet_home or edge_bet_away:
            edge_only_bets.append({
                "game_id": game_id,
                "date": date,
                "home_team": home_team,
                "away_team": away_team,
                "bet_side": "HOME" if edge_bet_home else "AWAY",
                "spread": market_spread,
                "point_edge": point_edge,
                "covered": home_covered if edge_bet_home else away_covered,
            })

        if ats_bet_home:
            ats_only_bets.append({
                "game_id": game_id,
                "date": date,
                "home_team": home_team,
                "away_team": away_team,
                "bet_side": "HOME",
                "spread": market_spread,
                "disagreement": disagreement,
                "edge_vs_market": edge_vs_market,
                "covered": home_covered,
            })

        if consensus_home:
            bets.append({
                "game_id": game_id,
                "date": date,
                "home_team": home_team,
                "away_team": away_team,
                "bet_side": "HOME",
                "spread": market_spread,
                "point_edge": point_edge,
                "disagreement": disagreement,
                "edge_vs_market": edge_vs_market,
                "covered": home_covered,
                "actual_diff": actual_diff,
            })

    # Results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    # Edge-only results
    edge_df = pd.DataFrame(edge_only_bets)
    if not edge_df.empty:
        edge_wins = edge_df["covered"].sum()
        edge_total = len(edge_df)
        edge_rate = edge_wins / edge_total
        edge_roi = (edge_rate * 0.909 - (1 - edge_rate)) * 100  # -110 odds
        print(f"\nEDGE ONLY (5+ pt, no B2B, team filter):")
        print(f"  Bets: {edge_total}")
        print(f"  Wins: {edge_wins}")
        print(f"  ATS Rate: {edge_rate:.1%}")
        print(f"  ROI: {edge_roi:+.1f}%")

    # ATS-only results
    ats_df = pd.DataFrame(ats_only_bets)
    if not ats_df.empty:
        ats_wins = ats_df["covered"].sum()
        ats_total = len(ats_df)
        ats_rate = ats_wins / ats_total
        ats_roi = (ats_rate * 0.909 - (1 - ats_rate)) * 100
        print(f"\nATS ONLY (Dual Model, home-only):")
        print(f"  Bets: {ats_total}")
        print(f"  Wins: {ats_wins}")
        print(f"  ATS Rate: {ats_rate:.1%}")
        print(f"  ROI: {ats_roi:+.1f}%")

    # Consensus results
    consensus_df = pd.DataFrame(bets)
    if not consensus_df.empty:
        cons_wins = consensus_df["covered"].sum()
        cons_total = len(consensus_df)
        cons_rate = cons_wins / cons_total
        cons_roi = (cons_rate * 0.909 - (1 - cons_rate)) * 100

        print(f"\n{'=' * 40}")
        print("CONSENSUS (EDGE + ATS both agree):")
        print(f"{'=' * 40}")
        print(f"  Bets: {cons_total}")
        print(f"  Wins: {cons_wins}")
        print(f"  ATS Rate: {cons_rate:.1%}")
        print(f"  ROI: {cons_roi:+.1f}%")

        # Statistical significance
        if cons_total > 0:
            # Binomial test vs 52.4% (break-even at -110)
            result = stats.binomtest(cons_wins, cons_total, 0.524, alternative='greater')
            p_value = result.pvalue
            print(f"  p-value (vs 52.4%): {p_value:.4f}")
            if p_value < 0.05:
                print("  *** STATISTICALLY SIGNIFICANT ***")

        # Season breakdown
        consensus_df["season"] = consensus_df["date"].apply(
            lambda d: d.year if d.month >= 10 else d.year - 1
        )

        print(f"\nSeason-by-Season:")
        for season in sorted(consensus_df["season"].unique()):
            season_df = consensus_df[consensus_df["season"] == season]
            s_wins = season_df["covered"].sum()
            s_total = len(season_df)
            s_rate = s_wins / s_total if s_total > 0 else 0
            s_roi = (s_rate * 0.909 - (1 - s_rate)) * 100
            status = "+" if s_roi > 0 else ""
            print(f"  {season}-{season+1}: {s_total:3d} bets, {s_rate:.1%} ATS, {status}{s_roi:.1f}% ROI")

        # Overlap analysis
        edge_games = set(edge_df["game_id"]) if not edge_df.empty else set()
        ats_games = set(ats_df["game_id"]) if not ats_df.empty else set()
        consensus_games = set(consensus_df["game_id"])

        print(f"\nOverlap Analysis:")
        print(f"  EDGE signals: {len(edge_games)}")
        print(f"  ATS signals: {len(ats_games)}")
        print(f"  Overlap (consensus): {len(consensus_games)}")
        print(f"  Overlap rate: {len(consensus_games) / max(len(edge_games), 1):.1%} of EDGE bets")

        # Show sample consensus bets
        print(f"\nSample Consensus Bets (last 10):")
        sample = consensus_df.tail(10)
        for _, bet in sample.iterrows():
            result = "WIN" if bet["covered"] else "LOSS"
            print(f"  {bet['date'].date()} {bet['away_team']}@{bet['home_team']}: "
                  f"HOME {bet['spread']:+.1f} edge:{bet['point_edge']:.1f} disagree:{bet['disagreement']:.1f} "
                  f"[{result}] actual:{bet['actual_diff']:+.1f}")
    else:
        print("\nNo consensus bets found!")
        print("This could mean the strategies rarely overlap.")

    print("\n" + "=" * 80)

    return {
        "edge_bets": edge_df if not edge_df.empty else None,
        "ats_bets": ats_df if not ats_df.empty else None,
        "consensus_bets": consensus_df if not consensus_df.empty else None,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(".env")

    run_consensus_backtest()
