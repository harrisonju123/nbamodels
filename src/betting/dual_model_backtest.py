"""
Dual Model ATS Backtest with Kelly Criterion

Combines the MLP/XGBoost dual model (good for ATS edge finding)
with Kelly criterion bet sizing for realistic backtesting.

Key insight: When MLP disagrees with XGBoost significantly,
there's often ATS value because MLP captures compound advantages.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from loguru import logger

from ..models.dual_model import DualPredictionModel
from .kelly import KellyBetSizer


@dataclass
class ATSBet:
    """Single ATS bet record."""
    game_id: str
    date: pd.Timestamp
    home_team: str
    away_team: str
    bet_side: str  # 'HOME' or 'AWAY'
    spread: float  # Betting spread (negative = home favored)
    mlp_spread: float
    xgb_spread: float
    disagreement: float
    confidence: float
    bet_size: float
    covered: bool  # Did the bet cover the spread?
    pnl: float
    bankroll_after: float


@dataclass
class DualModelBacktestResult:
    """Results from dual model ATS backtesting."""
    bets: List[ATSBet]
    initial_bankroll: float
    final_bankroll: float
    total_wagered: float
    total_pnl: float
    roi: float
    ats_win_rate: float
    num_bets: int
    max_drawdown: float
    sharpe_ratio: float
    kelly_fraction_used: float
    min_disagreement: float
    bankroll_history: List[float]

    # Model-specific metrics
    avg_disagreement: float
    home_bets: int
    away_bets: int
    home_ats_pct: float
    away_ats_pct: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "initial_bankroll": float(self.initial_bankroll),
            "final_bankroll": float(self.final_bankroll),
            "total_wagered": float(self.total_wagered),
            "total_pnl": float(self.total_pnl),
            "roi": float(self.roi),
            "ats_win_rate": float(self.ats_win_rate),
            "num_bets": int(self.num_bets),
            "max_drawdown": float(self.max_drawdown),
            "sharpe_ratio": float(self.sharpe_ratio),
            "kelly_fraction_used": float(self.kelly_fraction_used),
            "min_disagreement": float(self.min_disagreement),
            "avg_disagreement": float(self.avg_disagreement),
            "home_bets": int(self.home_bets),
            "away_bets": int(self.away_bets),
            "home_ats_pct": float(self.home_ats_pct),
            "away_ats_pct": float(self.away_ats_pct),
        }

    def summary(self) -> str:
        """Human-readable summary."""
        return f"""
=== Dual Model ATS Backtest Results ===
Bets Placed: {self.num_bets}
ATS Win Rate: {self.ats_win_rate:.1%}

Initial Bankroll: ${self.initial_bankroll:,.2f}
Final Bankroll: ${self.final_bankroll:,.2f}
Total P&L: ${self.total_pnl:+,.2f}

ROI: {self.roi:+.2%}
Max Drawdown: {self.max_drawdown:.1%}
Sharpe Ratio: {self.sharpe_ratio:.2f}

Bet Distribution:
  Home Bets: {self.home_bets} ({self.home_ats_pct:.1%} ATS)
  Away Bets: {self.away_bets} ({self.away_ats_pct:.1%} ATS)

Model Signals:
  Avg Disagreement: {self.avg_disagreement:.1f} pts
  Min Disagreement Threshold: {self.min_disagreement:.1f} pts

Settings:
  Kelly Fraction: {self.kelly_fraction_used}
"""


class DualModelATSBacktester:
    """
    Backtest dual model for ATS betting with Kelly sizing.

    Uses MLP/XGBoost disagreement as the primary signal.
    When disagreement is high, MLP has historically been more accurate
    for ATS predictions.
    """

    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        kelly_fraction: float = 0.15,  # Conservative for ATS
        min_disagreement: float = 5.0,  # Minimum spread disagreement in points (stricter)
        min_edge_vs_market: float = 3.0,  # Minimum edge vs Elo market proxy
        max_bet_fraction: float = 0.05,
        spread_odds: int = -110,  # Standard spread odds
        retrain_frequency: int = 100,  # Retrain every N games
        min_train_games: int = 1000,  # Minimum games before betting
        home_only: bool = False,  # Only bet on home teams (based on asymmetric performance)
    ):
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.min_disagreement = min_disagreement
        self.min_edge_vs_market = min_edge_vs_market
        self.max_bet_fraction = max_bet_fraction
        self.spread_odds = spread_odds
        self.retrain_frequency = retrain_frequency
        self.min_train_games = min_train_games
        self.home_only = home_only

        self.kelly_sizer = KellyBetSizer(
            fraction=kelly_fraction,
            max_bet_pct=max_bet_fraction,
        )

    def run_backtest(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        test_seasons: List[int],
        date_col: str = "date",
        season_col: str = "season",
    ) -> DualModelBacktestResult:
        """
        Run ATS backtest with dual model and Kelly sizing.

        Args:
            df: DataFrame with features, point_diff, and elo_prob
            feature_cols: Feature columns for model
            test_seasons: Seasons to test on
            date_col: Date column name
            season_col: Season column name

        Returns:
            DualModelBacktestResult with performance metrics
        """
        # Sort by date
        df = df.sort_values(date_col).reset_index(drop=True)

        bankroll = self.initial_bankroll
        bankroll_history = [bankroll]
        bets = []
        total_wagered = 0.0
        games_since_retrain = 0
        model = None

        # Calculate implied probability from spread odds
        if self.spread_odds < 0:
            implied_prob = abs(self.spread_odds) / (abs(self.spread_odds) + 100)
        else:
            implied_prob = 100 / (self.spread_odds + 100)

        # Decimal odds for payout calculation
        decimal_odds = 1 + (100 / abs(self.spread_odds)) if self.spread_odds < 0 else 1 + (self.spread_odds / 100)

        for i, row in df.iterrows():
            # Skip if not enough training data
            if i < self.min_train_games:
                continue

            # Retrain model if needed
            if model is None or games_since_retrain >= self.retrain_frequency:
                train_data = df.iloc[:i]

                model = DualPredictionModel()
                model.fit(
                    train_data[feature_cols],
                    train_data["home_win"],
                    dates=train_data[date_col],
                    feature_columns=feature_cols,
                )
                games_since_retrain = 0
                logger.debug(f"Retrained dual model at game {i}")

            games_since_retrain += 1

            # Only bet on test seasons
            if row.get(season_col) not in test_seasons:
                continue

            # Get model predictions
            X_pred = pd.DataFrame([row[feature_cols]])
            preds = model.predict_proba(X_pred)

            mlp_spread = preds['mlp_spread'][0]
            xgb_spread = preds['xgb_spread'][0]
            disagreement = preds['disagreement'][0]  # mlp_spread - xgb_spread

            # Get market spread (use Elo as proxy)
            if "home_elo_prob" in row.index and pd.notna(row.get("home_elo_prob")):
                elo_prob = row["home_elo_prob"]
            elif "elo_prob" in row.index and pd.notna(row.get("elo_prob")):
                elo_prob = row["elo_prob"]
            else:
                elo_prob = 0.55  # Default home advantage

            market_spread = -((elo_prob - 0.5) * 28)  # Convert prob to spread

            # Calculate edge vs market (MLP spread vs market spread)
            # Positive = MLP thinks home is better than market
            edge_vs_market = mlp_spread - market_spread

            # Determine bet based on disagreement AND edge vs market
            # Positive disagreement = MLP thinks home is better than XGB does
            # This often indicates compound advantages the market misses

            bet_side = None
            if disagreement >= self.min_disagreement and edge_vs_market >= self.min_edge_vs_market:
                # MLP sees home value AND edge vs market, bet HOME ATS
                bet_side = "HOME"
                edge_points = disagreement  # Points of disagreement
            elif not self.home_only and disagreement <= -self.min_disagreement and edge_vs_market <= -self.min_edge_vs_market:
                # MLP sees away value AND edge vs market, bet AWAY ATS (if not home_only)
                bet_side = "AWAY"
                edge_points = -disagreement  # Absolute disagreement
            else:
                continue  # No signal or filtered out

            # Convert spread disagreement to probability edge
            # ~3 points of disagreement ≈ 10% probability difference
            prob_edge = edge_points / 28  # 28 points per 100% probability

            # Adjusted win probability
            adjusted_prob = implied_prob + prob_edge
            adjusted_prob = np.clip(adjusted_prob, 0.3, 0.7)  # Reasonable bounds

            # Kelly bet sizing
            kelly = self.kelly_sizer.calculate_kelly(
                win_prob=adjusted_prob,
                odds=self.spread_odds,
                odds_format="american",
            )

            bet_size = bankroll * kelly

            # Skip tiny bets
            if bet_size < 1:
                continue

            # Determine if bet covered
            point_diff = row.get("point_diff", 0)  # Home - Away

            if bet_side == "HOME":
                # Home covers if point_diff > -spread (spread is negative when home favored)
                covered = point_diff > -market_spread
            else:
                # Away covers if point_diff < -spread
                covered = point_diff < -market_spread

            # Calculate P&L (standard -110 odds)
            if covered:
                pnl = bet_size * (decimal_odds - 1)
            else:
                pnl = -bet_size

            bankroll += pnl
            total_wagered += bet_size
            bankroll_history.append(bankroll)

            # Record bet
            bet = ATSBet(
                game_id=str(row.get("game_id", i)),
                date=pd.Timestamp(row[date_col]),
                home_team=str(row.get("home_team", "")),
                away_team=str(row.get("away_team", "")),
                bet_side=bet_side,
                spread=market_spread,
                mlp_spread=mlp_spread,
                xgb_spread=xgb_spread,
                disagreement=disagreement,
                confidence=abs(disagreement) / 10,
                bet_size=bet_size,
                covered=covered,
                pnl=pnl,
                bankroll_after=bankroll,
            )
            bets.append(bet)

        return self._calculate_metrics(bets, bankroll_history, total_wagered)

    def _calculate_metrics(
        self,
        bets: List[ATSBet],
        bankroll_history: List[float],
        total_wagered: float,
    ) -> DualModelBacktestResult:
        """Calculate backtest metrics."""
        num_bets = len(bets)

        if num_bets == 0:
            return DualModelBacktestResult(
                bets=[],
                initial_bankroll=self.initial_bankroll,
                final_bankroll=self.initial_bankroll,
                total_wagered=0,
                total_pnl=0,
                roi=0,
                ats_win_rate=0,
                num_bets=0,
                max_drawdown=0,
                sharpe_ratio=0,
                kelly_fraction_used=self.kelly_fraction,
                min_disagreement=self.min_disagreement,
                bankroll_history=bankroll_history,
                avg_disagreement=0,
                home_bets=0,
                away_bets=0,
                home_ats_pct=0,
                away_ats_pct=0,
            )

        wins = sum(1 for b in bets if b.covered)
        final_bankroll = bankroll_history[-1]
        total_pnl = final_bankroll - self.initial_bankroll
        roi = total_pnl / total_wagered if total_wagered > 0 else 0

        # Max drawdown
        peak = self.initial_bankroll
        max_dd = 0
        for value in bankroll_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)

        # Sharpe ratio
        returns = []
        for i in range(1, len(bankroll_history)):
            ret = (bankroll_history[i] - bankroll_history[i-1]) / bankroll_history[i-1]
            returns.append(ret)

        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0

        # Bet distribution
        home_bets = [b for b in bets if b.bet_side == "HOME"]
        away_bets = [b for b in bets if b.bet_side == "AWAY"]
        home_wins = sum(1 for b in home_bets if b.covered)
        away_wins = sum(1 for b in away_bets if b.covered)

        return DualModelBacktestResult(
            bets=bets,
            initial_bankroll=self.initial_bankroll,
            final_bankroll=final_bankroll,
            total_wagered=total_wagered,
            total_pnl=total_pnl,
            roi=roi,
            ats_win_rate=wins / num_bets,
            num_bets=num_bets,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            kelly_fraction_used=self.kelly_fraction,
            min_disagreement=self.min_disagreement,
            bankroll_history=bankroll_history,
            avg_disagreement=np.mean([abs(b.disagreement) for b in bets]),
            home_bets=len(home_bets),
            away_bets=len(away_bets),
            home_ats_pct=home_wins / len(home_bets) if home_bets else 0,
            away_ats_pct=away_wins / len(away_bets) if away_bets else 0,
        )


def run_dual_model_ats_backtest(
    features_path: str = "data/features/game_features.parquet",
    test_seasons: List[int] = None,
    initial_bankroll: float = 1000.0,
    kelly_fraction: float = 0.15,
    min_disagreement: float = 5.0,
    min_edge_vs_market: float = 3.0,
    home_only: bool = False,
    output_path: str = "data/backtest/dual_model_ats_backtest.json",
) -> Tuple[DualModelBacktestResult, pd.DataFrame]:
    """
    Run dual model ATS backtest.

    Args:
        features_path: Path to game features
        test_seasons: Seasons to test on
        initial_bankroll: Starting bankroll
        kelly_fraction: Fractional Kelly to use
        min_disagreement: Minimum MLP/XGB disagreement in points
        min_edge_vs_market: Minimum edge vs Elo market proxy
        home_only: Only bet on home teams
        output_path: Path to save results

    Returns:
        Tuple of (BacktestResult, test DataFrame)
    """
    test_seasons = test_seasons or [2024]

    # Load features
    df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(df)} games")

    # Get feature columns - CRITICAL: exclude leakage
    exclude_cols = {
        # Identifiers
        "game_id", "date", "season", "home_team", "away_team",
        "home_team_id", "away_team_id",
        # Targets and outcomes (DATA LEAKAGE!)
        "home_win", "point_diff", "total_points",
        "home_score", "away_score", "total_score",
        # Keep Elo for market simulation, exclude from model
        "home_elo", "away_elo", "elo_diff", "elo_prob", "home_elo_prob",
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]
    logger.info(f"Using {len(feature_cols)} features")

    # Run backtest
    backtester = DualModelATSBacktester(
        initial_bankroll=initial_bankroll,
        kelly_fraction=kelly_fraction,
        min_disagreement=min_disagreement,
        min_edge_vs_market=min_edge_vs_market,
        home_only=home_only,
    )

    result = backtester.run_backtest(
        df=df,
        feature_cols=feature_cols,
        test_seasons=test_seasons,
    )

    # Get test data
    test_df = df[df["season"].isin(test_seasons)].copy()

    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        import json
        from datetime import datetime

        output = {
            "run_date": datetime.now().isoformat(),
            "test_seasons": test_seasons,
            "settings": {
                "kelly_fraction": kelly_fraction,
                "min_disagreement": min_disagreement,
                "min_edge_vs_market": min_edge_vs_market,
                "home_only": home_only,
                "spread_odds": -110,
            },
            "results": result.to_dict(),
            "bankroll_history": [float(x) for x in result.bankroll_history[-100:]],
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved results to {output_path}")

    return result, test_df


if __name__ == "__main__":
    result, df = run_dual_model_ats_backtest(
        test_seasons=[2024],
        kelly_fraction=0.15,
        min_disagreement=3.0,
    )

    print(result.summary())
