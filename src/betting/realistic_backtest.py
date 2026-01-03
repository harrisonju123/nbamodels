"""
Realistic Kelly Criterion Backtesting Module

Key improvements over naive backtesting:
1. Walk-forward validation (retrain model as we go)
2. Simulated market odds with realistic vig
3. Market efficiency modeling (market is close to true prob)
4. Proper handling of betting limits and bankroll management
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from loguru import logger
from datetime import timedelta
import pickle
import os

from .kelly import KellyBetSizer


@dataclass
class RealisticBet:
    """Single bet record with market context."""
    game_id: str
    date: pd.Timestamp
    home_team: str
    away_team: str
    bet_side: str  # 'home' or 'away'
    model_prob: float
    market_prob: float  # Simulated fair market probability
    offered_odds: int  # American odds with vig
    edge: float
    kelly_fraction: float
    bet_size: float
    won: bool
    pnl: float
    bankroll_after: float


@dataclass
class RealisticBacktestResult:
    """Results from realistic backtesting."""
    bets: List[RealisticBet]
    initial_bankroll: float
    final_bankroll: float
    total_wagered: float
    total_pnl: float
    roi: float
    win_rate: float
    num_bets: int
    max_drawdown: float
    sharpe_ratio: float
    kelly_fraction_used: float
    min_edge_threshold: float
    bankroll_history: List[float]
    avg_vig_paid: float
    edge_decay: float  # How much edge decayed vs expected

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "initial_bankroll": float(self.initial_bankroll),
            "final_bankroll": float(self.final_bankroll),
            "total_wagered": float(self.total_wagered),
            "total_pnl": float(self.total_pnl),
            "roi": float(self.roi),
            "win_rate": float(self.win_rate),
            "num_bets": int(self.num_bets),
            "max_drawdown": float(self.max_drawdown),
            "sharpe_ratio": float(self.sharpe_ratio),
            "kelly_fraction_used": float(self.kelly_fraction_used),
            "min_edge_threshold": float(self.min_edge_threshold),
            "avg_vig_paid": float(self.avg_vig_paid),
            "edge_decay": float(self.edge_decay),
        }

    def summary(self) -> str:
        """Human-readable summary."""
        return f"""
=== Realistic Backtest Results ===
Bets Placed: {self.num_bets}
Win Rate: {self.win_rate:.1%}

Initial Bankroll: ${self.initial_bankroll:,.2f}
Final Bankroll: ${self.final_bankroll:,.2f}
Total P&L: ${self.total_pnl:+,.2f}

ROI: {self.roi:+.2%}
Max Drawdown: {self.max_drawdown:.1%}
Sharpe Ratio: {self.sharpe_ratio:.2f}

Market Reality:
  Avg Vig Paid: {self.avg_vig_paid:.2%}
  Edge Decay: {self.edge_decay:.1%}

Settings:
  Kelly Fraction: {self.kelly_fraction_used}
  Min Edge: {self.min_edge_threshold:.1%}
"""


class MarketSimulator:
    """
    Simulates realistic market odds.

    Real betting markets are HIGHLY efficient. They incorporate:
    - Power ratings (like Elo)
    - Injuries, rest, travel
    - Public betting patterns
    - Sharp money
    - Everything our model might use

    We simulate this by assuming the market "knows" a weighted average of
    Elo and our model's prediction, making edge hard to find.
    """

    def __init__(
        self,
        base_vig: float = 0.045,  # Standard 4.5% vig
        market_efficiency: float = 0.85,  # How much market reflects our model's info
        line_noise_std: float = 0.015,  # Small noise for line movement
        random_seed: int = 42,
    ):
        self.base_vig = base_vig
        self.market_efficiency = market_efficiency
        self.line_noise_std = line_noise_std
        self.rng = np.random.default_rng(random_seed)

    def get_market_prob(
        self,
        row: pd.Series,
        model_prob: float,
    ) -> float:
        """
        Get market probability - blend of Elo and model prediction.

        Real markets are efficient and incorporate most public info.
        We model this as: market_prob = efficiency * model_prob + (1-efficiency) * elo_prob

        Args:
            row: DataFrame row with game data
            model_prob: Our model's prediction

        Returns:
            Market fair probability for home team
        """
        # Get Elo-based baseline
        if "elo_prob" in row.index and pd.notna(row.get("elo_prob")):
            elo_prob = row["elo_prob"]
        elif "home_elo" in row.index and "away_elo" in row.index:
            home_elo = row["home_elo"] + 100  # Home advantage
            away_elo = row["away_elo"]
            elo_prob = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        else:
            elo_prob = 0.55  # Base rate

        # Market is efficient - incorporates most of our model's info
        # At 85% efficiency, market knows 85% of what our model knows
        market_prob = (
            self.market_efficiency * model_prob +
            (1 - self.market_efficiency) * elo_prob
        )

        # Add small noise for line movement
        noise = self.rng.normal(0, self.line_noise_std)
        market_prob = np.clip(market_prob + noise, 0.25, 0.75)

        return market_prob

    def get_offered_odds(self, market_prob: float) -> Tuple[int, int]:
        """Get American odds with vig for both sides."""
        home_odds = self._fair_prob_to_vig_odds(market_prob)
        away_odds = self._fair_prob_to_vig_odds(1 - market_prob)
        return home_odds, away_odds

    def _fair_prob_to_vig_odds(self, fair_prob: float) -> int:
        """Convert fair probability to American odds with vig."""
        # Apply half the vig to each side
        implied_prob = fair_prob + (self.base_vig / 2)
        implied_prob = np.clip(implied_prob, 0.05, 0.95)

        # Convert to American odds
        if implied_prob >= 0.5:
            # Favorite: negative odds
            return int(-100 * implied_prob / (1 - implied_prob))
        else:
            # Underdog: positive odds
            return int(100 * (1 - implied_prob) / implied_prob)


class RealisticKellyBacktester:
    """
    Realistic backtesting with walk-forward validation.

    Key features:
    1. Retrains model periodically as new data becomes available
    2. Uses simulated market odds with realistic vig
    3. Models the fact that the market is efficient
    """

    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        kelly_fraction: float = 0.2,
        min_edge: float = 0.03,
        max_bet_fraction: float = 0.05,
        base_vig: float = 0.045,
        retrain_frequency: int = 50,  # Retrain every N games
        min_train_games: int = 500,  # Minimum games before betting
    ):
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_bet_fraction = max_bet_fraction
        self.retrain_frequency = retrain_frequency
        self.min_train_games = min_train_games

        self.market_sim = MarketSimulator(base_vig=base_vig)

        self.kelly_sizer = KellyBetSizer(
            fraction=kelly_fraction,
            max_bet_pct=max_bet_fraction,
        )

    def run_backtest(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "home_win",
        date_col: str = "date",
        season_col: str = "season",
        test_seasons: List[int] = None,
        use_walk_forward: bool = True,
        model_class: type = None,
        model_params: dict = None,
    ) -> RealisticBacktestResult:
        """
        Run realistic backtest with optional walk-forward validation.

        Args:
            df: DataFrame with features and targets
            feature_cols: Columns to use as features
            target_col: Target column name
            date_col: Date column name
            season_col: Season column name
            test_seasons: Seasons to actually bet on (earlier seasons for training only)
            use_walk_forward: Whether to retrain model during backtest
            model_class: Model class to use (defaults to XGBClassifier)
            model_params: Parameters for model

        Returns:
            RealisticBacktestResult with performance metrics
        """
        # Import here to avoid circular imports
        try:
            from xgboost import XGBClassifier
            default_model = XGBClassifier
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            default_model = GradientBoostingClassifier

        model_class = model_class or default_model
        model_params = model_params or {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42,
        }

        # Sort by date
        df = df.sort_values(date_col).reset_index(drop=True)

        bankroll = self.initial_bankroll
        bankroll_history = [bankroll]
        bets = []
        total_wagered = 0.0
        total_vig_paid = 0.0
        games_since_retrain = 0
        model = None

        expected_edge_sum = 0.0
        actual_edge_sum = 0.0

        for i, row in df.iterrows():
            # Skip if not enough training data
            if i < self.min_train_games:
                continue

            # Retrain model if needed
            if use_walk_forward and (model is None or games_since_retrain >= self.retrain_frequency):
                train_data = df.iloc[:i]
                X_train = train_data[feature_cols].fillna(train_data[feature_cols].median())
                y_train = train_data[target_col]

                model = model_class(**model_params)
                model.fit(X_train, y_train)
                games_since_retrain = 0
                logger.debug(f"Retrained model at game {i} with {len(train_data)} samples")

            games_since_retrain += 1

            # Only bet on test seasons
            if test_seasons and row.get(season_col) not in test_seasons:
                continue

            # Get model prediction
            X_pred = pd.DataFrame([row[feature_cols]]).fillna(df[feature_cols].median())
            try:
                model_prob = model.predict_proba(X_pred)[0, 1]
            except Exception:
                continue

            # Get true outcome
            true_outcome = row[target_col]

            # Get market probability (market is efficient and knows most of what model knows)
            market_prob = self.market_sim.get_market_prob(row, model_prob)

            # Get offered odds with vig
            home_odds, away_odds = self.market_sim.get_offered_odds(market_prob)

            # Determine which side to bet (if any)
            home_edge = model_prob - market_prob
            away_edge = (1 - model_prob) - (1 - market_prob)

            # Decide bet side and edge
            if home_edge >= self.min_edge:
                bet_side = "home"
                edge = home_edge
                offered_odds = home_odds
                win = bool(true_outcome == 1)
            elif away_edge >= self.min_edge:
                bet_side = "away"
                edge = away_edge
                offered_odds = away_odds
                win = bool(true_outcome == 0)
            else:
                continue  # No edge, skip

            # Calculate Kelly bet size
            kelly = self.kelly_sizer.calculate_kelly(
                win_prob=model_prob if bet_side == "home" else (1 - model_prob),
                odds=offered_odds,
                odds_format="american",
            )

            bet_size = bankroll * kelly

            # Skip tiny bets
            if bet_size < 1:
                continue

            # Calculate P&L
            if win:
                if offered_odds > 0:
                    profit = bet_size * (offered_odds / 100)
                else:
                    profit = bet_size * (100 / abs(offered_odds))
                pnl = profit
            else:
                pnl = -bet_size

            # Calculate vig paid
            if offered_odds > 0:
                implied_prob = 100 / (offered_odds + 100)
            else:
                implied_prob = abs(offered_odds) / (abs(offered_odds) + 100)

            fair_prob = market_prob if bet_side == "home" else (1 - market_prob)
            vig_paid = implied_prob - fair_prob
            total_vig_paid += vig_paid * bet_size

            # Track edge decay
            expected_edge_sum += edge
            actual_edge_sum += pnl / bet_size if bet_size > 0 else 0

            bankroll += pnl
            total_wagered += bet_size
            bankroll_history.append(bankroll)

            # Record bet
            bet = RealisticBet(
                game_id=str(row.get("game_id", i)),
                date=pd.Timestamp(row[date_col]),
                home_team=str(row.get("home_team", "")),
                away_team=str(row.get("away_team", "")),
                bet_side=bet_side,
                model_prob=model_prob,
                market_prob=market_prob,
                offered_odds=offered_odds,
                edge=edge,
                kelly_fraction=kelly,
                bet_size=bet_size,
                won=win,
                pnl=pnl,
                bankroll_after=bankroll,
            )
            bets.append(bet)

        # Calculate metrics
        return self._calculate_metrics(
            bets=bets,
            bankroll_history=bankroll_history,
            total_wagered=total_wagered,
            total_vig_paid=total_vig_paid,
            expected_edge_sum=expected_edge_sum,
            actual_edge_sum=actual_edge_sum,
        )

    def _calculate_metrics(
        self,
        bets: List[RealisticBet],
        bankroll_history: List[float],
        total_wagered: float,
        total_vig_paid: float,
        expected_edge_sum: float,
        actual_edge_sum: float,
    ) -> RealisticBacktestResult:
        """Calculate backtest metrics."""
        num_bets = len(bets)

        if num_bets == 0:
            return RealisticBacktestResult(
                bets=[],
                initial_bankroll=self.initial_bankroll,
                final_bankroll=self.initial_bankroll,
                total_wagered=0,
                total_pnl=0,
                roi=0,
                win_rate=0,
                num_bets=0,
                max_drawdown=0,
                sharpe_ratio=0,
                kelly_fraction_used=self.kelly_fraction,
                min_edge_threshold=self.min_edge,
                bankroll_history=bankroll_history,
                avg_vig_paid=0,
                edge_decay=0,
            )

        wins = sum(1 for b in bets if b.won)
        final_bankroll = bankroll_history[-1]
        total_pnl = final_bankroll - self.initial_bankroll
        roi = total_pnl / total_wagered if total_wagered > 0 else 0

        # Calculate max drawdown
        peak = self.initial_bankroll
        max_dd = 0
        for value in bankroll_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)

        # Calculate Sharpe ratio
        returns = []
        for i in range(1, len(bankroll_history)):
            ret = (bankroll_history[i] - bankroll_history[i-1]) / bankroll_history[i-1]
            returns.append(ret)

        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0

        # Edge decay: how much worse actual ROI was vs expected edge
        edge_decay = 1 - (actual_edge_sum / expected_edge_sum) if expected_edge_sum > 0 else 0

        return RealisticBacktestResult(
            bets=bets,
            initial_bankroll=self.initial_bankroll,
            final_bankroll=final_bankroll,
            total_wagered=total_wagered,
            total_pnl=total_pnl,
            roi=roi,
            win_rate=wins / num_bets,
            num_bets=num_bets,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            kelly_fraction_used=self.kelly_fraction,
            min_edge_threshold=self.min_edge,
            bankroll_history=bankroll_history,
            avg_vig_paid=total_vig_paid / total_wagered if total_wagered > 0 else 0,
            edge_decay=edge_decay,
        )

    def run_parameter_sweep(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        kelly_fractions: List[float] = [0.1, 0.15, 0.2, 0.25],
        min_edges: List[float] = [0.02, 0.03, 0.04, 0.05],
        market_efficiencies: List[float] = [0.90, 0.95, 0.98],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Sweep parameters to understand sensitivity.

        Returns DataFrame with results for each parameter combination.
        """
        results = []

        for kf in kelly_fractions:
            for me_thresh in min_edges:
                for mkt_eff in market_efficiencies:
                    self.kelly_fraction = kf
                    self.min_edge = me_thresh
                    self.kelly_sizer = KellyBetSizer(fraction=kf, max_bet_pct=self.max_bet_fraction)
                    self.market_sim = MarketSimulator(
                        base_vig=0.045,
                        market_efficiency=mkt_eff,
                    )

                    result = self.run_backtest(df, feature_cols, **kwargs)

                    results.append({
                        "kelly_fraction": kf,
                        "min_edge": me_thresh,
                        "market_efficiency": mkt_eff,
                        "num_bets": result.num_bets,
                        "roi": result.roi,
                        "win_rate": result.win_rate,
                        "max_drawdown": result.max_drawdown,
                        "sharpe_ratio": result.sharpe_ratio,
                        "final_bankroll": result.final_bankroll,
                        "total_pnl": result.total_pnl,
                        "avg_vig_paid": result.avg_vig_paid,
                    })

        return pd.DataFrame(results)


def run_realistic_backtest(
    features_path: str = "data/features/game_features.parquet",
    test_seasons: List[int] = None,
    initial_bankroll: float = 1000.0,
    kelly_fraction: float = 0.2,
    min_edge: float = 0.03,
    use_walk_forward: bool = True,
    output_path: str = "data/backtest/realistic_backtest.json",
) -> Tuple[RealisticBacktestResult, pd.DataFrame]:
    """
    Run realistic backtest using game features.

    Market odds are simulated using Elo ratings (representing public consensus).
    Edge is the difference between ML model prediction and Elo-based market.

    Args:
        features_path: Path to game features
        test_seasons: Seasons to test on (default: [2024])
        initial_bankroll: Starting bankroll
        kelly_fraction: Fractional Kelly to use
        min_edge: Minimum edge threshold
        use_walk_forward: Whether to retrain model during backtest
        output_path: Path to save results

    Returns:
        Tuple of (RealisticBacktestResult, predictions DataFrame)
    """
    test_seasons = test_seasons or [2024]

    # Load features
    df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(df)} games")

    # Get feature columns - CRITICAL: exclude all leakage columns
    exclude_cols = {
        # Identifiers
        "game_id", "date", "season", "home_team", "away_team",
        "home_team_id", "away_team_id",
        # Targets and outcomes (DATA LEAKAGE!)
        "home_win", "point_diff", "total_points",
        "home_score", "away_score", "total_score",
        # Elo is used for market simulation, not model
        "home_elo", "away_elo", "elo_diff", "elo_prob", "home_elo_prob",
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]
    logger.info(f"Using {len(feature_cols)} features (excluding scores and Elo)")

    # Filter to test seasons (but need earlier data for training)
    df = df[df["season"] <= max(test_seasons)].copy()

    # Run backtest
    backtester = RealisticKellyBacktester(
        initial_bankroll=initial_bankroll,
        kelly_fraction=kelly_fraction,
        min_edge=min_edge,
    )

    result = backtester.run_backtest(
        df=df,
        feature_cols=feature_cols,
        test_seasons=test_seasons,
        use_walk_forward=use_walk_forward,
    )

    # Get test data for return
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
                "min_edge": min_edge,
                "use_walk_forward": use_walk_forward,
                "market_simulation": "elo_based",
            },
            "results": result.to_dict(),
            "bankroll_history": [float(x) for x in result.bankroll_history[-100:]],
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved results to {output_path}")

    return result, test_df


if __name__ == "__main__":
    result, df = run_realistic_backtest(
        test_seasons=[2024],
        kelly_fraction=0.2,
        min_edge=0.03,
        market_efficiency=0.95,
        use_walk_forward=True,
    )

    print(result.summary())
