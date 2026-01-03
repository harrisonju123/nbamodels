"""
Kelly Criterion Backtesting Module

Simulates betting performance using Kelly criterion for bet sizing.
Research shows fractional Kelly (1/4 to 1/5) reduces variance significantly.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class Bet:
    """Single bet record."""
    game_id: str
    date: pd.Timestamp
    pred_prob: float
    implied_prob: float
    odds: float  # Decimal odds
    edge: float
    kelly_fraction: float
    bet_size: float
    won: bool
    pnl: float
    bankroll_after: float


@dataclass
class BacktestResult:
    """Results from backtesting."""
    bets: List[Bet]
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
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "initial_bankroll": self.initial_bankroll,
            "final_bankroll": self.final_bankroll,
            "total_wagered": self.total_wagered,
            "total_pnl": self.total_pnl,
            "roi": self.roi,
            "win_rate": self.win_rate,
            "num_bets": self.num_bets,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "kelly_fraction_used": self.kelly_fraction_used,
            "min_edge_threshold": self.min_edge_threshold,
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        return f"""
=== Backtest Results ===
Bets Placed: {self.num_bets}
Win Rate: {self.win_rate:.1%}

Initial Bankroll: ${self.initial_bankroll:,.2f}
Final Bankroll: ${self.final_bankroll:,.2f}
Total P&L: ${self.total_pnl:+,.2f}

ROI: {self.roi:+.2%}
Max Drawdown: {self.max_drawdown:.1%}
Sharpe Ratio: {self.sharpe_ratio:.2f}

Settings:
  Kelly Fraction: {self.kelly_fraction_used}
  Min Edge: {self.min_edge_threshold:.1%}
"""


class KellyBacktester:
    """
    Backtest betting strategy using Kelly criterion.
    
    The Kelly criterion calculates optimal bet size:
        f* = (bp - q) / b
    where:
        b = decimal odds - 1 (profit per unit wagered)
        p = probability of winning
        q = 1 - p (probability of losing)
    
    Research shows full Kelly is too aggressive. We use fractional Kelly
    (typically 1/4 or 1/5) to reduce variance and risk of ruin.
    """
    
    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        kelly_fraction: float = 0.2,  # 1/5 Kelly (conservative)
        min_edge: float = 0.03,  # 3% minimum edge to bet
        max_bet_fraction: float = 0.05,  # Never bet more than 5% of bankroll
        default_odds: float = 1.91,  # -110 in decimal (standard spread odds)
    ):
        """
        Initialize backtester.
        
        Args:
            initial_bankroll: Starting bankroll
            kelly_fraction: Fraction of Kelly to use (0.2 = 1/5 Kelly)
            min_edge: Minimum edge required to place bet
            max_bet_fraction: Maximum bet as fraction of bankroll
            default_odds: Default decimal odds if not provided
        """
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_bet_fraction = max_bet_fraction
        self.default_odds = default_odds
    
    def american_to_decimal(self, american_odds: float) -> float:
        """Convert American odds to decimal."""
        if american_odds > 0:
            return 1 + (american_odds / 100)
        else:
            return 1 + (100 / abs(american_odds))
    
    def decimal_to_implied_prob(self, decimal_odds: float) -> float:
        """Convert decimal odds to implied probability."""
        return 1 / decimal_odds
    
    def calculate_kelly(
        self,
        pred_prob: float,
        decimal_odds: float,
    ) -> float:
        """
        Calculate Kelly fraction for a bet.
        
        Returns the fraction of bankroll to bet (before applying kelly_fraction).
        """
        b = decimal_odds - 1  # Profit per unit
        p = pred_prob
        q = 1 - p
        
        # Kelly formula
        kelly = (b * p - q) / b
        
        # Never bet negative (implies don't bet)
        return max(0, kelly)
    
    def run_backtest(
        self,
        predictions: pd.DataFrame,
        prob_column: str = "pred_prob",
        outcome_column: str = "home_win",
        odds_column: Optional[str] = None,
        date_column: str = "date",
    ) -> BacktestResult:
        """
        Run backtest on historical predictions.
        
        Args:
            predictions: DataFrame with predictions and outcomes
            prob_column: Column with predicted probabilities
            outcome_column: Column with actual outcomes (1 = win, 0 = loss)
            odds_column: Column with decimal odds (uses default if None)
            date_column: Column with dates for ordering
            
        Returns:
            BacktestResult with performance metrics
        """
        # Sort by date
        df = predictions.sort_values(date_column).copy()
        
        bankroll = self.initial_bankroll
        bankroll_history = [bankroll]
        bets = []
        total_wagered = 0.0
        
        for idx, row in df.iterrows():
            pred_prob = row[prob_column]
            actual = row[outcome_column]
            
            # Get odds
            if odds_column and odds_column in df.columns:
                decimal_odds = row[odds_column]
            else:
                decimal_odds = self.default_odds
            
            # Calculate implied probability and edge
            implied_prob = self.decimal_to_implied_prob(decimal_odds)
            edge = pred_prob - implied_prob
            
            # Only bet if edge exceeds threshold
            if edge < self.min_edge:
                continue
            
            # Calculate Kelly bet size
            kelly = self.calculate_kelly(pred_prob, decimal_odds)
            
            # Apply fractional Kelly and max bet cap
            bet_fraction = min(
                kelly * self.kelly_fraction,
                self.max_bet_fraction
            )
            
            bet_size = bankroll * bet_fraction
            
            # Skip tiny bets
            if bet_size < 1:
                continue
            
            # Determine outcome
            won = bool(actual == 1)
            
            if won:
                pnl = bet_size * (decimal_odds - 1)
            else:
                pnl = -bet_size
            
            bankroll += pnl
            total_wagered += bet_size
            bankroll_history.append(bankroll)
            
            # Record bet
            bet = Bet(
                game_id=str(row.get("game_id", idx)),
                date=pd.Timestamp(row[date_column]),
                pred_prob=pred_prob,
                implied_prob=implied_prob,
                odds=decimal_odds,
                edge=edge,
                kelly_fraction=kelly,
                bet_size=bet_size,
                won=won,
                pnl=pnl,
                bankroll_after=bankroll,
            )
            bets.append(bet)
        
        # Calculate metrics
        num_bets = len(bets)
        if num_bets == 0:
            return BacktestResult(
                bets=[],
                initial_bankroll=self.initial_bankroll,
                final_bankroll=bankroll,
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
            )
        
        wins = sum(1 for b in bets if b.won)
        total_pnl = bankroll - self.initial_bankroll
        roi = total_pnl / total_wagered if total_wagered > 0 else 0
        
        # Calculate max drawdown
        peak = self.initial_bankroll
        max_dd = 0
        for value in bankroll_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        # Calculate Sharpe ratio (daily returns)
        returns = []
        for i in range(1, len(bankroll_history)):
            ret = (bankroll_history[i] - bankroll_history[i-1]) / bankroll_history[i-1]
            returns.append(ret)
        
        if len(returns) > 1:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe = 0
        
        return BacktestResult(
            bets=bets,
            initial_bankroll=self.initial_bankroll,
            final_bankroll=bankroll,
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
        )
    
    def run_parameter_sweep(
        self,
        predictions: pd.DataFrame,
        kelly_fractions: List[float] = [0.1, 0.15, 0.2, 0.25, 0.33],
        min_edges: List[float] = [0.02, 0.03, 0.04, 0.05],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Sweep parameters to find optimal settings.
        
        Args:
            predictions: DataFrame with predictions
            kelly_fractions: Kelly fractions to test
            min_edges: Minimum edges to test
            
        Returns:
            DataFrame with results for each parameter combination
        """
        results = []
        
        for kf in kelly_fractions:
            for me in min_edges:
                self.kelly_fraction = kf
                self.min_edge = me
                
                result = self.run_backtest(predictions, **kwargs)
                
                results.append({
                    "kelly_fraction": kf,
                    "min_edge": me,
                    "num_bets": result.num_bets,
                    "roi": result.roi,
                    "win_rate": result.win_rate,
                    "max_drawdown": result.max_drawdown,
                    "sharpe_ratio": result.sharpe_ratio,
                    "final_bankroll": result.final_bankroll,
                    "total_pnl": result.total_pnl,
                })
        
        return pd.DataFrame(results)


def run_backtest_from_model(
    model_path: str = "models/spread_model_calibrated.pkl",
    features_path: str = "data/features/game_features.parquet",
    test_seasons: List[int] = [2024],
    initial_bankroll: float = 1000.0,
    kelly_fraction: float = 0.2,
    min_edge: float = 0.03,
) -> Tuple[BacktestResult, pd.DataFrame]:
    """
    Run backtest using saved model on test data.
    
    Args:
        model_path: Path to saved model
        features_path: Path to game features
        test_seasons: Seasons to use for testing
        initial_bankroll: Starting bankroll
        kelly_fraction: Fractional Kelly to use
        min_edge: Minimum edge threshold
        
    Returns:
        Tuple of (BacktestResult, predictions DataFrame)
    """
    import pickle
    
    # Load model
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data["model"]
    feature_cols = model_data["feature_cols"]
    
    # Load features
    df = pd.read_parquet(features_path)
    
    # Filter to test seasons
    test_df = df[df["season"].isin(test_seasons)].copy()
    logger.info(f"Testing on {len(test_df)} games from seasons {test_seasons}")
    
    # Prepare features
    X_test = test_df[feature_cols].copy()
    
    # Handle missing values
    X_test = X_test.fillna(X_test.median())
    
    # Get predictions
    probs = model.predict_proba(X_test)[:, 1]
    test_df["pred_prob"] = probs
    
    # Run backtest
    backtester = KellyBacktester(
        initial_bankroll=initial_bankroll,
        kelly_fraction=kelly_fraction,
        min_edge=min_edge,
    )
    
    result = backtester.run_backtest(
        test_df,
        prob_column="pred_prob",
        outcome_column="home_win",
    )
    
    return result, test_df


if __name__ == "__main__":
    # Example usage
    result, predictions = run_backtest_from_model(
        test_seasons=[2024],
        kelly_fraction=0.2,
        min_edge=0.03,
    )
    
    print(result.summary())
