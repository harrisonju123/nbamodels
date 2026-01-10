"""
Injury-Based Probability Adjustment

Adjusts model predictions based on current injury data.
Supports both heuristic-based adjustments and trained models
that learn coefficients from historical injury/outcome data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class InjuryCalibration:
    """Calibrated coefficients from historical data."""
    impact_to_prob: float  # Coefficient for probability adjustment
    impact_to_spread: float  # Coefficient for spread adjustment
    star_bonus_prob: float  # Star player probability bonus
    star_bonus_spread: float  # Star player spread bonus
    n_games_trained: int  # Number of games used for training
    validation_mae: float  # Mean absolute error on validation set
    calibration_date: str  # When calibration was performed


class InjuryAdjuster:
    """
    Adjusts win probabilities based on injury impact differential.

    Research basis:
    - Each point of spread ≈ 3% win probability change
    - Each point of injury impact ≈ 0.4 points of spread value
    - Therefore: injury_impact * 0.4 * 0.03 = probability adjustment

    Example:
    - Home team missing 10 points of impact, away missing 2
    - injury_diff = 2 - 10 = -8 (negative = hurts home team)
    - Adjustment = -8 * 0.012 = -0.096 (-9.6% to home win prob)
    """

    # Calibration parameters
    # Points of injury impact to probability conversion
    # Conservative estimate: 1 point impact ≈ 1.2% probability
    IMPACT_TO_PROB = 0.012

    # Maximum adjustment (don't swing more than 15%)
    MAX_ADJUSTMENT = 0.15

    # Minimum probability (never go below 5% or above 95%)
    MIN_PROB = 0.05
    MAX_PROB = 0.95

    def __init__(
        self,
        impact_to_prob: float = None,
        max_adjustment: float = None,
        star_bonus: float = 0.02,
    ):
        """
        Initialize the injury adjuster.

        Args:
            impact_to_prob: Conversion factor from impact points to probability
            max_adjustment: Maximum probability adjustment allowed
            star_bonus: Additional adjustment when a star player is out
        """
        self.impact_to_prob = impact_to_prob or self.IMPACT_TO_PROB
        self.max_adjustment = max_adjustment or self.MAX_ADJUSTMENT
        self.star_bonus = star_bonus

    def calculate_adjustment(
        self,
        home_injury_impact: float,
        away_injury_impact: float,
        home_star_out: bool = False,
        away_star_out: bool = False,
    ) -> float:
        """
        Calculate probability adjustment based on injuries.

        Positive adjustment = helps home team
        Negative adjustment = hurts home team

        Args:
            home_injury_impact: Expected point impact of home team injuries
            away_injury_impact: Expected point impact of away team injuries
            home_star_out: Whether home team has a star player out
            away_star_out: Whether away team has a star player out

        Returns:
            Probability adjustment (add to home win probability)
        """
        # Base adjustment from injury differential
        # Positive injury_diff means away is more hurt = good for home
        injury_diff = away_injury_impact - home_injury_impact
        base_adjustment = injury_diff * self.impact_to_prob

        # Star player bonus (additional adjustment for star absences)
        star_adjustment = 0.0
        if away_star_out and not home_star_out:
            star_adjustment = self.star_bonus  # Helps home
        elif home_star_out and not away_star_out:
            star_adjustment = -self.star_bonus  # Hurts home

        total_adjustment = base_adjustment + star_adjustment

        # Clip to maximum adjustment
        total_adjustment = np.clip(total_adjustment, -self.max_adjustment, self.max_adjustment)

        return total_adjustment

    def adjust_probability(
        self,
        base_prob: float,
        home_injury_impact: float,
        away_injury_impact: float,
        home_star_out: bool = False,
        away_star_out: bool = False,
    ) -> Tuple[float, float]:
        """
        Adjust a win probability based on injuries.

        Args:
            base_prob: Base model probability (home win)
            home_injury_impact: Expected point impact of home injuries
            away_injury_impact: Expected point impact of away injuries
            home_star_out: Whether home team has star out
            away_star_out: Whether away team has star out

        Returns:
            Tuple of (adjusted_prob, adjustment_amount)
        """
        adjustment = self.calculate_adjustment(
            home_injury_impact,
            away_injury_impact,
            home_star_out,
            away_star_out,
        )

        adjusted_prob = base_prob + adjustment

        # Clip to valid probability range
        adjusted_prob = np.clip(adjusted_prob, self.MIN_PROB, self.MAX_PROB)

        # Recalculate actual adjustment after clipping
        actual_adjustment = adjusted_prob - base_prob

        return adjusted_prob, actual_adjustment

    def calculate_spread_adjustment(
        self,
        home_injury_impact: float,
        away_injury_impact: float,
        home_star_out: bool = False,
        away_star_out: bool = False,
    ) -> float:
        """
        Calculate spread adjustment in points based on injuries.

        Positive adjustment = home team gets points added (injuries hurt them)
        Negative adjustment = home team loses points (injuries help them)

        Research basis:
        - Each point of injury impact ≈ 0.5 points of spread value
        - Star player out adds ~2 points of adjustment

        Args:
            home_injury_impact: Expected point impact of home team injuries
            away_injury_impact: Expected point impact of away team injuries
            home_star_out: Whether home team has a star player out
            away_star_out: Whether away team has a star player out

        Returns:
            Spread adjustment in points (positive = hurts home expected margin)
        """
        # Points per injury impact (conservative: 50% of impact translates to spread)
        IMPACT_TO_SPREAD = 0.5

        # Star player adjustment in points
        STAR_SPREAD_ADJUSTMENT = 2.0

        # Base adjustment: positive injury_diff means home is more hurt
        injury_diff = home_injury_impact - away_injury_impact
        base_adjustment = injury_diff * IMPACT_TO_SPREAD

        # Star player adjustment
        star_adjustment = 0.0
        if home_star_out and not away_star_out:
            star_adjustment = STAR_SPREAD_ADJUSTMENT  # Hurts home
        elif away_star_out and not home_star_out:
            star_adjustment = -STAR_SPREAD_ADJUSTMENT  # Helps home

        total_adjustment = base_adjustment + star_adjustment

        # Cap at reasonable maximum (15 points)
        total_adjustment = np.clip(total_adjustment, -15.0, 15.0)

        return total_adjustment

    def adjust_spread_prediction(
        self,
        expected_diff: float,
        home_injury_impact: float,
        away_injury_impact: float,
        home_star_out: bool = False,
        away_star_out: bool = False,
    ) -> Tuple[float, float]:
        """
        Adjust expected point differential based on injuries.

        Args:
            expected_diff: Base model expected diff (home - away)
            home_injury_impact: Expected point impact of home injuries
            away_injury_impact: Expected point impact of away injuries
            home_star_out: Whether home team has star out
            away_star_out: Whether away team has star out

        Returns:
            Tuple of (adjusted_diff, adjustment_amount)
        """
        adjustment = self.calculate_spread_adjustment(
            home_injury_impact,
            away_injury_impact,
            home_star_out,
            away_star_out,
        )

        # Adjustment is in points that HURT home team, so subtract
        adjusted_diff = expected_diff - adjustment

        return adjusted_diff, adjustment

    def adjust_predictions(self, predictions_df):
        """
        Adjust all predictions in a DataFrame.

        Expects columns:
        - model_home_prob: Base model probability
        - home_injury_impact: Home team injury impact
        - away_injury_impact: Away team injury impact
        - home_star_out: Home star out flag (optional)
        - away_star_out: Away star out flag (optional)

        Adds columns:
        - base_home_prob: Original probability
        - injury_adjustment: Adjustment applied
        - model_home_prob: Updated probability (overwrites original)
        - model_away_prob: Updated away probability
        """
        import pandas as pd

        if predictions_df.empty:
            return predictions_df

        # Check if injury columns exist
        if 'home_injury_impact' not in predictions_df.columns:
            logger.warning("No injury data in predictions, skipping adjustment")
            return predictions_df

        df = predictions_df.copy()

        # Save original probability
        df['base_home_prob'] = df['model_home_prob']

        # Apply adjustments
        adjustments = []
        adjusted_probs = []

        for _, row in df.iterrows():
            adj_prob, adjustment = self.adjust_probability(
                base_prob=row['model_home_prob'],
                home_injury_impact=row.get('home_injury_impact', 0) or 0,
                away_injury_impact=row.get('away_injury_impact', 0) or 0,
                home_star_out=bool(row.get('home_star_out', 0)),
                away_star_out=bool(row.get('away_star_out', 0)),
            )
            adjusted_probs.append(adj_prob)
            adjustments.append(adjustment)

        df['injury_adjustment'] = adjustments
        df['model_home_prob'] = adjusted_probs
        df['model_away_prob'] = 1 - df['model_home_prob']

        # Log summary
        non_zero = df[df['injury_adjustment'].abs() > 0.001]
        if len(non_zero) > 0:
            avg_adj = non_zero['injury_adjustment'].abs().mean()
            logger.info(
                f"Applied injury adjustments to {len(non_zero)} games "
                f"(avg adjustment: {avg_adj:.1%})"
            )

        return df


def test_injury_adjuster():
    """Test the injury adjustment logic."""
    adjuster = InjuryAdjuster()

    print("=" * 60)
    print("Testing Injury Probability Adjustment")
    print("=" * 60)

    # Test cases
    test_cases = [
        # (base_prob, home_impact, away_impact, home_star, away_star, description)
        (0.50, 0, 0, False, False, "No injuries"),
        (0.50, 10, 0, True, False, "Home star out (10 pts)"),
        (0.50, 0, 10, False, True, "Away star out (10 pts)"),
        (0.50, 5, 15, False, True, "Away more hurt"),
        (0.50, 15, 5, True, False, "Home more hurt"),
        (0.60, 8, 8, True, True, "Both teams equally hurt"),
        (0.75, 20, 0, True, False, "Home heavily injured (high base)"),
        (0.25, 0, 20, False, True, "Away heavily injured (low base)"),
    ]

    print(f"\nBase conversion: {adjuster.impact_to_prob:.1%} per impact point")
    print(f"Star bonus: {adjuster.star_bonus:.1%}")
    print(f"Max adjustment: {adjuster.max_adjustment:.1%}")
    print()

    for base, home_imp, away_imp, home_star, away_star, desc in test_cases:
        adj_prob, adjustment = adjuster.adjust_probability(
            base, home_imp, away_imp, home_star, away_star
        )

        arrow = "→" if adjustment >= 0 else "←"
        print(f"{desc:40} | {base:.1%} {arrow} {adj_prob:.1%} ({adjustment:+.1%})")

    print("\n" + "=" * 60)


class TrainedInjuryModel:
    """
    Learns injury adjustment coefficients from historical data.

    Uses historical games with known injury data and outcomes to
    calibrate the relationship between injury impacts and game results.
    """

    DEFAULT_CACHE_DIR = "data/cache/injury_model"

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.calibration: Optional[InjuryCalibration] = None
        self._spread_model = None
        self._prob_model = None
        self._scaler = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def prepare_training_data(
        self,
        games_df: pd.DataFrame,
        player_impacts: Dict[int, float],
        injury_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Prepare training data from historical games.

        Args:
            games_df: DataFrame with game results
                Required: game_id, home_team_id, away_team_id, home_score, away_score
            player_impacts: Dict mapping player_id -> impact value
            injury_data: Optional DataFrame with injury info
                Required: game_id, player_id, team_id, is_out

        Returns:
            DataFrame with features for training
        """
        training_data = []

        for _, game in games_df.iterrows():
            game_id = game.get('game_id')
            home_team_id = game.get('home_team_id')
            away_team_id = game.get('away_team_id')
            home_score = game.get('home_score')
            away_score = game.get('away_score')

            if pd.isna(home_score) or pd.isna(away_score):
                continue

            # Calculate point differential (positive = home won by more)
            point_diff = home_score - away_score
            home_won = 1 if point_diff > 0 else 0

            # Get injury impacts for this game
            home_impact = 0.0
            away_impact = 0.0
            home_star_out = False
            away_star_out = False

            if injury_data is not None and game_id in injury_data['game_id'].values:
                game_injuries = injury_data[
                    (injury_data['game_id'] == game_id) &
                    (injury_data['is_out'] == True)
                ]

                for _, inj in game_injuries.iterrows():
                    player_id = inj['player_id']
                    team_id = inj['team_id']
                    impact = player_impacts.get(player_id, 0)

                    if team_id == home_team_id:
                        home_impact += impact
                        if impact > 3.0:  # Star threshold
                            home_star_out = True
                    elif team_id == away_team_id:
                        away_impact += impact
                        if impact > 3.0:
                            away_star_out = True

            training_data.append({
                'game_id': game_id,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'point_diff': point_diff,
                'home_won': home_won,
                'home_injury_impact': home_impact,
                'away_injury_impact': away_impact,
                'injury_diff': away_impact - home_impact,  # Positive = helps home
                'home_star_out': int(home_star_out),
                'away_star_out': int(away_star_out),
                'star_diff': int(away_star_out) - int(home_star_out),  # Positive = helps home
            })

        return pd.DataFrame(training_data)

    def fit(
        self,
        training_df: pd.DataFrame,
        validation_split: float = 0.2,
    ) -> InjuryCalibration:
        """
        Fit the injury adjustment model on historical data.

        Args:
            training_df: DataFrame from prepare_training_data()
            validation_split: Fraction to hold out for validation

        Returns:
            InjuryCalibration with learned coefficients
        """
        if not HAS_SKLEARN:
            logger.warning("sklearn not available, using heuristic defaults")
            return self._get_default_calibration()

        # Filter to games with at least some injury impact
        df = training_df[
            (training_df['home_injury_impact'] > 0) |
            (training_df['away_injury_impact'] > 0)
        ].copy()

        if len(df) < 50:
            logger.warning(f"Only {len(df)} games with injuries, using heuristics")
            return self._get_default_calibration()

        # Shuffle and split
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df) * (1 - validation_split))
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        # Features: injury_diff, star_diff
        feature_cols = ['injury_diff', 'star_diff']
        X_train = train_df[feature_cols].values
        X_val = val_df[feature_cols].values

        # Target for spread model: point differential
        y_spread_train = train_df['point_diff'].values
        y_spread_val = val_df['point_diff'].values

        # Target for probability model: win/loss
        y_prob_train = train_df['home_won'].values
        y_prob_val = val_df['home_won'].values

        # Fit spread model (Ridge regression)
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_val_scaled = self._scaler.transform(X_val)

        self._spread_model = Ridge(alpha=1.0)
        self._spread_model.fit(X_train_scaled, y_spread_train)

        # Fit probability model (Logistic regression)
        self._prob_model = LogisticRegression(C=1.0, max_iter=1000)
        self._prob_model.fit(X_train_scaled, y_prob_train)

        # Extract coefficients (unscaled for interpretation)
        spread_coefs = self._spread_model.coef_ / self._scaler.scale_
        prob_coefs = self._prob_model.coef_[0] / self._scaler.scale_

        # Validate
        spread_pred = self._spread_model.predict(X_val_scaled)
        spread_mae = np.mean(np.abs(spread_pred - y_spread_val))

        # Create calibration
        from datetime import datetime

        self.calibration = InjuryCalibration(
            impact_to_prob=float(prob_coefs[0]),  # Coefficient for injury_diff
            impact_to_spread=float(spread_coefs[0]),  # Coefficient for injury_diff
            star_bonus_prob=float(prob_coefs[1]),  # Coefficient for star_diff
            star_bonus_spread=float(spread_coefs[1]),  # Coefficient for star_diff
            n_games_trained=len(train_df),
            validation_mae=float(spread_mae),
            calibration_date=datetime.now().strftime("%Y-%m-%d"),
        )

        self._is_fitted = True

        logger.info(
            f"Trained injury model on {len(train_df)} games. "
            f"Validation MAE: {spread_mae:.2f} points"
        )
        logger.info(
            f"Learned coefficients: "
            f"impact_to_spread={self.calibration.impact_to_spread:.4f}, "
            f"star_bonus_spread={self.calibration.star_bonus_spread:.2f}"
        )

        return self.calibration

    def _get_default_calibration(self) -> InjuryCalibration:
        """Return default heuristic calibration."""
        from datetime import datetime

        self.calibration = InjuryCalibration(
            impact_to_prob=0.012,
            impact_to_spread=0.5,
            star_bonus_prob=0.02,
            star_bonus_spread=2.0,
            n_games_trained=0,
            validation_mae=0.0,
            calibration_date=datetime.now().strftime("%Y-%m-%d"),
        )
        return self.calibration

    def get_adjuster(self) -> InjuryAdjuster:
        """
        Get an InjuryAdjuster configured with learned coefficients.

        Returns:
            InjuryAdjuster with calibrated parameters
        """
        if self.calibration is None:
            self.calibration = self._get_default_calibration()

        return InjuryAdjuster(
            impact_to_prob=self.calibration.impact_to_prob,
            max_adjustment=0.15,
            star_bonus=self.calibration.star_bonus_prob,
        )

    def predict_spread_adjustment(
        self,
        injury_diff: float,
        star_diff: float = 0,
    ) -> float:
        """
        Predict spread adjustment using the trained model.

        Args:
            injury_diff: away_impact - home_impact (positive = helps home)
            star_diff: away_star_out - home_star_out (positive = helps home)

        Returns:
            Expected spread adjustment in points (positive = helps home)
        """
        if not self._is_fitted or self._spread_model is None:
            # Use calibration coefficients directly
            if self.calibration is None:
                self.calibration = self._get_default_calibration()

            return (
                injury_diff * self.calibration.impact_to_spread +
                star_diff * self.calibration.star_bonus_spread
            )

        X = np.array([[injury_diff, star_diff]])
        X_scaled = self._scaler.transform(X)
        return float(self._spread_model.predict(X_scaled)[0])

    def predict_prob_adjustment(
        self,
        injury_diff: float,
        star_diff: float = 0,
    ) -> float:
        """
        Predict probability adjustment using the trained model.

        Args:
            injury_diff: away_impact - home_impact (positive = helps home)
            star_diff: away_star_out - home_star_out (positive = helps home)

        Returns:
            Probability adjustment (positive = helps home win probability)
        """
        if not self._is_fitted or self._prob_model is None:
            # Use calibration coefficients directly
            if self.calibration is None:
                self.calibration = self._get_default_calibration()

            return np.clip(
                injury_diff * self.calibration.impact_to_prob +
                star_diff * self.calibration.star_bonus_prob,
                -0.15, 0.15
            )

        # For logistic regression, we return the log-odds change
        # which approximates probability change for near-50% base probs
        X = np.array([[injury_diff, star_diff]])
        X_scaled = self._scaler.transform(X)

        # Get probability difference from 50% baseline
        prob = self._prob_model.predict_proba(X_scaled)[0, 1]
        adjustment = prob - 0.5

        return float(np.clip(adjustment, -0.15, 0.15))

    def save(self, path: Optional[str] = None) -> str:
        """Save the trained model to disk."""
        import json

        path = path or str(self.cache_dir / "trained_injury_model.json")

        if self.calibration is None:
            logger.warning("No calibration to save")
            return path

        data = {
            'calibration': {
                'impact_to_prob': self.calibration.impact_to_prob,
                'impact_to_spread': self.calibration.impact_to_spread,
                'star_bonus_prob': self.calibration.star_bonus_prob,
                'star_bonus_spread': self.calibration.star_bonus_spread,
                'n_games_trained': self.calibration.n_games_trained,
                'validation_mae': self.calibration.validation_mae,
                'calibration_date': self.calibration.calibration_date,
            },
            'scaler_mean': self._scaler.mean_.tolist() if self._scaler else None,
            'scaler_scale': self._scaler.scale_.tolist() if self._scaler else None,
            'spread_coef': self._spread_model.coef_.tolist() if self._spread_model else None,
            'spread_intercept': float(self._spread_model.intercept_) if self._spread_model else None,
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved trained injury model to {path}")
        return path

    def load(self, path: Optional[str] = None) -> "TrainedInjuryModel":
        """Load a trained model from disk."""
        import json

        path = path or str(self.cache_dir / "trained_injury_model.json")

        if not Path(path).exists():
            logger.warning(f"No trained model at {path}, using defaults")
            self.calibration = self._get_default_calibration()
            return self

        with open(path) as f:
            data = json.load(f)

        cal_data = data['calibration']
        self.calibration = InjuryCalibration(
            impact_to_prob=cal_data['impact_to_prob'],
            impact_to_spread=cal_data['impact_to_spread'],
            star_bonus_prob=cal_data['star_bonus_prob'],
            star_bonus_spread=cal_data['star_bonus_spread'],
            n_games_trained=cal_data['n_games_trained'],
            validation_mae=cal_data['validation_mae'],
            calibration_date=cal_data['calibration_date'],
        )

        # Restore sklearn models if we have the data
        if HAS_SKLEARN and data.get('scaler_mean') is not None:
            self._scaler = StandardScaler()
            self._scaler.mean_ = np.array(data['scaler_mean'])
            self._scaler.scale_ = np.array(data['scaler_scale'])

            if data.get('spread_coef') is not None:
                self._spread_model = Ridge()
                self._spread_model.coef_ = np.array(data['spread_coef'])
                self._spread_model.intercept_ = data['spread_intercept']
                self._is_fitted = True

        logger.info(
            f"Loaded trained injury model from {path} "
            f"(trained on {self.calibration.n_games_trained} games)"
        )
        return self


def build_trained_injury_model(
    games_path: str = "data/raw/games.parquet",
    player_impact_path: str = "data/cache/player_impact/player_impact_model.parquet",
    injury_data_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> TrainedInjuryModel:
    """
    Build a trained injury adjustment model from historical data.

    Args:
        games_path: Path to games data
        player_impact_path: Path to player impact model
        injury_data_path: Optional path to injury data
        output_path: Where to save the trained model

    Returns:
        Fitted TrainedInjuryModel
    """
    # Load games
    if not Path(games_path).exists():
        logger.warning(f"No games data at {games_path}")
        return TrainedInjuryModel()

    games_df = pd.read_parquet(games_path)
    logger.info(f"Loaded {len(games_df)} games")

    # Load player impacts
    player_impacts = {}
    if Path(player_impact_path).exists():
        try:
            from ..features.player_impact import PlayerImpactModel
            impact_model = PlayerImpactModel()
            impact_model.load(player_impact_path)

            impacts_df = impact_model.get_impacts_df()
            player_impacts = dict(zip(
                impacts_df['player_id'],
                impacts_df['impact']
            ))
            logger.info(f"Loaded impacts for {len(player_impacts)} players")
        except Exception as e:
            logger.warning(f"Could not load player impacts: {e}")

    # Load injury data if available
    injury_data = None
    if injury_data_path and Path(injury_data_path).exists():
        injury_data = pd.read_parquet(injury_data_path)
        logger.info(f"Loaded injury data with {len(injury_data)} records")

    # Create and train model
    model = TrainedInjuryModel()

    training_df = model.prepare_training_data(
        games_df, player_impacts, injury_data
    )

    if len(training_df) > 0:
        model.fit(training_df)

        if output_path:
            model.save(output_path)
    else:
        logger.warning("No training data available")

    return model


if __name__ == "__main__":
    test_injury_adjuster()

    print("\n" + "=" * 60)
    print("Testing TrainedInjuryModel")
    print("=" * 60)

    # Test with synthetic data
    model = TrainedInjuryModel()

    # Create synthetic training data
    np.random.seed(42)
    n_games = 500

    training_data = []
    for i in range(n_games):
        # Simulate injury impacts
        home_impact = np.random.exponential(2) if np.random.random() > 0.6 else 0
        away_impact = np.random.exponential(2) if np.random.random() > 0.6 else 0

        injury_diff = away_impact - home_impact

        # Simulate home star out
        home_star = np.random.random() < 0.1
        away_star = np.random.random() < 0.1
        star_diff = int(away_star) - int(home_star)

        # Simulate point differential with injury effect
        # True coefficients: 0.5 points per injury diff, 3 points per star diff
        true_effect = injury_diff * 0.5 + star_diff * 3.0
        point_diff = np.random.normal(3 + true_effect, 12)  # Home advantage + injury effect

        training_data.append({
            'game_id': f'game_{i}',
            'home_team_id': 1,
            'away_team_id': 2,
            'point_diff': point_diff,
            'home_won': 1 if point_diff > 0 else 0,
            'home_injury_impact': home_impact,
            'away_injury_impact': away_impact,
            'injury_diff': injury_diff,
            'home_star_out': int(home_star),
            'away_star_out': int(away_star),
            'star_diff': star_diff,
        })

    training_df = pd.DataFrame(training_data)
    print(f"\nCreated synthetic training data with {len(training_df)} games")
    print(f"Games with injuries: {len(training_df[training_df['home_injury_impact'] > 0])}")

    # Fit model
    calibration = model.fit(training_df)
    print(f"\nLearned coefficients:")
    print(f"  impact_to_spread: {calibration.impact_to_spread:.4f} (true: 0.5)")
    print(f"  star_bonus_spread: {calibration.star_bonus_spread:.2f} (true: 3.0)")
    print(f"  validation_mae: {calibration.validation_mae:.2f}")

    # Test predictions
    print("\nTest predictions:")
    test_cases = [
        (5.0, 0, "Away hurt by 5 pts"),
        (-5.0, 0, "Home hurt by 5 pts"),
        (0.0, 1, "Away star out"),
        (0.0, -1, "Home star out"),
        (3.0, 1, "Away hurt + star out"),
    ]

    for injury_diff, star_diff, desc in test_cases:
        pred = model.predict_spread_adjustment(injury_diff, star_diff)
        print(f"  {desc}: {pred:+.2f} points")
