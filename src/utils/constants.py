"""
Centralized constants for NBA betting models.

This module contains all constant values used throughout the codebase,
including team mappings, model thresholds, and configuration values.
"""

# ============================================================================
# TEAM NAME MAPPINGS
# ============================================================================

# Team full name to abbreviation mapping
TEAM_NAME_TO_ABBREV = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "LA Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

# Team abbreviation to full name (inverse mapping)
ABBREV_TO_TEAM_NAME = {v: k for k, v in TEAM_NAME_TO_ABBREV.items() if "LA " not in k}

# All valid team abbreviations
VALID_TEAM_ABBREVS = set(TEAM_NAME_TO_ABBREV.values())

# ============================================================================
# MODEL THRESHOLDS
# ============================================================================

# Betting strategy thresholds
MIN_DISAGREEMENT = 5.0  # Minimum model disagreement to consider betting
MIN_EDGE_VS_MARKET = 5.0  # Minimum edge vs market line (in points)
MIN_EDGE_VS_ELO = 5.0  # Minimum edge vs Elo baseline

# Kelly criterion
KELLY_FRACTION = 0.25  # Conservative Kelly fraction for bet sizing
KELLY_MAX_BET_PCT = 0.05  # Maximum bet as % of bankroll

# Confidence thresholds
HIGH_CONFIDENCE_DISAGREEMENT = 5.0  # High confidence bet threshold
MIN_CONFIDENCE_SCORE = 0.6  # Minimum confidence for any bet

# ============================================================================
# PLAYER IMPACT CONSTANTS
# ============================================================================

# Minutes thresholds
STARTER_MINUTES_THRESHOLD = 32.0  # Minutes to be considered high-impact starter
ROTATION_PLAYER_MINUTES = 20.0  # Minutes to be considered rotation player

# Impact scoring weights
SCORING_REPLACEMENT_RATE = 0.28  # % of scoring not replaced by bench
ASSIST_REPLACEMENT_RATE = 0.35  # % of assists not replaced
ASSIST_POINT_VALUE = 1.5  # Points created per assist
REBOUND_REPLACEMENT_RATE = 0.25  # % of rebounds not replaced
REBOUND_POINT_VALUE = 0.4  # Possession value per rebound
PLUS_MINUS_WEIGHT = 0.1  # Weight for plus/minus impact

# Star player PPG thresholds
MVP_LEVEL_PPG = 25.0  # MVP/superstar level
ALLSTAR_LEVEL_PPG = 20.0  # All-Star level
QUALITY_STARTER_PPG = 15.0  # Quality starter level

# Star player multipliers
MVP_MULTIPLIER = 1.4  # Multiplier for MVP-level players
ALLSTAR_MULTIPLIER = 1.25  # Multiplier for All-Star players
QUALITY_STARTER_MULTIPLIER = 1.1  # Multiplier for quality starters

# Minimum impact floor
MIN_ROTATION_PLAYER_IMPACT = 1.0  # Minimum impact for any rotation player

# ============================================================================
# INJURY STATUS PROBABILITIES
# ============================================================================

# Probability player misses game by injury status
STATUS_MISS_PROB = {
    "Out": 1.0,
    "Doubtful": 0.9,
    "Questionable": 0.6,
    "Day-To-Day": 0.5,
    "Probable": 0.15,
    "Available": 0.0,
    "Unknown": 0.5,
}

# ============================================================================
# SPREAD/PROBABILITY CONVERSION
# ============================================================================

# Standard conversion factor for spread to probability
SPREAD_TO_PROB_FACTOR = 28.0  # Points for full probability range (0 to 1)

# ============================================================================
# ODDS CONVERSION
# ============================================================================

# Standard vig for spread bets
STANDARD_SPREAD_ODDS = -110  # Standard juice on spreads
NO_VIG_SPREAD_PROB = 0.5  # Fair 50/50 probability

# ============================================================================
# API CONFIGURATION
# ============================================================================

# Request timeouts (seconds)
API_TIMEOUT_SECONDS = 10
API_LONG_TIMEOUT_SECONDS = 30

# Retry configuration
API_MAX_RETRIES = 3
API_RETRY_BACKOFF_BASE = 2  # Exponential backoff: 2^attempt seconds

# Rate limiting
API_RATE_LIMIT_DELAY = 0.5  # Seconds between requests

# ============================================================================
# CACHING CONFIGURATION
# ============================================================================

# Cache durations
STATS_CACHE_HOURS = 12  # Player stats cache
INJURY_CACHE_MINUTES = 30  # Injury data cache
ODDS_CACHE_MINUTES = 5  # Odds data cache

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# Database paths (relative to project root)
BETS_DB_PATH = "data/bets/bets.db"
GAMES_PARQUET_PATH = "data/raw/games.parquet"

# ============================================================================
# VALIDATION BOUNDS
# ============================================================================

# Player stats bounds (for validation)
MAX_PPG = 50.0  # Maximum reasonable PPG
MAX_RPG = 20.0  # Maximum reasonable RPG
MAX_APG = 15.0  # Maximum reasonable APG
MAX_MPG = 48.0  # Maximum possible MPG

MIN_PPG = 0.0
MIN_RPG = 0.0
MIN_APG = 0.0
MIN_MPG = 0.0

# Spread bounds
MAX_REASONABLE_SPREAD = 30.0  # Maximum reasonable NBA spread
MIN_REASONABLE_SPREAD = -30.0  # Minimum reasonable NBA spread

# Probability bounds
MAX_PROBABILITY = 0.999  # Maximum probability (avoid log(0))
MIN_PROBABILITY = 0.001  # Minimum probability (avoid log(0))

# ============================================================================
# ALTERNATIVE DATA CONSTANTS
# ============================================================================

# Referee constants
REF_TOTAL_BIAS_WEIGHT = 0.5  # Weight for referee total bias in predictions
REF_PACE_FACTOR_DEFAULT = 1.0  # Neutral pace factor

# Lineup constants
LINEUP_STARTER_COUNT = 5  # Expected number of starters
LINEUP_UNCERTAINTY_THRESHOLD = 0.4  # High uncertainty if 40%+ starters questionable

# News constants
NEWS_RECENCY_HOURS = 24  # Default lookback window for news
NEWS_BREAKING_THRESHOLD_HOURS = 2  # Consider "breaking" if within 2 hours
NEWS_HIGH_VOLUME_THRESHOLD = 10  # High news volume if 10+ articles

# Sentiment constants (Phase 4)
SENTIMENT_ENABLED = False  # Toggle when APIs configured
SENTIMENT_NEUTRAL = 0.0  # Neutral sentiment value
SENTIMENT_MIN = -1.0  # Most negative sentiment
SENTIMENT_MAX = 1.0  # Most positive sentiment
