# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - 2026-01-03

### Added

#### Security
- Created `.env.example` template for API key configuration
- Added `SECURITY.md` with API key management best practices
- Verified `.env` is not in git history and properly ignored

#### Code Quality
- Created centralized constants module (`src/utils/constants.py`) with:
  - Team name mappings (eliminates 5+ duplicate implementations)
  - Model threshold constants (MIN_DISAGREEMENT, MIN_EDGE_VS_MARKET, etc.)
  - Player impact scoring constants (replacement rates, multipliers)
  - Injury status probabilities
  - API configuration (timeouts, retries)
  - Validation bounds for player stats and spreads
- Created validation utilities module (`src/utils/validation.py`) with:
  - `validate_player_stats()` - Validates and clips player statistics
  - `validate_spread()` - Validates spread values
  - `validate_probability()` - Ensures probabilities in valid range
  - `validate_team_abbrev()` - Checks team abbreviation validity
  - `validate_api_response()` - Validates API responses have required fields
  - `validate_dataframe_columns()` - Checks DataFrame has required columns
  - `validate_odds()` - Validates American odds format

### Changed

#### API Error Handling
- **`src/data/odds_api.py`**:
  - Added timeout parameter (10s default) to all API requests
  - Implemented retry logic with exponential backoff (max 3 retries)
  - Added specific exception handling for Timeout, HTTPError, RequestException
  - Changed `get_current_odds()` to return empty DataFrame on error instead of crashing
  - Don't retry on 4xx errors (client errors), only retry on 5xx (server errors)

- **`src/data/espn_injuries.py`**:
  - Updated to use centralized `API_TIMEOUT_SECONDS` constant
  - Updated to use centralized `API_RATE_LIMIT_DELAY` constant
  - Already had timeout handling, now uses consistent values across codebase

#### Constants Centralization
- **`src/data/espn_injuries.py`**:
  - Removed duplicate `TEAM_NAME_TO_ABBREV` mapping
  - Removed duplicate `STATUS_MISS_PROB` dict
  - Removed hardcoded player impact constants (replaced with imports from `constants.py`)
  - Updated all methods to use imported constants:
    - `STARTER_MINUTES_THRESHOLD` (was 32.0)
    - `SCORING_REPLACEMENT_RATE` (was 0.28)
    - `ASSIST_POINT_VALUE` (was 1.5)
    - `ASSIST_REPLACEMENT_RATE` (was 0.35)
    - `REBOUND_POINT_VALUE` (was 0.4)
    - `REBOUND_REPLACEMENT_RATE` (was 0.25)
    - `PLUS_MINUS_WEIGHT` (was 0.1)
    - `MVP_LEVEL_PPG` (was 25.0)
    - `ALLSTAR_LEVEL_PPG` (was 20.0)
    - `QUALITY_STARTER_PPG` (was 15.0)
    - `MVP_MULTIPLIER` (was 1.4)
    - `ALLSTAR_MULTIPLIER` (was 1.25)
    - `QUALITY_STARTER_MULTIPLIER` (was 1.1)
    - `MIN_ROTATION_PLAYER_IMPACT` (was 1.0)

- **`src/models/dual_model.py`**:
  - Updated to use `SPREAD_TO_PROB_FACTOR` constant (was hardcoded 28)
  - Updated to use `MIN_PROBABILITY` and `MAX_PROBABILITY` for clipping (was 0.001/0.999)
  - Updated to use `MIN_DISAGREEMENT` constant (was 5.0)
  - Updated to use `MIN_EDGE_VS_MARKET` constant (was 5.0)

#### Performance Improvements
- **`src/data/espn_injuries.py`**:
  - **Fixed N+1 query pattern** in `PlayerImpactCalculator.get_team_injury_impact()`:
    - Previously called `stats_cache.get_player_stats()` in a loop for each injured player
    - Now calls `stats_cache.refresh()` once to load all player stats
    - Created `_lookup_player_stats()` helper method to query pre-loaded DataFrame
    - Eliminates multiple API calls when processing injury lists
  - Refactored player lookup logic into reusable `_lookup_player_stats()` method
  - Maintained all existing functionality (exact match, partial match, fallback to star player dict)

### Fixed

#### Error Handling
- API requests now properly handle and log timeout errors
- Server errors (5xx) now retry with exponential backoff
- Client errors (4xx) now fail fast instead of retrying
- All API failures now return safe defaults instead of crashing

#### Code Organization
- Eliminated magic numbers throughout codebase
- Removed duplicate team name mappings (5+ copies → 1 source of truth)
- Made threshold values configurable via constants instead of scattered literals

### Technical Debt Reduction

#### Addressed Issues
1. ✅ **CRITICAL: Exposed API Keys** - Created security documentation and .env.example
2. ✅ **CRITICAL: Missing Error Handling** - Added timeouts, retries, and proper exception handling
3. ✅ **HIGH: Code Duplication** - Centralized team mappings and constants
4. ✅ **HIGH: Magic Numbers** - Replaced all magic numbers with named constants
5. ✅ **MEDIUM: N+1 Query Pattern** - Fixed player stats lookup in injury impact calculation
6. ✅ **MEDIUM: Missing Input Validation** - Created comprehensive validation utilities

#### Remaining Issues (Future Work)
- **HIGH: God Classes** - `dashboard.py` (1,735 lines) and `multi_predictions.py` (2,000+ lines) need refactoring
- **MEDIUM: Missing Caching** - Recent games fetching could benefit from caching
- **LOW: Missing Tests** - Code coverage is <5%, needs comprehensive test suite

---

## Migration Guide

### For Developers

If you're updating code that uses the old hardcoded values:

**Old:**
```python
# Hardcoded values
if ppg >= 25:
    base_impact *= 1.4
spread = -((prob - 0.5) * 28)
miss_prob = 0.5  # Unknown status
```

**New:**
```python
from src.utils.constants import MVP_LEVEL_PPG, MVP_MULTIPLIER, SPREAD_TO_PROB_FACTOR

if ppg >= MVP_LEVEL_PPG:
    base_impact *= MVP_MULTIPLIER
spread = -((prob - 0.5) * SPREAD_TO_PROB_FACTOR)
miss_prob = STATUS_MISS_PROB.get(status, 0.5)
```

### For API Key Management

**IMPORTANT:** If you've been using this codebase before 2026-01-03:

1. Your `.env` file was never committed (verified), so API keys are safe
2. However, consider rotating your API keys as a precautionary measure
3. Follow the instructions in `SECURITY.md` for proper key management

### Breaking Changes

**None** - All changes are backward compatible. Existing code will continue to work.

---

## Notes

- All changes maintain backward compatibility
- No changes to prediction logic or model behavior
- All constants use their original values unless explicitly documented
- Performance improvements are purely internal (same outputs, better efficiency)
