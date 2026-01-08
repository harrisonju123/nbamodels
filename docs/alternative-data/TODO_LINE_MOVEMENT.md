# Line Movement Tracking - TODO List

**Last Updated:** 2026-01-03
**Code Review Score:** 8.5/10

---

## 🔴 High Priority (Performance & Reliability)

### 1. Implement Stale Lock File Cleanup
**Status:** 🔲 Not Started
**Priority:** High
**Effort:** 2-3 hours
**File:** `src/data/line_snapshot_collector.py`

**Issue:**
If the snapshot collection process crashes between acquiring the lock (line 76) and releasing it (line 141), the lock file `/tmp/nba_snapshot_collector.lock` persists forever, blocking all future runs.

**Impact:**
- Manual intervention required to delete lock file
- Cron job stops working until lock removed
- No automatic recovery

**Solution:**
```python
import atexit
import os
import signal

# Option 1: PID-based lock check
def acquire_lock_with_pid_check():
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, 'r') as f:
                old_pid = int(f.read().strip())
            # Check if process still exists
            os.kill(old_pid, 0)
            # Process exists, bail
            logger.warning(f"Lock held by running process {old_pid}")
            return None
        except (OSError, ValueError):
            # Process dead or invalid PID, remove stale lock
            logger.info("Removing stale lock file")
            os.remove(LOCK_FILE)

    # Write current PID to lock file
    lock_fd = open(LOCK_FILE, 'w')
    lock_fd.write(str(os.getpid()))
    lock_fd.flush()
    return lock_fd

# Option 2: Register cleanup on exit
def cleanup_lock():
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except Exception as e:
        logger.warning(f"Could not cleanup lock: {e}")

atexit.register(cleanup_lock)
signal.signal(signal.SIGTERM, lambda s, f: cleanup_lock())
```

**Testing:**
- [ ] Test normal execution (lock acquired and released)
- [ ] Test process kill (SIGKILL) - verify lock cleaned up on next run
- [ ] Test concurrent execution (lock prevents overlap)

---

### 2. Optimize Reversal Detection Query
**Status:** 🔲 Not Started
**Priority:** Medium
**Effort:** 1-2 hours
**File:** `src/data/line_history.py:286-340`

**Issue:**
`detect_line_reversals()` fetches ALL snapshots for a game via `get_line_history()`, even though it only needs 3+ records to detect reversals.

**Impact:**
- Fetches 100+ snapshot records unnecessarily
- Slow for games with extensive history
- Wasted database I/O

**Solution:**
```python
def detect_line_reversals(
    self,
    game_id: str,
    bet_type: str = 'spread',
    min_reversal_pts: float = 0.5
) -> List[Dict]:
    """Detect line reversals with optimized query."""

    conn = self._get_connection()

    # Quick count check to avoid fetching data
    count_query = """
        SELECT COUNT(*) as count
        FROM line_snapshots
        WHERE game_id = ? AND bet_type = ?
    """
    count = conn.execute(count_query, [game_id, bet_type]).fetchone()['count']

    if count < 3:
        conn.close()
        return []

    conn.close()

    # Now fetch full history (only if we have enough data)
    df = self.get_line_history(game_id, bet_type)

    # ... rest of existing logic
```

**Testing:**
- [ ] Benchmark query time with 10, 50, 100+ snapshots
- [ ] Verify reversals detected correctly
- [ ] Test with games having <3 snapshots

---

## 🟡 Medium Priority (Code Quality)

### 3. Enhanced HTTP Error Logging
**Status:** 🔲 Not Started
**Priority:** Medium
**Effort:** 30 minutes
**File:** `src/data/line_snapshot_collector.py:95-111`

**Issue:**
HTTP error handling provides status codes but doesn't log response body for debugging.

**Impact:**
- Harder to debug API issues
- Missing context when rate limits hit

**Solution:**
```python
except requests.exceptions.HTTPError as e:
    response_preview = e.response.text[:200] if e.response else "No response body"

    if e.response.status_code == 429:
        logger.error(f"Rate limit exceeded (429): {response_preview}")
        # Extract rate limit reset time if available
        reset_time = e.response.headers.get('X-RateLimit-Reset', 'unknown')
        logger.info(f"Rate limit resets at: {reset_time}")
    elif e.response.status_code == 401:
        logger.error(f"Authentication failed (401): {response_preview}")
    else:
        logger.error(f"HTTP {e.response.status_code}: {response_preview}")
    raise
```

**Testing:**
- [ ] Simulate 429 rate limit response
- [ ] Simulate 401 auth failure
- [ ] Verify response body truncated to 200 chars

---

### 4. Standardize Naming Conventions
**Status:** 🔲 Not Started
**Priority:** Low
**Effort:** 15 minutes
**File:** `src/data/line_history.py:23-30`

**Issue:**
Inconsistent abbreviations in `LineMovement` dataclass.

**Current:**
```python
@dataclass
class LineMovement:
    movement_pts: float  # Abbreviation
    movement_time: str   # Full word
```

**Solution:**
```python
@dataclass
class LineMovement:
    from_line: float
    to_line: float
    from_odds: int
    to_odds: int
    movement_points: float  # Changed from movement_pts
    movement_time: str
    bookmaker: str
```

**Testing:**
- [ ] Update all references to `movement_pts`
- [ ] Run type checking with mypy

---

### 5. Fix Moneyline Bet Type Mapping
**Status:** 🔲 Not Started
**Priority:** Low
**Effort:** 15 minutes
**File:** `src/data/line_history.py:490`

**Issue:**
Hardcoded 'h2h' string instead of using bet_type mapping constant.

**Current:**
```python
'moneyline_closing': df[df['bet_type'] == 'h2h'].to_dict('records')
```

**Solution:**
```python
# Add constant at module level
BET_TYPE_API_MAPPING = {
    'moneyline': 'h2h',
    'spread': 'spreads',
    'totals': 'totals'
}

# Use in function
api_bet_type = BET_TYPE_API_MAPPING.get(bet_type, bet_type)
'moneyline_closing': df[df['bet_type'] == api_bet_type].to_dict('records')
```

**Testing:**
- [ ] Verify closing line summary works for all bet types

---

## 🟢 Future Enhancements (Nice to Have)

### 6. Add Query Result Caching
**Status:** 💡 Idea
**Priority:** Low
**Effort:** 3-4 hours
**File:** `src/data/line_history.py`

**Rationale:**
Dashboard may query same game's line history multiple times in a single session.

**Solution:**
```python
from functools import lru_cache
from datetime import datetime, timedelta

class LineHistoryManager:
    def __init__(self):
        self._cache = {}
        self._cache_ttl = timedelta(minutes=5)

    def get_line_history(self, game_id, bet_type, side, bookmaker=None):
        cache_key = f"{game_id}:{bet_type}:{side}:{bookmaker}"

        # Check cache
        if cache_key in self._cache:
            cached_time, cached_df = self._cache[cache_key]
            if datetime.now() - cached_time < self._cache_ttl:
                return cached_df

        # Fetch from DB
        df = self._fetch_line_history(game_id, bet_type, side, bookmaker)

        # Cache result
        self._cache[cache_key] = (datetime.now(), df)
        return df
```

**Benefits:**
- Faster dashboard response
- Reduced database load

---

### 7. Add Line Movement Alerts
**Status:** 💡 Idea
**Priority:** Low
**Effort:** 4-6 hours
**Files:** New file `src/betting/line_alerts.py`

**Feature:**
Notify users when significant line movements occur on their bets.

**Implementation:**
```python
class LineMovementAlert:
    """Alert when line moves against/with bet."""

    def check_bet_line_movement(self, bet_id: str) -> Optional[Alert]:
        """Check if line moved significantly since bet placed."""
        # Get bet details
        # Get current line
        # Compare movement
        # Return alert if threshold exceeded
        pass
```

**Alert Triggers:**
- Line moves >1.5 pts against bet (warning)
- Line moves >1.5 pts with bet (positive signal)
- Steam move detected on bet's game
- RLM signal conflicts with bet

---

### 8. Add Movement Pattern Backtesting
**Status:** 💡 Idea
**Priority:** Low
**Effort:** 6-8 hours
**Files:** `src/betting/pattern_backtest.py`

**Feature:**
Validate if betting with/against certain patterns improves ROI.

**Patterns to Test:**
- Bet when pattern = 'late_steam' (hypothesis: sharp money)
- Fade 'volatile' lines (hypothesis: public confusion)
- Wait for 'stable' → 'trending' transition
- Bet opposite of 'drifting' direction

**Metrics:**
- ROI by pattern type
- Win rate by pattern
- Optimal timing based on pattern

---

## 📊 Testing Checklist

### Unit Tests Needed
- [ ] `test_line_history.py`
  - [ ] `test_get_line_history_empty()` - Empty result handling
  - [ ] `test_get_opening_line_fallback()` - Fallback to first snapshot
  - [ ] `test_detect_reversals()` - Reversal detection logic
  - [ ] `test_analyze_movement_pattern()` - Pattern classification
  - [ ] `test_get_line_at_time()` - Time-based lookup accuracy

- [ ] `test_snapshot_collector.py`
  - [ ] `test_capture_opening_lines()` - New game detection
  - [ ] `test_map_market_to_bet_type()` - Market name mapping
  - [ ] `test_extract_sides_data()` - Side data extraction
  - [ ] `test_concurrent_execution()` - File locking

### Integration Tests Needed
- [ ] End-to-end snapshot collection
- [ ] Dashboard line movement page rendering
- [ ] Opening line capture → query → display flow

### Performance Tests
- [ ] Benchmark line_history queries with 1000+ snapshots
- [ ] Test dashboard load time with multiple games
- [ ] Verify bulk insert performance in snapshot collector

---

## 📈 Metrics to Track

### After Each Fix
- [ ] Code coverage %
- [ ] Query execution time (ms)
- [ ] Dashboard page load time (s)
- [ ] Database size growth rate

### Production Monitoring
- [ ] Snapshot collection success rate
- [ ] Opening line capture rate
- [ ] Dashboard error rate
- [ ] Query response times (p50, p95, p99)

---

## 🚀 Deployment Checklist

### Before Production
- [ ] Fix stale lock file handling (Issue #1)
- [ ] Run full test suite
- [ ] Test with production-like data volume
- [ ] Set up cron jobs with proper error handling
- [ ] Configure log rotation for snapshot collector
- [ ] Document API rate limit expectations

### After Production Launch
- [ ] Monitor snapshot collection logs for 7 days
- [ ] Review opening line capture accuracy
- [ ] Gather user feedback on dashboard performance
- [ ] Optimize slow queries identified in logs

---

## 📝 Notes

### Code Review History
- **2026-01-03:** Initial review (Score: 7.5/10)
  - Critical issues: 1 (table creation - false positive)
  - Major issues: 9
  - Fixed: Index optimization, DataFrame checks, timezone consistency, magic numbers

- **2026-01-03:** Post-fix review (Score: 8.5/10)
  - Remaining: 3 high priority, 3 medium priority, 3 future enhancements

### Reference Files
- Code review output: See conversation history
- Implementation plan: `/Users/harrisonju/.claude/plans/swift-questing-lake.md`
- Original schema: `src/bet_tracker.py:102-129`

### Contact
For questions about these TODOs, refer to the code review conversation from 2026-01-03.
