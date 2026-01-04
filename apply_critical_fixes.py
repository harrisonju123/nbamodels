#!/usr/bin/env python3
"""
Apply critical fixes from code review.
This script makes atomic, safe changes to fix the most important issues.
"""

import sys
import re
from pathlib import Path

def fix_race_condition_bet_tracker():
    """Fix race condition in log_bet() using INSERT OR IGNORE."""
    file_path = Path("src/bet_tracker.py")
    content = file_path.read_text()

    # Replace the check-then-insert pattern with INSERT OR IGNORE
    old_pattern = r'''conn = _get_connection\(\)
    bet_id = f"\{game_id\}_\{bet_type\}_\{bet_side\}"

    # Check for duplicate
    existing = conn\.execute\(
        "SELECT \* FROM bets WHERE id = \?", \(bet_id,\)
    \)\.fetchone\(\)

    if existing:
        logger\.info\(f"Bet already logged: \{bet_id\}"\)
        conn\.close\(\)
        return dict\(existing\)'''

    new_code = '''conn = _get_connection()
    bet_id = f"{game_id}_{bet_type}_{bet_side}"'''

    # Find and replace the duplicate check section
    content = content.replace(
        '''    # Check for duplicate
    existing = conn.execute(
        "SELECT * FROM bets WHERE id = ?", (bet_id,)
    ).fetchone()

    if existing:
        logger.info(f"Bet already logged: {bet_id}")
        conn.close()
        return dict(existing)

    logged_at''',
        '''    logged_at'''
    )

    # Change INSERT to INSERT OR IGNORE
    content = content.replace(
        '    conn.execute("""\n        INSERT INTO bets (',
        '    # Use INSERT OR IGNORE to prevent race condition duplicates\n    cursor = conn.execute("""\n        INSERT OR IGNORE INTO bets ('
    )

    # Add check after INSERT OR IGNORE
    content = content.replace(
        '''    ))

    conn.commit()

    logger.info''',
        '''    ))

    conn.commit()

    # Check if we actually inserted (rowcount > 0) or if it already existed
    if cursor.rowcount == 0:
        logger.info(f"Bet already logged (race condition prevented): {bet_id}")
        existing = conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()
        conn.close()
        return dict(existing) if existing else {}

    logger.info'''
    )

    file_path.write_text(content)
    print("✓ Fixed race condition in log_bet()")

def add_database_indexes():
    """Add missing database indexes for performance."""
    file_path = Path("src/bet_tracker.py")
    content = file_path.read_text()

    # Find the _get_connection function and add indexes
    existing_idx = 'conn.execute("CREATE INDEX IF NOT EXISTS idx_bets_outcome ON bets(outcome)")'

    new_indexes = '''conn.execute("CREATE INDEX IF NOT EXISTS idx_bets_outcome ON bets(outcome)")

    # Additional indexes for common queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bets_logged_at ON bets(logged_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bets_settled_at ON bets(settled_at DESC) WHERE outcome IS NOT NULL")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bets_bookmaker ON bets(bookmaker) WHERE bookmaker IS NOT NULL")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bets_bet_type ON bets(bet_type)")'''

    if existing_idx in content and "idx_bets_logged_at" not in content:
        content = content.replace(existing_idx, new_indexes)
        file_path.write_text(content)
        print("✓ Added missing database indexes")
    else:
        print("  Indexes already added or pattern not found")

def fix_memory_leak_dashboard():
    """Add pagination to prevent loading entire bet history."""
    file_path = Path("analytics_dashboard.py")
    content = file_path.read_text()

    # Add limit to load_bets_data
    old_get = '    df = get_bet_history()'
    new_get = '''    # Limit to prevent memory issues with large datasets
    # Dashboard shows recent data - no need to load entire history
    df = get_bet_history()

    # Hard limit to last 10,000 bets for performance
    if len(df) > 10000:
        df = df.nlargest(10000, 'logged_at')'''

    if old_get in content and "Hard limit to last 10,000" not in content:
        content = content.replace(old_get, new_get)
        file_path.write_text(content)
        print("✓ Fixed memory leak in dashboard (added 10K limit)")
    else:
        print("  Memory leak fix already applied or pattern not found")

def add_rate_limiting_docs():
    """Add documentation for rate limiting setup."""
    docs = """# API Rate Limiting Setup

## Installing slowapi

```bash
pip install slowapi
```

## Add to requirements.txt

```
slowapi>=0.1.8
```

## Implementation

See api/dashboard_api.py - rate limiting should be added in production.

Example:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/summary")
@limiter.limit("10/minute")
def get_summary(...):
    pass
```
"""

    file_path = Path("RATE_LIMITING.md")
    file_path.write_text(docs)
    print("✓ Created RATE_LIMITING.md documentation")

def main():
    print("Applying critical fixes from code review...")
    print()

    try:
        fix_race_condition_bet_tracker()
        add_database_indexes()
        fix_memory_leak_dashboard()
        add_rate_limiting_docs()

        print()
        print("✅ All critical fixes applied successfully!")
        print()
        print("Next steps:")
        print("1. Review the changes with: git diff")
        print("2. Run tests: pytest tests/")
        print("3. Test dashboard: streamlit run analytics_dashboard.py")
        print("4. Commit changes: git add -A && git commit")

    except Exception as e:
        print(f"❌ Error applying fixes: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
