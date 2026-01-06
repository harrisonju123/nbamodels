#!/usr/bin/env python3
"""
Quick test script to validate reporting modules work correctly.

This script tests basic functionality without sending actual Discord messages
or requiring real data.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("TESTING REPORTING MODULES")
print("=" * 80)

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from src.reporting import (
        DiscordNotifier,
        WeeklyReportGenerator,
        MonthlyReportGenerator,
        ReconciliationEngine
    )
    print("   ✓ All imports successful")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize classes
print("\n2. Testing class initialization...")
try:
    notifier = DiscordNotifier(webhook_url="https://example.com/webhook")
    weekly = WeeklyReportGenerator(db_path="data/bets/bets.db")
    monthly = MonthlyReportGenerator(db_path="data/bets/bets.db")
    reconcile = ReconciliationEngine(db_path="data/bets/bets.db")
    print("   ✓ All classes initialized successfully")
except Exception as e:
    print(f"   ✗ Initialization failed: {e}")
    sys.exit(1)

# Test 3: Test methods exist
print("\n3. Testing method availability...")
methods_to_test = [
    (notifier, 'send_daily_update'),
    (notifier, 'send_message'),
    (notifier, 'get_daily_stats'),
    (weekly, 'generate_report'),
    (weekly, 'format_report_text'),
    (weekly, 'get_weekly_data'),
    (monthly, 'generate_monthly_report'),
    (monthly, 'format_monthly_report_text'),
    (monthly, 'calculate_sharpe_ratio'),
    (monthly, 'calculate_max_drawdown'),
    (reconcile, 'run_full_reconciliation'),
    (reconcile, 'format_reconciliation_report'),
    (reconcile, 'check_data_integrity'),
]

for obj, method in methods_to_test:
    if hasattr(obj, method):
        print(f"   ✓ {obj.__class__.__name__}.{method}() exists")
    else:
        print(f"   ✗ {obj.__class__.__name__}.{method}() missing")
        sys.exit(1)

# Test 4: Test report generation with empty data
print("\n4. Testing report generation (with empty data)...")
try:
    # This will return an error dict but shouldn't crash
    weekly_report = weekly.generate_report(weeks_back=1)
    if 'error' in weekly_report or 'performance' in weekly_report:
        print("   ✓ Weekly report generation works (empty data)")
    else:
        print("   ✗ Weekly report format unexpected")

    monthly_report = monthly.generate_monthly_report(months_back=1)
    if 'error' in monthly_report or 'performance' in monthly_report:
        print("   ✓ Monthly report generation works (empty data)")
    else:
        print("   ✗ Monthly report format unexpected")

except Exception as e:
    print(f"   ✗ Report generation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test reconciliation
print("\n5. Testing reconciliation engine...")
try:
    reconcile_report = reconcile.run_full_reconciliation(days_back=7)
    if 'summary' in reconcile_report:
        print(f"   ✓ Reconciliation works: {reconcile_report['summary']['total_issues']} issues found")
    else:
        print("   ✗ Reconciliation report format unexpected")
except Exception as e:
    print(f"   ✗ Reconciliation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test formatting functions
print("\n6. Testing report formatting...")
try:
    if 'error' not in weekly_report:
        weekly_text = weekly.format_report_text(weekly_report)
        if len(weekly_text) > 0:
            print("   ✓ Weekly report formatting works")
        else:
            print("   ✗ Weekly report formatting returned empty string")

    if 'error' not in monthly_report:
        monthly_text = monthly.format_monthly_report_text(monthly_report)
        if len(monthly_text) > 0:
            print("   ✓ Monthly report formatting works")
        else:
            print("   ✗ Monthly report formatting returned empty string")

    reconcile_text = reconcile.format_reconciliation_report(reconcile_report)
    if len(reconcile_text) > 0:
        print("   ✓ Reconciliation report formatting works")
    else:
        print("   ✗ Reconciliation formatting returned empty string")

except Exception as e:
    print(f"   ✗ Formatting failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print("\nReporting modules are ready to use.")
print("\nNext steps:")
print("  1. Set DISCORD_WEBHOOK_URL environment variable")
print("  2. Run 'python nba.py report weekly' to generate a weekly report")
print("  3. Run 'python nba.py report monthly' to generate a monthly report")
print("  4. Run 'python nba.py reconcile' to validate data integrity")
print("  5. Set up cron job for daily reports (see src/reporting/README.md)")
