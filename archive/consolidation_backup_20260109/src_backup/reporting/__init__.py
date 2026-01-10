"""
Automated Reporting Module

Provides automated reporting capabilities for the NBA betting system:
- Daily Discord updates
- Weekly strategy reviews
- Monthly investor-grade reports
- Automated reconciliation

Usage Examples:
    # Send daily Discord update
    from src.reporting import DiscordNotifier
    notifier = DiscordNotifier()
    notifier.send_daily_update()

    # Generate weekly report
    from src.reporting import WeeklyReportGenerator
    weekly = WeeklyReportGenerator()
    report = weekly.generate_report(weeks_back=1)
    print(weekly.format_report_text(report))

    # Generate monthly report
    from src.reporting import MonthlyReportGenerator
    monthly = MonthlyReportGenerator()
    report = monthly.generate_monthly_report(months_back=1)
    print(monthly.format_monthly_report_text(report))

    # Run reconciliation
    from src.reporting import ReconciliationEngine
    engine = ReconciliationEngine()
    report = engine.run_full_reconciliation(days_back=7)
    print(engine.format_reconciliation_report(report))
"""

from .discord_notifier import DiscordNotifier
from .weekly_report import WeeklyReportGenerator
from .monthly_report import MonthlyReportGenerator
from .reconciliation import ReconciliationEngine

__all__ = [
    'DiscordNotifier',
    'WeeklyReportGenerator',
    'MonthlyReportGenerator',
    'ReconciliationEngine',
]
