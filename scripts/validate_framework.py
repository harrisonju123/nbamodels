"""
Multi-Strategy Framework Validation

Runs comprehensive tests and generates a validation report.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from datetime import datetime
from loguru import logger

# Import test modules
from tests import test_strategies, test_orchestrator, test_database


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def check_imports():
    """Check that all required modules can be imported."""
    print_section("1. IMPORT CHECKS")

    imports_to_check = [
        ("Base Strategy Classes", "src.betting.strategies.base", ["BettingStrategy", "BetSignal", "StrategyType"]),
        ("Totals Strategy", "src.betting.strategies.totals_strategy", ["TotalsStrategy"]),
        ("Live Strategy", "src.betting.strategies.live_strategy", ["LiveBettingStrategy"]),
        ("Arbitrage Strategy", "src.betting.strategies.arbitrage_strategy", ["ArbitrageStrategy"]),
        ("Player Props Strategy", "src.betting.strategies.player_props_strategy", ["PlayerPropsStrategy"]),
        ("Orchestrator", "src.betting.orchestrator", ["StrategyOrchestrator", "OrchestratorConfig"]),
        ("Player Prop Models", "src.models.player_props", ["BasePlayerPropModel"]),
        ("Odds API", "src.data.odds_api", ["OddsAPIClient"]),
    ]

    results = []
    for name, module_path, classes in imports_to_check:
        try:
            module = __import__(module_path, fromlist=classes)
            for cls in classes:
                if not hasattr(module, cls):
                    raise ImportError(f"{cls} not found")
            print(f"✓ {name}: OK")
            results.append((name, True, None))
        except Exception as e:
            print(f"✗ {name}: FAILED - {e}")
            results.append((name, False, str(e)))

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    print(f"\nImport Checks: {passed}/{total} passed")

    return results


def check_file_structure():
    """Check that all required files exist."""
    print_section("2. FILE STRUCTURE CHECKS")

    files_to_check = [
        # Strategies
        ("strategies/base.py", "src/betting/strategies/base.py"),
        ("strategies/totals_strategy.py", "src/betting/strategies/totals_strategy.py"),
        ("strategies/live_strategy.py", "src/betting/strategies/live_strategy.py"),
        ("strategies/arbitrage_strategy.py", "src/betting/strategies/arbitrage_strategy.py"),
        ("strategies/player_props_strategy.py", "src/betting/strategies/player_props_strategy.py"),
        # Orchestrator
        ("orchestrator.py", "src/betting/orchestrator.py"),
        # Models
        ("player_props/base_prop_model.py", "src/models/player_props/base_prop_model.py"),
        ("player_props/points_model.py", "src/models/player_props/points_model.py"),
        ("player_props/rebounds_model.py", "src/models/player_props/rebounds_model.py"),
        ("player_props/assists_model.py", "src/models/player_props/assists_model.py"),
        ("player_props/threes_model.py", "src/models/player_props/threes_model.py"),
        # Tests
        ("tests/test_strategies.py", "tests/test_strategies.py"),
        ("tests/test_orchestrator.py", "tests/test_orchestrator.py"),
        ("tests/test_database.py", "tests/test_database.py"),
        # Documentation
        ("MULTI_STRATEGY_FRAMEWORK.md", "MULTI_STRATEGY_FRAMEWORK.md"),
    ]

    results = []
    for name, path in files_to_check:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {'OK' if exists else 'MISSING'}")
        results.append((name, exists))

    passed = sum(1 for _, exists in results if exists)
    total = len(results)
    print(f"\nFile Structure: {passed}/{total} files found")

    return results


def run_unit_tests():
    """Run all unit tests."""
    print_section("3. UNIT TESTS")

    test_modules = [
        ("Strategy Tests", test_strategies),
        ("Orchestrator Tests", test_orchestrator),
        ("Database Tests", test_database),
    ]

    all_results = []

    for name, module in test_modules:
        print(f"\n--- {name} ---")
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)

        all_results.append((name, result))

        print(f"  Tests run: {result.testsRun}")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Success: {result.wasSuccessful()}")

    return all_results


def check_models_exist():
    """Check if trained models exist."""
    print_section("4. MODEL AVAILABILITY")

    models_to_check = [
        ("Totals Model", "models/totals_model.pkl"),
        ("Points Prop Model", "models/player_props/pts_model.pkl"),
        ("Rebounds Prop Model", "models/player_props/reb_model.pkl"),
        ("Assists Prop Model", "models/player_props/ast_model.pkl"),
        ("Threes Prop Model", "models/player_props/3pm_model.pkl"),
    ]

    results = []
    for name, path in models_to_check:
        exists = os.path.exists(path)
        status = "✓" if exists else "⚠"
        message = "Found" if exists else "Not trained (optional)"
        print(f"{status} {name}: {message}")
        results.append((name, exists))

    return results


def generate_validation_report(
    import_results,
    file_results,
    test_results,
    model_results
):
    """Generate a comprehensive validation report."""
    print_section("VALIDATION REPORT")

    # Summary
    import_passed = sum(1 for _, success, _ in import_results if success)
    import_total = len(import_results)

    file_passed = sum(1 for _, exists in file_results if exists)
    file_total = len(file_results)

    test_passed = sum(1 for _, result in test_results if result.wasSuccessful())
    test_total = len(test_results)

    models_found = sum(1 for _, exists in model_results if exists)
    models_total = len(model_results)

    print(f"Imports:        {import_passed}/{import_total} passed")
    print(f"Files:          {file_passed}/{file_total} found")
    print(f"Test Suites:    {test_passed}/{test_total} passed")
    print(f"Models:         {models_found}/{models_total} trained")

    # Overall status
    print("\n" + "-" * 70)

    core_ready = (
        import_passed == import_total and
        file_passed == file_total and
        test_passed == test_total
    )

    if core_ready:
        print("✅ FRAMEWORK STATUS: READY FOR PRODUCTION")
        print("\nCore infrastructure is fully validated and ready to use.")

        if models_found == models_total:
            print("✅ All models trained - full feature set available")
        elif models_found > 0:
            print(f"⚠️  Some models missing ({models_found}/{models_total} trained)")
            print("   Player props will have limited functionality")
        else:
            print("⚠️  No models trained")
            print("   Run training scripts to enable predictions")

        print("\nNext Steps:")
        print("  1. Integrate with existing orchestrator/dashboard")
        print("  2. Enable desired strategies in production config")
        print("  3. Monitor performance via Discord reports")

    else:
        print("❌ FRAMEWORK STATUS: ISSUES DETECTED")

        if import_passed < import_total:
            print(f"\n⚠️  {import_total - import_passed} import failures detected")
            print("   Check error messages above")

        if file_passed < file_total:
            print(f"\n⚠️  {file_total - file_passed} files missing")
            print("   Re-run implementation to create missing files")

        if test_passed < test_total:
            print(f"\n⚠️  {test_total - test_passed} test suites failed")
            print("   Review test output above for details")

    print("-" * 70)

    # Save report to file
    report_path = "VALIDATION_REPORT.txt"
    with open(report_path, 'w') as f:
        f.write("Multi-Strategy Framework Validation Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Imports:        {import_passed}/{import_total} passed\n")
        f.write(f"Files:          {file_passed}/{file_total} found\n")
        f.write(f"Test Suites:    {test_passed}/{test_total} passed\n")
        f.write(f"Models:         {models_found}/{models_total} trained\n\n")

        f.write(f"Status: {'READY' if core_ready else 'ISSUES DETECTED'}\n")

    print(f"\nReport saved to {report_path}")

    return core_ready


def main():
    """Run full validation suite."""
    print("\n" + "=" * 70)
    print("  MULTI-STRATEGY FRAMEWORK VALIDATION")
    print("=" * 70)

    try:
        # Run all checks
        import_results = check_imports()
        file_results = check_file_structure()
        test_results = run_unit_tests()
        model_results = check_models_exist()

        # Generate report
        ready = generate_validation_report(
            import_results,
            file_results,
            test_results,
            model_results
        )

        # Exit with appropriate code
        sys.exit(0 if ready else 1)

    except Exception as e:
        logger.error(f"Validation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
