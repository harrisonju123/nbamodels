#!/bin/bash
# Quick test to verify dashboard is ready to run

echo "=== Dashboard Readiness Check ==="
echo ""

# Check Python version
echo "✓ Checking Python..."
~/.asdf/installs/python/3.12.4/bin/python --version

# Check dependencies
echo ""
echo "✓ Checking dependencies..."
~/.asdf/installs/python/3.12.4/bin/python -c "
import streamlit
import plotly
import fastapi
import uvicorn
import pydantic
print('  - streamlit:', streamlit.__version__)
print('  - plotly:', plotly.__version__)
print('  - fastapi:', fastapi.__version__)
print('  - uvicorn:', uvicorn.__version__)
print('  - pydantic:', pydantic.__version__)
"

# Check file structure
echo ""
echo "✓ Checking files..."
ls -lh analytics_dashboard.py api/dashboard_api.py DASHBOARD_README.md | awk '{print "  -", $9, "("$5")"}'

# Check imports
echo ""
echo "✓ Checking API imports..."
~/.asdf/installs/python/3.12.4/bin/python -c "
from api.dashboard_api import app
print('  - API module loads successfully')
"

echo ""
echo "=== Dashboard Ready! ==="
echo ""
echo "To run the dashboard:"
echo "  streamlit run analytics_dashboard.py"
echo ""
echo "To run the API:"
echo "  uvicorn api.dashboard_api:app --reload --port 8000"
echo ""
echo "For full documentation:"
echo "  cat DASHBOARD_README.md"
