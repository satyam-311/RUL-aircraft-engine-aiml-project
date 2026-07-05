@echo off
cd /d "%~dp0"
echo Starting Aircraft Engine Health Monitor dashboard...
echo Open your browser at http://localhost:8501
echo Press Ctrl+C to stop.
uv run streamlit run app/dashboard.py
pause
