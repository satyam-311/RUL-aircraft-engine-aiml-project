@echo off
REM Start the FastAPI inference endpoint (Phase 10)
REM Docs available at http://127.0.0.1:8000/docs once running
uv run uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
