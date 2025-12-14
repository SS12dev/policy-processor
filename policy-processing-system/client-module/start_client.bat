@echo off
REM Startup script for Policy Processor Client
REM Starts both Streamlit UI and PDF Server

echo ================================================================================
echo Starting Policy Processor Client
echo ================================================================================
echo.

echo Starting PDF Server (port 8502)...
start "PDF Server" cmd /k "python pdf_server.py"

echo Waiting for PDF server to start...
timeout /t 3 /nobreak >nul

echo.
echo Starting Streamlit UI (port 8501)...
streamlit run app.py

echo.
echo ================================================================================
echo Client started successfully!
echo ================================================================================
echo.
echo Streamlit UI: http://localhost:8501
echo PDF Server:   http://localhost:8502
echo.
