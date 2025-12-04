@echo off
echo ========================================
echo Starting Prior Authorization Agents
echo ========================================
echo.

echo Installing monitoring requirements...
pip install psutil --quiet
echo.

echo Starting Policy Analysis Agent (Port 10001)...
start "Policy Analysis Agent" cmd /k "cd agents\policy-analysis-agent\policy_analysis_agent && python main.py --host localhost --port 10001"
timeout /t 3 /nobreak > nul

echo Starting Application Processing Agent (Port 10002)...
start "Application Processing Agent" cmd /k "cd agents\application-processing-agent\application_processing_agent && python main.py --host localhost --port 10002"
timeout /t 3 /nobreak > nul

echo Starting Decision Making Agent (Port 10003)...
start "Decision Making Agent" cmd /k "cd agents\decision-making-agent\decision_making_agent && python main.py --host localhost --port 10003"
timeout /t 5 /nobreak > nul

echo.
echo ========================================
echo All agents started with memory optimization!
echo.
echo Policy Analysis Agent: http://localhost:10001/.well-known/agent
echo Application Processing Agent: http://localhost:10002/.well-known/agent
echo Decision Making Agent: http://localhost:10003/.well-known/agent
echo ========================================
echo.

echo Starting Real-Time Agent Monitor...
echo This will show live CPU and memory usage for all agents.
echo Press Ctrl+C in the monitor window to stop monitoring.
echo.
start "Agent Monitor" cmd /k "python monitor_agents.py --refresh 2 --save"

echo.
echo ========================================
echo System Ready!
echo - All 3 agents are running
echo - Real-time monitor is active
echo - Memory optimization is enabled
echo ========================================
echo.
echo Press any key to exit this launcher...
pause > nul