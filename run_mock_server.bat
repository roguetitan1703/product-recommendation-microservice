@echo off
echo Starting Mock Recommendation Server...
echo.
echo This is a simple mock server for testing without dependencies
echo.
cd %~dp0
py -m pip install flask
py mock_server.py
echo.
pause 