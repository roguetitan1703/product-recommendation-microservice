@echo off
echo Starting Recommendation Service...
echo.
echo This will start the Flask application for product recommendations
echo If you see any errors related to missing modules, make sure to install them:
echo   py -m pip install flask flask-cors redis python-dotenv
echo.
cd %~dp0
set PYTHONPATH=%~dp0..
py flask_app.py
echo.
echo Press any key to exit...
pause > nul 