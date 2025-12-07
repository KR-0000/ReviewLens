@echo off
REM Setup script for ReviewLens (Windows)

echo Setting up ReviewLens...

REM Check Python
python --version || (echo Python 3 is required && exit /b 1)

REM Install backend dependencies
echo Installing backend dependencies...
cd backend
pip install -r requirements.txt
cd ..

REM Install spaCy model
echo Installing spaCy model...
python -m spacy download en_core_web_sm

REM Install frontend dependencies
echo Installing frontend dependencies...
cd frontend
call npm install
cd ..

echo Setup complete!
echo.
echo To run the application:
echo   1. Start backend: python run_backend.py
echo   2. Start frontend: cd frontend ^&^& npm run dev

