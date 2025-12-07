#!/bin/bash
# Setup script for ReviewLens

echo "Setting up ReviewLens..."

# Check Python version
python3 --version || { echo "Python 3 is required"; exit 1; }

# Install backend dependencies
echo "Installing backend dependencies..."
cd backend
pip install -r requirements.txt
cd ..

# Install spaCy model
echo "Installing spaCy model..."
python3 -m spacy download en_core_web_sm

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo "Setup complete!"
echo ""
echo "To run the application:"
echo "  1. Start backend: python run_backend.py"
echo "  2. Start frontend: cd frontend && npm run dev"
echo ""
echo "Or use: python run_all.py (if available)"

