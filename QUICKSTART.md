# Quick Start Guide

## Prerequisites
- Python 3.8+
- Node.js 16+ and npm

## Quick Setup

1. **Install Dependencies:**

   Backend:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

   Frontend:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

2. **Load Data (First Time Only):**
   ```bash
   python load_data.py
   python process_reviews.py
   ```

3. **Run the Application:**

   Terminal 1 - Backend:
   ```bash
   python run_backend.py
   ```

   Terminal 2 - Frontend:
   ```bash
   cd frontend
   npm run dev
   ```

4. **Access:**
   - Frontend: http://localhost:3000
   - API Docs: http://localhost:8000/api/docs

## That's it!

The application is now running. Search for a product name to see insights!

