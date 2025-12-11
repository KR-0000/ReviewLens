# ReviewLens

Web application for analyzing a Amazon product reviews dataset. ReviewLens provides insights such as sentiment analysis, keyword extraction, and summaries.


### Backend
FastAPI, SQLite, NLTK (Vader sentiment analysis), spaCy for phrase extraction, Transformers (hugging face model for summarization)

### Frontend
React, Vite, CSS

## Installation

### Prerequisites
- Python 3.8+
- Node.js 16+ and npm
- Git

### Setup

#### Option 1: Automated Setup (Recommended)

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

#### Option 2: Manual Setup

1. **Install Backend Dependencies:**
```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cd ..
```

2. **Install Frontend Dependencies:**
```bash
cd frontend
npm install
cd ..
```

3. **Load Data (First Time Only):**
```bash
python load_data.py
python process_reviews.py
```

## Running the Application

### Development Mode

1. **Start Backend Server:**
```bash
python run_backend.py
```
Backend will run on `http://localhost:8000`

2. **Start Frontend Dev Server:**
```bash
cd frontend
npm run dev
```
Frontend will run on `http://localhost:3000`

3. **Access the Application:**
Open your browser and navigate to `http://localhost:3000`

### Production Mode

1. **Build Frontend:**
```bash
cd frontend
npm run build
```

2. **Run Backend with Production Settings:**
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

## Data Pipeline

1. **Load Data**: `load_data.py` loads CSV into SQLite
2. **Process Reviews**: `process_reviews.py` performs:
   - Sentiment analysis (VADER)
   - Keyword extraction (spaCy)
   - Summary generation (Transformers)
3. Display results


### Transformer Models
The application will automatically try to load transformer models in this order:
1. t5-small
2. facebook/bart-large-cnn
3. google/pegasus-xsum

If all fail, it falls back to extractive summarization.

### Database Issues
If you need to reload data:
```bash
python load_data.py
python process_reviews.py
```


