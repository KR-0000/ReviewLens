# ReviewLens

A modern web application for analyzing Amazon product reviews using NLP and machine learning. ReviewLens provides comprehensive insights including sentiment analysis, keyword extraction, and AI-generated summaries.

## Features

- 🔍 **Product Search**: Search for products by name with autocomplete suggestions
- 📊 **Statistics**: View total reviews, average rating, and sentiment scores
- ✅ **Positive Keywords**: Top 5 positive keyword phrases extracted using spaCy
- ❌ **Negative Keywords**: Top 5 negative keyword phrases with sentiment scoring
- 📝 **AI Summaries**: Transformer-based abstractive summaries of review themes
- 🎨 **Modern UI**: Clean, responsive interface built with React

## Architecture

### Backend
- **FastAPI**: Modern, fast Python web framework
- **SQLite**: Database for storing reviews and processed insights
- **NLTK**: Natural language processing (VADER sentiment analysis)
- **spaCy**: Advanced NLP for phrase extraction (noun chunks, verb phrases)
- **Transformers**: Hugging Face models for abstractive summarization

### Frontend
- **React**: Modern UI framework
- **Vite**: Fast build tool and dev server
- **CSS**: Custom styling with responsive design

## Project Structure

```
ReviewLens/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py          # API endpoints
│   │   ├── models/
│   │   │   └── schemas.py         # Pydantic models
│   │   ├── services/
│   │   │   └── review_service.py  # Business logic
│   │   ├── config.py              # Configuration
│   │   └── main.py                # FastAPI app
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/            # React components
│   │   ├── App.jsx                # Main app component
│   │   └── main.jsx               # Entry point
│   ├── package.json
│   └── vite.config.js
├── data/
│   └── Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv
├── load_data.py                   # Data loading script
├── process_reviews.py             # Review processing
├── query_interface.py             # Query interface
├── run_backend.py                 # Backend server runner
└── README.md
```

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

## API Endpoints

### Health Check
```
GET /api/health
```

### Search Products
```
GET /api/products/search?q={search_term}
```

### Get Product Insights
```
GET /api/products/{product_name}/insights
```

### API Documentation
- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

## Data Pipeline

1. **Load Data**: `load_data.py` loads CSV into SQLite
2. **Process Reviews**: `process_reviews.py` performs:
   - Sentiment analysis (VADER)
   - Keyword extraction (spaCy)
   - Summary generation (Transformers)
3. **Query**: API endpoints serve processed insights

## Technologies Used

### Backend
- FastAPI
- SQLite
- NLTK (VADER sentiment)
- spaCy (phrase extraction)
- Transformers (Hugging Face)
- PyTorch

### Frontend
- React 18
- Vite
- Modern CSS

## Configuration

Environment variables (optional):
- `DATABASE_PATH`: Path to SQLite database (default: `reviews.db`)
- `CSV_PATH`: Path to CSV file
- `API_HOST`: Backend host (default: `0.0.0.0`)
- `API_PORT`: Backend port (default: `8000`)
- `DEBUG`: Enable debug mode (default: `False`)

## Troubleshooting

### spaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
```

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

## License

This project is for educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
