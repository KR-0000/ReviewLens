"""Configuration settings for the application."""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

# Database
DATABASE_PATH = os.getenv('DATABASE_PATH', str(BASE_DIR / 'reviews.db'))
CSV_PATH = os.getenv('CSV_PATH', str(BASE_DIR / 'data' / 'Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv'))

# API Settings
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# CORS Settings
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

