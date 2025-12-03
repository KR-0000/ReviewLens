"""
ReviewLens: Aspect-Based Sentiment Analysis for Amazon Product Reviews
Authors: Rishabh Sheth, Kushagra Dhall, Joshua Dsouza

This starter code provides the foundation for:
1. Data loading and preprocessing
2. Aspect extraction from reviews
3. Aspect-based sentiment analysis
4. Database integration (SQL + MongoDB)
5. Visualization dashboard
"""

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy

# Database
import sqlite3
from pymongo import MongoClient

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    print("Please install spaCy model: python -m spacy download en_core_web_sm")
    nlp = None


# =====================================================================
# 1. DATA LOADING AND PREPROCESSING
# =====================================================================

class DataLoader:
    """Handles loading and initial preprocessing of Amazon reviews dataset"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        
    def load_data(self):
        """Load the CSV file"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.filepath)
        print(f"Loaded {len(self.df)} reviews")
        return self.df
    
    def clean_data(self):
        """Clean and preprocess the dataset"""
        print("Cleaning data...")
        
        # Remove duplicates
        initial_len = len(self.df)
        self.df = self.df.drop_duplicates(subset=['reviews.text'])
        print(f"Removed {initial_len - len(self.df)} duplicate reviews")
        
        # Remove rows with missing review text
        self.df = self.df.dropna(subset=['reviews.text'])
        
        # Convert rating to numeric
        if 'reviews.rating' in self.df.columns:
            self.df['reviews.rating'] = pd.to_numeric(self.df['reviews.rating'], errors='coerce')
        
        # Create a unique review ID
        self.df['review_id'] = range(len(self.df))
        
        return self.df
    
    def get_sample(self, n=1000):
        """Get a sample of reviews for testing"""
        return self.df.sample(n=min(n, len(self.df)))


class TextPreprocessor:
    """Handles text preprocessing for reviews"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Keep some sentiment-bearing words
        self.stop_words -= {'not', 'no', 'but', 'however', 'very', 'too'}
    
    def clean_text(self, text):
        """Basic text cleaning"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stopwords but keep important ones"""
        return [t for t in tokens if t not in self.stop_words]


# =====================================================================
# 2. ASPECT EXTRACTION
# =====================================================================

class AspectExtractor:
    """Extracts product aspects from reviews using multiple methods"""
    
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        
        # Predefined aspect categories for electronics (customize per product category)
        self.aspect_keywords = {
            'battery': ['battery', 'charge', 'charging', 'power'],
            'quality': ['quality', 'build', 'material', 'construction'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'worth'],
            'sound': ['sound', 'audio', 'volume', 'speaker', 'noise'],
            'design': ['design', 'look', 'appearance', 'style', 'color'],
            'screen': ['screen', 'display', 'resolution', 'brightness'],
            'performance': ['performance', 'speed', 'fast', 'slow', 'lag'],
            'durability': ['durability', 'durable', 'sturdy', 'fragile', 'break'],
            'size': ['size', 'big', 'small', 'large', 'compact', 'weight'],
            'ease_of_use': ['easy', 'simple', 'complicated', 'difficult', 'intuitive']
        }
    
    def extract_noun_phrases(self, text):
        """Extract noun phrases using spaCy"""
        if self.nlp is None:
            return []
        
        doc = self.nlp(text)
        noun_phrases = []
        
        for chunk in doc.noun_chunks:
            # Filter out very short or very long phrases
            if 1 <= len(chunk.text.split()) <= 3:
                noun_phrases.append(chunk.text.lower())
        
        return noun_phrases
    
    def identify_aspects(self, text):
        """Identify which predefined aspects are mentioned in the text"""
        text_lower = text.lower()
        found_aspects = []
        
        for aspect, keywords in self.aspect_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_aspects.append(aspect)
        
        return found_aspects
    
    def extract_aspect_sentences(self, text, aspect):
        """Extract sentences that mention a specific aspect"""
        sentences = sent_tokenize(text)
        aspect_sentences = []
        
        keywords = self.aspect_keywords.get(aspect, [])
        
        for sent in sentences:
            if any(keyword in sent.lower() for keyword in keywords):
                aspect_sentences.append(sent)
        
        return aspect_sentences


# =====================================================================
# 3. SENTIMENT ANALYSIS
# =====================================================================

class SentimentAnalyzer:
    """Performs sentiment analysis on reviews and aspects"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, text):
        """Get sentiment scores using VADER"""
        scores = self.sia.polarity_scores(text)
        return scores
    
    def get_overall_sentiment(self, text):
        """Get simple positive/negative/neutral label"""
        scores = self.analyze_sentiment(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_aspect_sentiment(self, aspect_sentences):
        """Analyze sentiment for a specific aspect"""
        if not aspect_sentences:
            return None
        
        # Average sentiment across all sentences mentioning the aspect
        sentiments = [self.analyze_sentiment(sent)['compound'] for sent in aspect_sentences]
        avg_sentiment = np.mean(sentiments)
        
        return {
            'compound': avg_sentiment,
            'num_mentions': len(aspect_sentences),
            'label': 'positive' if avg_sentiment >= 0.05 else ('negative' if avg_sentiment <= -0.05 else 'neutral')
        }


# =====================================================================
# 4. ASPECT-BASED SENTIMENT ANALYSIS (ABSA)
# =====================================================================

class AspectBasedAnalyzer:
    """Combines aspect extraction and sentiment analysis"""
    
    def __init__(self, nlp_model):
        self.aspect_extractor = AspectExtractor(nlp_model)
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def analyze_review(self, review_text):
        """Perform complete ABSA on a single review"""
        # Extract aspects mentioned
        aspects = self.aspect_extractor.identify_aspects(review_text)
        
        # Get overall sentiment
        overall_sentiment = self.sentiment_analyzer.analyze_sentiment(review_text)
        
        # Analyze sentiment for each aspect
        aspect_sentiments = {}
        for aspect in aspects:
            aspect_sentences = self.aspect_extractor.extract_aspect_sentences(review_text, aspect)
            sentiment = self.sentiment_analyzer.analyze_aspect_sentiment(aspect_sentences)
            if sentiment:
                aspect_sentiments[aspect] = sentiment
        
        return {
            'overall_sentiment': overall_sentiment,
            'aspects': aspect_sentiments
        }
    
    def analyze_product(self, reviews_df):
        """Analyze all reviews for a product and aggregate"""
        product_analysis = defaultdict(lambda: {'sentiments': [], 'mentions': 0})
        
        for _, row in reviews_df.iterrows():
            review_analysis = self.analyze_review(row['reviews.text'])
            
            for aspect, sentiment_data in review_analysis['aspects'].items():
                product_analysis[aspect]['sentiments'].append(sentiment_data['compound'])
                product_analysis[aspect]['mentions'] += sentiment_data['num_mentions']
        
        # Aggregate results
        aspect_summary = {}
        for aspect, data in product_analysis.items():
            aspect_summary[aspect] = {
                'avg_sentiment': np.mean(data['sentiments']),
                'total_mentions': data['mentions'],
                'num_reviews': len(data['sentiments'])
            }
        
        return aspect_summary


# =====================================================================
# 5. DATABASE INTEGRATION
# =====================================================================

class DatabaseManager:
    """Manages SQL and MongoDB databases"""
    
    def __init__(self, sql_db='reviewlens.db', mongo_uri='mongodb://localhost:27017/', mongo_db='reviewlens'):
        self.sql_db = sql_db
        self.mongo_client = None
        self.mongo_db = None
        
        # Initialize SQL
        self.init_sql()
        
        # Try to initialize MongoDB (optional)
        try:
            self.mongo_client = MongoClient(mongo_uri)
            self.mongo_db = self.mongo_client[mongo_db]
            print("MongoDB connected")
        except:
            print("MongoDB not available - using SQL only")
    
    def init_sql(self):
        """Initialize SQL database schema"""
        conn = sqlite3.connect(self.sql_db)
        cursor = conn.cursor()
        
        # Products table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                product_id TEXT PRIMARY KEY,
                name TEXT,
                brand TEXT,
                category TEXT,
                avg_rating REAL
            )
        ''')
        
        # Reviews table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                review_id INTEGER PRIMARY KEY,
                product_id TEXT,
                rating REAL,
                date TEXT,
                overall_sentiment REAL,
                sentiment_label TEXT,
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            )
        ''')
        
        # Aspects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS aspects (
                aspect_id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT,
                aspect_name TEXT,
                avg_sentiment REAL,
                total_mentions INTEGER,
                num_reviews INTEGER,
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("SQL database initialized")
    
    def insert_product(self, product_data):
        """Insert product metadata into SQL"""
        conn = sqlite3.connect(self.sql_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO products (product_id, name, brand, category, avg_rating)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            product_data['product_id'],
            product_data.get('name', ''),
            product_data.get('brand', ''),
            product_data.get('category', ''),
            product_data.get('avg_rating', None)
        ))
        
        conn.commit()
        conn.close()
    
    def insert_review(self, review_data):
        """Insert review into SQL and MongoDB"""
        # SQL: structured data
        conn = sqlite3.connect(self.sql_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO reviews (review_id, product_id, rating, date, overall_sentiment, sentiment_label)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            review_data['review_id'],
            review_data['product_id'],
            review_data.get('rating', None),
            review_data.get('date', ''),
            review_data.get('overall_sentiment', None),
            review_data.get('sentiment_label', '')
        ))
        
        conn.commit()
        conn.close()
        
        # MongoDB: full text and analysis
        if self.mongo_db is not None:
            self.mongo_db.reviews.insert_one(review_data)
    
    def insert_aspects(self, product_id, aspect_summary):
        """Insert aspect analysis results"""
        conn = sqlite3.connect(self.sql_db)
        cursor = conn.cursor()
        
        for aspect_name, data in aspect_summary.items():
            cursor.execute('''
                INSERT INTO aspects (product_id, aspect_name, avg_sentiment, total_mentions, num_reviews)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                product_id,
                aspect_name,
                data['avg_sentiment'],
                data['total_mentions'],
                data['num_reviews']
            ))
        
        conn.commit()
        conn.close()
    
    def get_product_aspects(self, product_id):
        """Retrieve aspect analysis for a product"""
        conn = sqlite3.connect(self.sql_db)
        query = '''
            SELECT aspect_name, avg_sentiment, total_mentions, num_reviews
            FROM aspects
            WHERE product_id = ?
            ORDER BY total_mentions DESC
        '''
        df = pd.read_sql_query(query, conn, params=(product_id,))
        conn.close()
        return df


# =====================================================================
# 6. VISUALIZATION
# =====================================================================

class Visualizer:
    """Creates visualizations for review analysis"""
    
    def __init__(self):
        sns.set_style('whitegrid')
    
    def plot_aspect_sentiments(self, aspect_df):
        """Plot sentiment scores for each aspect"""
        if aspect_df.empty:
            print("No data to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort by mentions
        aspect_df = aspect_df.sort_values('total_mentions', ascending=True)
        
        # Create horizontal bar chart
        colors = ['red' if x < -0.05 else 'green' if x > 0.05 else 'gray' 
                  for x in aspect_df['avg_sentiment']]
        
        ax.barh(aspect_df['aspect_name'], aspect_df['avg_sentiment'], color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Average Sentiment Score', fontsize=12)
        ax.set_ylabel('Aspect', fontsize=12)
        ax.set_title('Product Aspect Sentiment Analysis', fontsize=14, fontweight='bold')
        ax.set_xlim(-1, 1)
        
        plt.tight_layout()
        plt.show()
    
    def plot_sentiment_distribution(self, reviews_df):
        """Plot distribution of review sentiments"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Rating distribution
        if 'reviews.rating' in reviews_df.columns:
            reviews_df['reviews.rating'].value_counts().sort_index().plot(kind='bar', ax=ax1, color='steelblue')
            ax1.set_title('Rating Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Rating', fontsize=12)
            ax1.set_ylabel('Count', fontsize=12)
        
        # Sentiment distribution
        if 'sentiment_label' in reviews_df.columns:
            reviews_df['sentiment_label'].value_counts().plot(kind='bar', ax=ax2, 
                                                               color=['green', 'gray', 'red'])
            ax2.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Sentiment', fontsize=12)
            ax2.set_ylabel('Count', fontsize=12)
        
        plt.tight_layout()
        plt.show()


# =====================================================================
# 7. MAIN PIPELINE
# =====================================================================

class ReviewLensPipeline:
    """Main pipeline that orchestrates the entire analysis"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_loader = DataLoader(data_path)
        self.preprocessor = TextPreprocessor()
        self.absa_analyzer = AspectBasedAnalyzer(nlp)
        self.db_manager = DatabaseManager()
        self.visualizer = Visualizer()
        self.df = None
    
    def run(self, sample_size=None):
        """Run the complete pipeline"""
        print("="*70)
        print("REVIEWLENS: ASPECT-BASED SENTIMENT ANALYSIS")
        print("="*70)
        
        # Step 1: Load and clean data
        self.df = self.data_loader.load_data()
        self.df = self.data_loader.clean_data()
        
        if sample_size:
            self.df = self.df.sample(n=min(sample_size, len(self.df)))
            print(f"\nUsing sample of {len(self.df)} reviews for analysis")
        
        # Step 2: Preprocess text
        print("\nPreprocessing text...")
        self.df['cleaned_text'] = self.df['reviews.text'].apply(self.preprocessor.clean_text)
        
        # Step 3: Analyze sentiments
        print("Analyzing sentiments...")
        sentiment_results = []
        for text in self.df['cleaned_text']:
            sentiment = self.absa_analyzer.sentiment_analyzer.get_overall_sentiment(text)
            sentiment_results.append(sentiment)
        self.df['sentiment_label'] = sentiment_results
        
        # Step 4: Analyze by product
        print("\nAnalyzing products...")
        if 'id' in self.df.columns:
            products = self.df['id'].unique()[:5]  # Analyze first 5 products
            
            for product_id in products:
                print(f"\n  Processing product: {product_id}")
                product_reviews = self.df[self.df['id'] == product_id]
                
                # Run ABSA
                aspect_summary = self.absa_analyzer.analyze_product(product_reviews)
                
                # Store in database
                self.db_manager.insert_aspects(product_id, aspect_summary)
                
                # Visualize
                aspect_df = self.db_manager.get_product_aspects(product_id)
                if not aspect_df.empty:
                    print(f"\n  Top aspects for product {product_id}:")
                    print(aspect_df.head())
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        
        return self.df
    
    def visualize_results(self, product_id=None):
        """Visualize results for a specific product or overall"""
        if product_id:
            aspect_df = self.db_manager.get_product_aspects(product_id)
            self.visualizer.plot_aspect_sentiments(aspect_df)
        
        if self.df is not None:
            self.visualizer.plot_sentiment_distribution(self.df)


# =====================================================================
# USAGE EXAMPLE
# =====================================================================

if __name__ == "__main__":
    # Initialize pipeline
    # Replace with your actual file path


    DATA_PATH = "data/raw_kaggle_data.csv"
    
    # Create pipeline
    pipeline = ReviewLensPipeline(DATA_PATH)
    
    # Run analysis on a sample (use sample_size=None for full dataset)
    results = pipeline.run(sample_size=500)
    
    # Visualize results
    pipeline.visualize_results()
    
    print("\n✅ Pipeline complete! Check the 'reviewlens.db' database for results.")