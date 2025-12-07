"""
Main script to run the complete pipeline.
This script orchestrates the entire workflow: load data, process reviews, and query insights.
"""

import os
import sys
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import nltk
        import sqlite3
        return True
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False


def run_pipeline():
    """Run the complete pipeline."""
    print("=" * 80)
    print("Amazon Product Review Analysis - Complete Pipeline")
    print("=" * 80)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    csv_path = 'data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv'
    db_path = 'reviews.db'
    
    # Check if CSV exists
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
    
    # Step 1: Load data
    print("\n[Step 1/3] Loading CSV data into SQLite database...")
    print("-" * 80)
    from load_data import load_csv_to_db
    load_csv_to_db(csv_path, db_path)
    
    # Step 2: Process reviews
    print("\n[Step 2/3] Processing reviews (sentiment, keywords, summaries)...")
    print("-" * 80)
    from process_reviews import ReviewProcessor
    processor = ReviewProcessor(db_path)
    processor.process_all_reviews()
    processor.process_products()
    
    # Step 3: Query interface
    print("\n[Step 3/3] Starting query interface...")
    print("-" * 80)
    from query_interface import interactive_mode
    interactive_mode()


if __name__ == '__main__':
    run_pipeline()

