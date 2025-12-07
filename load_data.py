"""
Load CSV data into SQLite database.
This script reads the Amazon reviews CSV and stores it in a SQLite database.
"""

import sqlite3
import csv
import os
from pathlib import Path


def create_database(db_path='reviews.db'):
    """Create SQLite database and reviews table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop table if exists
    cursor.execute('DROP TABLE IF EXISTS reviews')
    
    # Create table with all columns from CSV
    create_table_sql = """
    CREATE TABLE reviews (
        id TEXT,
        dateAdded TEXT,
        dateUpdated TEXT,
        name TEXT,
        asins TEXT,
        brand TEXT,
        categories TEXT,
        primaryCategories TEXT,
        imageURLs TEXT,
        keys TEXT,
        manufacturer TEXT,
        manufacturerNumber TEXT,
        reviews_date TEXT,
        reviews_dateSeen TEXT,
        reviews_didPurchase TEXT,
        reviews_doRecommend TEXT,
        reviews_id TEXT,
        reviews_numHelpful TEXT,
        reviews_rating TEXT,
        reviews_sourceURLs TEXT,
        reviews_text TEXT,
        reviews_title TEXT,
        reviews_username TEXT,
        sourceURLs TEXT,
        sentiment_score REAL,
        sentiment_label TEXT,
        processed INTEGER DEFAULT 0
    )
    """
    
    cursor.execute(create_table_sql)
    
    # Create index for faster lookups
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_name ON reviews(name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed ON reviews(processed)')
    
    conn.commit()
    return conn, cursor


def load_csv_to_db(csv_path, db_path='reviews.db', batch_size=1000):
    """Load CSV data into SQLite database in batches."""
    conn, cursor = create_database(db_path)
    
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Loading data from {csv_path}...")
    
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        
        batch = []
        total_rows = 0
        
        for row in reader:
            # Convert column names with dots to underscores for SQLite
            processed_row = {
                'id': row.get('id', ''),
                'dateAdded': row.get('dateAdded', ''),
                'dateUpdated': row.get('dateUpdated', ''),
                'name': row.get('name', ''),
                'asins': row.get('asins', ''),
                'brand': row.get('brand', ''),
                'categories': row.get('categories', ''),
                'primaryCategories': row.get('primaryCategories', ''),
                'imageURLs': row.get('imageURLs', ''),
                'keys': row.get('keys', ''),
                'manufacturer': row.get('manufacturer', ''),
                'manufacturerNumber': row.get('manufacturerNumber', ''),
                'reviews_date': row.get('reviews.date', ''),
                'reviews_dateSeen': row.get('reviews.dateSeen', ''),
                'reviews_didPurchase': row.get('reviews.didPurchase', ''),
                'reviews_doRecommend': row.get('reviews.doRecommend', ''),
                'reviews_id': row.get('reviews.id', ''),
                'reviews_numHelpful': row.get('reviews.numHelpful', ''),
                'reviews_rating': row.get('reviews.rating', ''),
                'reviews_sourceURLs': row.get('reviews.sourceURLs', ''),
                'reviews_text': row.get('reviews.text', ''),
                'reviews_title': row.get('reviews.title', ''),
                'reviews_username': row.get('reviews.username', ''),
                'sourceURLs': row.get('sourceURLs', '')
            }
            
            batch.append(processed_row)
            
            if len(batch) >= batch_size:
                insert_batch(cursor, batch)
                total_rows += len(batch)
                print(f"Loaded {total_rows} rows...")
                batch = []
        
        # Insert remaining rows
        if batch:
            insert_batch(cursor, batch)
            total_rows += len(batch)
    
    conn.commit()
    print(f"Successfully loaded {total_rows} rows into database.")
    
    # Print some statistics
    cursor.execute('SELECT COUNT(*) FROM reviews')
    count = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(DISTINCT name) FROM reviews')
    unique_products = cursor.fetchone()[0]
    
    print(f"\nDatabase Statistics:")
    print(f"  Total reviews: {count}")
    print(f"  Unique products: {unique_products}")
    
    conn.close()
    return db_path


def insert_batch(cursor, batch):
    """Insert a batch of rows into the database."""
    insert_sql = """
    INSERT INTO reviews (
        id, dateAdded, dateUpdated, name, asins, brand, categories,
        primaryCategories, imageURLs, keys, manufacturer, manufacturerNumber,
        reviews_date, reviews_dateSeen, reviews_didPurchase, reviews_doRecommend,
        reviews_id, reviews_numHelpful, reviews_rating, reviews_sourceURLs,
        reviews_text, reviews_title, reviews_username, sourceURLs
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    for row in batch:
        cursor.execute(insert_sql, (
            row['id'], row['dateAdded'], row['dateUpdated'], row['name'],
            row['asins'], row['brand'], row['categories'], row['primaryCategories'],
            row['imageURLs'], row['keys'], row['manufacturer'], row['manufacturerNumber'],
            row['reviews_date'], row['reviews_dateSeen'], row['reviews_didPurchase'],
            row['reviews_doRecommend'], row['reviews_id'], row['reviews_numHelpful'],
            row['reviews_rating'], row['reviews_sourceURLs'], row['reviews_text'],
            row['reviews_title'], row['reviews_username'], row['sourceURLs']
        ))


if __name__ == '__main__':
    csv_path = 'data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv'
    db_path = 'reviews.db'
    
    print("=" * 60)
    print("Loading CSV data into SQLite database")
    print("=" * 60)
    
    load_csv_to_db(csv_path, db_path)
    
    print("\nData loading complete!")

