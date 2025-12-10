"""
Load CSV data into SQLite database.
This script reads the Amazon reviews CSV and stores it in a SQLite database.
"""

import sqlite3
import csv
import os
from pathlib import Path


# -------------------------------------------------------------
# 1. DATABASE CREATION
# -------------------------------------------------------------
def create_database(cursor):
    """Create the base 'reviews' table."""
    cursor.execute("DROP TABLE IF EXISTS reviews")

    cursor.execute("""
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
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON reviews(name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed ON reviews(processed)")


# -------------------------------------------------------------
# 2. CSV LOADING
# -------------------------------------------------------------
def load_csv_to_db(csv_path, cursor, batch_size=1000):
    """Load CSV rows into the reviews table."""

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Loading data from {csv_path}...")

    insert_sql = """
        INSERT INTO reviews (
            id, dateAdded, dateUpdated, name, asins, brand, categories,
            primaryCategories, imageURLs, keys, manufacturer, manufacturerNumber,
            reviews_date, reviews_dateSeen, reviews_didPurchase, reviews_doRecommend,
            reviews_id, reviews_numHelpful, reviews_rating, reviews_sourceURLs,
            reviews_text, reviews_title, reviews_username, sourceURLs
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    batch = []
    total_rows = 0

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)

        for row in reader:
            processed_row = [
                row.get("id", ""),
                row.get("dateAdded", ""),
                row.get("dateUpdated", ""),
                row.get("name", ""),
                row.get("asins", ""),
                row.get("brand", ""),
                row.get("categories", ""),
                row.get("primaryCategories", ""),
                row.get("imageURLs", ""),
                row.get("keys", ""),
                row.get("manufacturer", ""),
                row.get("manufacturerNumber", ""),
                row.get("reviews.date", ""),
                row.get("reviews.dateSeen", ""),
                row.get("reviews.didPurchase", ""),
                row.get("reviews.doRecommend", ""),
                row.get("reviews.id", ""),
                row.get("reviews.numHelpful", ""),
                row.get("reviews.rating", ""),
                row.get("reviews.sourceURLs", ""),
                row.get("reviews.text", ""),
                row.get("reviews.title", ""),
                row.get("reviews.username", ""),
                row.get("sourceURLs", "")
            ]

            batch.append(processed_row)

            if len(batch) >= batch_size:
                cursor.executemany(insert_sql, batch)
                total_rows += len(batch)
                print(f"Loaded {total_rows} rows...")
                batch = []

    # leftover rows
    if batch:
        cursor.executemany(insert_sql, batch)
        total_rows += len(batch)

    print(f"Successfully loaded {total_rows} rows\n")

    cursor.execute("SELECT COUNT(*), COUNT(DISTINCT name) FROM reviews")
    total, unique = cursor.fetchone()

    print("Database Stats:")
    print(f"  Total reviews: {total}")
    print(f"  Unique products: {unique}\n")


# -------------------------------------------------------------
# 3. NORMALIZED TABLES
# -------------------------------------------------------------
def create_normalized_tables(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            brand TEXT,
            manufacturer TEXT,
            manufacturerNumber TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews_normalized (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            user_id INTEGER,
            rating REAL,
            text TEXT,
            title TEXT,
            date TEXT,
            didPurchase TEXT,
            numHelpful TEXT,
            FOREIGN KEY(product_id) REFERENCES products(product_id),
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS product_categories (
            product_id INTEGER,
            category_id INTEGER,
            PRIMARY KEY(product_id, category_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            url TEXT,
            UNIQUE(product_id, url)
        )
    """)


# -------------------------------------------------------------
# 4. NORMALIZATION PROCESS
# -------------------------------------------------------------
def normalize_data(cursor):
    print("Normalizing...")

    cursor.execute("SELECT * FROM reviews")
    rows = cursor.fetchall()
    total = len(rows)

    for i, row in enumerate(rows):
        (
            id, dateAdded, dateUpdated, name, asins, brand, categories,
            primaryCategories, imageURLs, keys, manufacturer, manufacturerNumber,
            reviews_date, reviews_dateSeen, reviews_didPurchase, reviews_doRecommend,
            reviews_id, reviews_numHelpful, reviews_rating, reviews_sourceURLs,
            reviews_text, reviews_title, reviews_username, sourceURLs,
            sentiment_score, sentiment_label, processed
        ) = row

        # PRODUCTS
        cursor.execute(
            "SELECT product_id FROM products WHERE name=? AND brand=?",
            (name, brand)
        )
        res = cursor.fetchone()

        if res:
            product_id = res[0]
        else:
            cursor.execute("""
                INSERT INTO products (name, brand, manufacturer, manufacturerNumber)
                VALUES (?, ?, ?, ?)
            """, (name, brand, manufacturer, manufacturerNumber))
            product_id = cursor.lastrowid

        # USERS
        cursor.execute("SELECT user_id FROM users WHERE username=?", (reviews_username,))
        user_row = cursor.fetchone()

        if user_row:
            user_id = user_row[0]
        else:
            cursor.execute("INSERT INTO users (username) VALUES (?)", (reviews_username,))
            user_id = cursor.lastrowid

        # REVIEWS NORMALIZED
        cursor.execute("""
            INSERT INTO reviews_normalized
            (product_id, user_id, rating, text, title, date, didPurchase, numHelpful)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            product_id, user_id, reviews_rating, reviews_text,
            reviews_title, reviews_date, reviews_didPurchase, reviews_numHelpful
        ))

        # CATEGORIES
        if categories:
            for cat in categories.split(","):
                cat = cat.strip()
                if not cat:
                    continue

                cursor.execute("SELECT category_id FROM categories WHERE category_name=?", (cat,))
                cat_row = cursor.fetchone()

                if cat_row:
                    category_id = cat_row[0]
                else:
                    cursor.execute("INSERT INTO categories (category_name) VALUES (?)", (cat,))
                    category_id = cursor.lastrowid

                cursor.execute("""
                    INSERT OR IGNORE INTO product_categories (product_id, category_id)
                    VALUES (?, ?)
                """, (product_id, category_id))

        # IMAGES
        if imageURLs:
            for url in imageURLs.split(","):
                url = url.strip()
                if url:
                    cursor.execute("""
                        INSERT OR IGNORE INTO images (product_id, url)
                        VALUES (?, ?)
                    """, (product_id, url))

        if i % 5000 == 0:
            print(f"{i}/{total} normalized...")

    print("Normalization complete!\n")


# -------------------------------------------------------------
# 5. RECREATED VIEW
# -------------------------------------------------------------
def create_reviews_view(cursor):
    cursor.execute("DROP VIEW IF EXISTS reviews_reconstructed")

    cursor.execute("""
        CREATE VIEW reviews_reconstructed AS
        SELECT 
            rn.review_id,
            p.name,
            p.brand,
            p.manufacturer,
            p.manufacturerNumber,
            u.username AS reviews_username,
            rn.rating AS reviews_rating,
            rn.text AS reviews_text,
            rn.title AS reviews_title,
            rn.date AS reviews_date,
            rn.didPurchase AS reviews_didPurchase,
            rn.numHelpful AS reviews_numHelpful,

            (SELECT GROUP_CONCAT(c.category_name, ', ')
             FROM product_categories pc
             JOIN categories c ON pc.category_id = c.category_id
             WHERE pc.product_id = p.product_id) AS categories,

            (SELECT GROUP_CONCAT(i.url, ', ')
             FROM images i
             WHERE i.product_id = p.product_id) AS imageURLs

        FROM reviews_normalized rn
        JOIN products p ON rn.product_id = p.product_id
        JOIN users u ON rn.user_id = u.user_id
    """)

    print("View 'reviews_reconstructed' created!\n")


# -------------------------------------------------------------
# MAIN EXECUTION FLOW
# -------------------------------------------------------------
if __name__ == "__main__":
    csv_path = "data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
    db_path = "reviews.db"

    if os.path.exists(db_path):
        os.remove(db_path)

    # CENTRALIZED CONNECTION (everything runs inside one context!)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        print("=" * 60)
        print("Loading CSV into SQLite")
        print("=" * 60)

        create_database(cursor)
        load_csv_to_db(csv_path, cursor)

        print("Normalizing into new tables...")
        create_normalized_tables(cursor)
        normalize_data(cursor)

        print("Creating reconstructed view...")
        create_reviews_view(cursor)

        conn.commit()

    print("\nAll done! 🎉")
