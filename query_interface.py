"""
Query interface for product insights.
This script provides a simple interface to look up products and view insights.
"""

import sqlite3
from typing import Dict, Optional, List


class ProductQueryInterface:
    """Interface for querying product insights."""
    
    def __init__(self, db_path='reviews.db'):
        self.db_path = db_path
    
    def get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def search_products(self, search_term: str) -> List[str]:
        """Search for products by name (partial match with improved robustness)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Normalize search term: remove extra spaces, handle punctuation
        search_term_clean = ' '.join(search_term.split()).strip()
        
        # Create multiple search patterns for robustness
        search_patterns = [
            f'%{search_term_clean}%',  # Exact match
            f'%{search_term_clean.replace(" ", "%")}%',  # Handle spaces
        ]
        
        # Also try case-insensitive search
        products_set = set()
        for pattern in search_patterns:
            cursor.execute(
                'SELECT DISTINCT name FROM reviews WHERE LOWER(name) LIKE LOWER(?) ORDER BY name LIMIT 20',
                (pattern,)
            )
            results = cursor.fetchall()
            for row in results:
                products_set.add(row[0])
        
        # Sort results by relevance (exact matches first, then partial)
        products_list = list(products_set)
        search_term_lower = search_term_clean.lower()
        
        # Sort: exact matches first, then by position of match
        def sort_key(name):
            name_lower = name.lower()
            if name_lower.startswith(search_term_lower):
                return (0, name_lower.find(search_term_lower))
            elif search_term_lower in name_lower:
                return (1, name_lower.find(search_term_lower))
            else:
                return (2, 0)
        
        products_list.sort(key=sort_key)
        
        conn.close()
        return products_list[:20]  # Limit to 20 results
    
    def get_product_insights(self, product_name: str) -> Optional[Dict]:
        """Get comprehensive insights for a product."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get basic statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_reviews,
                AVG(CAST(reviews_rating AS REAL)) as avg_rating,
                AVG(sentiment_score) as avg_sentiment
            FROM reviews
            WHERE name = ? AND reviews_rating IS NOT NULL AND reviews_rating != ''
        ''', (product_name,))
        
        stats = cursor.fetchone()
        if not stats or stats[0] == 0:
            return None
        
        total_reviews, avg_rating, avg_sentiment = stats
        
        # Get keywords and summary
        cursor.execute('''
            SELECT positive_keywords, negative_keywords, summary
            FROM product_keywords
            WHERE product_name = ?
        ''', (product_name,))
        
        keywords_row = cursor.fetchone()
        positive_keywords = keywords_row[0] if keywords_row and keywords_row[0] else "No positive keywords found"
        negative_keywords = keywords_row[1] if keywords_row and keywords_row[1] else "No negative keywords found"
        summary = keywords_row[2] if keywords_row and keywords_row[2] else "No summary available"
        
        # Format keywords
        pos_keywords_list = self._parse_keywords(positive_keywords)
        neg_keywords_list = self._parse_keywords(negative_keywords)
        
        insights = {
            'product_name': product_name,
            'total_reviews': int(total_reviews) if total_reviews else 0,
            'average_rating': round(float(avg_rating), 2) if avg_rating else 0.0,
            'average_sentiment': round(float(avg_sentiment), 3) if avg_sentiment else 0.0,
            'sentiment_label': self._get_sentiment_label(float(avg_sentiment) if avg_sentiment else 0.0),
            'top_positive_keywords': pos_keywords_list[:5],
            'top_negative_keywords': neg_keywords_list[:5],
            'summary': summary
        }
        
        conn.close()
        return insights
    
    def _parse_keywords(self, keywords_str: str) -> List[str]:
        """Parse keywords string into list, removing duplicates."""
        if not keywords_str or keywords_str == "No positive keywords found" or keywords_str == "No negative keywords found":
            return []
        
        # Format: "phrase1(score1), phrase2(score2), ..." (scores are sentiment scores, not counts)
        keywords = []
        seen = set()  # Track seen keywords (case-insensitive)
        
        for item in keywords_str.split(','):
            item = item.strip()
            if '(' in item:
                keyword = item.split('(')[0].strip()
                keyword_lower = keyword.lower()
                # Only add if not already seen (case-insensitive)
                if keyword_lower not in seen:
                    seen.add(keyword_lower)
                    keywords.append(keyword)
        
        return keywords
    
    def _get_sentiment_label(self, score: float) -> str:
        """Get sentiment label from score."""
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    def display_insights(self, insights: Dict):
        """Display product insights in a formatted way."""
        print("\n" + "=" * 80)
        print(f"PRODUCT INSIGHTS: {insights['product_name']}")
        print("=" * 80)
        
        print(f"\n📊 Statistics:")
        print(f"   Total Reviews: {insights['total_reviews']}")
        print(f"   Average Rating: {insights['average_rating']:.2f} / 5.0")
        print(f"   Average Sentiment: {insights['average_sentiment']:.3f} ({insights['sentiment_label']})")
        
        print(f"\n✅ Top Positive Keywords:")
        if insights['top_positive_keywords']:
            for i, keyword in enumerate(insights['top_positive_keywords'], 1):
                print(f"   {i}. {keyword}")
        else:
            print("   No positive keywords found")
        
        print(f"\n❌ Top Negative Keywords:")
        if insights['top_negative_keywords']:
            for i, keyword in enumerate(insights['top_negative_keywords'], 1):
                print(f"   {i}. {keyword}")
        else:
            print("   No negative keywords found")
        
        print(f"\n📝 Summary:")
        print(f"   {insights['summary']}")
        
        print("\n" + "=" * 80 + "\n")


def interactive_mode():
    """Run interactive query interface."""
    interface = ProductQueryInterface('reviews.db')
    
    print("=" * 80)
    print("Amazon Product Review Insights - Query Interface")
    print("=" * 80)
    print("\nEnter a product name (or partial name) to search, or 'quit' to exit.\n")
    
    while True:
        search_term = input("Search for product: ").strip()
        
        if search_term.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not search_term:
            print("Please enter a search term.\n")
            continue
        
        # Search for products
        products = interface.search_products(search_term)
        
        if not products:
            print(f"\nNo products found matching '{search_term}'.\n")
            continue
        
        # Display matching products
        if len(products) == 1:
            # Auto-select if only one match
            selected_product = products[0]
            print(f"\nFound product: {selected_product}")
        else:
            print(f"\nFound {len(products)} products:")
            for i, product in enumerate(products, 1):
                print(f"  {i}. {product}")
            
            try:
                choice = input(f"\nSelect a product (1-{len(products)}) or 'back' to search again: ").strip()
                if choice.lower() == 'back':
                    continue
                choice_num = int(choice)
                if 1 <= choice_num <= len(products):
                    selected_product = products[choice_num - 1]
                else:
                    print("Invalid choice.\n")
                    continue
            except ValueError:
                print("Invalid input.\n")
                continue
        
        # Get and display insights
        insights = interface.get_product_insights(selected_product)
        
        if insights:
            interface.display_insights(insights)
        else:
            print(f"\nNo insights available for '{selected_product}'.\n")


if __name__ == '__main__':
    interactive_mode()

