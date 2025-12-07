"""Service layer for review analysis."""
import sys
import os
from pathlib import Path

# Add parent directory to path to import ReviewProcessor
BASE_DIR = Path(__file__).parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import from root level modules
from process_reviews import ReviewProcessor
from query_interface import ProductQueryInterface
from backend.app.config import DATABASE_PATH


class ReviewService:
    """Service for product review analysis."""
    
    def __init__(self):
        """Initialize the review service."""
        self.processor = ReviewProcessor(db_path=DATABASE_PATH)
        self.query_interface = ProductQueryInterface(db_path=DATABASE_PATH)
    
    def search_products(self, search_term: str) -> list[str]:
        """Search for products by name."""
        return self.query_interface.search_products(search_term)
    
    def get_product_insights(self, product_name: str) -> dict:
        """Get comprehensive insights for a product."""
        insights = self.query_interface.get_product_insights(product_name)
        if not insights:
            return None
        
        # Convert to API format
        return {
            "product_name": insights["product_name"],
            "total_reviews": insights["total_reviews"],
            "average_rating": insights["average_rating"],
            "average_sentiment": insights["average_sentiment"],
            "sentiment_label": insights["sentiment_label"],
            "top_positive_keywords": [
                {"phrase": kw, "score": 0.0} for kw in insights["top_positive_keywords"]
            ],
            "top_negative_keywords": [
                {"phrase": kw, "score": 0.0} for kw in insights["top_negative_keywords"]
            ],
            "summary": insights["summary"]
        }

