"""Pydantic models for API requests and responses."""
from typing import List, Optional
from pydantic import BaseModel


class KeywordItem(BaseModel):
    phrase: str
    score: float


class ProductInsights(BaseModel):
    product_name: str
    total_reviews: int
    average_rating: float
    average_sentiment: float
    sentiment_label: str
    top_positive_keywords: List[KeywordItem]
    top_negative_keywords: List[KeywordItem]
    summary: str


class ProductSearchResult(BaseModel):
    products: List[str]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

