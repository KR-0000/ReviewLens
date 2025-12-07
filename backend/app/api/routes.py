"""API routes for the review analysis application."""
from fastapi import APIRouter, HTTPException
from typing import List
from backend.app.models.schemas import ProductInsights, ProductSearchResult, ErrorResponse
from backend.app.services.review_service import ReviewService

router = APIRouter()
service = ReviewService()


@router.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.get("/products/search", response_model=ProductSearchResult, tags=["products"])
async def search_products(q: str):
    """
    Search for products by name.
    
    - **q**: Search query (product name or partial name)
    """
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")
    
    try:
        products = service.search_products(q.strip())
        return ProductSearchResult(products=products)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching products: {str(e)}")


@router.get("/products/{product_name}/insights", response_model=ProductInsights, tags=["products"])
async def get_product_insights(product_name: str):
    """
    Get comprehensive insights for a specific product.
    
    - **product_name**: Exact product name
    """
    if not product_name or not product_name.strip():
        raise HTTPException(status_code=400, detail="Product name cannot be empty")
    
    try:
        insights = service.get_product_insights(product_name.strip())
        if not insights:
            raise HTTPException(
                status_code=404, 
                detail=f"No insights found for product: {product_name}"
            )
        
        return ProductInsights(**insights)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving product insights: {str(e)}"
        )

