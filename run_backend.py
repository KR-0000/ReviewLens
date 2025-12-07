#!/usr/bin/env python3
"""Run the FastAPI backend server."""
import uvicorn
from backend.app.config import API_HOST, API_PORT, DEBUG

if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG,
        log_level="info"
    )

