"""
SmartStandOrg API Server - Run this script to start the FastAPI server
"""

import uvicorn

if __name__ == "__main__":
    print("Starting SmartStandOrg API server...")
    print("Access the API documentation at http://localhost:8000/docs")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)