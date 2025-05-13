import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import aiplatform
from dotenv import load_dotenv
import numpy as np

load_dotenv()

PROJECT_ID = os.environ.get("PROJECT_ID", "gen-lang-client-0471694923")
LOCATION = os.environ.get("LOCATION", "asia-northeast1")
INDEX_ENDPOINT_ID = os.environ.get("INDEX_ENDPOINT_ID", "8864297927502200832")
DEPLOYED_INDEX_ID = os.environ.get("DEPLOYED_INDEX_ID", "vs_hybridsearch_ja_deployed")

aiplatform.init(project=PROJECT_ID, location=LOCATION)

app = FastAPI(
    title="Vector Search API",
    description="API for product vector search using Vertex AI",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

np.random.seed(42)  # For reproducibility
product_embs = {
    str(i): np.random.rand(768).tolist() for i in range(1000, 2000)
}

product_names = {
    str(i): f"Product {i}" for i in range(1000, 2000)
}

class SearchRequest(BaseModel):
    product_id: str
    num_neighbors: int = 10

class SearchResult(BaseModel):
    product_id: str
    product_name: str
    distance: float

class SearchResponse(BaseModel):
    results: List[SearchResult]

def get_index_endpoint():
    endpoint_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/indexEndpoints/{INDEX_ENDPOINT_ID}"
    return aiplatform.MatchingEngineIndexEndpoint(endpoint_name)

@app.get("/")
async def root():
    return {"message": "Vector Search API is running"}

@app.get("/products", response_model=Dict[str, str])
async def get_products(limit: int = 10):
    """Get a list of product IDs and names"""
    return {k: product_names[k] for k in list(product_names.keys())[:limit]}

@app.post("/search", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    """Search for similar products based on a product ID"""
    if request.product_id not in product_embs:
        raise HTTPException(status_code=404, detail=f"Product ID {request.product_id} not found")
    
    try:
        query_emb = product_embs[request.product_id]
        
        my_index_endpoint = get_index_endpoint()
        
        response = my_index_endpoint.find_neighbors(
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[query_emb],
            num_neighbors=request.num_neighbors
        )
        
        results = []
        for neighbor in response[0]:
            product_id = neighbor.id
            results.append(SearchResult(
                product_id=product_id,
                product_name=product_names.get(product_id, f"Unknown Product {product_id}"),
                distance=float(neighbor.distance)
            ))
        
        return SearchResponse(results=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching products: {str(e)}")

@app.post("/search_demo")
async def search_demo():
    """Demo endpoint that runs the exact code from the example"""
    try:
        product_id = '6523'
        if product_id not in product_embs:
            product_embs[product_id] = np.random.rand(768).tolist()
            product_names[product_id] = f"Product {product_id}"
        
        query_emb = product_embs[product_id]
        
        my_index_endpoint = get_index_endpoint()
        
        response = my_index_endpoint.find_neighbors(
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[query_emb],
            num_neighbors=10
        )
        
        results = []
        for idx, neighbor in enumerate(response[0]):
            result = f"{neighbor.distance:.2f} {product_names[neighbor.id]}"
            results.append(result)
        
        return {"results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in demo search: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
