# Vector Search FastAPI Application

This FastAPI application provides an API for product vector search using Vertex AI's Vector Search capabilities.

## Features

- REST API for product vector search
- Integration with Vertex AI Vector Search
- Docker container for deployment to GCP Cloud Run
- Sample product embeddings for demonstration

## API Endpoints

- `GET /`: Health check endpoint
- `GET /products`: Get a list of product IDs and names
- `POST /search`: Search for similar products based on a product ID
- `POST /search_demo`: Demo endpoint that runs the exact code from the example

## Environment Variables

The application uses the following environment variables:

- `PROJECT_ID`: GCP project ID (default: "gen-lang-client-0471694923")
- `LOCATION`: GCP region (default: "asia-northeast1")
- `INDEX_ENDPOINT_ID`: Vector Search index endpoint ID (default: "8864297927502200832")
- `DEPLOYED_INDEX_ID`: Deployed index ID (default: "vs_hybridsearch_ja_deployed")

## Local Development

1. Install dependencies:

```bash
cd CloudRun_FastAPI
poetry install
```

2. Run the application:

```bash
poetry run uvicorn app.main:app --reload
```

3. Access the API documentation at http://localhost:8000/docs

## Deployment to Cloud Run

1. Build the Docker image:

```bash
cd CloudRun_FastAPI
docker build -t vector-search-api .
```

2. Deploy to Cloud Run:

```bash
gcloud run deploy vector-search-api \
  --image vector-search-api \
  --platform managed \
  --region asia-northeast1 \
  --allow-unauthenticated
```

## Example Usage

```python
import requests

# Search for similar products
response = requests.post(
    "http://localhost:8000/search",
    json={"product_id": "6523", "num_neighbors": 10}
)

# Print results
for result in response.json()["results"]:
    print(f"{result['distance']:.2f} {result['product_name']}")
```
