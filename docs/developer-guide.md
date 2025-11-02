# Developer Guide

## 🛠️ Development Setup

### Prerequisites
```bash
# Required software
- Python 3.8+
- Git
- Modern web browser
- Text editor/IDE

# Install Pixi (package manager)
curl -fsSL https://pixi.sh/install.sh | bash
```

### Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd sim_search_chromadb

# Install dependencies
pixi install

# Verify installation
pixi run python --version
```

## 🏗️ Architecture Overview

### System Components
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   HTML Frontend │────│   FastAPI Server │────│   ChromaDB      │
│   (User Interface) │    │   (API Layer)    │    │   (Vector DB)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                         │                         │
         │                         │                         │
    ┌─────────┐              ┌──────────┐              ┌──────────┐
    │ Browser │              │ Pydantic │              │ HNSW     │
    │ JS/CSS  │              │ Models   │              │ Index    │
    └─────────┘              └──────────┘              └──────────┘
```

### Data Flow
1. **User Input** → HTML Form → JavaScript
2. **API Request** → FastAPI Endpoint → Validation
3. **Query Processing** → ChromaDB → Vector Search
4. **Results** → JSON Response → Frontend Display

## 💻 Development Workflow

### Running in Development Mode
```bash
# Start with auto-reload (recommended for development)
pixi run dev

# Start without auto-reload
pixi run serve

# Run original CLI script
pixi run run
```

### Development Tools
```bash
# Check API health
pixi run health

# Test API endpoints
curl http://localhost:8000/docs

# View available tasks
pixi task list
```

### File Watching
The development server automatically reloads when you modify:
- `api/main.py` (FastAPI server)
- Frontend files don't auto-reload (refresh browser manually)

## 🔧 Customization Guide

### Adding New Endpoints
```python
# In api/main.py
@app.get("/custom/endpoint")
async def custom_endpoint():
    """Your custom endpoint"""
    return {"message": "Custom functionality"}
```

### Modifying Search Logic
```python
# Custom similarity function
def custom_similarity_search(query: str, custom_params: dict):
    results = collection.query(
        query_texts=[query],
        n_results=custom_params.get('limit', 5),
        where=build_custom_filter(custom_params)
    )
    return process_custom_results(results)
```

### Adding New Metadata Fields
```python
# 1. Update employee data structure
employees = [
    {
        "id": "employee_1",
        "name": "John Doe",
        # ...existing fields...
        "salary_range": "80k-100k",     # New field
        "remote_friendly": True,        # New field
        "team_size": 5                  # New field
    }
]

# 2. Update metadata in collection.add()
metadatas=[{
    # ...existing metadata...
    "salary_range": employee["salary_range"],
    "remote_friendly": employee["remote_friendly"],
    "team_size": employee["team_size"]
} for employee in employees]

# 3. Update Pydantic models
class MetadataFilter(BaseModel):
    # ...existing fields...
    salary_range: Optional[str] = None
    remote_friendly: Optional[bool] = None
    team_size_min: Optional[int] = None
```

### Custom Embedding Models
```python
# Option 1: Different SentenceTransformer model
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"  # Higher quality, slower
)

# Option 2: OpenAI embeddings (requires API key)
ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-api-key",
    model_name="text-embedding-ada-002"
)

# Option 3: Custom embedding function
class CustomEmbeddingFunction:
    def __call__(self, input: list[str]) -> list[list[float]]:
        # Your custom embedding logic
        return embeddings
```

## 🧪 Testing

### Manual Testing
```bash
# Test similarity search
curl -X POST "http://localhost:8000/search/similarity" \
     -H "Content-Type: application/json" \
     -d '{"query": "Python developer", "n_results": 3}'

# Test filtering
curl "http://localhost:8000/search/filter?department=Engineering"

# Test advanced search
curl -X POST "http://localhost:8000/search/advanced" \
     -H "Content-Type: application/json" \
     -d '{"query": "senior developer", "filters": {"department": "Engineering"}}'
```

### Automated Testing
```python
# Create tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_similarity_search():
    response = client.post(
        "/search/similarity",
        json={"query": "Python developer", "n_results": 3}
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) <= 3

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

### Frontend Testing
```javascript
// Test in browser console
// Check if API is accessible
fetch('http://localhost:8000/health')
    .then(response => response.json())
    .then(data => console.log('API Status:', data));

// Test search functionality
async function testSearch() {
    const response = await fetch('http://localhost:8000/search/similarity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: 'Python developer', n_results: 3 })
    });
    const data = await response.json();
    console.log('Search Results:', data);
}
```

## 🚀 Deployment

### Production Configuration
```python
# Production settings in api/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install pixi
RUN pixi install

EXPOSE 8000
CMD ["pixi", "run", "prod"]
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  similarity-search:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
    volumes:
      - ./data:/app/data  # Persist ChromaDB data
```

### Cloud Deployment

#### AWS EC2
```bash
# Install dependencies
sudo apt update
sudo apt install python3.11 python3-pip

# Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Clone and setup
git clone <your-repo>
cd sim_search_chromadb
pixi install

# Run production server
pixi run prod
```

#### Heroku
```bash
# Create Procfile
echo "web: pixi run prod" > Procfile

# Deploy
git add .
git commit -m "Deploy to Heroku"
heroku create your-app-name
git push heroku main
```

## 🔍 Debugging

### Common Issues

#### ChromaDB Connection Issues
```python
# Debug ChromaDB state
print(f"Collection count: {collection.count()}")
print(f"Collection metadata: {collection.metadata}")

# Reset collection if needed
try:
    client.delete_collection("employee_collection")
except:
    pass
collection = client.create_collection("employee_collection")
```

#### Embedding Model Issues
```python
# Test embedding function
test_texts = ["Python developer", "Marketing manager"]
embeddings = ef(test_texts)
print(f"Embedding dimensions: {len(embeddings[0])}")
print(f"Embedding type: {type(embeddings[0][0])}")
```

#### API Issues
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug endpoints
@app.get("/debug/collection")
async def debug_collection():
    return {
        "count": collection.count(),
        "metadata": collection.metadata
    }
```

### Performance Monitoring
```python
import time

# Add timing to search functions
def timed_search(query: str):
    start_time = time.time()
    results = collection.query(query_texts=[query], n_results=5)
    end_time = time.time()
    
    return {
        "results": results,
        "search_time_ms": (end_time - start_time) * 1000
    }
```

## 🔧 Advanced Features

### Custom Scoring
```python
def custom_relevance_score(results, query_context):
    """Apply custom scoring based on business logic"""
    for i, result in enumerate(results['metadatas'][0]):
        base_score = results['distances'][0][i]
        
        # Boost recent hires
        if result.get('hire_date') > '2023-01-01':
            base_score *= 0.9
        
        # Boost high performers
        if result.get('performance_rating') == 'excellent':
            base_score *= 0.8
        
        results['distances'][0][i] = base_score
    
    return results
```

### Batch Processing
```python
async def batch_search(queries: list[str]):
    """Process multiple queries efficiently"""
    results = []
    for query in queries:
        result = collection.query(query_texts=[query], n_results=5)
        results.append(result)
    return results
```

### Caching Layer
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query: str, n_results: int = 5):
    """Cache frequent searches"""
    return collection.query(query_texts=[query], n_results=n_results)
```

## 📊 Performance Optimization

### Database Optimization
```python
# Optimize HNSW parameters for your dataset size
collection = client.create_collection(
    name="employee_collection",
    configuration={
        "hnsw": {
            "space": "cosine",
            "ef": 200,      # Higher for better recall
            "M": 32         # Higher for better connectivity
        }
    }
)
```

### Memory Management
```python
# Monitor memory usage
import psutil

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

# Implement memory-efficient batch processing
def process_large_dataset(employees, batch_size=100):
    for i in range(0, len(employees), batch_size):
        batch = employees[i:i+batch_size]
        # Process batch
        yield batch
```

## 🔐 Security Considerations

### Input Validation
```python
from pydantic import validator

class SecureSearchRequest(BaseModel):
    query: str
    n_results: int = Field(default=5, ge=1, le=50)
    
    @validator('query')
    def validate_query(cls, v):
        if len(v) > 500:
            raise ValueError('Query too long')
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()
```

### Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/search/similarity")
@limiter.limit("10/minute")
async def similarity_search(request: Request, search_request: EmployeeSearchRequest):
    # Rate-limited search
    pass
```

## 📈 Monitoring and Analytics

### Search Analytics
```python
import json
from datetime import datetime

def log_search(query: str, results_count: int, response_time: float):
    """Log search analytics"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "results_count": results_count,
        "response_time_ms": response_time,
        "user_agent": "web_dashboard"
    }
    
    with open("search_analytics.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
```

### Health Monitoring
```python
@app.get("/health/detailed")
async def detailed_health():
    """Comprehensive health check"""
    return {
        "api_status": "healthy",
        "chromadb_count": collection.count(),
        "memory_usage_mb": get_memory_usage(),
        "model_loaded": ef is not None,
        "timestamp": datetime.now().isoformat()
    }
```