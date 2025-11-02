# API Reference and Implementation Guide

## Table of Contents
1. [HTML Frontend Dashboard](#html-frontend-dashboard)
2. [FastAPI Web Service](#fastapi-web-service)
3. [ChromaDB API Reference](#chromadb-api-reference)
4. [SentenceTransformers API](#sentencetransformers-api)
5. [Custom Extensions](#custom-extensions)
6. [Integration Patterns](#integration-patterns)
7. [Testing Framework](#testing-framework)

---

## HTML Frontend Dashboard

### Quick Access
The easiest way to use the similarity search system is through the HTML dashboard:

```bash
# Start the API server
pixi run dev

# Open the dashboard
xdg-open frontend/index.html
```

### Dashboard Features
- **🎯 Similarity Search**: Natural language queries with semantic understanding
- **🔍 Filter Search**: Precise metadata filtering (department, experience, location)
- **⚡ Advanced Search**: Combines semantic search with metadata filters
- **📊 Visual Results**: Employee cards with similarity scores and details
- **🔄 Real-time**: Instant search with loading states and error handling

### Browser Requirements
- Modern browser with JavaScript enabled
- Local file access (works with file:// URLs)
- Network access to localhost:8000 (FastAPI server)

---

## FastAPI Web Service

The HTML dashboard connects to a FastAPI backend that provides REST endpoints.

---

## ChromaDB API Reference

### Client Initialization

#### Basic Client
```python
import chromadb

# In-memory client (development)
client = chromadb.Client()

# Persistent client (production)
client = chromadb.PersistentClient(path="./database")

# HTTP client (remote server)
client = chromadb.HttpClient(host="localhost", port=8000)
```

#### Client Configuration
```python
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(
        # Authentication
        chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
        chroma_client_auth_credentials="username:password",
        
        # Performance
        chroma_segment_cache_policy="LRU",
        chroma_segment_cache_size=1000000,
        
        # Networking
        chroma_server_grpc_port=8001,
        chroma_server_http_port=8000,
        chroma_server_ssl_enabled=True,
        
        # Telemetry
        anonymized_telemetry=False
    )
)
```

### Collection Management

#### Create Collection
```python
collection = client.create_collection(
    name="my_collection",
    metadata={"description": "Collection description"},
    embedding_function=embedding_function,  # Optional
    configuration={
        "hnsw": {
            "space": "cosine",           # cosine, euclidean, manhattan
            "M": 16,                     # Max connections (4-64)
            "ef_construction": 200,      # Build quality (100-800)
            "ef": 10,                    # Query quality (10-500)
            "max_elements": 100000       # Capacity limit
        }
    }
)
```

#### Get Existing Collection
```python
# Get collection by name
collection = client.get_collection(name="my_collection")

# Get collection with embedding function
collection = client.get_collection(
    name="my_collection", 
    embedding_function=embedding_function
)

# List all collections
collections = client.list_collections()
print([c.name for c in collections])
```

#### Delete Collection
```python
# Delete collection
client.delete_collection(name="my_collection")

# Check if collection exists
try:
    collection = client.get_collection(name="my_collection")
    print("Collection exists")
except chromadb.errors.InvalidCollectionException:
    print("Collection does not exist")
```

### Metadata Requirements and Constraints

#### Supported Data Types
ChromaDB metadata fields support only scalar values:
```python
# ✅ Supported types
metadata = {
    "title": "Book Title",           # string
    "year": 2023,                   # integer
    "rating": 4.5,                  # float
    "is_fiction": True,             # boolean
    "notes": None                   # None/null
}

# ❌ Unsupported types (will cause errors)
metadata = {
    "tags": ["sci-fi", "adventure"],     # list - NOT SUPPORTED
    "authors": {"primary": "Author"},    # dict - NOT SUPPORTED
    "metadata": {"nested": "value"}      # nested object - NOT SUPPORTED
}
```

#### Converting Complex Data Types
```python
# Convert lists to strings
book_data = {
    "awards": ["Hugo Award", "Nebula Award", "Locus Award"]
}

# ✅ Correct approach
metadata = {
    "awards": ", ".join(book_data["awards"]) if book_data["awards"] else "None"
}
# Result: "Hugo Award, Nebula Award, Locus Award"

# Convert nested objects to JSON strings
complex_data = {
    "publication_info": {
        "publisher": "Penguin",
        "isbn": "978-0123456789",
        "edition": 2
    }
}

# ✅ Correct approach
import json
metadata = {
    "publication_info": json.dumps(complex_data["publication_info"])
}

# Later retrieve and parse
import json
pub_info = json.loads(metadata["publication_info"])
```

#### Common Metadata Patterns
```python
# Book recommendation system metadata
book_metadata = {
    "title": book["title"],
    "author": book["author"], 
    "genre": book["genre"],
    "year": book["year"],                    # int
    "rating": book["rating"],                # float
    "pages": book["pages"],                  # int
    "themes": book["themes"],                # string (comma-separated)
    "awards": ", ".join(book["awards"]),     # converted list
    "is_classic": book["year"] < 1970,       # boolean
    "reading_level": book.get("level", "Adult")  # string with default
}
```
### Document Operations

#### Add Documents
```python
# Simple addition
collection.add(
    documents=["Document text 1", "Document text 2"],
    ids=["id1", "id2"]
)

# With metadata
collection.add(
    documents=["Document text"],
    metadatas=[{"category": "news", "author": "John"}],
    ids=["doc1"]
)

# With custom embeddings
collection.add(
    documents=["Document text"],
    embeddings=[[0.1, 0.2, 0.3, ...]],  # Pre-computed embeddings
    metadatas=[{"key": "value"}],
    ids=["doc1"]
)
```

#### Update Documents
```python
# Update existing documents
collection.update(
    ids=["id1"],
    documents=["Updated document text"],
    metadatas=[{"updated": True}]
)

# Upsert (add or update)
collection.upsert(
    documents=["New or updated document"],
    metadatas=[{"status": "current"}],
    ids=["doc1"]
)
```

#### Delete Documents
```python
# Delete by IDs
collection.delete(ids=["id1", "id2"])

# Delete by metadata filter
collection.delete(where={"category": "obsolete"})

# Delete all documents
collection.delete()
```

### Query Operations

#### Basic Query
```python
results = collection.query(
    query_texts=["Search query"],
    n_results=10,
    include=['documents', 'metadatas', 'distances']  # Default: all
)

# Access results
print(f"Found {len(results['ids'][0])} results")
for i, doc_id in enumerate(results['ids'][0]):
    print(f"ID: {doc_id}")
    print(f"Document: {results['documents'][0][i]}")
    print(f"Distance: {results['distances'][0][i]}")
    print(f"Metadata: {results['metadatas'][0][i]}")
```

#### Query with Pre-computed Embeddings
```python
# If you have pre-computed embeddings
results = collection.query(
    query_embeddings=[[0.1, 0.2, 0.3, ...]],
    n_results=5
)
```

#### Metadata Filtering
```python
# Exact match
results = collection.query(
    query_texts=["Python programming"],
    where={"language": "Python"},
    n_results=10
)

# Range queries
results = collection.query(
    query_texts=["experienced developer"],
    where={"experience_years": {"$gte": 5}},
    n_results=10
)

# Complex filters
results = collection.query(
    query_texts=["software engineer"],
    where={
        "$and": [
            {"department": "Engineering"},
            {"experience_years": {"$gte": 3}},
            {"location": {"$in": ["SF", "NYC"]}}
        ]
    },
    n_results=10
)
```

#### Get Documents (No Vector Search)
```python
# Get all documents
all_docs = collection.get()

# Get specific documents
docs = collection.get(ids=["id1", "id2"])

# Get with metadata filtering
filtered_docs = collection.get(
    where={"category": "important"},
    limit=100,
    offset=0
)
```

### Metadata Query Operators

#### Comparison Operators
```python
# Greater than
{"age": {"$gt": 25}}

# Greater than or equal
{"age": {"$gte": 25}}

# Less than
{"age": {"$lt": 65}}

# Less than or equal
{"age": {"$lte": 65}}

# Not equal
{"status": {"$ne": "inactive"}}
```

#### Array Operators
```python
# In array
{"category": {"$in": ["tech", "science", "engineering"]}}

# Not in array
{"category": {"$nin": ["spam", "deleted"]}}
```

#### Logical Operators
```python
# AND operation
{
    "$and": [
        {"age": {"$gte": 25}},
        {"department": "Engineering"}
    ]
}

# OR operation
{
    "$or": [
        {"priority": "high"},
        {"urgent": True}
    ]
}

# NOT operation
{
    "$not": {"status": "archived"}
}
```

---

## SentenceTransformers API

### Model Selection and Initialization

#### Popular Models
```python
from sentence_transformers import SentenceTransformer

# General purpose models
models = {
    "all-MiniLM-L6-v2": {
        "dimensions": 384,
        "size": "90MB", 
        "quality": "Good",
        "speed": "Fast"
    },
    "all-mpnet-base-v2": {
        "dimensions": 768,
        "size": "420MB",
        "quality": "High", 
        "speed": "Medium"
    },
    "all-distilroberta-v1": {
        "dimensions": 768,
        "size": "290MB",
        "quality": "High",
        "speed": "Medium"
    }
}

# Initialize model
model = SentenceTransformer('all-MiniLM-L6-v2')
```

#### Model Configuration
```python
# Custom model configuration
model = SentenceTransformer(
    'all-MiniLM-L6-v2',
    device='cpu',  # or 'cuda' for GPU
    cache_folder='./model_cache'
)

# Model information
print(f"Max sequence length: {model.max_seq_length}")
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
```

### Embedding Generation

#### Basic Encoding
```python
# Single sentence
embedding = model.encode("Hello World")
print(f"Embedding shape: {embedding.shape}")  # (384,)

# Multiple sentences
sentences = ["This is sentence 1", "This is sentence 2"]
embeddings = model.encode(sentences)
print(f"Embeddings shape: {embeddings.shape}")  # (2, 384)
```

#### Advanced Encoding Options
```python
embeddings = model.encode(
    sentences,
    batch_size=32,           # Process in batches
    show_progress_bar=True,  # Show progress
    convert_to_tensor=True,  # Return PyTorch tensor
    normalize_embeddings=True # L2 normalize embeddings
)
```

#### Similarity Calculation
```python
from sentence_transformers.util import cos_sim

# Calculate cosine similarity
sentence1 = "The weather is nice today"
sentence2 = "It's a beautiful day outside"

embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)

similarity = cos_sim(embedding1, embedding2)
print(f"Similarity: {similarity.item():.4f}")
```

### ChromaDB Integration

#### Embedding Function Implementation
```python
from chromadb.utils import embedding_functions

# Built-in SentenceTransformer embedding function
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Custom embedding function
class CustomEmbeddingFunction:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def __call__(self, input):
        # Custom preprocessing
        if isinstance(input, str):
            input = [input]
        
        # Add custom logic here
        processed_input = [text.lower().strip() for text in input]
        
        # Generate embeddings
        embeddings = self.model.encode(processed_input)
        return embeddings.tolist()

# Use custom function
custom_ef = CustomEmbeddingFunction()
collection = client.create_collection(
    name="custom_embeddings",
    embedding_function=custom_ef
)
```

---

## Custom Extensions

### Advanced Search Functions

#### Fuzzy Search with Edit Distance
```python
import difflib
from typing import List, Tuple

def fuzzy_text_search(
    collection, 
    query: str, 
    text_fields: List[str] = ['documents'],
    cutoff: float = 0.6
) -> List[Tuple[str, float, dict]]:
    """
    Perform fuzzy text matching in addition to semantic search.
    
    Args:
        collection: ChromaDB collection
        query: Search query
        text_fields: Fields to search in metadata
        cutoff: Minimum similarity threshold (0-1)
    
    Returns:
        List of (document_id, similarity_score, metadata) tuples
    """
    
    # Get all documents
    all_docs = collection.get()
    
    fuzzy_matches = []
    
    for i, doc_id in enumerate(all_docs['ids']):
        document = all_docs['documents'][i]
        metadata = all_docs['metadatas'][i]
        
        # Calculate fuzzy similarity
        similarity = difflib.SequenceMatcher(None, query.lower(), document.lower()).ratio()
        
        if similarity >= cutoff:
            fuzzy_matches.append((doc_id, similarity, metadata))
        
        # Also check metadata fields
        for field in text_fields:
            if field in metadata:
                field_similarity = difflib.SequenceMatcher(
                    None, query.lower(), str(metadata[field]).lower()
                ).ratio()
                
                if field_similarity >= cutoff:
                    fuzzy_matches.append((doc_id, field_similarity, metadata))
    
    # Sort by similarity (descending)
    fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
    
    # Remove duplicates (keep highest scoring)
    seen_ids = set()
    unique_matches = []
    for match in fuzzy_matches:
        if match[0] not in seen_ids:
            unique_matches.append(match)
            seen_ids.add(match[0])
    
    return unique_matches

# Usage
fuzzy_results = fuzzy_text_search(
    collection, 
    "pythoon develper",  # Typos in query
    cutoff=0.7
)
```

#### Semantic Clustering
```python
from sklearn.cluster import KMeans
import numpy as np

def cluster_documents(collection, n_clusters: int = 5):
    """
    Cluster documents based on their embeddings.
    
    Args:
        collection: ChromaDB collection
        n_clusters: Number of clusters
    
    Returns:
        Dictionary mapping cluster_id to list of document_ids
    """
    
    # Get all documents with embeddings
    all_docs = collection.get(include=['documents', 'metadatas', 'embeddings'])
    
    if not all_docs['embeddings']:
        # Generate embeddings if not stored
        embeddings = []
        for doc in all_docs['documents']:
            # This assumes you have access to the embedding function
            embedding = ef([doc])[0]  # Assuming ef is your embedding function
            embeddings.append(embedding)
    else:
        embeddings = all_docs['embeddings']
    
    # Perform clustering
    embeddings_array = np.array(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    
    # Group documents by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        
        clusters[label].append({
            'id': all_docs['ids'][i],
            'document': all_docs['documents'][i],
            'metadata': all_docs['metadatas'][i]
        })
    
    return clusters, kmeans.cluster_centers_

# Usage
document_clusters, cluster_centers = cluster_documents(collection, n_clusters=3)

for cluster_id, documents in document_clusters.items():
    print(f"\nCluster {cluster_id} ({len(documents)} documents):")
    for doc in documents[:3]:  # Show first 3 documents
        print(f"  - {doc['id']}: {doc['document'][:50]}...")
```

### Performance Monitoring

#### Query Performance Tracker
```python
import time
import statistics
from collections import defaultdict
from contextlib import contextmanager

class QueryPerformanceTracker:
    """Track and analyze query performance metrics."""
    
    def __init__(self):
        self.query_times = defaultdict(list)
        self.query_counts = defaultdict(int)
    
    @contextmanager
    def track_query(self, query_type: str, query_text: str = None):
        """Context manager to track query execution time."""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.query_times[query_type].append(duration)
            self.query_counts[query_type] += 1
    
    def get_stats(self, query_type: str = None):
        """Get performance statistics."""
        if query_type:
            times = self.query_times[query_type]
            if not times:
                return None
            
            return {
                'count': len(times),
                'avg_time': statistics.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'median_time': statistics.median(times)
            }
        else:
            stats = {}
            for qt in self.query_times:
                stats[qt] = self.get_stats(qt)
            return stats
    
    def print_report(self):
        """Print performance report."""
        print("\n=== Query Performance Report ===")
        
        for query_type, stats in self.get_stats().items():
            print(f"\n{query_type}:")
            print(f"  Count: {stats['count']}")
            print(f"  Average: {stats['avg_time']:.3f}s")
            print(f"  Min: {stats['min_time']:.3f}s")
            print(f"  Max: {stats['max_time']:.3f}s")
            print(f"  Median: {stats['median_time']:.3f}s")

# Usage
perf_tracker = QueryPerformanceTracker()

# Track semantic search
with perf_tracker.track_query("semantic_search", "python developer"):
    results = collection.query(
        query_texts=["python developer"],
        n_results=5
    )

# Track metadata filtering
with perf_tracker.track_query("metadata_filter"):
    results = collection.get(
        where={"department": "Engineering"}
    )

# Print performance report
perf_tracker.print_report()
```

---

## Integration Patterns

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

app = FastAPI(title="Employee Search API")

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

class EmployeeCreate(BaseModel):
    id: str
    name: str
    role: str
    department: str
    experience: int
    skills: str
    location: str
    employment_type: str

class SearchQuery(BaseModel):
    query: str
    n_results: int = 10
    filters: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    id: str
    document: str
    metadata: Dict[str, Any]
    distance: float

@app.on_event("startup")
async def startup_event():
    """Initialize collection on startup."""
    global collection
    try:
        collection = client.get_collection(
            name="employees",
            embedding_function=ef
        )
    except:
        collection = client.create_collection(
            name="employees",
            embedding_function=ef
        )

@app.post("/employees/", status_code=201)
async def add_employee(employee: EmployeeCreate):
    """Add a new employee to the search index."""
    
    # Create document text
    document = (f"{employee.role} with {employee.experience} years "
                f"in {employee.department}. Skills: {employee.skills}. "
                f"Located in {employee.location}.")
    
    # Prepare metadata
    metadata = employee.dict()
    del metadata['id']  # Remove ID from metadata
    
    try:
        collection.add(
            documents=[document],
            metadatas=[metadata],
            ids=[employee.id]
        )
        return {"message": "Employee added successfully", "id": employee.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search/", response_model=List[SearchResult])
async def search_employees(search_query: SearchQuery):
    """Search for employees using semantic similarity."""
    
    try:
        results = collection.query(
            query_texts=[search_query.query],
            n_results=search_query.n_results,
            where=search_query.filters
        )
        
        # Format results
        search_results = []
        for i in range(len(results['ids'][0])):
            search_results.append(SearchResult(
                id=results['ids'][0][i],
                document=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                distance=results['distances'][0][i]
            ))
        
        return search_results
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/employees/{employee_id}")
async def get_employee(employee_id: str):
    """Get specific employee by ID."""
    
    try:
        results = collection.get(ids=[employee_id])
        
        if not results['ids']:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        return {
            "id": results['ids'][0],
            "document": results['documents'][0],
            "metadata": results['metadatas'][0]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/employees/{employee_id}")
async def delete_employee(employee_id: str):
    """Delete employee from search index."""
    
    try:
        collection.delete(ids=[employee_id])
        return {"message": "Employee deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run with: uvicorn api:app --reload
```

### Streamlit Dashboard

```python
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="Employee Search Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Initialize ChromaDB
@st.cache_resource
def init_chromadb():
    client = chromadb.PersistentClient(path="./chroma_db")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    try:
        collection = client.get_collection("employees", embedding_function=ef)
    except:
        collection = client.create_collection("employees", embedding_function=ef)
    
    return collection

collection = init_chromodb()

# Sidebar for filters
st.sidebar.header("Search Filters")

department_filter = st.sidebar.selectbox(
    "Department",
    ["All", "Engineering", "Marketing", "HR", "Sales"]
)

experience_range = st.sidebar.slider(
    "Experience (years)",
    min_value=0,
    max_value=25,
    value=(0, 25)
)

location_filter = st.sidebar.multiselect(
    "Locations",
    ["New York", "San Francisco", "Los Angeles", "Chicago", "Boston", "Seattle"]
)

# Main interface
st.title("🔍 Employee Search Dashboard")

# Search input
col1, col2 = st.columns([3, 1])

with col1:
    search_query = st.text_input(
        "Search for employees",
        placeholder="e.g., 'Python developer with machine learning experience'"
    )

with col2:
    search_button = st.button("Search", type="primary")

# Build filters
filters = {}
if department_filter != "All":
    filters["department"] = department_filter

if experience_range != (0, 25):
    filters["experience"] = {
        "$gte": experience_range[0],
        "$lte": experience_range[1]
    }

if location_filter:
    filters["location"] = {"$in": location_filter}

# Perform search
if search_button and search_query:
    with st.spinner("Searching..."):
        try:
            # Build where clause
            where_clause = None
            if filters:
                where_clause = {"$and": [{k: v} for k, v in filters.items()]}
            
            results = collection.query(
                query_texts=[search_query],
                n_results=20,
                where=where_clause
            )
            
            if results['ids'][0]:
                st.success(f"Found {len(results['ids'][0])} results")
                
                # Create results dataframe
                results_data = []
                for i in range(len(results['ids'][0])):
                    metadata = results['metadatas'][0][i]
                    results_data.append({
                        "ID": results['ids'][0][i],
                        "Name": metadata['name'],
                        "Role": metadata['role'],
                        "Department": metadata['department'],
                        "Experience": metadata['experience'],
                        "Location": metadata['location'],
                        "Similarity Score": f"{1 - results['distances'][0][i]/2:.3f}",
                        "Document": results['documents'][0][i][:100] + "..."
                    })
                
                df = pd.DataFrame(results_data)
                
                # Display results table
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Department distribution
                    dept_counts = df['Department'].value_counts()
                    fig1 = px.pie(
                        values=dept_counts.values,
                        names=dept_counts.index,
                        title="Results by Department"
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Experience distribution
                    fig2 = px.histogram(
                        df,
                        x="Experience",
                        title="Experience Distribution"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            
            else:
                st.warning("No results found. Try adjusting your search query or filters.")
        
        except Exception as e:
            st.error(f"Search error: {str(e)}")

# Collection statistics
with st.expander("Collection Statistics"):
    try:
        all_items = collection.get()
        st.metric("Total Documents", len(all_items['ids']))
        
        # Department breakdown
        if all_items['metadatas']:
            departments = [meta.get('department', 'Unknown') for meta in all_items['metadatas']]
            dept_df = pd.DataFrame({'Department': departments})
            dept_counts = dept_df['Department'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Department Breakdown")
                for dept, count in dept_counts.items():
                    st.write(f"**{dept}**: {count}")
            
            with col2:
                fig = px.bar(x=dept_counts.index, y=dept_counts.values, 
                           title="Employees by Department")
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading statistics: {str(e)}")

# Run with: streamlit run dashboard.py
```

---

## Testing Framework

### Unit Tests with Pytest

```python
import pytest
import chromadb
from chromadb.utils import embedding_functions
import tempfile
import shutil
import os

class TestEmployeeSearch:
    """Test suite for employee search functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def client_and_collection(self, temp_db_path):
        """Set up test client and collection."""
        client = chromadb.PersistentClient(path=temp_db_path)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        collection = client.create_collection(
            name="test_employees",
            embedding_function=ef
        )
        
        return client, collection
    
    @pytest.fixture
    def sample_employees(self):
        """Sample employee data for testing."""
        return [
            {
                "id": "emp_1",
                "name": "Alice Johnson",
                "role": "Software Engineer",
                "department": "Engineering",
                "experience": 5,
                "skills": "Python, React, Node.js",
                "location": "San Francisco",
                "employment_type": "Full-time"
            },
            {
                "id": "emp_2", 
                "name": "Bob Smith",
                "role": "Data Scientist",
                "department": "Engineering",
                "experience": 8,
                "skills": "Python, Machine Learning, Statistics",
                "location": "New York",
                "employment_type": "Full-time"
            }
        ]
    
    def test_collection_creation(self, client_and_collection):
        """Test collection creation."""
        client, collection = client_and_collection
        
        assert collection.name == "test_employees"
        assert collection.count() == 0
    
    def test_add_employees(self, client_and_collection, sample_employees):
        """Test adding employees to collection."""
        client, collection = client_and_collection
        
        # Add employees
        documents = []
        metadatas = []
        ids = []
        
        for emp in sample_employees:
            doc = f"{emp['role']} with {emp['experience']} years in {emp['department']}"
            documents.append(doc)
            
            metadata = {k: v for k, v in emp.items() if k != 'id'}
            metadatas.append(metadata)
            
            ids.append(emp['id'])
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        # Verify addition
        assert collection.count() == len(sample_employees)
        
        # Test retrieval
        all_items = collection.get()
        assert len(all_items['ids']) == len(sample_employees)
    
    def test_semantic_search(self, client_and_collection, sample_employees):
        """Test semantic similarity search."""
        client, collection = client_and_collection
        
        # Add test data
        self._add_sample_data(collection, sample_employees)
        
        # Perform search
        results = collection.query(
            query_texts=["Python developer"],
            n_results=2
        )
        
        # Verify results
        assert len(results['ids'][0]) > 0
        assert all(isinstance(d, float) for d in results['distances'][0])
    
    def test_metadata_filtering(self, client_and_collection, sample_employees):
        """Test metadata-based filtering."""
        client, collection = client_and_collection
        
        # Add test data
        self._add_sample_data(collection, sample_employees)
        
        # Test department filtering
        results = collection.get(
            where={"department": "Engineering"}
        )
        
        assert len(results['ids']) == 2  # Both employees are in Engineering
        
        # Test experience filtering
        results = collection.get(
            where={"experience": {"$gte": 7}}
        )
        
        assert len(results['ids']) == 1  # Only Bob has 8+ years
    
    def test_combined_search(self, client_and_collection, sample_employees):
        """Test combined semantic and metadata search."""
        client, collection = client_and_collection
        
        # Add test data
        self._add_sample_data(collection, sample_employees)
        
        # Combined search
        results = collection.query(
            query_texts=["Python programming"],
            where={"experience": {"$gte": 5}},
            n_results=5
        )
        
        # Both employees should match (both have Python skills and 5+ years)
        assert len(results['ids'][0]) == 2
    
    def test_empty_results(self, client_and_collection):
        """Test handling of empty search results."""
        client, collection = client_and_collection
        
        # Search empty collection
        results = collection.query(
            query_texts=["nonexistent query"],
            n_results=10
        )
        
        assert len(results['ids'][0]) == 0
        assert len(results['documents'][0]) == 0
        assert len(results['distances'][0]) == 0
    
    def test_update_employee(self, client_and_collection, sample_employees):
        """Test updating employee data."""
        client, collection = client_and_collection
        
        # Add initial data
        self._add_sample_data(collection, sample_employees)
        
        # Update employee
        collection.update(
            ids=["emp_1"],
            documents=["Senior Software Engineer with 6 years in Engineering"],
            metadatas=[{"name": "Alice Johnson", "role": "Senior Software Engineer", 
                       "experience": 6, "department": "Engineering"}]
        )
        
        # Verify update
        results = collection.get(ids=["emp_1"])
        assert "Senior" in results['documents'][0]
        assert results['metadatas'][0]['experience'] == 6
    
    def test_delete_employee(self, client_and_collection, sample_employees):
        """Test deleting employee data."""
        client, collection = client_and_collection
        
        # Add initial data
        self._add_sample_data(collection, sample_employees)
        
        initial_count = collection.count()
        
        # Delete employee
        collection.delete(ids=["emp_1"])
        
        # Verify deletion
        assert collection.count() == initial_count - 1
        
        # Verify employee is gone
        results = collection.get(ids=["emp_1"])
        assert len(results['ids']) == 0
    
    def _add_sample_data(self, collection, sample_employees):
        """Helper method to add sample employee data."""
        documents = []
        metadatas = []
        ids = []
        
        for emp in sample_employees:
            doc = f"{emp['role']} with {emp['experience']} years in {emp['department']}. Skills: {emp['skills']}"
            documents.append(doc)
            
            metadata = {k: v for k, v in emp.items() if k != 'id'}
            metadatas.append(metadata)
            
            ids.append(emp['id'])
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

# Performance Tests
class TestPerformance:
    """Performance testing for large datasets."""
    
    def test_large_dataset_performance(self, client_and_collection):
        """Test performance with larger dataset."""
        client, collection = client_and_collection
        
        # Generate large dataset
        import time
        
        documents = [f"Employee {i} is a software engineer with Python skills" for i in range(1000)]
        metadatas = [{"id": i, "department": "Engineering", "experience": i % 20} for i in range(1000)]
        ids = [f"emp_{i}" for i in range(1000)]
        
        # Time the addition
        start_time = time.time()
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        add_time = time.time() - start_time
        
        # Time the search
        start_time = time.time()
        results = collection.query(
            query_texts=["Python engineer"],
            n_results=10
        )
        search_time = time.time() - start_time
        
        # Performance assertions
        assert add_time < 30.0  # Should add 1000 docs in under 30 seconds
        assert search_time < 1.0  # Should search in under 1 second
        assert len(results['ids'][0]) == 10

# Run tests with: pytest test_employee_search.py -v
```

### Integration Tests

```python
import pytest
import requests
import time
from concurrent.futures import ThreadPoolExecutor
import threading

class TestAPIIntegration:
    """Integration tests for the FastAPI service."""
    
    @pytest.fixture(scope="class")
    def api_url(self):
        """Base API URL for testing."""
        return "http://localhost:8000"
    
    def test_add_and_search_employee(self, api_url):
        """Test complete workflow: add employee and search."""
        
        # Add employee
        employee_data = {
            "id": "test_emp_1",
            "name": "Test User",
            "role": "Software Engineer",
            "department": "Engineering", 
            "experience": 5,
            "skills": "Python, FastAPI, ChromaDB",
            "location": "San Francisco",
            "employment_type": "Full-time"
        }
        
        response = requests.post(f"{api_url}/employees/", json=employee_data)
        assert response.status_code == 201
        
        # Search for employee
        search_data = {
            "query": "Python developer",
            "n_results": 5
        }
        
        response = requests.post(f"{api_url}/search/", json=search_data)
        assert response.status_code == 200
        
        results = response.json()
        assert len(results) > 0
        assert any(result["id"] == "test_emp_1" for result in results)
    
    def test_concurrent_searches(self, api_url):
        """Test API under concurrent load."""
        
        def search_request():
            search_data = {"query": "engineer", "n_results": 10}
            response = requests.post(f"{api_url}/search/", json=search_data)
            return response.status_code == 200
        
        # Execute concurrent searches
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(search_request) for _ in range(50)]
            results = [future.result() for future in futures]
        
        # All requests should succeed
        assert all(results)

# Load Testing
def test_search_performance():
    """Basic performance test for search operations."""
    
    # This would typically use tools like locust or pytest-benchmark
    import chromadb
    from chromadb.utils import embedding_functions
    
    client = chromadb.Client()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = client.create_collection(
        name="perf_test",
        embedding_function=ef
    )
    
    # Add test data
    documents = [f"Software engineer with Python skills {i}" for i in range(100)]
    ids = [f"emp_{i}" for i in range(100)]
    
    collection.add(documents=documents, ids=ids)
    
    # Time multiple searches
    import time
    search_times = []
    
    for _ in range(10):
        start = time.time()
        results = collection.query(
            query_texts=["Python developer"],
            n_results=10
        )
        end = time.time()
        search_times.append(end - start)
    
    avg_time = sum(search_times) / len(search_times)
    
    # Performance assertion
    assert avg_time < 0.5  # Average search should be under 500ms
    print(f"Average search time: {avg_time:.3f}s")

# Run with: pytest test_integration.py -v -s
```

This comprehensive API reference and testing framework provides everything needed to build, test, and deploy production-ready vector search applications with ChromaDB!
