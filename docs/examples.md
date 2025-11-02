# Usage Examples and Tutorials

## Table of Contents
1. [Quick Start Guide](#quick-start-guide)
2. [Basic Usage Examples](#basic-usage-examples)
3. [Advanced Search Patterns](#advanced-search-patterns)
4. [Customization Examples](#customization-examples)
5. [Production Deployment](#production-deployment)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start Guide

### 1. Installation and Setup
```bash
# Clone the repository
git clone <repository-url>
cd sim_search_chromadb

# Install dependencies
pixi install

# Run the demo
pixi run python script.py
```

### 2. Expected Output
```
Collection created: employee_collection
Collection contents:
Number of documents: 15

=== Similarity Search Examples ===

1. Searching for Python developers:
Query: 'Python developer with web development experience'
  1. John Doe (employee_1) - Distance: 0.3245
     Role: Software Engineer, Department: Engineering
     Document: Software Engineer with 5 years of experience...
```

---

## Basic Usage Examples

### Example 1: Simple Text Search
```python
import chromadb
from chromadb.utils import embedding_functions

# Initialize
client = chromadb.Client()
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create collection
collection = client.create_collection(
    name="simple_search",
    embedding_function=ef
)

# Add documents
collection.add(
    documents=["I love programming in Python", "Java is great for enterprise"],
    ids=["doc1", "doc2"]
)

# Search
results = collection.query(
    query_texts=["Python development"],
    n_results=1
)

print(f"Best match: {results['documents'][0][0]}")
# Output: "I love programming in Python"
```

### Example 2: Search with Metadata
```python
# Add documents with metadata
collection.add(
    documents=["Senior Python Developer needed", "Junior Java Developer position"],
    metadatas=[
        {"language": "Python", "level": "Senior"},
        {"language": "Java", "level": "Junior"}
    ],
    ids=["job1", "job2"]
)

# Search with filtering
results = collection.query(
    query_texts=["Python programming job"],
    where={"level": "Senior"},
    n_results=5
)
```

### Example 3: Batch Operations
```python
# Batch document addition
documents = [
    "Machine Learning Engineer with TensorFlow experience",
    "Data Scientist skilled in Python and R",
    "Full Stack Developer using React and Node.js"
]

metadatas = [
    {"role": "ML Engineer", "skills": ["TensorFlow", "Python"]},
    {"role": "Data Scientist", "skills": ["Python", "R", "Statistics"]},
    {"role": "Full Stack Developer", "skills": ["React", "Node.js", "JavaScript"]}
]

ids = ["ml_eng_1", "data_sci_1", "fullstack_1"]

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)
```

---

## Advanced Search Patterns

### Pattern 1: Multi-Field Semantic Search
```python
# Create rich documents combining multiple fields
def create_employee_document(employee):
    """Generate comprehensive search document from employee data."""
    skills_text = f"Skills include {employee['skills']}"
    role_text = f"Works as {employee['role']} in {employee['department']}"
    experience_text = f"Has {employee['experience']} years of experience"
    location_text = f"Located in {employee['location']}"
    
    return f"{role_text}. {experience_text}. {skills_text}. {location_text}."

# Usage
employee = {
    "role": "Senior Data Scientist",
    "department": "Analytics",
    "experience": 8,
    "skills": "Python, Machine Learning, Statistics, SQL",
    "location": "San Francisco"
}

document = create_employee_document(employee)
# Result: "Works as Senior Data Scientist in Analytics. Has 8 years of experience. 
#          Skills include Python, Machine Learning, Statistics, SQL. Located in San Francisco."
```

### Pattern 2: Hierarchical Search
```python
def hierarchical_search(collection, query, filters=None):
    """Perform search with fallback strategies."""
    
    # First: Try with strict filters
    if filters:
        results = collection.query(
            query_texts=[query],
            where=filters,
            n_results=5
        )
        
        if len(results['ids'][0]) > 0:
            return results, "strict"
    
    # Second: Try with relaxed filters
    if filters and "experience" in filters:
        relaxed_filters = {k: v for k, v in filters.items() if k != "experience"}
        results = collection.query(
            query_texts=[query],
            where=relaxed_filters,
            n_results=5
        )
        
        if len(results['ids'][0]) > 0:
            return results, "relaxed"
    
    # Third: Pure semantic search
    results = collection.query(
        query_texts=[query],
        n_results=10
    )
    
    return results, "semantic_only"

# Usage
query = "experienced Python developer"
filters = {
    "$and": [
        {"department": "Engineering"},
        {"experience": {"$gte": 5}},
        {"location": "San Francisco"}
    ]
}

results, search_type = hierarchical_search(collection, query, filters)
print(f"Search completed using {search_type} strategy")
```

### Pattern 3: Similarity Threshold Filtering
```python
def search_with_threshold(collection, query, threshold=0.7, max_results=10):
    """Return only results above similarity threshold."""
    
    results = collection.query(
        query_texts=[query],
        n_results=max_results
    )
    
    filtered_results = {
        'ids': [[]],
        'documents': [[]],
        'metadatas': [[]],
        'distances': [[]]
    }
    
    for i, distance in enumerate(results['distances'][0]):
        # ChromaDB uses distance (lower = more similar)
        # Convert to similarity: similarity = 1 - (distance / 2)
        similarity = 1 - (distance / 2)
        
        if similarity >= threshold:
            filtered_results['ids'][0].append(results['ids'][0][i])
            filtered_results['documents'][0].append(results['documents'][0][i])
            filtered_results['metadatas'][0].append(results['metadatas'][0][i])
            filtered_results['distances'][0].append(distance)
    
    return filtered_results

# Usage
high_quality_results = search_with_threshold(
    collection, 
    "senior software engineer", 
    threshold=0.8
)
```

### Pattern 4: Multi-Query Ensemble Search
```python
def ensemble_search(collection, queries, weights=None):
    """Combine results from multiple related queries."""
    
    if weights is None:
        weights = [1.0] * len(queries)
    
    all_results = {}
    
    # Collect results for each query
    for i, query in enumerate(queries):
        results = collection.query(
            query_texts=[query],
            n_results=10
        )
        
        weight = weights[i]
        
        for j, doc_id in enumerate(results['ids'][0]):
            distance = results['distances'][0][j]
            weighted_score = distance * weight
            
            if doc_id in all_results:
                all_results[doc_id]['total_score'] += weighted_score
                all_results[doc_id]['query_count'] += 1
            else:
                all_results[doc_id] = {
                    'total_score': weighted_score,
                    'query_count': 1,
                    'document': results['documents'][0][j],
                    'metadata': results['metadatas'][0][j]
                }
    
    # Sort by average score
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]['total_score'] / x[1]['query_count']
    )
    
    return sorted_results

# Usage
queries = [
    "Python web developer",
    "backend engineer with API experience", 
    "full stack developer"
]
weights = [1.0, 0.8, 0.6]  # Prioritize first query

ensemble_results = ensemble_search(collection, queries, weights)
```

---

## Customization Examples

### Custom Embedding Function
```python
class CustomEmbeddingFunction:
    """Custom embedding function with preprocessing."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def __call__(self, texts):
        """Generate embeddings with custom preprocessing."""
        
        # Custom preprocessing
        processed_texts = []
        for text in texts:
            # Normalize text
            text = text.lower().strip()
            
            # Remove special characters (optional)
            import re
            text = re.sub(r'[^\w\s]', '', text)
            
            # Add context markers
            text = f"Employee profile: {text}"
            
            processed_texts.append(text)
        
        # Generate embeddings
        embeddings = self.model.encode(processed_texts)
        return embeddings.tolist()

# Usage
custom_ef = CustomEmbeddingFunction()
collection = client.create_collection(
    name="custom_embeddings",
    embedding_function=custom_ef
)
```

### Dynamic Collection Management
```python
class EmployeeSearchSystem:
    """Complete search system with collection management."""
    
    def __init__(self, persist_directory="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collections = {}
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    
    def create_department_collection(self, department_name):
        """Create department-specific collection."""
        collection_name = f"employees_{department_name.lower()}"
        
        try:
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"department": department_name}
            )
        except Exception:
            # Collection exists, get it
            collection = self.client.get_collection(collection_name)
        
        self.collections[department_name] = collection
        return collection
    
    def add_employee(self, employee, department=None):
        """Add employee to appropriate collection."""
        dept = department or employee.get('department', 'general')
        
        if dept not in self.collections:
            self.create_department_collection(dept)
        
        collection = self.collections[dept]
        
        # Generate document
        document = self._create_document(employee)
        
        collection.add(
            documents=[document],
            metadatas=[employee],
            ids=[employee['id']]
        )
    
    def search_all_departments(self, query, n_results=5):
        """Search across all department collections."""
        all_results = []
        
        for dept, collection in self.collections.items():
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Add department info to results
            for i, doc_id in enumerate(results['ids'][0]):
                result = {
                    'id': doc_id,
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'department_collection': dept
                }
                all_results.append(result)
        
        # Sort by distance
        all_results.sort(key=lambda x: x['distance'])
        return all_results[:n_results]
    
    def _create_document(self, employee):
        """Create searchable document from employee data."""
        return (f"{employee['role']} with {employee['experience']} years "
                f"in {employee['department']}. Skills: {employee['skills']}. "
                f"Located in {employee['location']}.")

# Usage
search_system = EmployeeSearchSystem()

# Add employees
employees = [
    {"id": "eng_1", "name": "Alice", "department": "Engineering", 
     "role": "Software Engineer", "experience": 5, "skills": "Python, React"},
    {"id": "mkt_1", "name": "Bob", "department": "Marketing",
     "role": "Marketing Manager", "experience": 8, "skills": "SEO, Analytics"}
]

for emp in employees:
    search_system.add_employee(emp)

# Search across departments
results = search_system.search_all_departments("Python developer")
```

---

## Production Deployment

### Configuration for Production
```python
import chromadb
from chromadb.config import Settings

# Production client configuration
client = chromadb.PersistentClient(
    path="./production_chroma_db",
    settings=Settings(
        # Enable authentication
        chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
        chroma_client_auth_credentials="admin:secure_password",
        
        # Performance settings
        chroma_server_grpc_port=8001,
        chroma_server_http_port=8000,
        
        # Security settings
        chroma_server_ssl_enabled=True,
        
        # Resource limits
        chroma_segment_cache_policy="LRU",
        chroma_segment_cache_size=1000000
    )
)
```

### Monitoring and Logging
```python
import logging
import time
from functools import wraps

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('search_system.log'),
        logging.StreamHandler()
    ]
)

def monitor_search_performance(func):
    """Decorator to monitor search performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            logging.info(f"Search completed in {duration:.3f}s - "
                        f"Query: {kwargs.get('query_texts', 'Unknown')}")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logging.error(f"Search failed after {duration:.3f}s - "
                         f"Error: {str(e)}")
            raise
    
    return wrapper

# Apply monitoring to collection methods
original_query = chromadb.Collection.query
chromadb.Collection.query = monitor_search_performance(original_query)
```

### Batch Processing for Large Datasets
```python
def bulk_add_employees(collection, employees, batch_size=100):
    """Add employees in batches for better performance."""
    
    for i in range(0, len(employees), batch_size):
        batch = employees[i:i + batch_size]
        
        # Prepare batch data
        documents = []
        metadatas = []
        ids = []
        
        for emp in batch:
            documents.append(create_employee_document(emp))
            metadatas.append({k: v for k, v in emp.items() if k != 'id'})
            ids.append(emp['id'])
        
        # Add batch to collection
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logging.info(f"Added batch {i//batch_size + 1}: {len(batch)} employees")
            
        except Exception as e:
            logging.error(f"Failed to add batch {i//batch_size + 1}: {str(e)}")
            raise

# Usage
large_employee_dataset = [...] # List of 10,000+ employees
bulk_add_employees(collection, large_employee_dataset, batch_size=500)
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Error: SentenceTransformers
```bash
# Error
ModuleNotFoundError: No module named 'sentence_transformers'

# Solution
pixi add sentence-transformers
# or
pip install sentence-transformers
```

#### 2. Collection Already Exists
```python
# Error
chromadb.errors.DuplicateIDError: Collection 'employee_collection' already exists

# Solution
try:
    collection = client.create_collection(name="employee_collection")
except chromadb.errors.DuplicateIDError:
    collection = client.get_collection(name="employee_collection")
```

#### 3. Empty Search Results
```python
def debug_empty_results(collection, query):
    """Debug function for empty search results."""
    
    # Check collection size
    all_items = collection.get()
    print(f"Collection contains {len(all_items['ids'])} documents")
    
    if len(all_items['ids']) == 0:
        print("Collection is empty - add documents first")
        return
    
    # Check query embedding
    results = collection.query(
        query_texts=[query],
        n_results=len(all_items['ids'])  # Get all results
    )
    
    print(f"Query returned {len(results['ids'][0])} results")
    
    if len(results['ids'][0]) > 0:
        print(f"Best match distance: {results['distances'][0][0]:.4f}")
        print(f"Worst match distance: {results['distances'][0][-1]:.4f}")
    
    # Show sample documents
    print("\nSample documents in collection:")
    for i in range(min(3, len(all_items['documents']))):
        print(f"  {i+1}: {all_items['documents'][i][:100]}...")
```

#### 4. Performance Issues
```python
# Monitor query performance
def profile_search(collection, queries, n_results=5):
    """Profile search performance across multiple queries."""
    
    import time
    
    results = {}
    
    for query in queries:
        start_time = time.time()
        
        search_results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        duration = time.time() - start_time
        results[query] = {
            'duration': duration,
            'result_count': len(search_results['ids'][0])
        }
    
    # Print performance summary
    avg_duration = sum(r['duration'] for r in results.values()) / len(results)
    print(f"Average query time: {avg_duration:.3f}s")
    
    for query, stats in results.items():
        print(f"Query: '{query[:50]}...' - {stats['duration']:.3f}s - {stats['result_count']} results")
    
    return results

# Usage
test_queries = [
    "Python developer",
    "team leader with experience",
    "marketing manager",
    "senior engineer"
]

profile_search(collection, test_queries)
```

#### 5. Memory Issues with Large Collections
```python
# Memory-efficient search for large collections
def memory_efficient_search(collection, query, batch_size=1000):
    """Search large collections in batches to manage memory."""
    
    # Get total document count
    all_items = collection.get(limit=1)  # Just get count
    # Note: ChromaDB doesn't directly provide count, 
    # so we estimate based on batch retrieval
    
    results = []
    offset = 0
    
    while True:
        # Get batch of documents
        batch_items = collection.get(
            limit=batch_size,
            offset=offset
        )
        
        if len(batch_items['ids']) == 0:
            break
        
        # Search within batch
        batch_results = collection.query(
            query_texts=[query],
            n_results=min(10, len(batch_items['ids'])),
            include=['documents', 'metadatas', 'distances']
        )
        
        results.extend(zip(
            batch_results['ids'][0],
            batch_results['documents'][0], 
            batch_results['metadatas'][0],
            batch_results['distances'][0]
        ))
        
        offset += batch_size
    
    # Sort all results by distance
    results.sort(key=lambda x: x[3])  # Sort by distance
    
    return results[:10]  # Return top 10
```

### Performance Optimization Tips

1. **Use Persistent Client**: For production workloads
2. **Batch Operations**: Add documents in batches of 100-1000
3. **Optimize HNSW Parameters**: Tune based on your use case
4. **Pre-filter with Metadata**: Reduce vector search space
5. **Cache Frequent Queries**: Store common search results
6. **Monitor Memory Usage**: Use appropriate batch sizes
7. **Use Appropriate Model**: Balance quality vs speed requirements

---

This comprehensive guide should help you understand and implement various patterns with the ChromaDB similarity search system!

# HTML Dashboard Usage Examples

## Getting Started with the Web Interface

### 1. Setup and Launch
```bash
# Start the FastAPI server
pixi run dev

# Open the HTML dashboard in your browser
xdg-open frontend/index.html
```

### 2. Similarity Search Examples

**Tab: 🎯 Similarity Search**

Try these natural language queries:

- **"Python developer with web experience"**
  - Finds: John Doe (Software Engineer), Alex Rodriguez (Lead Software Engineer)
  - Shows semantic understanding of programming skills

- **"team leader with management experience"**
  - Finds: David Lee (Engineering Manager), Rachel Brown (Marketing Director)
  - Identifies leadership roles across departments

- **"marketing professional with social media skills"**
  - Finds: Jane Smith (Marketing Manager), Emily Wilson (Marketing Assistant)
  - Matches domain expertise and specific skills

### 3. Filter Search Examples

**Tab: 🔍 Filter Search**

Use precise criteria to filter employees:

- **Engineering Department + 5+ Years Experience:**
  - Department: "Engineering"
  - Min Experience: 5
  - Results: Senior engineers and architects

- **California Employees:**
  - Location: "San Francisco" or "Los Angeles"
  - Shows geographic filtering

- **Part-time Employees:**
  - Employment Type: "Part-time"
  - Filters by work arrangement

### 4. Advanced Search Examples

**Tab: ⚡ Advanced Search**

Combine semantic search with filters:

- **"senior developer" + Engineering + 8+ years:**
  - Query: "senior developer with architecture experience"
  - Department: "Engineering"
  - Min Experience: 8
  - Finds: Michael Brown, Chris Evans, Alex Rodriguez

- **"marketing manager" + California:**
  - Query: "marketing manager with leadership skills"
  - Location: "Los Angeles"
  - Finds: Jane Smith and similar profiles

### 5. Understanding Results

Each result card shows:
- **Employee name and role**
- **Match score** (higher = better match)
- **Department, experience, location**
- **Full description** with skills and background
- **Hover effects** for better interaction

### 6. Pro Tips

- **Use descriptive queries** - "Python web developer" works better than just "Python"
- **Combine filters wisely** - Don't over-constrain your search
- **Check similarity scores** - Scores below 0.5 indicate very good matches
- **Try different phrasings** - "team lead" vs "manager" vs "supervisor"
