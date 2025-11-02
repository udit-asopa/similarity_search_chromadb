# Employee Similarity Search with ChromaDB

A comprehensive Python application demonstrating semantic similarity search and metadata filtering using ChromaDB vector database with SentenceTransformers embeddings.

## 🎯 Overview

This project showcases how to build a powerful employee search system that combines:
- **Semantic similarity search** using natural language queries
- **Metadata filtering** for precise results
- **Combined search capabilities** for advanced filtering
- **Vector embeddings** for understanding context and meaning

## 🚀 Features

### Core Capabilities
- ✨ **Natural Language Search**: Find employees using conversational queries like "Python developer with web experience"
- 🔍 **Metadata Filtering**: Filter by department, experience level, location, and employment type
- 🎯 **Combined Search**: Merge semantic search with metadata filters for precise results
- 📊 **Similarity Scoring**: Get relevance scores to understand match quality
- 🛡️ **Error Handling**: Robust error handling for edge cases and empty results

### Search Examples
1. **Skill-based Search**: "Python developer with web development experience"
2. **Role-based Search**: "team leader manager with experience"
3. **Department Filtering**: Find all Engineering employees
4. **Experience Filtering**: Employees with 10+ years experience
5. **Location Filtering**: Employees in California
6. **Advanced Combined**: Senior Python developers in tech cities with 8+ years experience

## 🛠️ Technology Stack

- **ChromaDB**: Vector database for similarity search
- **SentenceTransformers**: Embedding model (`all-MiniLM-L6-v2`)
- **Python 3.8+**: Core programming language
- **Pixi**: Package and environment management

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Pixi package manager

### Setup Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd sim_search_chromadb
```

2. **Install dependencies using Pixi**
```bash
pixi install
```

3. **Run the application**
```bash
pixi run python script.py
```

## 🏗️ Project Structure

```
sim_search_chromadb/
├── script.py              # Main application script
├── README.md              # This file
├── pixi.toml             # Pixi configuration
├── pixi.lock             # Dependency lock file
└── docs/                 # Documentation folder
    ├── concepts.md       # Core concepts explained
    ├── code-structure.md # Code architecture
    └── examples.md       # Usage examples
```

## 📋 Usage

### Basic Usage

Run the complete demonstration:
```bash
pixi run python script.py
```

### Code Structure

The application consists of two main functions:

#### `main()` Function
- Creates ChromaDB collection with cosine similarity
- Defines employee dataset (15 employees with diverse roles)
- Generates comprehensive text documents for each employee
- Adds data to the collection with metadata
- Calls the search demonstration function

#### `perform_advanced_search()` Function
- Demonstrates similarity search examples
- Shows metadata filtering capabilities
- Performs combined search operations
- Handles edge cases and error scenarios

### Employee Data Structure

Each employee record contains:
```python
{
    "id": "employee_1",
    "name": "John Doe",
    "experience": 5,
    "department": "Engineering",
    "role": "Software Engineer",
    "skills": "Python, JavaScript, React, Node.js, databases",
    "location": "New York",
    "employment_type": "Full-time"
}
```

### Generated Documents

Rich text descriptions are created for similarity search:
```
"Software Engineer with 5 years of experience in Engineering. 
Skills: Python, JavaScript, React, Node.js, databases. 
Located in New York. Employment type: Full-time."
```

## 🔧 Configuration

### ChromaDB Collection Setup
```python
collection = client.create_collection(
    name="employee_collection",
    metadata={"description": "A collection for storing employee data"},
    configuration={
        "hnsw": {"space": "cosine"},
        "embedding_function": ef
    }
)
```

### Embedding Function
```python
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
```

## 📊 Example Outputs

### Similarity Search Results
```
=== Similarity Search Examples ===

1. Searching for Python developers:
Query: 'Python developer with web development experience'
  1. John Doe (employee_1) - Distance: 0.3245
     Role: Software Engineer, Department: Engineering
     Document: Software Engineer with 5 years of experience in Engineering...
```

### Metadata Filtering Results
```
=== Metadata Filtering Examples ===

3. Finding all Engineering employees:
Found 9 Engineering employees:
  - John Doe: Software Engineer (5 years)
  - Michael Brown: Senior Software Engineer (12 years)
  - David Lee: Engineering Manager (15 years)
```

## 🎓 Key Concepts

### Vector Embeddings
- Converts text into numerical vectors that capture semantic meaning
- Similar concepts have similar vector representations
- Enables "understanding" of context beyond keyword matching

### Cosine Similarity
- Measures similarity between vectors using cosine of the angle
- Values range from 0 (identical) to 2 (completely different)
- Lower distances indicate higher similarity

### HNSW (Hierarchical Navigable Small World)
- Efficient algorithm for approximate nearest neighbor search
- Provides fast similarity search on large datasets
- Configurable for different distance metrics (cosine, euclidean, etc.)

## 🔍 Advanced Features

### Query Types Supported
- **Text Similarity**: Natural language queries
- **Metadata Exact Match**: `{"department": "Engineering"}`
- **Metadata Range**: `{"experience": {"$gte": 10}}`
- **Metadata Array**: `{"location": {"$in": ["San Francisco", "New York"]}}`
- **Complex Logic**: `{"$and": [condition1, condition2]}`

### Error Handling
- Empty result detection
- Invalid query handling
- Collection creation errors
- Embedding generation failures

## 🐛 Troubleshooting

### ChromaDB Metadata Issues

**Problem**: `Expected metadata value to be a str, int, float, bool, or None, got [...] which is a list`

**Solution**: ChromaDB only supports scalar metadata values. Convert lists to strings:
```python
# ❌ This will fail
metadata = {"awards": ["Hugo Award", "Nebula Award"]}

# ✅ Convert to string
metadata = {"awards": "Hugo Award, Nebula Award"}
```

### Performance Issues

**Slow Search Performance**:
- Use smaller embedding models: `all-MiniLM-L6-v2` (384 dim) vs `all-mpnet-base-v2` (768 dim)
- Adjust HNSW parameters: lower `ef` for faster search, higher for accuracy
- Limit `n_results` to reasonable numbers (5-20)

**High Memory Usage**:
- Use CPU-only models if GPU memory is limited
- Consider model caching strategies
- Batch large operations

### Environment Setup

**Installation Issues**:
```bash
# Update pixi and reinstall dependencies
pixi self-update
rm -rf .pixi
pixi install
```

**Model Download Failures**:
- Ensure internet connection for first-time model download
- Models are cached locally after first download (~90MB-420MB)
- Check disk space for model storage
````
