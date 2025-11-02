# Employee Similarity Search with ChromaDB

A comprehensive Python application demonstrating semantic similarity search and metadata filtering using ChromaDB vector database with SentenceTransformers embeddings.

## 🎯 Overview

This project showcases how to build a powerful employee search system that combines:
- **Semantic similarity search** using natural language queries
- **Metadata filtering** for precise results

## 🚀 Quick Start

### 🎯 **3-Minute Setup**

1. **Clone and install:**
```bash
git clone <repository-url>
cd sim_search_chromadb
pixi install
```

2. **Start the server:**
```bash
pixi run dev
```

3. **Open the dashboard:**
```bash
xdg-open frontend/index.html
```

4. **Try your first search:**
   - Click "🎯 Similarity Search" tab
   - Enter: "Python developer with web experience"
   - Click "🔍 Search Similar Employees"
   - See instant results with similarity scores!

### Why Use the HTML Dashboard?
- User-friendly interface - no coding required
- Real-time search with instant results
- Visual employee cards with similarity scores
- Multiple search modes in one interface

## 🛠️ Technology Stack

- **ChromaDB**: Vector database for similarity search
- **SentenceTransformers**: Embedding model (`all-MiniLM-L6-v2`)
- **FastAPI**: Modern web API framework for backend
- **HTML/CSS/JavaScript**: Interactive web dashboard
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

3. **Start the FastAPI server**
```bash
pixi run dev
```

4. **Open the Web Dashboard**
```bash
# Open the interactive HTML dashboard
xdg-open frontend/index.html
# or navigate to frontend/index.html in your file manager
```

### Alternative: Command Line Usage
```bash
# Run the original script (command-line only)
pixi run python script.py
```

## 🏗️ Project Structure

```
sim_search_chromadb/
├── script.py              # Original CLI application
├── README.md              # This file
├── pixi.toml             # Pixi configuration
├── pixi.lock             # Dependency lock file
├── api/                  # FastAPI web service
│   └── main.py          # FastAPI application
├── frontend/             # Web dashboard
│   └── index.html       # Interactive HTML interface
├── exercises/            # Practice exercises
│   ├── books_advanced_search.py
│   ├── books_advanced_search_solution.py
│   └── books_bonus_features.py
└── docs/                 # Documentation folder
    ├── concepts.md       # Core concepts explained
    ├── code-structure.md # Code architecture
    ├── examples.md       # Usage examples
    └── api-reference.md  # API documentation
```

## 📋 Usage

### Web Dashboard (Recommended)

1. **Start the API server:**
```bash
pixi run dev
```

2. **Open the interactive web dashboard:**
```bash
# Open the HTML dashboard in your browser
xdg-open frontend/index.html
```

3. **Use the three search modes:**
   - **🎯 Similarity Search**: Natural language queries like "Python developer with web experience"
   - **🔍 Filter Search**: Precise filtering by department, experience, location, etc.
   - **⚡ Advanced Search**: Combine semantic search with metadata filters

### Command Line Usage

Run the original CLI demonstration:
```bash
pixi run python script.py
```

### API Endpoints

Access the auto-generated API documentation:
- **Interactive docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

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

## 📚 Documentation

- **[Features Guide](docs/features.md)** - Complete feature overview and capabilities
- **[Usage Examples](docs/examples.md)** - HTML dashboard and search examples
- **[Developer Guide](docs/developer-guide.md)** - Development, customization, deployment
- **[API Reference](docs/api-reference.md)** - Complete API documentation
- **[Wiki Index](docs/wiki-index.md)** - Complete navigation guide

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

## 🔧 Available Tasks

```bash
# Web interface
pixi run dev          # Start development server
pixi run prod         # Production server

# Command line
pixi run run          # Original CLI script

# Exercises
pixi run books        # Book search exercises
pixi run books-solution
pixi run books-bonus

# Utilities
pixi run health       # Check API health
```

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
