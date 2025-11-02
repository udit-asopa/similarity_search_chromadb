# Test Configuration and Requirements

## Testing Framework Dependencies
pytest = ">=7.4.0"
pytest-asyncio = ">=0.21.0"
pytest-cov = ">=4.1.0"
httpx = ">=0.24.0"
pytest-mock = ">=3.11.0"
faker = ">=19.0.0"

## Test Structure
```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── unit/                    # Unit tests
│   ├── test_embedding.py    # Embedding function tests
│   ├── test_filters.py      # Metadata filter logic tests
│   └── test_models.py       # Pydantic model tests
├── integration/             # Integration tests
│   ├── test_chromadb.py     # ChromaDB integration tests
│   └── test_collection.py   # Collection operations tests
├── api/                     # API endpoint tests
│   ├── test_endpoints.py    # FastAPI endpoint tests
│   ├── test_search.py       # Search functionality tests
│   └── test_validation.py   # Request validation tests
├── e2e/                     # End-to-end tests
│   └── test_workflows.py    # Complete user workflows
└── performance/             # Performance tests
    └── test_benchmarks.py   # Performance benchmarks
```

## Running Tests
```bash
# Run all tests
pixi run test

# Run with coverage
pixi run test-cov

# Run specific test categories
pixi run test-unit
pixi run test-api
pixi run test-e2e

# Run performance benchmarks
pixi run test-perf
```