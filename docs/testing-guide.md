# Testing Guide

## 🧪 Comprehensive Test Suite

This document describes the testing strategy and implementation for the Employee Similarity Search system. The tests are designed like a senior software engineer would create - comprehensive, maintainable, and production-ready.

## 📁 Test Structure

```
tests/
├── conftest.py                  # Shared fixtures and configuration
├── README.md                   # Test documentation
├── unit/                       # Unit tests (fast, isolated)
│   ├── test_embedding.py       # Embedding functions and core logic
│   └── test_filters.py         # Metadata filters and Pydantic models
├── integration/                # Integration tests (component interactions)
│   └── test_chromadb.py        # ChromaDB operations and collection management
├── api/                        # API endpoint tests
│   └── test_endpoints.py       # FastAPI endpoint testing
├── e2e/                        # End-to-end tests (complete workflows)
│   └── test_workflows.py       # User workflows and system integration
└── performance/                # Performance benchmarks
    └── test_benchmarks.py      # Performance and scalability tests
```

## 🎯 Testing Philosophy

### Senior Engineer Principles Applied

1. **Comprehensive Coverage**
   - Unit tests for core logic
   - Integration tests for component interactions
   - API tests for all endpoints
   - E2E tests for user workflows
   - Performance tests for scalability

2. **Realistic Test Data**
   - Faker library for generating realistic test data
   - Mock objects that behave like production systems
   - Edge cases and boundary conditions covered

3. **Production-Like Testing**
   - Actual ChromaDB operations in integration tests
   - Real HTTP requests in API tests
   - Concurrent load testing
   - Memory and resource monitoring

4. **Maintainable Test Code**
   - Shared fixtures for common setup
   - Clear test organization and naming
   - Comprehensive documentation
   - Proper mocking and isolation

## 🚀 Running Tests

### Quick Test Commands

```bash
# Run all tests
pixi run test

# Run with coverage report
pixi run test-cov

# Run specific test categories
pixi run test-unit          # Unit tests only
pixi run test-integration   # Integration tests only
pixi run test-api-tests     # API tests only
pixi run test-e2e          # End-to-end tests only
pixi run test-perf         # Performance benchmarks only

# Run tests with specific markers
pytest -m "unit"           # Unit tests
pytest -m "not slow"       # Skip slow tests
pytest -m "performance"    # Performance tests only
```

### Advanced Test Options

```bash
# Verbose output with details
pytest tests/ -v -s

# Run tests in parallel (if pytest-xdist installed)
pytest tests/ -n auto

# Run tests with coverage and generate HTML report
pytest tests/ --cov=api --cov-report=html

# Run specific test file
pytest tests/unit/test_embedding.py -v

# Run specific test method
pytest tests/api/test_endpoints.py::TestSimilaritySearchEndpoint::test_similarity_search_success -v

# Watch for file changes and re-run tests
pytest tests/ --watch
```

## 🧪 Test Categories

### 1. Unit Tests (`tests/unit/`)

**Purpose**: Test individual functions and classes in isolation.

**Characteristics**:
- Fast execution (< 1 second each)
- No external dependencies
- Use mocks for ChromaDB and embedding functions
- Test edge cases and error conditions

**Key Test Classes**:
- `TestEmbeddingFunction`: Embedding function behavior
- `TestMetadataFilters`: Filter logic and edge cases
- `TestSearchResultFormatting`: Result processing
- `TestInputValidation`: Pydantic model validation
- `TestErrorHandling`: Error scenarios and recovery

**Example**:
```python
def test_build_metadata_filter_same_min_max_experience(self):
    """Test filter building when min and max experience are same."""
    filters = MetadataFilter(experience_min=5, experience_max=5)
    result = build_metadata_filter(filters)
    
    expected = {"experience": 5}
    assert result == expected
```

### 2. Integration Tests (`tests/integration/`)

**Purpose**: Test component interactions with real dependencies.

**Characteristics**:
- Use actual ChromaDB operations
- Test real embedding functions
- Verify data persistence and retrieval
- Test complex query scenarios

**Key Test Classes**:
- `TestChromaDBIntegration`: Database client operations
- `TestCollectionOperations`: CRUD operations on collections
- `TestEmbeddingIntegration`: Real embedding function integration
- `TestCollectionMetadata`: Metadata handling
- `TestErrorHandling`: Database error scenarios

**Example**:
```python
def test_query_similarity_search(self, test_collection):
    """Test similarity search queries."""
    results = test_collection.query(
        query_texts=["software engineer programming"],
        n_results=2
    )
    
    assert len(results["ids"][0]) <= 2
    for distance in results["distances"][0]:
        assert 0 <= distance <= 2.0  # Cosine distance range
```

### 3. API Tests (`tests/api/`)

**Purpose**: Test HTTP endpoints and API behavior.

**Characteristics**:
- Use FastAPI TestClient
- Test all HTTP methods and endpoints
- Validate request/response formats
- Test error handling and status codes

**Key Test Classes**:
- `TestHealthEndpoints`: Health check endpoints
- `TestSimilaritySearchEndpoint`: Similarity search API
- `TestFilterSearchEndpoint`: Metadata filtering API
- `TestAdvancedSearchEndpoint`: Combined search API
- `TestEmployeesEndpoint`: Employee listing API
- `TestStatsEndpoint`: Statistics API
- `TestCORSAndHeaders`: HTTP headers and CORS
- `TestRequestValidation`: Input validation

**Example**:
```python
def test_similarity_search_success(self, api_client, mock_search_results):
    """Test successful similarity search."""
    request_data = {
        "query": "Python developer", 
        "n_results": 3
    }
    
    response = api_client.post("/search/similarity", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["search_type"] == "similarity"
    assert len(data["results"]) == 2
```

### 4. End-to-End Tests (`tests/e2e/`)

**Purpose**: Test complete user workflows and system integration.

**Characteristics**:
- Test realistic user scenarios
- Multi-step workflows
- Cross-component integration
- Error recovery testing

**Key Test Classes**:
- `TestCompleteUserWorkflows`: New user discovery workflows
- `TestErrorRecoveryWorkflows`: System resilience
- `TestIntegrationWithExternalSystems`: Microservice integration
- `TestPerformanceWorkflows`: Performance under load

**Example**:
```python
async def test_new_user_complete_workflow(self, async_api_client):
    """Test complete workflow for a new user discovering the system."""
    
    # Step 1: Check health
    health_response = await async_api_client.get("/health")
    assert health_response.status_code == 200
    
    # Step 2: Explore employees
    employees_response = await async_api_client.get("/employees?limit=10")
    assert employees_response.status_code == 200
    
    # Step 3: Perform similarity search
    search_response = await async_api_client.post("/search/similarity", json={
        "query": "Python developer with web experience",
        "n_results": 5
    })
    assert search_response.status_code == 200
```

### 5. Performance Tests (`tests/performance/`)

**Purpose**: Benchmark system performance and scalability.

**Characteristics**:
- Measure response times and throughput
- Test under concurrent load
- Monitor resource usage
- Validate performance requirements

**Key Test Classes**:
- `TestSearchPerformanceBenchmarks`: Response time benchmarks
- `TestConcurrencyBenchmarks`: Concurrent load testing
- `TestScalabilityBenchmarks`: Scaling characteristics
- `TestMemoryAndResourceBenchmarks`: Resource usage monitoring

**Example**:
```python
def test_similarity_search_latency(self, api_client):
    """Benchmark similarity search response times."""
    latencies = []
    
    for query in test_queries:
        start_time = time.perf_counter()
        response = api_client.post("/search/similarity", json={
            "query": query,
            "n_results": 5
        })
        end_time = time.perf_counter()
        
        latency = (end_time - start_time) * 1000  # ms
        latencies.append(latency)
    
    avg_latency = statistics.mean(latencies)
    assert avg_latency < 100  # Under 100ms average
```

## 🔧 Test Configuration and Fixtures

### Shared Fixtures (`conftest.py`)

**Key Fixtures**:
- `mock_embedding_function`: Consistent mock embeddings
- `test_chromadb_client`: In-memory ChromaDB client
- `test_collection`: Pre-populated test collection
- `api_client`: FastAPI TestClient
- `async_api_client`: Async HTTP client
- `sample_employee_data`: Realistic test data
- `test_data_generator`: Utility for generating test data

**Configuration**:
- Automatic test environment setup
- Pytest markers for test categorization
- Coverage reporting configuration
- Async test support

## 📊 Test Metrics and Coverage

### Coverage Targets

- **Overall**: > 90% line coverage
- **Core Logic**: > 95% coverage
- **API Endpoints**: 100% endpoint coverage
- **Error Paths**: > 85% error handling coverage

### Performance Benchmarks

- **Similarity Search**: < 100ms average latency
- **Filter Search**: < 50ms average latency
- **Advanced Search**: < 150ms average latency
- **Concurrent Load**: > 95% success rate under 10 concurrent users
- **Memory Usage**: < 50MB increase during sustained load

## 🐛 Test-Driven Development Workflow

### 1. Red-Green-Refactor Cycle

```bash
# 1. Write failing test
pytest tests/unit/test_new_feature.py::test_new_functionality -v

# 2. Implement minimum code to pass
# Edit source code...

# 3. Verify test passes
pytest tests/unit/test_new_feature.py::test_new_functionality -v

# 4. Refactor and ensure all tests still pass
pytest tests/ -x
```

### 2. Test-First API Development

```python
# 1. Write API test first
def test_new_endpoint(self, api_client):
    response = api_client.post("/new-endpoint", json={
        "param": "value"
    })
    assert response.status_code == 200
    assert response.json()["result"] == "expected"

# 2. Implement endpoint to pass test
@app.post("/new-endpoint")
async def new_endpoint(request: NewRequest):
    return {"result": "expected"}
```

## 🔍 Debugging Tests

### Common Issues and Solutions

**Test Failures**:
```bash
# Run single failing test with verbose output
pytest tests/path/to/test.py::test_name -v -s

# Debug with pdb
pytest tests/path/to/test.py::test_name --pdb

# Show print statements
pytest tests/path/to/test.py::test_name -s
```

**Mock Issues**:
```python
# Verify mock calls
with patch('api.main.collection') as mock_collection:
    mock_collection.query.return_value = {...}
    
    # Your test code
    
    # Verify mock was called correctly
    mock_collection.query.assert_called_once_with(
        query_texts=["expected query"],
        n_results=5
    )
```

**Async Test Issues**:
```python
# Use async fixtures and proper async syntax
@pytest.mark.asyncio
async def test_async_endpoint(self, async_api_client):
    response = await async_api_client.get("/endpoint")
    assert response.status_code == 200
```

## 📈 Continuous Integration

### Test Pipeline

```bash
# CI Pipeline stages
1. Install dependencies: pixi install
2. Lint and format: ruff check .
3. Unit tests: pixi run test-unit
4. Integration tests: pixi run test-integration  
5. API tests: pixi run test-api-tests
6. E2E tests: pixi run test-e2e
7. Performance tests: pixi run test-perf (optional)
8. Coverage report: pixi run test-cov
```

### Quality Gates

- All tests must pass
- Coverage > 90%
- No critical security issues
- Performance benchmarks within limits
- No breaking API changes

## 🎓 Best Practices

### Test Writing Guidelines

1. **Descriptive Test Names**
   ```python
   # Good
   def test_similarity_search_returns_empty_results_for_nonexistent_query(self):
   
   # Bad  
   def test_search(self):
   ```

2. **Clear Test Structure**
   ```python
   def test_something(self):
       # Arrange
       test_data = create_test_data()
       
       # Act
       result = function_under_test(test_data)
       
       # Assert
       assert result.status == "expected"
   ```

3. **Independent Tests**
   - Each test should be independent
   - Use fixtures for shared setup
   - Clean up after tests

4. **Test Edge Cases**
   - Empty inputs
   - Boundary values
   - Error conditions
   - Concurrent access

5. **Mock External Dependencies**
   - Use mocks for ChromaDB in unit tests
   - Mock network calls
   - Control external system behavior

This comprehensive test suite ensures the similarity search system is robust, performant, and maintainable - exactly what a senior software engineer would implement for a production system.