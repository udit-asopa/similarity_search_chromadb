"""
Pytest configuration and shared fixtures for the similarity search system.
Provides common test utilities, fixtures, and configuration.
"""
import os
import pytest
import asyncio
from typing import Generator, AsyncGenerator
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import chromadb
from chromadb.utils import embedding_functions

# Set test environment variables
os.environ["TESTING"] = "1"
os.environ["CHROMADB_TEST_MODE"] = "1"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def mock_embedding_function():
    """Mock embedding function that returns predictable vectors."""
    class MockEmbeddingFunction:
        def __call__(self, input):
            # ChromaDB expects 'input' parameter name
            texts = input if isinstance(input, list) else [input]
            embeddings = []
            for text in texts:
                # Create reproducible embeddings based on text hash
                hash_val = hash(text) % 1000000
                # Generate 384-dimensional vector (matching all-MiniLM-L6-v2)
                vector = [(hash_val + i) / 1000000.0 for i in range(384)]
                embeddings.append(vector)
            return embeddings
    
    return MockEmbeddingFunction()


@pytest.fixture(scope="function")
def test_chromadb_client():
    """Create a test ChromaDB client with in-memory storage."""
    client = chromadb.Client()
    yield client
    # Cleanup: delete all collections
    try:
        collections = client.list_collections()
        for collection in collections:
            client.delete_collection(collection.name)
    except Exception:
        pass


@pytest.fixture(scope="function")
def test_collection(test_chromadb_client, mock_embedding_function):
    """Create a test collection with sample data."""
    try:
        # Delete collection if it exists
        test_chromadb_client.delete_collection("test_collection")
    except Exception:
        pass
    
    collection = test_chromadb_client.create_collection(
        name="test_collection",
        embedding_function=mock_embedding_function
    )
    
    # Add sample test data
    sample_employees = [
        {
            "id": "test_emp_1",
            "name": "Alice Test",
            "department": "Engineering", 
            "role": "Software Engineer",
            "experience": 5,
            "location": "San Francisco",
            "employment_type": "Full-time"
        },
        {
            "id": "test_emp_2", 
            "name": "Bob Test",
            "department": "Marketing",
            "role": "Marketing Manager", 
            "experience": 8,
            "location": "New York",
            "employment_type": "Full-time"
        }
    ]
    
    documents = [
        f"{emp['role']} with {emp['experience']} years in {emp['department']}. Located in {emp['location']}."
        for emp in sample_employees
    ]
    
    collection.add(
        ids=[emp["id"] for emp in sample_employees],
        documents=documents,
        metadatas=[{
            "name": emp["name"],
            "department": emp["department"], 
            "role": emp["role"],
            "experience": emp["experience"],
            "location": emp["location"],
            "employment_type": emp["employment_type"]
        } for emp in sample_employees]
    )
    
    yield collection


@pytest.fixture(scope="function") 
def api_client():
    """Create a test client for the FastAPI application."""
    import sys
    import os
    
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        from api.main import app
        with TestClient(app) as client:
            yield client
    except ImportError:
        # If we can't import the API, skip API tests
        pytest.skip("API module not available - skipping API tests")


@pytest.fixture(scope="function")
async def async_api_client():
    """Create an async test client for the FastAPI application."""
    import sys
    import os
    import httpx
    
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        from api.main import app
        from httpx import ASGITransport
        
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    except ImportError:
        # If we can't import the API, skip API tests
        pytest.skip("API module not available - skipping async API tests")


@pytest.fixture(scope="function")
def sample_employee_data():
    """Generate sample employee data for testing."""
    return [
        {
            "id": "emp_1",
            "name": "John Developer",
            "experience": 5,
            "department": "Engineering",
            "role": "Senior Software Engineer", 
            "skills": "Python, React, Node.js, PostgreSQL",
            "location": "San Francisco",
            "employment_type": "Full-time"
        },
        {
            "id": "emp_2",
            "name": "Jane Manager", 
            "experience": 10,
            "department": "Engineering",
            "role": "Engineering Manager",
            "skills": "Team leadership, Architecture, Mentoring",
            "location": "Seattle", 
            "employment_type": "Full-time"
        },
        {
            "id": "emp_3",
            "name": "Mike Marketing",
            "experience": 3,
            "department": "Marketing", 
            "role": "Marketing Specialist",
            "skills": "SEO, Content strategy, Analytics",
            "location": "Austin",
            "employment_type": "Part-time"
        }
    ]


@pytest.fixture(scope="function")
def mock_search_results():
    """Mock search results for testing API responses."""
    return {
        "ids": [["emp_1", "emp_2"]],
        "distances": [[0.2, 0.4]], 
        "documents": [
            [
                "Senior Software Engineer with 5 years in Engineering. Located in San Francisco.",
                "Engineering Manager with 10 years in Engineering. Located in Seattle."
            ]
        ],
        "metadatas": [
            [
                {
                    "name": "John Developer",
                    "department": "Engineering",
                    "role": "Senior Software Engineer",
                    "experience": 5,
                    "location": "San Francisco", 
                    "employment_type": "Full-time"
                },
                {
                    "name": "Jane Manager",
                    "department": "Engineering", 
                    "role": "Engineering Manager",
                    "experience": 10,
                    "location": "Seattle",
                    "employment_type": "Full-time" 
                }
            ]
        ]
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables and cleanup."""
    # Set test-specific environment variables
    original_env = os.environ.copy()
    os.environ["TESTING"] = "1"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def create_employee(
        emp_id: str = "test_emp",
        name: str = "Test Employee", 
        department: str = "Engineering",
        role: str = "Software Engineer",
        experience: int = 5,
        location: str = "San Francisco",
        employment_type: str = "Full-time"
    ) -> dict:
        """Create a single employee record for testing."""
        return {
            "id": emp_id,
            "name": name,
            "department": department,
            "role": role,
            "experience": experience, 
            "location": location,
            "employment_type": employment_type
        }
    
    @staticmethod
    def create_employees(count: int = 5) -> list[dict]:
        """Create multiple employee records for testing."""
        from faker import Faker
        fake = Faker()
        
        departments = ["Engineering", "Marketing", "HR", "Sales"]
        roles = {
            "Engineering": ["Software Engineer", "Senior Engineer", "Engineering Manager"],
            "Marketing": ["Marketing Manager", "Marketing Specialist", "Content Creator"], 
            "HR": ["HR Coordinator", "HR Manager", "Recruiter"],
            "Sales": ["Sales Representative", "Sales Manager", "Account Manager"]
        }
        locations = ["San Francisco", "New York", "Seattle", "Austin", "Boston"]
        
        employees = []
        for i in range(count):
            dept = fake.random_element(departments)
            employees.append({
                "id": f"test_emp_{i+1}",
                "name": fake.name(),
                "department": dept,
                "role": fake.random_element(roles[dept]),
                "experience": fake.random_int(min=1, max=20),
                "location": fake.random_element(locations),
                "employment_type": fake.random_element(["Full-time", "Part-time"])
            })
        
        return employees


@pytest.fixture
def test_data_generator():
    """Provide the test data generator utility."""
    return TestDataGenerator


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    import sys
    import os
    
    # Add project root to Python path
    project_root = os.path.dirname(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test" 
    )
    config.addinivalue_line(
        "markers", "api: mark test as an API test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file paths."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "api" in str(item.fspath):
            item.add_marker(pytest.mark.api)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e) 
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)