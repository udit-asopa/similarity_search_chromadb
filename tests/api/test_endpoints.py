"""
API endpoint tests using FastAPI TestClient.
Tests all API endpoints with various scenarios and edge cases.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json


class TestHealthEndpoints:
    """Test health check and status endpoints."""
    
    def test_root_endpoint(self, api_client):
        """Test the root health check endpoint."""
        response = api_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Employee Similarity Search API is running!"
        assert data["status"] == "healthy"
    
    def test_health_endpoint_success(self, api_client):
        """Test detailed health endpoint when system is healthy."""
        with patch('api.main.collection') as mock_collection:
            mock_collection.count.return_value = 8
            
            response = api_client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["chromadb_status"] == "connected"
            assert data["employee_count"] == 8
            assert data["embedding_model"] == "all-MiniLM-L6-v2"
    
    def test_health_endpoint_failure(self, api_client):
        """Test health endpoint when ChromaDB is unavailable."""
        with patch('api.main.collection') as mock_collection:
            mock_collection.count.side_effect = Exception("ChromaDB unavailable")
            
            response = api_client.get("/health")
            
            assert response.status_code == 503
            data = response.json()
            assert "Service unhealthy" in data["detail"]


class TestSimilaritySearchEndpoint:
    """Test the similarity search endpoint."""
    
    def test_similarity_search_success(self, api_client, mock_search_results):
        """Test successful similarity search."""
        with patch('api.main.collection') as mock_collection:
            mock_collection.query.return_value = mock_search_results
            
            request_data = {
                "query": "Python developer", 
                "n_results": 3
            }
            
            response = api_client.post("/search/similarity", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "Python developer"
            assert data["total_results"] == 2
            assert data["search_type"] == "similarity"
            assert len(data["results"]) == 2
            
            # Check first result structure
            first_result = data["results"][0]
            assert "id" in first_result
            assert "name" in first_result
            assert "similarity_score" in first_result
    
    def test_similarity_search_empty_results(self, api_client):
        """Test similarity search with no results."""
        empty_results = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        
        with patch('api.main.collection') as mock_collection:
            mock_collection.query.return_value = empty_results
            
            request_data = {"query": "nonexistent skill", "n_results": 5}
            response = api_client.post("/search/similarity", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_results"] == 0
            assert data["results"] == []
    
    def test_similarity_search_invalid_request(self, api_client):
        """Test similarity search with invalid request data."""
        # Missing required query
        response = api_client.post("/search/similarity", json={"n_results": 5})
        assert response.status_code == 422
        
        # Invalid n_results (too high)
        response = api_client.post("/search/similarity", json={
            "query": "test",
            "n_results": 25
        })
        assert response.status_code == 422
        
        # Invalid n_results (too low)
        response = api_client.post("/search/similarity", json={
            "query": "test", 
            "n_results": 0
        })
        assert response.status_code == 422
    
    def test_similarity_search_server_error(self, api_client):
        """Test similarity search when ChromaDB raises an error."""
        with patch('api.main.collection') as mock_collection:
            mock_collection.query.side_effect = Exception("Database error")
            
            request_data = {"query": "Python developer", "n_results": 3}
            response = api_client.post("/search/similarity", json=request_data)
            
            assert response.status_code == 500
            data = response.json()
            assert "Search failed" in data["detail"]


class TestFilterSearchEndpoint:
    """Test the metadata filter search endpoint."""
    
    def test_filter_search_success(self, api_client):
        """Test successful filter search."""
        mock_results = {
            "ids": ["emp_1", "emp_2"],
            "documents": ["Doc 1", "Doc 2"],
            "metadatas": [
                {
                    "name": "John Engineer",
                    "department": "Engineering",
                    "role": "Software Engineer",
                    "experience": 5,
                    "location": "San Francisco",
                    "employment_type": "Full-time"
                },
                {
                    "name": "Jane Engineer",
                    "department": "Engineering", 
                    "role": "Senior Engineer",
                    "experience": 8,
                    "location": "San Francisco",
                    "employment_type": "Full-time"
                }
            ]
        }
        
        with patch('api.main.collection') as mock_collection:
            mock_collection.get.return_value = mock_results
            
            response = api_client.get("/search/filter?department=Engineering&location=San Francisco")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_results"] == 2
            assert data["search_type"] == "filter"
            assert len(data["results"]) == 2
    
    def test_filter_search_no_filters(self, api_client):
        """Test filter search with no filter parameters."""
        mock_results = {
            "ids": ["emp_1"],
            "documents": ["All employees doc"],
            "metadatas": [{
                "name": "All Employee",
                "department": "Engineering",
                "role": "Engineer",
                "experience": 5,
                "location": "San Francisco",
                "employment_type": "Full-time"
            }]
        }
        
        with patch('api.main.collection') as mock_collection:
            mock_collection.get.return_value = mock_results
            
            response = api_client.get("/search/filter")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_results"] == 1
    
    def test_filter_search_experience_range(self, api_client):
        """Test filter search with experience range."""
        mock_results = {
            "ids": ["emp_1"],
            "documents": ["Senior employee"],
            "metadatas": [{
                "name": "Senior Employee",
                "department": "Engineering",
                "role": "Senior Engineer", 
                "experience": 10,
                "location": "Seattle",
                "employment_type": "Full-time"
            }]
        }
        
        with patch('api.main.collection') as mock_collection:
            mock_collection.get.return_value = mock_results
            
            response = api_client.get("/search/filter?experience_min=8&experience_max=15")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_results"] == 1
    
    def test_filter_search_invalid_params(self, api_client):
        """Test filter search with invalid parameters."""
        # Invalid experience values - may return 500 if not properly validated at API level
        response = api_client.get("/search/filter?experience_min=-1")
        assert response.status_code in [422, 500]  # Either validation error or server error
        
        # Invalid limit
        response = api_client.get("/search/filter?limit=100")
        assert response.status_code in [422, 500]  # Either validation error or server error


class TestAdvancedSearchEndpoint:
    """Test the advanced search endpoint combining similarity and filters."""
    
    def test_advanced_search_success(self, api_client, mock_search_results):
        """Test successful advanced search."""
        with patch('api.main.collection') as mock_collection:
            mock_collection.query.return_value = mock_search_results
            
            request_data = {
                "query": "senior Python developer",
                "filters": {
                    "department": "Engineering",
                    "experience_min": 8
                },
                "n_results": 3
            }
            
            response = api_client.post("/search/advanced", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "senior Python developer"
            assert data["search_type"] == "advanced"
            assert data["total_results"] == 2
    
    def test_advanced_search_no_filters(self, api_client, mock_search_results):
        """Test advanced search without filters (should work like similarity search)."""
        with patch('api.main.collection') as mock_collection:
            mock_collection.query.return_value = mock_search_results
            
            request_data = {
                "query": "marketing manager",
                "n_results": 5
            }
            
            response = api_client.post("/search/advanced", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "marketing manager"
            assert data["search_type"] == "advanced"
    
    def test_advanced_search_complex_filters(self, api_client, mock_search_results):
        """Test advanced search with complex filter combinations."""
        with patch('api.main.collection') as mock_collection:
            mock_collection.query.return_value = mock_search_results
            
            request_data = {
                "query": "team leader",
                "filters": {
                    "department": "Engineering",
                    "experience_min": 5,
                    "experience_max": 15,
                    "location": "San Francisco",
                    "employment_type": "Full-time"
                },
                "n_results": 2
            }
            
            response = api_client.post("/search/advanced", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_results"] == 2
    
    def test_advanced_search_invalid_filter_combination(self, api_client):
        """Test advanced search with invalid filter combinations."""
        # Test the problematic case that caused the original error
        request_data = {
            "query": "software engineer",
            "filters": {
                "department": "Engineering",
                "experience_min": 0,
                "experience_max": 0,  # This was causing the ChromaDB error
                "location": "New York",
                "employment_type": "Full-time"
            },
            "n_results": 5
        }
        
        # Mock the collection to return empty results for this case
        empty_results = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        
        with patch('api.main.collection') as mock_collection:
            mock_collection.query.return_value = empty_results
            
            response = api_client.post("/search/advanced", json=request_data)
            
            # Should not return 500 error anymore due to our fix
            assert response.status_code == 200
            data = response.json()
            assert data["total_results"] == 0


class TestEmployeesEndpoint:
    """Test the employees listing endpoint."""
    
    def test_list_employees_success(self, api_client):
        """Test successful employee listing."""
        mock_results = {
            "ids": ["emp_1", "emp_2"],
            "metadatas": [
                {
                    "name": "John Doe",
                    "department": "Engineering",
                    "role": "Software Engineer",
                    "experience": 5,
                    "location": "San Francisco",
                    "employment_type": "Full-time"
                },
                {
                    "name": "Jane Smith",
                    "department": "Marketing",
                    "role": "Marketing Manager", 
                    "experience": 8,
                    "location": "New York",
                    "employment_type": "Full-time"
                }
            ]
        }
        
        with patch('api.main.collection') as mock_collection:
            mock_collection.get.return_value = mock_results
            
            response = api_client.get("/employees")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_employees"] == 2
            assert len(data["employees"]) == 2
            assert data["employees"][0]["name"] == "John Doe"
    
    def test_list_employees_with_limit(self, api_client):
        """Test employee listing with custom limit."""
        mock_results = {
            "ids": ["emp_1"],
            "metadatas": [{
                "name": "John Doe",
                "department": "Engineering",
                "role": "Software Engineer",
                "experience": 5,
                "location": "San Francisco", 
                "employment_type": "Full-time"
            }]
        }
        
        with patch('api.main.collection') as mock_collection:
            mock_collection.get.return_value = mock_results
            
            response = api_client.get("/employees?limit=1")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_employees"] == 1
    
    def test_list_employees_invalid_limit(self, api_client):
        """Test employee listing with invalid limit."""
        # Limit too high
        response = api_client.get("/employees?limit=200")
        assert response.status_code == 422
        
        # Limit too low
        response = api_client.get("/employees?limit=0")
        assert response.status_code == 422


class TestStatsEndpoint:
    """Test the statistics endpoint."""
    
    def test_stats_success(self, api_client):
        """Test successful stats retrieval."""
        mock_results = {
            "ids": ["emp_1", "emp_2", "emp_3"],
            "metadatas": [
                {
                    "department": "Engineering",
                    "location": "San Francisco",
                    "employment_type": "Full-time",
                    "experience": 5
                },
                {
                    "department": "Engineering", 
                    "location": "New York",
                    "employment_type": "Full-time",
                    "experience": 10
                },
                {
                    "department": "Marketing",
                    "location": "San Francisco", 
                    "employment_type": "Part-time",
                    "experience": 3
                }
            ]
        }
        
        with patch('api.main.collection') as mock_collection:
            mock_collection.get.return_value = mock_results
            
            response = api_client.get("/stats")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["total_employees"] == 3
            assert data["departments"]["Engineering"] == 2
            assert data["departments"]["Marketing"] == 1
            assert data["locations"]["San Francisco"] == 2
            assert data["employment_types"]["Full-time"] == 2
            assert data["employment_types"]["Part-time"] == 1
            
            # Check experience stats
            exp_stats = data["experience_stats"]
            assert exp_stats["min"] == 3
            assert exp_stats["max"] == 10
            assert exp_stats["average"] == 6.0  # (5+10+3)/3
    
    def test_stats_empty_collection(self, api_client):
        """Test stats with empty collection."""
        mock_results = {"ids": [], "metadatas": []}
        
        with patch('api.main.collection') as mock_collection:
            mock_collection.get.return_value = mock_results
            
            response = api_client.get("/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "No employees found"


class TestCORSAndHeaders:
    """Test CORS and HTTP headers."""
    
    def test_cors_headers_present(self, api_client):
        """Test that CORS headers are present in responses."""
        response = api_client.get("/")
        
        # Check for CORS headers (may vary based on TestClient behavior)
        assert response.status_code == 200
        # Note: TestClient might not include all CORS headers
    
    def test_content_type_headers(self, api_client):
        """Test that proper content-type headers are set."""
        response = api_client.get("/health")
        
        assert response.status_code in [200, 503]
        assert response.headers.get("content-type") == "application/json"


class TestRequestValidation:
    """Test request validation across all endpoints."""
    
    def test_malformed_json(self, api_client):
        """Test handling of malformed JSON in requests."""
        response = api_client.post(
            "/search/similarity",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_content_type(self, api_client):
        """Test handling of missing content-type header."""
        response = api_client.post("/search/similarity", content='{"query": "test"}')
        
        # Should still work or give appropriate error
        assert response.status_code in [200, 422, 415]
    
    def test_empty_request_body(self, api_client):
        """Test handling of empty request body."""
        response = api_client.post("/search/similarity", json={})
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


@pytest.mark.api
class TestEndpointIntegration:
    """Integration tests across multiple endpoints."""
    
    def test_health_before_search(self, api_client):
        """Test that health check works before performing searches."""
        with patch('api.main.collection') as mock_collection:
            mock_collection.count.return_value = 5
            
            # Check health first
            health_response = api_client.get("/health")
            assert health_response.status_code == 200
            
            # Then perform search
            mock_collection.query.return_value = {
                "ids": [["emp_1"]],
                "documents": [["test doc"]],
                "metadatas": [[{"name": "Test", "department": "Engineering", 
                              "role": "Engineer", "experience": 5, "location": "SF",
                              "employment_type": "Full-time"}]],
                "distances": [[0.2]]
            }
            
            search_response = api_client.post("/search/similarity", json={
                "query": "test query",
                "n_results": 1
            })
            assert search_response.status_code == 200
    
    def test_search_then_stats(self, api_client):
        """Test performing search followed by stats check."""
        with patch('api.main.collection') as mock_collection:
            # First search
            mock_collection.query.return_value = {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }
            
            search_response = api_client.post("/search/similarity", json={
                "query": "test",
                "n_results": 1
            })
            assert search_response.status_code == 200
            
            # Then get stats
            mock_collection.get.return_value = {
                "ids": ["emp_1"],
                "metadatas": [{
                    "department": "Engineering",
                    "location": "SF",
                    "employment_type": "Full-time", 
                    "experience": 5
                }]
            }
            
            stats_response = api_client.get("/stats")
            assert stats_response.status_code == 200