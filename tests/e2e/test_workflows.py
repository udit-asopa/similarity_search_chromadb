"""
End-to-end tests covering complete user workflows and system integration.
Tests real user scenarios from start to finish.
"""
import pytest
import asyncio
import time
from unittest.mock import patch
import httpx


@pytest.mark.e2e
class TestCompleteUserWorkflows:
    """Test complete user workflows from start to finish."""
    
    @pytest.mark.asyncio
    async def test_new_user_complete_workflow(self, async_api_client):
        """Test complete workflow for a new user discovering the system."""
        
        # Step 1: User checks if API is healthy
        health_response = await async_api_client.get("/health")
        
        if health_response.status_code == 200:
            health_data = health_response.json()
            assert health_data["status"] == "healthy"
            employee_count = health_data["employee_count"]
        else:
            # If health check fails, we can't proceed with meaningful tests
            pytest.skip("API health check failed - system not ready")
        
        # Step 2: User explores available employees
        employees_response = await async_api_client.get("/employees?limit=10")
        assert employees_response.status_code == 200
        
        employees_data = employees_response.json()
        assert "employees" in employees_data
        assert employees_data["total_employees"] > 0
        
        # Step 3: User performs first similarity search
        search_request = {
            "query": "Python developer with web experience",
            "n_results": 5
        }
        
        search_response = await async_api_client.post("/search/similarity", json=search_request)
        assert search_response.status_code == 200
        
        search_data = search_response.json()
        assert search_data["search_type"] == "similarity"
        assert "results" in search_data
        
        # Step 4: User refines search with filters
        filter_response = await async_api_client.get("/search/filter?department=Engineering&experience_min=3")
        assert filter_response.status_code == 200
        
        filter_data = filter_response.json()
        assert filter_data["search_type"] == "filter"
        
        # Step 5: User performs advanced search combining both
        advanced_request = {
            "query": "senior software engineer",
            "filters": {
                "department": "Engineering",
                "experience_min": 5
            },
            "n_results": 3
        }
        
        advanced_response = await async_api_client.post("/search/advanced", json=advanced_request)
        assert advanced_response.status_code == 200
        
        advanced_data = advanced_response.json()
        assert advanced_data["search_type"] == "advanced"
        
        # Step 6: User checks system statistics
        stats_response = await async_api_client.get("/stats")
        assert stats_response.status_code == 200
        
        stats_data = stats_response.json()
        if "total_employees" in stats_data:
            assert stats_data["total_employees"] > 0
            assert "departments" in stats_data
    
    def test_hr_recruiter_workflow(self, api_client):
        """Test workflow for HR recruiter finding candidates."""
        
        # HR recruiter looking for specific role
        # Step 1: Search for Python developers
        python_search = {
            "query": "Python developer full stack web development",
            "n_results": 10
        }
        
        with patch('api.main.collection') as mock_collection:
            mock_collection.query.return_value = {
                "ids": [["emp_1", "emp_2"]],
                "documents": [["Python dev doc", "Full stack doc"]],
                "metadatas": [[
                    {"name": "Alice Python", "department": "Engineering", "role": "Python Developer", 
                     "experience": 5, "location": "San Francisco", "employment_type": "Full-time"},
                    {"name": "Bob Fullstack", "department": "Engineering", "role": "Full Stack Developer",
                     "experience": 7, "location": "Seattle", "employment_type": "Full-time"}
                ]],
                "distances": [[0.2, 0.3]]
            }
            
            response = api_client.post("/search/similarity", json=python_search)
            assert response.status_code == 200
            
            data = response.json()
            python_developers = data["results"]
            assert len(python_developers) == 2
        
        # Step 2: Filter by location preferences
        location_filter_response = api_client.get("/search/filter?department=Engineering&location=San Francisco")
        assert location_filter_response.status_code == 200
        
        # Step 3: Look for senior candidates
        senior_search = {
            "query": "senior python architect technical lead",
            "filters": {
                "department": "Engineering",
                "experience_min": 8
            },
            "n_results": 5
        }
        
        with patch('api.main.collection') as mock_collection:
            mock_collection.query.return_value = {
                "ids": [["senior_1"]],
                "documents": [["Senior architect doc"]],
                "metadatas": [[
                    {"name": "Carol Senior", "department": "Engineering", "role": "Senior Architect",
                     "experience": 12, "location": "San Francisco", "employment_type": "Full-time"}
                ]],
                "distances": [[0.15]]
            }
            
            senior_response = api_client.post("/search/advanced", json=senior_search)
            assert senior_response.status_code == 200
            
            senior_data = senior_response.json()
            senior_candidates = senior_data["results"]
            
            # Validate senior candidates meet criteria
            for candidate in senior_candidates:
                assert candidate["experience"] >= 8
                assert candidate["department"] == "Engineering"
    
    def test_team_lead_finding_team_members(self, api_client):
        """Test workflow for team lead building a diverse team."""
        
        # Team lead wants to build a team with different skill sets
        skill_searches = [
            {"query": "frontend developer React Vue.js", "n_results": 3},
            {"query": "backend developer API microservices", "n_results": 3},
            {"query": "DevOps engineer CI/CD cloud infrastructure", "n_results": 2},
            {"query": "UX designer user interface design", "n_results": 2}
        ]
        
        team_candidates = {}
        
        for i, search in enumerate(skill_searches):
            with patch('api.main.collection') as mock_collection:
                mock_collection.query.return_value = {
                    "ids": [[f"skill_{i}_1", f"skill_{i}_2"]],
                    "documents": [[f"Skill {i} doc 1", f"Skill {i} doc 2"]],
                    "metadatas": [[
                        {"name": f"Expert {i}_1", "department": "Engineering", "role": f"Specialist {i}",
                         "experience": 4, "location": "San Francisco", "employment_type": "Full-time"},
                        {"name": f"Expert {i}_2", "department": "Engineering", "role": f"Senior {i}",
                         "experience": 6, "location": "Seattle", "employment_type": "Full-time"}
                    ]],
                    "distances": [[0.2, 0.3]]
                }
                
                response = api_client.post("/search/similarity", json=search)
                assert response.status_code == 200
                
                data = response.json()
                team_candidates[f"skill_{i}"] = data["results"]
        
        # Verify we found candidates for each skill area
        assert len(team_candidates) == 4
        for skill_area, candidates in team_candidates.items():
            assert len(candidates) >= 1
    
    def test_system_admin_monitoring_workflow(self, api_client):
        """Test workflow for system administrator monitoring system health."""
        
        # Step 1: Check overall system health
        health_response = api_client.get("/health")
        
        if health_response.status_code == 200:
            health_data = health_response.json()
            assert "chromadb_status" in health_data
            assert "employee_count" in health_data
            assert "embedding_model" in health_data
        else:
            # System is down - admin needs to investigate
            assert health_response.status_code == 503
            error_data = health_response.json()
            assert "detail" in error_data
        
        # Step 2: Check system statistics
        stats_response = api_client.get("/stats")
        
        if stats_response.status_code == 200:
            stats_data = stats_response.json()
            
            # Validate data distribution looks reasonable
            if "departments" in stats_data:
                departments = stats_data["departments"]
                assert isinstance(departments, dict)
                assert len(departments) > 0
                
                # Check for reasonable department distribution
                total_employees = sum(departments.values())
                assert total_employees > 0
        
        # Step 3: Test search functionality
        test_search = {
            "query": "test search for monitoring",
            "n_results": 1
        }
        
        search_response = api_client.post("/search/similarity", json=test_search)
        
        # Should either work or fail gracefully
        assert search_response.status_code in [200, 500, 503]
        
        if search_response.status_code == 200:
            # Search is working
            search_data = search_response.json()
            assert "results" in search_data
        else:
            # Search is failing - admin should investigate
            error_data = search_response.json()
            assert "detail" in error_data


@pytest.mark.e2e
class TestErrorRecoveryWorkflows:
    """Test error recovery and resilience workflows."""
    
    def test_graceful_degradation_workflow(self, api_client):
        """Test system behavior when components are failing."""
        
        # Test 1: ChromaDB is unavailable
        with patch('api.main.collection') as mock_collection:
            mock_collection.query.side_effect = Exception("ChromaDB connection failed")
            
            response = api_client.post("/search/similarity", json={
                "query": "test query",
                "n_results": 5
            })
            
            # Should return 500 error with meaningful message
            assert response.status_code == 500
            error_data = response.json()
            assert "Search failed" in error_data["detail"]
        
        # Test 2: Empty results handling
        with patch('api.main.collection') as mock_collection:
            mock_collection.query.return_value = {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }
            
            response = api_client.post("/search/similarity", json={
                "query": "nonexistent query",
                "n_results": 5
            })
            
            # Should handle empty results gracefully
            assert response.status_code == 200
            data = response.json()
            assert data["total_results"] == 0
            assert data["results"] == []
    
    def test_invalid_input_recovery(self, api_client):
        """Test recovery from various invalid inputs."""
        
        # Test invalid JSON
        response = api_client.post(
            "/search/similarity",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        
        # Test missing required fields
        response = api_client.post("/search/similarity", json={})
        assert response.status_code == 422
        
        # Test invalid field values
        response = api_client.post("/search/similarity", json={
            "query": "test",
            "n_results": -1
        })
        assert response.status_code == 422
        
        # Test very long query
        long_query = "x" * 10000
        response = api_client.post("/search/similarity", json={
            "query": long_query,
            "n_results": 5
        })
        # Should either work or fail gracefully
        assert response.status_code in [200, 422, 500]
    
    def test_concurrent_request_handling(self, api_client):
        """Test system behavior under concurrent requests."""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request():
            try:
                with patch('api.main.collection') as mock_collection:
                    mock_collection.query.return_value = {
                        "ids": [["emp_1"]],
                        "documents": [["Test doc"]],
                        "metadatas": [[{"name": "Test", "department": "Engineering", 
                                      "role": "Engineer", "experience": 5, "location": "SF",
                                      "employment_type": "Full-time"}]],
                        "distances": [[0.5]]
                    }
                    
                    response = api_client.post("/search/similarity", json={
                        "query": "concurrent test query",
                        "n_results": 1
                    })
                    results.append((response.status_code, response.json()))
            except Exception as e:
                errors.append(str(e))
        
        # Make 10 concurrent requests
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout
        
        # Verify most requests succeeded
        successful_requests = [r for r in results if r[0] == 200]
        assert len(successful_requests) >= 8  # At least 80% success rate
        
        # Verify no major errors
        assert len(errors) <= 2  # Allow for minor threading issues


@pytest.mark.e2e
class TestIntegrationWithExternalSystems:
    """Test integration scenarios with external systems."""
    
    def test_api_as_microservice(self, api_client):
        """Test using the API as part of a larger microservice architecture."""
        
        # Simulate external service calling our API
        def external_service_call(employee_query, filters=None):
            """Simulate an external service using our API."""
            if filters:
                request_data = {
                    "query": employee_query,
                    "filters": filters,
                    "n_results": 5
                }
                response = api_client.post("/search/advanced", json=request_data)
            else:
                request_data = {
                    "query": employee_query,
                    "n_results": 5
                }
                response = api_client.post("/search/similarity", json=request_data)
            
            return response.status_code, response.json()
        
        # Test various external service scenarios
        test_scenarios = [
            ("Python developer", None),
            ("marketing manager", {"department": "Marketing"}),
            ("senior engineer", {"experience_min": 8}),
            ("team lead architect", {"department": "Engineering", "experience_min": 10})
        ]
        
        for query, filters in test_scenarios:
            with patch('api.main.collection') as mock_collection:
                mock_collection.query.return_value = {
                    "ids": [["result_1"]],
                    "documents": [["Mock result"]],
                    "metadatas": [[{"name": "Mock Employee", "department": "Engineering",
                                  "role": "Mock Role", "experience": 5, "location": "Mock City",
                                  "employment_type": "Full-time"}]],
                    "distances": [[0.3]]
                }
                
                status_code, response_data = external_service_call(query, filters)
                
                # External service should get reliable responses
                assert status_code == 200
                assert "results" in response_data
                assert response_data["total_results"] >= 0
    
    def test_batch_processing_integration(self, api_client):
        """Test batch processing scenarios for bulk operations."""
        
        # Simulate batch processing of multiple queries
        batch_queries = [
            "Python developer",
            "Java engineer", 
            "React frontend developer",
            "DevOps engineer",
            "Data scientist"
        ]
        
        batch_results = []
        
        for query in batch_queries:
            with patch('api.main.collection') as mock_collection:
                mock_collection.query.return_value = {
                    "ids": [[f"batch_result_{hash(query) % 1000}"]],
                    "documents": [[f"Result for {query}"]],
                    "metadatas": [[{"name": f"Employee for {query}", "department": "Engineering",
                                  "role": query, "experience": 5, "location": "Test City",
                                  "employment_type": "Full-time"}]],
                    "distances": [[0.2]]
                }
                
                response = api_client.post("/search/similarity", json={
                    "query": query,
                    "n_results": 3
                })
                
                assert response.status_code == 200
                batch_results.append(response.json())
        
        # Verify batch processing completed successfully
        assert len(batch_results) == len(batch_queries)
        
        # Verify each result has expected structure
        for result in batch_results:
            assert "query" in result
            assert "results" in result
            assert "search_type" in result


@pytest.mark.e2e
@pytest.mark.slow
class TestPerformanceWorkflows:
    """Test performance characteristics under realistic workloads."""
    
    def test_sustained_load_workflow(self, api_client):
        """Test system behavior under sustained load."""
        import time
        
        # Perform sustained requests over time
        request_times = []
        error_count = 0
        
        for i in range(50):  # 50 requests
            start_time = time.time()
            
            with patch('api.main.collection') as mock_collection:
                mock_collection.query.return_value = {
                    "ids": [[f"load_test_{i}"]],
                    "documents": [[f"Load test document {i}"]],
                    "metadatas": [[{"name": f"Load Test {i}", "department": "Engineering",
                                  "role": "Test Role", "experience": 5, "location": "Test",
                                  "employment_type": "Full-time"}]],
                    "distances": [[0.3]]
                }
                
                try:
                    response = api_client.post("/search/similarity", json={
                        "query": f"load test query {i}",
                        "n_results": 3
                    })
                    
                    if response.status_code != 200:
                        error_count += 1
                        
                except Exception:
                    error_count += 1
            
            end_time = time.time()
            request_times.append(end_time - start_time)
            
            # Small delay between requests
            time.sleep(0.01)
        
        # Analyze performance
        avg_response_time = sum(request_times) / len(request_times)
        max_response_time = max(request_times)
        
        # Performance assertions
        assert avg_response_time < 0.5  # Average under 500ms
        assert max_response_time < 2.0   # Max under 2 seconds
        assert error_count < 5           # Less than 10% error rate
    
    def test_large_result_set_workflow(self, api_client):
        """Test handling of large result sets."""
        
        # Test requesting maximum allowed results
        with patch('api.main.collection') as mock_collection:
            # Mock large result set
            large_results = {
                "ids": [[f"large_result_{i}" for i in range(20)]],
                "documents": [[f"Large result document {i}" for i in range(20)]],
                "metadatas": [[{
                    "name": f"Large Result {i}",
                    "department": "Engineering",
                    "role": "Test Role",
                    "experience": i % 15,
                    "location": "Test City",
                    "employment_type": "Full-time"
                } for i in range(20)]],
                "distances": [[0.1 + (i * 0.01) for i in range(20)]]
            }
            
            mock_collection.query.return_value = large_results
            
            start_time = time.time()
            response = api_client.post("/search/similarity", json={
                "query": "large result set test",
                "n_results": 20  # Maximum allowed
            })
            end_time = time.time()
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify large result set is handled properly
            assert data["total_results"] == 20
            assert len(data["results"]) == 20
            
            # Performance should still be reasonable
            response_time = end_time - start_time
            assert response_time < 1.0  # Under 1 second even for large results