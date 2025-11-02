"""
Performance benchmark tests for the similarity search system.
Measures and validates performance characteristics under various conditions.
"""
import pytest
import time
import statistics
from unittest.mock import patch, MagicMock
import threading
import concurrent.futures


@pytest.mark.performance
class TestSearchPerformanceBenchmarks:
    """Benchmark search performance across different scenarios."""
    
    def test_similarity_search_latency(self, api_client):
        """Benchmark similarity search response times."""
        latencies = []
        
        test_queries = [
            "Python developer with web experience",
            "Marketing manager with social media skills",
            "Senior software engineer with leadership",
            "DevOps engineer with cloud expertise",
            "Data scientist with machine learning"
        ]
        
        for query in test_queries:
            with patch('api.main.collection') as mock_collection:
                mock_collection.query.return_value = {
                    "ids": [["perf_1", "perf_2", "perf_3"]],
                    "documents": [["Perf doc 1", "Perf doc 2", "Perf doc 3"]],
                    "metadatas": [[
                        {"name": "Perf 1", "department": "Engineering", "role": "Engineer",
                         "experience": 5, "location": "SF", "employment_type": "Full-time"},
                        {"name": "Perf 2", "department": "Marketing", "role": "Manager", 
                         "experience": 7, "location": "NY", "employment_type": "Full-time"},
                        {"name": "Perf 3", "department": "Engineering", "role": "Senior",
                         "experience": 10, "location": "Seattle", "employment_type": "Full-time"}
                    ]],
                    "distances": [[0.2, 0.3, 0.4]]
                }
                
                start_time = time.perf_counter()
                response = api_client.post("/search/similarity", json={
                    "query": query,
                    "n_results": 5
                })
                end_time = time.perf_counter()
                
                assert response.status_code == 200
                latency = (end_time - start_time) * 1000  # Convert to milliseconds
                latencies.append(latency)
        
        # Performance analysis
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        max_latency = max(latencies)
        
        # Performance assertions
        assert avg_latency < 100  # Average under 100ms
        assert p95_latency < 200  # 95th percentile under 200ms
        assert max_latency < 500  # Max under 500ms
        
        print(f"Similarity Search Performance:")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  95th percentile: {p95_latency:.2f}ms")
        print(f"  Maximum latency: {max_latency:.2f}ms")
    
    def test_filter_search_latency(self, api_client):
        """Benchmark filter search response times."""
        latencies = []
        
        filter_scenarios = [
            "?department=Engineering",
            "?experience_min=5&experience_max=10",
            "?location=San Francisco&employment_type=Full-time",
            "?department=Marketing&experience_min=3",
            "?employment_type=Full-time&experience_min=8"
        ]
        
        for filters in filter_scenarios:
            with patch('api.main.collection') as mock_collection:
                mock_collection.get.return_value = {
                    "ids": ["filter_1", "filter_2"],
                    "documents": ["Filter doc 1", "Filter doc 2"],
                    "metadatas": [
                        {"name": "Filter 1", "department": "Engineering", "role": "Engineer",
                         "experience": 6, "location": "San Francisco", "employment_type": "Full-time"},
                        {"name": "Filter 2", "department": "Engineering", "role": "Senior", 
                         "experience": 9, "location": "San Francisco", "employment_type": "Full-time"}
                    ]
                }
                
                start_time = time.perf_counter()
                response = api_client.get(f"/search/filter{filters}")
                end_time = time.perf_counter()
                
                assert response.status_code == 200
                latency = (end_time - start_time) * 1000
                latencies.append(latency)
        
        # Performance analysis
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        
        # Filter searches should be faster than similarity searches
        assert avg_latency < 50   # Average under 50ms
        assert max_latency < 100  # Max under 100ms
        
        print(f"Filter Search Performance:")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  Maximum latency: {max_latency:.2f}ms")
    
    def test_advanced_search_latency(self, api_client):
        """Benchmark advanced search (similarity + filters) response times."""
        latencies = []
        
        advanced_queries = [
            {
                "query": "senior Python developer",
                "filters": {"department": "Engineering", "experience_min": 8},
                "n_results": 5
            },
            {
                "query": "marketing manager with leadership",
                "filters": {"department": "Marketing", "experience_min": 5},
                "n_results": 3
            },
            {
                "query": "DevOps engineer cloud",
                "filters": {"department": "Engineering", "location": "San Francisco"},
                "n_results": 4
            }
        ]
        
        for query_data in advanced_queries:
            with patch('api.main.collection') as mock_collection:
                mock_collection.query.return_value = {
                    "ids": [["adv_1", "adv_2"]],
                    "documents": [["Advanced doc 1", "Advanced doc 2"]],
                    "metadatas": [[
                        {"name": "Advanced 1", "department": "Engineering", "role": "Senior Engineer",
                         "experience": 10, "location": "San Francisco", "employment_type": "Full-time"},
                        {"name": "Advanced 2", "department": "Engineering", "role": "Architect",
                         "experience": 12, "location": "Seattle", "employment_type": "Full-time"}
                    ]],
                    "distances": [[0.15, 0.25]]
                }
                
                start_time = time.perf_counter()
                response = api_client.post("/search/advanced", json=query_data)
                end_time = time.perf_counter()
                
                assert response.status_code == 200
                latency = (end_time - start_time) * 1000
                latencies.append(latency)
        
        # Performance analysis
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        
        # Advanced search should be comparable to similarity search
        assert avg_latency < 150  # Average under 150ms
        assert max_latency < 300  # Max under 300ms
        
        print(f"Advanced Search Performance:")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  Maximum latency: {max_latency:.2f}ms")


@pytest.mark.performance
class TestConcurrencyBenchmarks:
    """Benchmark system performance under concurrent load."""
    
    def test_concurrent_similarity_searches(self, api_client):
        """Test performance with multiple concurrent similarity searches."""
        num_threads = 10
        requests_per_thread = 5
        results = []
        
        def worker():
            thread_results = []
            for i in range(requests_per_thread):
                with patch('api.main.collection') as mock_collection:
                    mock_collection.query.return_value = {
                        "ids": [[f"conc_{threading.current_thread().ident}_{i}"]],
                        "documents": [[f"Concurrent doc {i}"]],
                        "metadatas": [[{
                            "name": f"Concurrent {i}",
                            "department": "Engineering",
                            "role": "Engineer",
                            "experience": 5,
                            "location": "Test City",
                            "employment_type": "Full-time"
                        }]],
                        "distances": [[0.3]]
                    }
                    
                    start_time = time.perf_counter()
                    try:
                        response = api_client.post("/search/similarity", json={
                            "query": f"concurrent query {i}",
                            "n_results": 3
                        })
                        end_time = time.perf_counter()
                        
                        latency = (end_time - start_time) * 1000
                        success = response.status_code == 200
                        thread_results.append((success, latency))
                        
                    except Exception as e:
                        end_time = time.perf_counter()
                        latency = (end_time - start_time) * 1000
                        thread_results.append((False, latency))
            
            return thread_results
        
        # Execute concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_threads)]
            
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        
        # Analyze results
        successful_requests = [r for r in results if r[0]]
        failed_requests = [r for r in results if not r[0]]
        
        success_rate = len(successful_requests) / len(results)
        avg_latency = statistics.mean([r[1] for r in successful_requests]) if successful_requests else 0
        
        # Performance assertions
        assert success_rate >= 0.95  # At least 95% success rate
        assert avg_latency < 200     # Average latency under 200ms under load
        
        print(f"Concurrent Search Performance:")
        print(f"  Total requests: {len(results)}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Failed requests: {len(failed_requests)}")
        print(f"  Average latency: {avg_latency:.2f}ms")
    
    def test_mixed_workload_performance(self, api_client):
        """Test performance with mixed search types under load."""
        results = {"similarity": [], "filter": [], "advanced": []}
        
        def similarity_worker():
            with patch('api.main.collection') as mock_collection:
                mock_collection.query.return_value = {
                    "ids": [["mix_sim_1"]],
                    "documents": [["Mixed similarity doc"]],
                    "metadatas": [[{"name": "Mixed Sim", "department": "Engineering", 
                                  "role": "Engineer", "experience": 5, "location": "SF",
                                  "employment_type": "Full-time"}]],
                    "distances": [[0.3]]
                }
                
                start_time = time.perf_counter()
                response = api_client.post("/search/similarity", json={
                    "query": "mixed workload test",
                    "n_results": 3
                })
                end_time = time.perf_counter()
                
                latency = (end_time - start_time) * 1000
                return response.status_code == 200, latency
        
        def filter_worker():
            with patch('api.main.collection') as mock_collection:
                mock_collection.get.return_value = {
                    "ids": ["mix_filter_1"],
                    "documents": ["Mixed filter doc"],
                    "metadatas": [{"name": "Mixed Filter", "department": "Engineering",
                                 "role": "Engineer", "experience": 5, "location": "SF", 
                                 "employment_type": "Full-time"}]
                }
                
                start_time = time.perf_counter()
                response = api_client.get("/search/filter?department=Engineering")
                end_time = time.perf_counter()
                
                latency = (end_time - start_time) * 1000
                return response.status_code == 200, latency
        
        def advanced_worker():
            with patch('api.main.collection') as mock_collection:
                mock_collection.query.return_value = {
                    "ids": [["mix_adv_1"]],
                    "documents": [["Mixed advanced doc"]], 
                    "metadatas": [[{"name": "Mixed Advanced", "department": "Engineering",
                                  "role": "Senior Engineer", "experience": 8, "location": "SF",
                                  "employment_type": "Full-time"}]],
                    "distances": [[0.2]]
                }
                
                start_time = time.perf_counter()
                response = api_client.post("/search/advanced", json={
                    "query": "mixed advanced test",
                    "filters": {"department": "Engineering"},
                    "n_results": 3
                })
                end_time = time.perf_counter()
                
                latency = (end_time - start_time) * 1000
                return response.status_code == 200, latency
        
        # Execute mixed workload
        workers = [
            ("similarity", similarity_worker),
            ("filter", filter_worker), 
            ("advanced", advanced_worker)
        ]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = []
            
            # Submit multiple requests of each type
            for _ in range(5):  # 5 iterations
                for search_type, worker_func in workers:
                    future = executor.submit(worker_func)
                    futures.append((search_type, future))
            
            # Collect results
            for search_type, future in futures:
                try:
                    success, latency = future.result(timeout=5)
                    results[search_type].append((success, latency))
                except Exception:
                    results[search_type].append((False, 5000))  # Timeout as 5s latency
        
        # Analyze mixed workload performance
        for search_type, type_results in results.items():
            if type_results:
                successful = [r for r in type_results if r[0]]
                success_rate = len(successful) / len(type_results)
                avg_latency = statistics.mean([r[1] for r in successful]) if successful else 0
                
                print(f"{search_type.capitalize()} Search in Mixed Workload:")
                print(f"  Success rate: {success_rate:.2%}")
                print(f"  Average latency: {avg_latency:.2f}ms")
                
                # All search types should maintain good performance
                assert success_rate >= 0.8  # At least 80% success rate
                assert avg_latency < 300     # Under 300ms average


@pytest.mark.performance
class TestScalabilityBenchmarks:
    """Benchmark system scalability characteristics."""
    
    def test_result_set_size_scaling(self, api_client):
        """Test how performance scales with result set size."""
        result_sizes = [1, 5, 10, 15, 20]
        latencies = []
        
        for size in result_sizes:
            with patch('api.main.collection') as mock_collection:
                # Generate mock results of specified size
                mock_ids = [f"scale_test_{i}" for i in range(size)]
                mock_docs = [f"Scale test document {i}" for i in range(size)]
                mock_metadata = [{
                    "name": f"Scale Employee {i}",
                    "department": "Engineering",
                    "role": "Test Role",
                    "experience": i % 15,
                    "location": "Test City",
                    "employment_type": "Full-time"
                } for i in range(size)]
                mock_distances = [0.1 + (i * 0.02) for i in range(size)]
                
                mock_collection.query.return_value = {
                    "ids": [mock_ids],
                    "documents": [mock_docs],
                    "metadatas": [mock_metadata],
                    "distances": [mock_distances]
                }
                
                start_time = time.perf_counter()
                response = api_client.post("/search/similarity", json={
                    "query": "scalability test query",
                    "n_results": size
                })
                end_time = time.perf_counter()
                
                assert response.status_code == 200
                latency = (end_time - start_time) * 1000
                latencies.append(latency)
                
                # Verify correct number of results
                data = response.json()
                assert data["total_results"] == size
        
        # Analyze scaling characteristics
        print("Result Set Size Scaling:")
        for size, latency in zip(result_sizes, latencies):
            print(f"  {size} results: {latency:.2f}ms")
        
        # Performance should scale reasonably (not exponentially)
        max_latency = max(latencies)
        assert max_latency < 500  # Even largest result set under 500ms
        
        # Latency increase should be reasonable
        latency_increase = latencies[-1] / latencies[0]
        assert latency_increase < 5  # No more than 5x increase for 20x more results
    
    def test_query_complexity_scaling(self, api_client):
        """Test how performance scales with query complexity."""
        queries = [
            "Python",  # Simple
            "Python developer",  # Medium
            "Python developer with web development experience",  # Complex
            "Senior Python developer with full stack web development experience and team leadership skills"  # Very complex
        ]
        
        latencies = []
        
        for query in queries:
            with patch('api.main.collection') as mock_collection:
                mock_collection.query.return_value = {
                    "ids": [["complex_1", "complex_2", "complex_3"]],
                    "documents": [["Complex doc 1", "Complex doc 2", "Complex doc 3"]],
                    "metadatas": [[
                        {"name": "Complex 1", "department": "Engineering", "role": "Developer",
                         "experience": 5, "location": "SF", "employment_type": "Full-time"},
                        {"name": "Complex 2", "department": "Engineering", "role": "Senior Dev", 
                         "experience": 8, "location": "NY", "employment_type": "Full-time"},
                        {"name": "Complex 3", "department": "Engineering", "role": "Lead Dev",
                         "experience": 12, "location": "Seattle", "employment_type": "Full-time"}
                    ]],
                    "distances": [[0.2, 0.3, 0.4]]
                }
                
                start_time = time.perf_counter()
                response = api_client.post("/search/similarity", json={
                    "query": query,
                    "n_results": 5
                })
                end_time = time.perf_counter()
                
                assert response.status_code == 200
                latency = (end_time - start_time) * 1000
                latencies.append(latency)
        
        print("Query Complexity Scaling:")
        complexity_levels = ["Simple", "Medium", "Complex", "Very Complex"]
        for level, latency in zip(complexity_levels, latencies):
            print(f"  {level}: {latency:.2f}ms")
        
        # Query complexity should not dramatically affect performance
        # (since we're using pre-computed embeddings)
        max_latency = max(latencies)
        assert max_latency < 200  # All queries under 200ms
        
        # Variation should be minimal
        latency_range = max(latencies) - min(latencies)
        assert latency_range < 100  # Less than 100ms variation


@pytest.mark.performance
class TestMemoryAndResourceBenchmarks:
    """Benchmark memory usage and resource consumption."""
    
    def test_memory_usage_stability(self, api_client):
        """Test that memory usage remains stable under load."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            pytest.skip("psutil not available - skipping memory usage test")
            return
        
        # Perform many requests
        for i in range(100):
            with patch('api.main.collection') as mock_collection:
                mock_collection.query.return_value = {
                    "ids": [[f"memory_test_{i}"]],
                    "documents": [[f"Memory test document {i}"]],
                    "metadatas": [[{
                        "name": f"Memory Employee {i}",
                        "department": "Engineering",
                        "role": "Test Role", 
                        "experience": i % 15,
                        "location": "Test City",
                        "employment_type": "Full-time"
                    }]],
                    "distances": [[0.3]]
                }
                
                response = api_client.post("/search/similarity", json={
                    "query": f"memory test query {i}",
                    "n_results": 5
                })
                
                assert response.status_code == 200
        
        try:
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            print(f"Memory Usage:")
            print(f"  Initial: {initial_memory:.2f} MB")
            print(f"  Final: {final_memory:.2f} MB")
            print(f"  Increase: {memory_increase:.2f} MB")
            
            # Memory increase should be minimal (no major leaks)
            assert memory_increase < 50  # Less than 50MB increase
        except NameError:
            # psutil not available, test was skipped
            pass
    
    @pytest.mark.slow
    def test_sustained_load_performance(self, api_client):
        """Test performance under sustained load over time."""
        duration_seconds = 30  # Run for 30 seconds
        start_time = time.time()
        request_count = 0
        latencies = []
        errors = 0
        
        while time.time() - start_time < duration_seconds:
            with patch('api.main.collection') as mock_collection:
                mock_collection.query.return_value = {
                    "ids": [[f"sustained_{request_count}"]],
                    "documents": [[f"Sustained test doc {request_count}"]],
                    "metadatas": [[{
                        "name": f"Sustained Employee {request_count}",
                        "department": "Engineering",
                        "role": "Test Role",
                        "experience": 5,
                        "location": "Test City", 
                        "employment_type": "Full-time"
                    }]],
                    "distances": [[0.3]]
                }
                
                req_start = time.perf_counter()
                try:
                    response = api_client.post("/search/similarity", json={
                        "query": f"sustained load test {request_count}",
                        "n_results": 3
                    })
                    req_end = time.perf_counter()
                    
                    if response.status_code == 200:
                        latency = (req_end - req_start) * 1000
                        latencies.append(latency)
                    else:
                        errors += 1
                        
                except Exception:
                    errors += 1
                
                request_count += 1
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        requests_per_second = request_count / total_time
        avg_latency = statistics.mean(latencies) if latencies else 0
        error_rate = errors / request_count if request_count > 0 else 0
        
        print(f"Sustained Load Performance ({total_time:.1f}s):")
        print(f"  Total requests: {request_count}")
        print(f"  Requests/second: {requests_per_second:.2f}")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  Error rate: {error_rate:.2%}")
        
        # Performance assertions for sustained load
        assert requests_per_second > 10   # At least 10 RPS
        assert avg_latency < 300          # Average under 300ms
        assert error_rate < 0.05          # Less than 5% errors