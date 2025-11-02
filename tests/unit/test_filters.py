"""
Unit tests for filter logic and Pydantic model validation.
Tests the metadata filtering system and request/response models.
"""
import pytest
from pydantic import ValidationError
from typing import Dict, Any, Optional


class TestMetadataFilterLogic:
    """Test metadata filter construction and logic."""
    
    def test_empty_filter_conditions(self):
        """Test building filters with no conditions."""
        from api.main import build_metadata_filter, MetadataFilter
        
        # All None values
        filters = MetadataFilter()
        result = build_metadata_filter(filters)
        
        assert result is None
    
    def test_single_department_filter(self):
        """Test building filter with only department."""
        from api.main import build_metadata_filter, MetadataFilter
        
        filters = MetadataFilter(department="Engineering")
        result = build_metadata_filter(filters)
        
        assert result == {"department": "Engineering"}
    
    def test_experience_range_filters(self):
        """Test various experience range scenarios."""
        from api.main import build_metadata_filter, MetadataFilter
        
        # Min only
        filters = MetadataFilter(experience_min=5)
        result = build_metadata_filter(filters)
        assert result == {"experience": {"$gte": 5}}
        
        # Max only  
        filters = MetadataFilter(experience_max=10)
        result = build_metadata_filter(filters)
        assert result == {"experience": {"$lte": 10}}
        
        # Both min and max (valid range)
        filters = MetadataFilter(experience_min=3, experience_max=8)
        result = build_metadata_filter(filters)
        assert result == {"experience": {"$gte": 3, "$lte": 8}}
        
        # Same min and max (exact match)
        filters = MetadataFilter(experience_min=5, experience_max=5)
        result = build_metadata_filter(filters)
        assert result == {"experience": 5}
    
    def test_experience_edge_cases(self):
        """Test edge cases in experience filtering."""
        from api.main import build_metadata_filter, MetadataFilter
        
        # Zero values
        filters = MetadataFilter(experience_min=0, experience_max=0)
        result = build_metadata_filter(filters)
        assert result == {"experience": 0}
        
        # Min > Max (invalid range - should be ignored)
        filters = MetadataFilter(experience_min=10, experience_max=5)
        result = build_metadata_filter(filters)
        assert result is None
        
        # Zero min only
        filters = MetadataFilter(experience_min=0)
        result = build_metadata_filter(filters)
        # Zero should be ignored for min-only filters
        assert result is None
    
    def test_location_and_employment_filters(self):
        """Test location and employment type filters."""
        from api.main import build_metadata_filter, MetadataFilter
        
        # Location only
        filters = MetadataFilter(location="San Francisco")
        result = build_metadata_filter(filters)
        assert result == {"location": "San Francisco"}
        
        # Employment type only
        filters = MetadataFilter(employment_type="Full-time")
        result = build_metadata_filter(filters)
        assert result == {"employment_type": "Full-time"}
    
    def test_multiple_conditions_and_logic(self):
        """Test combining multiple filter conditions."""
        from api.main import build_metadata_filter, MetadataFilter
        
        filters = MetadataFilter(
            department="Engineering",
            experience_min=5,
            location="San Francisco"
        )
        result = build_metadata_filter(filters)
        
        expected = {
            "$and": [
                {"department": "Engineering"},
                {"experience": {"$gte": 5}},
                {"location": "San Francisco"}
            ]
        }
        assert result == expected
    
    def test_complex_filter_combinations(self):
        """Test complex filter combinations."""
        from api.main import build_metadata_filter, MetadataFilter
        
        # All possible filters
        filters = MetadataFilter(
            department="Marketing",
            experience_min=3,
            experience_max=10,
            location="New York",
            employment_type="Part-time"
        )
        result = build_metadata_filter(filters)
        
        expected = {
            "$and": [
                {"department": "Marketing"},
                {"experience": {"$gte": 3, "$lte": 10}},
                {"location": "New York"},
                {"employment_type": "Part-time"}
            ]
        }
        assert result == expected


class TestPydanticModels:
    """Test Pydantic model validation and serialization."""
    
    def test_employee_search_request_valid(self):
        """Test valid EmployeeSearchRequest creation."""
        from api.main import EmployeeSearchRequest
        
        # Minimal valid request
        request = EmployeeSearchRequest(query="Python developer")
        assert request.query == "Python developer"
        assert request.n_results == 5  # Default value
        
        # Full valid request
        request = EmployeeSearchRequest(
            query="Senior engineer",
            n_results=10
        )
        assert request.query == "Senior engineer"
        assert request.n_results == 10
    
    def test_employee_search_request_validation_errors(self):
        """Test EmployeeSearchRequest validation errors."""
        from api.main import EmployeeSearchRequest
        
        # Missing required query
        with pytest.raises(ValidationError) as exc_info:
            EmployeeSearchRequest(n_results=5)
        assert "query" in str(exc_info.value)
        
        # n_results too low
        with pytest.raises(ValidationError):
            EmployeeSearchRequest(query="test", n_results=0)
        
        # n_results too high
        with pytest.raises(ValidationError):
            EmployeeSearchRequest(query="test", n_results=25)
    
    def test_metadata_filter_valid(self):
        """Test valid MetadataFilter creation."""
        from api.main import MetadataFilter
        
        # Empty filter (all optional)
        filter_obj = MetadataFilter()
        assert filter_obj.department is None
        assert filter_obj.experience_min is None
        
        # Partial filter
        filter_obj = MetadataFilter(
            department="Engineering",
            experience_min=5
        )
        assert filter_obj.department == "Engineering"
        assert filter_obj.experience_min == 5
        assert filter_obj.experience_max is None
        
        # Complete filter
        filter_obj = MetadataFilter(
            department="Marketing",
            experience_min=2,
            experience_max=15,
            location="Seattle",
            employment_type="Full-time"
        )
        assert filter_obj.department == "Marketing"
        assert filter_obj.experience_min == 2
        assert filter_obj.experience_max == 15
        assert filter_obj.location == "Seattle"
        assert filter_obj.employment_type == "Full-time"
    
    def test_metadata_filter_validation_errors(self):
        """Test MetadataFilter validation errors."""
        from api.main import MetadataFilter
        
        # Negative experience
        with pytest.raises(ValidationError):
            MetadataFilter(experience_min=-1)
        
        with pytest.raises(ValidationError):
            MetadataFilter(experience_max=-5)
    
    def test_advanced_search_request_valid(self):
        """Test valid AdvancedSearchRequest creation."""
        from api.main import AdvancedSearchRequest, MetadataFilter
        
        # Without filters
        request = AdvancedSearchRequest(query="senior developer")
        assert request.query == "senior developer"
        assert request.filters is None
        assert request.n_results == 5
        
        # With filters
        filters = MetadataFilter(department="Engineering")
        request = AdvancedSearchRequest(
            query="Python expert",
            filters=filters,
            n_results=3
        )
        assert request.query == "Python expert"
        assert request.filters.department == "Engineering"
        assert request.n_results == 3
    
    def test_employee_response_model(self):
        """Test EmployeeResponse model creation."""
        from api.main import EmployeeResponse
        
        employee = EmployeeResponse(
            id="emp_1",
            name="John Doe",
            role="Software Engineer",
            department="Engineering",
            experience=5,
            location="San Francisco",
            employment_type="Full-time",
            similarity_score=0.25,
            document="Software Engineer with 5 years experience..."
        )
        
        assert employee.id == "emp_1"
        assert employee.name == "John Doe"
        assert employee.similarity_score == 0.25
        assert isinstance(employee.experience, int)
        assert isinstance(employee.similarity_score, float)
    
    def test_search_response_model(self):
        """Test SearchResponse model creation."""
        from api.main import SearchResponse, EmployeeResponse
        
        employees = [
            EmployeeResponse(
                id="emp_1",
                name="John Doe", 
                role="Software Engineer",
                department="Engineering",
                experience=5,
                location="San Francisco",
                employment_type="Full-time",
                similarity_score=0.25,
                document="Test document"
            )
        ]
        
        response = SearchResponse(
            query="Python developer",
            total_results=1,
            results=employees,
            search_type="similarity"
        )
        
        assert response.query == "Python developer"
        assert response.total_results == 1
        assert len(response.results) == 1
        assert response.search_type == "similarity"
        assert response.results[0].name == "John Doe"


class TestFilterLogicEdgeCases:
    """Test edge cases and boundary conditions in filter logic."""
    
    def test_empty_string_filters(self):
        """Test behavior with empty string values."""
        from api.main import build_metadata_filter, MetadataFilter
        
        # Empty strings should be treated as None
        filters = MetadataFilter(
            department="",
            location="",
            employment_type=""
        )
        result = build_metadata_filter(filters)
        
        # Empty strings should not create filter conditions
        assert result is None
    
    def test_whitespace_only_filters(self):
        """Test behavior with whitespace-only values."""
        from api.main import build_metadata_filter, MetadataFilter
        
        # Whitespace-only strings should be treated as None
        filters = MetadataFilter(
            department="   ",
            location="\t\n",
            employment_type="  \n  "
        )
        result = build_metadata_filter(filters)
        
        # Should not create any conditions
        assert result is None
    
    def test_boundary_experience_values(self):
        """Test boundary values for experience filtering."""
        from api.main import build_metadata_filter, MetadataFilter
        
        # Maximum reasonable experience
        filters = MetadataFilter(experience_min=50, experience_max=60)
        result = build_metadata_filter(filters)
        assert result == {"experience": {"$gte": 50, "$lte": 60}}
        
        # Single year experience
        filters = MetadataFilter(experience_min=1, experience_max=1)
        result = build_metadata_filter(filters)
        assert result == {"experience": 1}
    
    def test_filter_condition_ordering(self):
        """Test that filter conditions maintain consistent ordering."""
        from api.main import build_metadata_filter, MetadataFilter
        
        # Create filters multiple times with same values
        filters1 = MetadataFilter(
            department="Engineering",
            experience_min=5,
            location="San Francisco"
        )
        filters2 = MetadataFilter(
            location="San Francisco", 
            department="Engineering",
            experience_min=5
        )
        
        result1 = build_metadata_filter(filters1)
        result2 = build_metadata_filter(filters2)
        
        # Results should be equivalent (order in $and array may vary)
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
        assert "$and" in result1
        assert "$and" in result2
        assert len(result1["$and"]) == len(result2["$and"])


class TestModelSerialization:
    """Test model serialization and deserialization."""
    
    def test_search_request_json_serialization(self):
        """Test SearchRequest JSON serialization."""
        from api.main import EmployeeSearchRequest
        
        request = EmployeeSearchRequest(
            query="Python developer",
            n_results=3
        )
        
        # Test dict conversion
        request_dict = request.dict()
        assert request_dict["query"] == "Python developer"
        assert request_dict["n_results"] == 3
        
        # Test JSON serialization
        json_str = request.json()
        assert "Python developer" in json_str
        assert "3" in json_str
    
    def test_search_response_json_serialization(self):
        """Test SearchResponse JSON serialization."""
        from api.main import SearchResponse, EmployeeResponse
        
        employee = EmployeeResponse(
            id="emp_1",
            name="John Doe",
            role="Software Engineer", 
            department="Engineering",
            experience=5,
            location="San Francisco",
            employment_type="Full-time",
            similarity_score=0.25,
            document="Test document"
        )
        
        response = SearchResponse(
            query="test query",
            total_results=1,
            results=[employee],
            search_type="similarity"
        )
        
        # Test dict conversion
        response_dict = response.dict()
        assert response_dict["total_results"] == 1
        assert len(response_dict["results"]) == 1
        assert response_dict["results"][0]["name"] == "John Doe"
        
        # Test JSON serialization
        json_str = response.json()
        assert "John Doe" in json_str
        assert "similarity" in json_str
    
    def test_model_field_validation_messages(self):
        """Test that validation error messages are informative."""
        from api.main import EmployeeSearchRequest
        
        try:
            EmployeeSearchRequest(query="test", n_results=100)
        except ValidationError as e:
            error_msg = str(e)
            assert "n_results" in error_msg
            # Should mention the constraint
            assert "20" in error_msg or "less than" in error_msg.lower()


class TestFilterPerformance:
    """Test performance characteristics of filter building."""
    
    def test_filter_building_speed(self):
        """Test that filter building is reasonably fast."""
        import time
        from api.main import build_metadata_filter, MetadataFilter
        
        # Create complex filter
        filters = MetadataFilter(
            department="Engineering",
            experience_min=5,
            experience_max=15,
            location="San Francisco",
            employment_type="Full-time"
        )
        
        # Time the filter building
        start_time = time.time()
        for _ in range(1000):  # Build 1000 filters
            result = build_metadata_filter(filters)
        end_time = time.time()
        
        # Should be very fast (less than 1 second for 1000 operations)
        total_time = end_time - start_time
        assert total_time < 1.0
        
        # Result should be correct
        assert result is not None
        assert "$and" in result
    
    def test_model_validation_speed(self):
        """Test that model validation is reasonably fast."""
        import time
        from api.main import EmployeeSearchRequest
        
        # Time model creation and validation
        start_time = time.time()
        for i in range(1000):
            request = EmployeeSearchRequest(
                query=f"test query {i}",
                n_results=5
            )
        end_time = time.time()
        
        # Should be very fast (less than 1 second for 1000 operations) 
        total_time = end_time - start_time
        assert total_time < 1.0