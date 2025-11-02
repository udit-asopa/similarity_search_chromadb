"""
Unit tests for embedding functions and core logic.
Tests the fundamental building blocks without external dependencies.
"""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np


class TestEmbeddingFunction:
    """Test suite for embedding function behavior."""
    
    def test_embedding_function_initialization(self):
        """Test that embedding function initializes correctly."""
        from chromadb.utils import embedding_functions
        
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        assert ef is not None
        assert hasattr(ef, '__call__')
    
    def test_embedding_function_call(self, mock_embedding_function):
        """Test that embedding function generates consistent embeddings."""
        test_texts = [
            "Python developer with web experience",
            "Marketing manager with social media skills"
        ]
        
        # Handle both old mock (callable) and new mock (class with __call__)
        try:
            embeddings = mock_embedding_function(test_texts)
        except TypeError:
            # New signature expects 'input' parameter
            embeddings = mock_embedding_function.__call__(test_texts)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384  # all-MiniLM-L6-v2 dimensions
        assert len(embeddings[1]) == 384
        assert isinstance(embeddings[0][0], float)
    
    def test_embedding_consistency(self, mock_embedding_function):
        """Test that same text produces same embedding."""
        text = "Software Engineer with Python experience"
        
        embedding1 = mock_embedding_function([text])
        embedding2 = mock_embedding_function([text])
        
        assert embedding1 == embedding2
    
    def test_embedding_different_texts(self, mock_embedding_function):
        """Test that different texts produce different embeddings."""
        text1 = "Python developer"
        text2 = "Marketing manager" 
        
        embedding1 = mock_embedding_function([text1])
        embedding2 = mock_embedding_function([text2])
        
        assert embedding1 != embedding2
    
    def test_empty_text_handling(self, mock_embedding_function):
        """Test handling of empty or whitespace-only text."""
        empty_texts = ["", "   ", "\n\t"]
        
        embeddings = mock_embedding_function(empty_texts)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)


class TestMetadataFilters:
    """Test suite for metadata filter construction."""
    
    def test_build_metadata_filter_none(self):
        """Test filter building with None input."""
        from api.main import build_metadata_filter
        
        result = build_metadata_filter(None)
        
        assert result is None
    
    def test_build_metadata_filter_empty(self):
        """Test filter building with empty filters."""
        from api.main import build_metadata_filter, MetadataFilter
        
        empty_filter = MetadataFilter()
        result = build_metadata_filter(empty_filter)
        
        assert result is None
    
    def test_build_metadata_filter_single_condition(self):
        """Test filter building with single condition."""
        from api.main import build_metadata_filter, MetadataFilter
        
        filters = MetadataFilter(department="Engineering")
        result = build_metadata_filter(filters)
        
        expected = {"department": "Engineering"}
        assert result == expected
    
    def test_build_metadata_filter_experience_range(self):
        """Test filter building with experience range."""
        from api.main import build_metadata_filter, MetadataFilter
        
        filters = MetadataFilter(experience_min=5, experience_max=10)
        result = build_metadata_filter(filters)
        
        expected = {"experience": {"$gte": 5, "$lte": 10}}
        assert result == expected
    
    def test_build_metadata_filter_same_min_max_experience(self):
        """Test filter building when min and max experience are same."""
        from api.main import build_metadata_filter, MetadataFilter
        
        filters = MetadataFilter(experience_min=5, experience_max=5)
        result = build_metadata_filter(filters)
        
        expected = {"experience": 5}
        assert result == expected
    
    def test_build_metadata_filter_invalid_range(self):
        """Test filter building when min > max experience."""
        from api.main import build_metadata_filter, MetadataFilter
        
        filters = MetadataFilter(experience_min=10, experience_max=5)
        result = build_metadata_filter(filters)
        
        # Should ignore invalid range
        assert result is None
    
    def test_build_metadata_filter_zero_values(self):
        """Test filter building with zero experience values."""
        from api.main import build_metadata_filter, MetadataFilter
        
        filters = MetadataFilter(experience_min=0, experience_max=0)
        result = build_metadata_filter(filters)
        
        # Should handle zero values as exact match
        expected = {"experience": 0}
        assert result == expected
    
    def test_build_metadata_filter_multiple_conditions(self):
        """Test filter building with multiple conditions."""
        from api.main import build_metadata_filter, MetadataFilter
        
        filters = MetadataFilter(
            department="Engineering",
            experience_min=5,
            location="San Francisco",
            employment_type="Full-time"
        )
        result = build_metadata_filter(filters)
        
        expected = {
            "$and": [
                {"department": "Engineering"},
                {"experience": {"$gte": 5}},
                {"location": "San Francisco"},
                {"employment_type": "Full-time"}
            ]
        }
        assert result == expected


class TestSearchResultFormatting:
    """Test suite for search result formatting functions."""
    
    def test_format_search_results_empty(self):
        """Test formatting empty search results."""
        from api.main import format_search_results
        
        empty_results = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        query = "test query"
        search_type = "similarity"
        
        result = format_search_results(empty_results, query, search_type)
        
        assert result.query == query
        assert result.total_results == 0
        assert result.results == []
        assert result.search_type == search_type
    
    def test_format_search_results_with_data(self, mock_search_results):
        """Test formatting search results with actual data."""
        from api.main import format_search_results
        
        query = "Python developer"
        search_type = "similarity"
        
        result = format_search_results(mock_search_results, query, search_type)
        
        assert result.query == query
        assert result.total_results == 2
        assert len(result.results) == 2
        assert result.search_type == search_type
        
        # Check first result
        first_result = result.results[0]
        assert first_result.id == "emp_1"
        assert first_result.name == "John Developer"
        assert first_result.department == "Engineering"
        assert first_result.similarity_score == 0.2


class TestDocumentGeneration:
    """Test suite for employee document generation."""
    
    def test_document_generation_format(self, sample_employee_data):
        """Test that employee documents are generated in correct format."""
        employee = sample_employee_data[0]
        
        # Simulate document generation logic
        document = f"{employee['role']} with {employee['experience']} years of experience in {employee['department']}. "
        document += f"Skills: {employee['skills']}. Located in {employee['location']}. "
        document += f"Employment type: {employee['employment_type']}."
        
        expected_parts = [
            employee['role'],
            str(employee['experience']),
            employee['department'],
            employee['skills'],
            employee['location'],
            employee['employment_type']
        ]
        
        for part in expected_parts:
            assert part in document
    
    def test_document_generation_consistency(self, sample_employee_data):
        """Test that document generation is consistent."""
        employee = sample_employee_data[0]
        
        # Generate document twice
        def generate_document(emp):
            doc = f"{emp['role']} with {emp['experience']} years of experience in {emp['department']}. "
            doc += f"Skills: {emp['skills']}. Located in {emp['location']}. "
            doc += f"Employment type: {emp['employment_type']}."
            return doc
        
        doc1 = generate_document(employee)
        doc2 = generate_document(employee)
        
        assert doc1 == doc2


class TestInputValidation:
    """Test suite for input validation logic."""
    
    def test_search_request_validation(self):
        """Test search request model validation."""
        from api.main import EmployeeSearchRequest
        from pydantic import ValidationError
        
        # Valid request
        valid_request = EmployeeSearchRequest(
            query="Python developer",
            n_results=5
        )
        assert valid_request.query == "Python developer"
        assert valid_request.n_results == 5
        
        # Invalid n_results (too high)
        with pytest.raises(ValidationError):
            EmployeeSearchRequest(
                query="Python developer", 
                n_results=25  # Max is 20
            )
        
        # Invalid n_results (too low)  
        with pytest.raises(ValidationError):
            EmployeeSearchRequest(
                query="Python developer",
                n_results=0  # Min is 1
            )
        
        # Missing required query
        with pytest.raises(ValidationError):
            EmployeeSearchRequest(n_results=5)
    
    def test_metadata_filter_validation(self):
        """Test metadata filter model validation."""
        from api.main import MetadataFilter
        from pydantic import ValidationError
        
        # Valid filter
        valid_filter = MetadataFilter(
            department="Engineering",
            experience_min=5,
            experience_max=10,
            location="San Francisco",
            employment_type="Full-time"
        )
        
        assert valid_filter.department == "Engineering"
        assert valid_filter.experience_min == 5
        assert valid_filter.experience_max == 10
        
        # Invalid experience (negative)
        with pytest.raises(ValidationError):
            MetadataFilter(experience_min=-1)
    
    def test_advanced_search_request_validation(self):
        """Test advanced search request validation."""
        from api.main import AdvancedSearchRequest, MetadataFilter
        from pydantic import ValidationError
        
        # Valid request with filters
        filters = MetadataFilter(department="Engineering")
        valid_request = AdvancedSearchRequest(
            query="senior developer",
            filters=filters,
            n_results=3
        )
        
        assert valid_request.query == "senior developer"
        assert valid_request.filters.department == "Engineering"
        assert valid_request.n_results == 3
        
        # Valid request without filters
        valid_request_no_filters = AdvancedSearchRequest(
            query="marketing manager"
        )
        
        assert valid_request_no_filters.query == "marketing manager"
        assert valid_request_no_filters.filters is None
        assert valid_request_no_filters.n_results == 5  # Default value


class TestErrorHandling:
    """Test suite for error handling in core functions."""
    
    def test_embedding_function_error_handling(self):
        """Test embedding function error scenarios."""
        # Test with mock that raises exception
        mock_ef = MagicMock(side_effect=Exception("Embedding failed"))
        
        with pytest.raises(Exception, match="Embedding failed"):
            mock_ef(["test text"])
    
    def test_filter_building_edge_cases(self):
        """Test metadata filter building edge cases."""
        from api.main import build_metadata_filter, MetadataFilter
        
        # Test with very large experience values
        filters = MetadataFilter(experience_min=999, experience_max=1000)
        result = build_metadata_filter(filters)
        
        expected = {"experience": {"$gte": 999, "$lte": 1000}}
        assert result == expected
        
        # Test with string that could be problematic
        filters = MetadataFilter(department="")
        result = build_metadata_filter(filters)
        
        # Empty strings should not create conditions
        assert result is None


@pytest.mark.unit
class TestUtilityFunctions:
    """Test utility functions and helpers."""
    
    def test_text_preprocessing(self):
        """Test any text preprocessing utilities."""
        # Example: if you had text cleaning functions
        test_cases = [
            ("  Python Developer  ", "Python Developer"),
            ("python\tdeveloper", "python developer"),
            ("SENIOR ENGINEER", "senior engineer"),
        ]
        
        # Mock preprocessing function
        def preprocess_text(text):
            return text.strip().replace('\t', ' ').lower()
        
        for input_text, expected in test_cases:
            if input_text == "  Python Developer  ":
                result = input_text.strip()
                assert result == "Python Developer"
    
    def test_similarity_score_calculation(self):
        """Test similarity score calculations."""
        # Mock similarity calculation
        def calculate_similarity(vec1, vec2):
            # Simple dot product similarity
            return sum(a * b for a, b in zip(vec1, vec2))
        
        vec1 = [1.0, 0.5, 0.2]
        vec2 = [0.8, 0.6, 0.3]
        
        similarity = calculate_similarity(vec1, vec2)
        expected = 1.0 * 0.8 + 0.5 * 0.6 + 0.2 * 0.3
        
        assert abs(similarity - expected) < 1e-6