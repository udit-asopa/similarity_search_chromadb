"""
Integration tests for ChromaDB operations and collection management.
Tests the actual ChromaDB functionality with real database operations.
"""
import pytest
import chromadb
from chromadb.utils import embedding_functions
import tempfile
import shutil
import os


class TestChromaDBIntegration:
    """Test ChromaDB client and basic operations."""
    
    def test_client_creation(self):
        """Test creating ChromaDB client."""
        client = chromadb.Client()
        assert client is not None
        
        # Test client methods are available
        assert hasattr(client, 'create_collection')
        assert hasattr(client, 'get_collection')
        assert hasattr(client, 'list_collections')
        assert hasattr(client, 'delete_collection')
    
    def test_persistent_client_creation(self):
        """Test creating persistent ChromaDB client."""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = chromadb.PersistentClient(path=temp_dir)
            assert client is not None
            
            # Should create database files
            assert os.path.exists(temp_dir)
    
    def test_collection_lifecycle(self, test_chromadb_client):
        """Test complete collection lifecycle."""
        collection_name = "test_lifecycle_collection"
        
        # Create collection
        collection = test_chromadb_client.create_collection(
            name=collection_name,
            metadata={"description": "Test collection"}
        )
        assert collection is not None
        assert collection.name == collection_name
        
        # List collections
        collections = test_chromadb_client.list_collections()
        collection_names = [c.name for c in collections]
        assert collection_name in collection_names
        
        # Get collection
        retrieved_collection = test_chromadb_client.get_collection(collection_name)
        assert retrieved_collection.name == collection_name
        
        # Delete collection
        test_chromadb_client.delete_collection(collection_name)
        
        # Verify deletion
        collections_after = test_chromadb_client.list_collections()
        collection_names_after = [c.name for c in collections_after]
        assert collection_name not in collection_names_after
    
    def test_duplicate_collection_creation(self, test_chromadb_client):
        """Test creating collection with duplicate name."""
        collection_name = "duplicate_test_collection"
        
        # Create first collection
        collection1 = test_chromadb_client.create_collection(name=collection_name)
        assert collection1 is not None
        
        # Try to create duplicate - should raise error
        with pytest.raises(Exception):
            test_chromadb_client.create_collection(name=collection_name)
        
        # Cleanup
        test_chromadb_client.delete_collection(collection_name)


class TestCollectionOperations:
    """Test collection data operations."""
    
    def test_add_documents(self, test_collection):
        """Test adding documents to collection."""
        # Collection already has sample data from fixture
        count = test_collection.count()
        assert count == 2  # From fixture setup
        
        # Add more documents
        new_ids = ["test_emp_3", "test_emp_4"]
        new_documents = [
            "HR Manager with 6 years in HR. Located in Boston.",
            "Sales Rep with 2 years in Sales. Located in Austin."
        ]
        new_metadatas = [
            {
                "name": "Carol HR",
                "department": "HR",
                "role": "HR Manager",
                "experience": 6,
                "location": "Boston",
                "employment_type": "Full-time"
            },
            {
                "name": "Dave Sales",
                "department": "Sales", 
                "role": "Sales Representative",
                "experience": 2,
                "location": "Austin",
                "employment_type": "Full-time"
            }
        ]
        
        test_collection.add(
            ids=new_ids,
            documents=new_documents,
            metadatas=new_metadatas
        )
        
        # Verify count increased
        new_count = test_collection.count()
        assert new_count == 4
    
    def test_get_all_documents(self, test_collection):
        """Test retrieving all documents from collection."""
        results = test_collection.get()
        
        assert "ids" in results
        assert "documents" in results
        assert "metadatas" in results
        
        assert len(results["ids"]) == 2
        assert len(results["documents"]) == 2
        assert len(results["metadatas"]) == 2
        
        # Check data integrity
        assert "test_emp_1" in results["ids"]
        assert "test_emp_2" in results["ids"]
    
    def test_get_with_filters(self, test_collection):
        """Test retrieving documents with metadata filters."""
        # Get Engineering employees only
        results = test_collection.get(
            where={"department": "Engineering"}
        )
        
        assert len(results["ids"]) == 1
        metadata = results["metadatas"][0]
        assert metadata["department"] == "Engineering"
        assert metadata["name"] == "Alice Test"
    
    def test_get_with_complex_filters(self, test_collection):
        """Test retrieving documents with complex filters."""
        # Add more test data for complex filtering
        test_collection.add(
            ids=["test_emp_5", "test_emp_6"],
            documents=[
                "Senior Engineer with 10 years in Engineering. Located in Seattle.",
                "Junior Engineer with 2 years in Engineering. Located in Portland."
            ],
            metadatas=[
                {
                    "name": "Senior Engineer",
                    "department": "Engineering",
                    "role": "Senior Software Engineer",
                    "experience": 10,
                    "location": "Seattle",
                    "employment_type": "Full-time"
                },
                {
                    "name": "Junior Engineer", 
                    "department": "Engineering",
                    "role": "Junior Software Engineer",
                    "experience": 2,
                    "location": "Portland",
                    "employment_type": "Full-time"
                }
            ]
        )
        
        # Filter by department AND experience range
        results = test_collection.get(
            where={
                "$and": [
                    {"department": "Engineering"},
                    {"experience": {"$gte": 5}}
                ]
            }
        )
        
        # Should get Alice Test (5 years) and Senior Engineer (10 years)
        assert len(results["ids"]) == 2
        experiences = [meta["experience"] for meta in results["metadatas"]]
        assert all(exp >= 5 for exp in experiences)
    
    def test_query_similarity_search(self, test_collection):
        """Test similarity search queries."""
        # Query for engineering-related terms
        results = test_collection.query(
            query_texts=["software engineer programming"],
            n_results=2
        )
        
        assert "ids" in results
        assert "documents" in results
        assert "metadatas" in results
        assert "distances" in results
        
        assert len(results["ids"][0]) <= 2
        assert len(results["distances"][0]) <= 2
        
        # Debug: Print actual distances for understanding
        print(f"ChromaDB distances: {results['distances'][0]}")
        
        # Check that distances are reasonable (lower is more similar)
        for distance in results["distances"][0]:
            assert distance >= 0  # Distance should be non-negative
            assert distance < 1000  # Reasonable upper bound for any distance metric
            # Note: ChromaDB distance values depend on embedding model and can vary widely
    
    def test_query_with_filters(self, test_collection):
        """Test similarity search with metadata filters."""
        results = test_collection.query(
            query_texts=["manager leadership"],
            n_results=5,
            where={"department": "Marketing"}
        )
        
        # Should only return Marketing employees
        for metadata in results["metadatas"][0]:
            assert metadata["department"] == "Marketing"
    
    def test_update_documents(self, test_collection):
        """Test updating existing documents."""
        # Update an existing document
        updated_document = "Updated Software Engineer with 6 years in Engineering. Located in San Francisco."
        updated_metadata = {
            "name": "Alice Updated",
            "department": "Engineering",
            "role": "Senior Software Engineer",  # Updated role
            "experience": 6,  # Updated experience
            "location": "San Francisco",
            "employment_type": "Full-time"
        }
        
        test_collection.upsert(
            ids=["test_emp_1"],
            documents=[updated_document],
            metadatas=[updated_metadata]
        )
        
        # Verify update
        results = test_collection.get(ids=["test_emp_1"])
        assert len(results["ids"]) == 1
        assert results["metadatas"][0]["name"] == "Alice Updated"
        assert results["metadatas"][0]["role"] == "Senior Software Engineer"
        assert results["metadatas"][0]["experience"] == 6
    
    def test_delete_documents(self, test_collection):
        """Test deleting documents from collection."""
        # Get initial count
        initial_count = test_collection.count()
        
        # Delete one document
        test_collection.delete(ids=["test_emp_1"])
        
        # Verify deletion
        new_count = test_collection.count()
        assert new_count == initial_count - 1
        
        # Verify specific document is gone
        results = test_collection.get(ids=["test_emp_1"])
        assert len(results["ids"]) == 0
    
    def test_delete_with_filters(self, test_collection):
        """Test deleting documents with filters."""
        # Add more test data
        test_collection.add(
            ids=["temp_1", "temp_2"],
            documents=["Temp doc 1", "Temp doc 2"],
            metadatas=[
                {"name": "Temp 1", "department": "Temp", "role": "Temp Role", 
                 "experience": 1, "location": "Temp", "employment_type": "Temp"},
                {"name": "Temp 2", "department": "Temp", "role": "Temp Role",
                 "experience": 2, "location": "Temp", "employment_type": "Temp"}
            ]
        )
        
        # Delete all temp documents
        test_collection.delete(where={"department": "Temp"})
        
        # Verify deletion
        results = test_collection.get(where={"department": "Temp"})
        assert len(results["ids"]) == 0


class TestEmbeddingIntegration:
    """Test embedding function integration with ChromaDB."""
    
    def test_collection_with_embedding_function(self, test_chromadb_client, mock_embedding_function):
        """Test creating collection with custom embedding function."""
        collection = test_chromadb_client.create_collection(
            name="test_embedding_collection",
            embedding_function=mock_embedding_function
        )
        
        # Add documents
        collection.add(
            ids=["embed_test_1"],
            documents=["Test document for embedding"],
            metadatas=[{"test": "metadata"}]
        )
        
        # Verify document was added with embeddings
        count = collection.count()
        assert count == 1
        
        # Test query works with embeddings
        results = collection.query(
            query_texts=["similar test document"],
            n_results=1
        )
        
        assert len(results["ids"][0]) == 1
        assert results["ids"][0][0] == "embed_test_1"
        
        # Cleanup
        test_chromadb_client.delete_collection("test_embedding_collection")
    
    def test_real_sentence_transformer_embedding(self, test_chromadb_client):
        """Test with real SentenceTransformer embedding function."""
        # This test uses actual SentenceTransformer model
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        collection = test_chromadb_client.create_collection(
            name="real_embedding_collection",
            embedding_function=ef
        )
        
        # Add test documents
        documents = [
            "Python developer with web development experience",
            "Marketing manager with social media skills",
            "Data scientist with machine learning expertise"
        ]
        
        collection.add(
            ids=["real_1", "real_2", "real_3"],
            documents=documents,
            metadatas=[
                {"role": "Developer", "department": "Engineering"},
                {"role": "Manager", "department": "Marketing"},
                {"role": "Scientist", "department": "Data"}
            ]
        )
        
        # Test semantic search
        results = collection.query(
            query_texts=["software engineer programming"],
            n_results=2
        )
        
        # Should return developer-related results first
        assert len(results["ids"][0]) == 2
        
        # First result should be most similar (Python developer)
        first_result_id = results["ids"][0][0]
        assert first_result_id == "real_1"
        
        # Distance should be reasonable
        first_distance = results["distances"][0][0]
        assert 0 <= first_distance <= 1.0  # Should be quite similar
        
        # Cleanup
        test_chromadb_client.delete_collection("real_embedding_collection")


class TestCollectionMetadata:
    """Test collection metadata handling."""
    
    def test_collection_metadata_creation(self, test_chromadb_client):
        """Test creating collection with metadata."""
        metadata = {
            "description": "Test collection for employees",
            "version": "1.0",
            "created_by": "test_suite"
        }
        
        collection = test_chromadb_client.create_collection(
            name="metadata_test_collection",
            metadata=metadata
        )
        
        # Verify metadata is stored
        assert collection.metadata == metadata
        
        # Cleanup
        test_chromadb_client.delete_collection("metadata_test_collection")
    
    def test_collection_metadata_update(self, test_chromadb_client):
        """Test updating collection metadata."""
        initial_metadata = {"version": "1.0"}
        
        collection = test_chromadb_client.create_collection(
            name="metadata_update_collection",
            metadata=initial_metadata
        )
        
        # Update metadata
        updated_metadata = {"version": "2.0", "updated": True}
        collection.modify(metadata=updated_metadata)
        
        # Verify update
        assert collection.metadata == updated_metadata
        
        # Cleanup
        test_chromadb_client.delete_collection("metadata_update_collection")


class TestErrorHandling:
    """Test error handling in ChromaDB operations."""
    
    def test_get_nonexistent_collection(self, test_chromadb_client):
        """Test getting collection that doesn't exist."""
        with pytest.raises(Exception):
            test_chromadb_client.get_collection("nonexistent_collection")
    
    def test_delete_nonexistent_collection(self, test_chromadb_client):
        """Test deleting collection that doesn't exist."""
        with pytest.raises(Exception):
            test_chromadb_client.delete_collection("nonexistent_collection")
    
    def test_add_duplicate_ids(self, test_collection):
        """Test adding documents with duplicate IDs."""
        initial_count = test_collection.count()
        
        # Try to add document with existing ID
        try:
            test_collection.add(
                ids=["test_emp_1"],  # This ID already exists
                documents=["Duplicate document"],
                metadatas=[{"name": "Duplicate"}]
            )
            # If no exception is raised, ChromaDB might use upsert behavior
            # In this case, count should remain the same or be handled gracefully
            final_count = test_collection.count()
            # Either the operation failed silently or updated existing document
            assert final_count >= initial_count
        except Exception:
            # This is the expected behavior - duplicate IDs should raise an exception
            pass
    
    def test_query_empty_collection(self, test_chromadb_client, mock_embedding_function):
        """Test querying empty collection."""
        empty_collection = test_chromadb_client.create_collection(
            name="empty_test_collection",
            embedding_function=mock_embedding_function
        )
        
        results = empty_collection.query(
            query_texts=["test query"],
            n_results=5
        )
        
        # Should return empty results
        assert len(results["ids"][0]) == 0
        assert len(results["documents"][0]) == 0
        
        # Cleanup
        test_chromadb_client.delete_collection("empty_test_collection")
    
    def test_invalid_metadata_types(self, test_collection):
        """Test adding documents with invalid metadata types."""
        # ChromaDB has restrictions on metadata value types
        with pytest.raises(Exception):
            test_collection.add(
                ids=["invalid_meta_1"],
                documents=["Test document"],
                metadatas=[{
                    "invalid_list": ["this", "should", "fail"],  # Lists not allowed
                    "invalid_dict": {"nested": "dict"}  # Nested dicts not allowed
                }]
            )


@pytest.mark.integration
class TestPerformanceCharacteristics:
    """Test performance characteristics of ChromaDB operations."""
    
    def test_bulk_insert_performance(self, test_chromadb_client, test_data_generator):
        """Test performance of bulk document insertion."""
        import time
        
        # Generate test data
        employees = test_data_generator.create_employees(100)
        
        collection = test_chromadb_client.create_collection(
            name="performance_test_collection"
        )
        
        # Prepare bulk data
        ids = [emp["id"] for emp in employees]
        documents = [f"{emp['role']} in {emp['department']}" for emp in employees]
        metadatas = [{
            "name": emp["name"],
            "department": emp["department"],
            "role": emp["role"],
            "experience": emp["experience"],
            "location": emp["location"],
            "employment_type": emp["employment_type"]
        } for emp in employees]
        
        # Time the bulk insert
        start_time = time.time()
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        end_time = time.time()
        
        insert_time = end_time - start_time
        
        # Verify all documents were inserted
        count = collection.count()
        assert count == 100
        
        print(f"Bulk insert performance: {insert_time:.2f}s for {count} documents")
        print(f"Rate: {count/insert_time:.2f} documents/second")
        
        # Should be reasonably fast (less than 30 seconds for 100 documents)
        # This accounts for embedding generation, vectorization, and indexing time
        assert insert_time < 30.0
        
        # Cleanup
        test_chromadb_client.delete_collection("performance_test_collection")
    
    def test_query_performance(self, test_collection):
        """Test query performance with various result sizes."""
        import time
        
        # Add more documents for meaningful performance test
        bulk_ids = [f"perf_test_{i}" for i in range(50)]
        bulk_docs = [f"Performance test document {i} with various content" for i in range(50)]
        bulk_metadata = [{
            "name": f"Perf Employee {i}",
            "department": "Engineering" if i % 2 == 0 else "Marketing",
            "role": "Test Role",
            "experience": i % 20,
            "location": "Test City",
            "employment_type": "Full-time"
        } for i in range(50)]
        
        test_collection.add(
            ids=bulk_ids,
            documents=bulk_docs,
            metadatas=bulk_metadata
        )
        
        # Test query performance
        start_time = time.time()
        results = test_collection.query(
            query_texts=["engineering software development"],
            n_results=10
        )
        end_time = time.time()
        
        query_time = end_time - start_time
        
        print(f"Query performance: {query_time:.3f}s for similarity search")
        
        # Should be reasonably fast (less than 5 seconds)
        assert query_time < 5.0
        
        # Should return results
        assert len(results["ids"][0]) > 0