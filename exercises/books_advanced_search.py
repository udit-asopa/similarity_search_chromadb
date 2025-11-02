# Advanced Book Recommendation System - Exercise Template
# 
# In this exercise, you will implement a complete book recommendation system
# using ChromaDB and SentenceTransformers. Follow the TODO comments to
# complete each section.

import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB components
# TODO: Create ChromaDB client
client = None  # Replace with client initialization

# TODO: Create embedding function using SentenceTransformers
ef = None  # Replace with embedding function

# Book dataset for the exercise
books = [
    {
        "id": "book_1",
        "title": "The Great Gatsby",
        "author": "F. Scott Fitzgerald",
        "genre": "Classic",
        "year": 1925,
        "rating": 4.1,
        "pages": 180,
        "description": "A tragic tale of wealth, love, and the American Dream in the Jazz Age",
        "themes": "wealth, corruption, American Dream, social class",
        "setting": "New York, 1920s"
    },
    {
        "id": "book_2",
        "title": "To Kill a Mockingbird",
        "author": "Harper Lee",
        "genre": "Classic",
        "year": 1960,
        "rating": 4.3,
        "pages": 376,
        "description": "A powerful story of racial injustice and moral growth in the American South",
        "themes": "racism, justice, moral courage, childhood innocence",
        "setting": "Alabama, 1930s"
    },
    {
        "id": "book_3",
        "title": "1984",
        "author": "George Orwell",
        "genre": "Dystopian",
        "year": 1949,
        "rating": 4.4,
        "pages": 328,
        "description": "A chilling vision of totalitarian control and surveillance society",
        "themes": "totalitarianism, surveillance, freedom, truth",
        "setting": "Oceania, dystopian future"
    },
    {
        "id": "book_4",
        "title": "Harry Potter and the Philosopher's Stone",
        "author": "J.K. Rowling",
        "genre": "Fantasy",
        "year": 1997,
        "rating": 4.5,
        "pages": 223,
        "description": "A young wizard discovers his magical heritage and begins his education at Hogwarts",
        "themes": "friendship, courage, good vs evil, coming of age",
        "setting": "England, magical world"
    },
    {
        "id": "book_5",
        "title": "The Lord of the Rings",
        "author": "J.R.R. Tolkien",
        "genre": "Fantasy",
        "year": 1954,
        "rating": 4.5,
        "pages": 1216,
        "description": "An epic fantasy quest to destroy a powerful ring and save Middle-earth",
        "themes": "heroism, friendship, good vs evil, power corruption",
        "setting": "Middle-earth, fantasy realm"
    },
    {
        "id": "book_6",
        "title": "The Hitchhiker's Guide to the Galaxy",
        "author": "Douglas Adams",
        "genre": "Science Fiction",
        "year": 1979,
        "rating": 4.2,
        "pages": 224,
        "description": "A humorous space adventure following Arthur Dent across the galaxy",
        "themes": "absurdity, technology, existence, humor",
        "setting": "Space, various planets"
    },
    {
        "id": "book_7",
        "title": "Dune",
        "author": "Frank Herbert",
        "genre": "Science Fiction",
        "year": 1965,
        "rating": 4.3,
        "pages": 688,
        "description": "A complex tale of politics, religion, and ecology on a desert planet",
        "themes": "power, ecology, religion, politics",
        "setting": "Arrakis, distant future"
    },
    {
        "id": "book_8",
        "title": "The Hunger Games",
        "author": "Suzanne Collins",
        "genre": "Dystopian",
        "year": 2008,
        "rating": 4.2,
        "pages": 374,
        "description": "A teenage girl fights for survival in a brutal televised competition",
        "themes": "survival, oppression, sacrifice, rebellion",
        "setting": "Panem, dystopian future"
    },
]

def create_book_collection():
    """
    Exercise 1: Create a ChromaDB collection for books
    
    TODO: Implement collection creation with proper configuration
    """
    collection = None  # Replace with collection creation
    
    # TODO: Configure collection with:
    # - Name: "book_recommendations"
    # - Embedding function
    # - Cosine similarity
    # - Descriptive metadata
    
    return collection

def generate_book_documents(books):
    """
    Exercise 2: Create comprehensive text documents for semantic search
    
    TODO: Generate rich text descriptions that combine:
    - Title and author
    - Description
    - Themes
    - Setting
    
    Args:
        books: List of book dictionaries
        
    Returns:
        List of text documents for similarity search
    """
    documents = []
    
    for book in books:
        # TODO: Create a comprehensive document string
        # Hint: Combine title, description, themes, and setting
        # Example format: "Title by Author. Description. Themes: themes. Set in: setting."
        
        document = ""  # Replace with document generation logic
        documents.append(document)
    
    return documents

def add_books_to_collection(collection, books, documents):
    """
    Exercise 3: Add books to the collection with metadata
    
    TODO: Add books to collection with comprehensive metadata
    
    Args:
        collection: ChromaDB collection
        books: List of book dictionaries
        documents: List of generated documents
    """
    # TODO: Prepare data for collection.add()
    # - Extract IDs
    # - Use generated documents
    # - Include all metadata fields
    
    pass  # Replace with implementation

def similarity_search(collection, query, n_results=5):
    """
    Exercise 4a: Implement similarity search
    
    TODO: Search for books using semantic similarity
    
    Args:
        collection: ChromaDB collection
        query: Search query string
        n_results: Number of results to return
        
    Returns:
        Search results
    """
    # TODO: Implement similarity search
    # - Use collection.query()
    # - Handle empty results
    # - Return formatted results
    
    results = None  # Replace with search implementation
    return results

def metadata_filter_search(collection, filters):
    """
    Exercise 4b: Implement metadata filtering
    
    TODO: Filter books by metadata criteria
    
    Args:
        collection: ChromaDB collection
        filters: Dictionary of filter criteria
        
    Returns:
        Filtered results
    """
    # TODO: Implement metadata filtering
    # - Use collection.get() with where clause
    # - Support various filter types
    # - Handle empty results
    
    results = None  # Replace with filter implementation
    return results

def combined_search(collection, query, filters, n_results=5):
    """
    Exercise 4c: Implement combined semantic + metadata search
    
    TODO: Combine similarity search with metadata filtering
    
    Args:
        collection: ChromaDB collection
        query: Search query string
        filters: Metadata filters
        n_results: Number of results to return
        
    Returns:
        Combined search results
    """
    # TODO: Implement combined search
    # - Use collection.query() with where parameter
    # - Combine semantic similarity with filters
    # - Handle edge cases
    
    results = None  # Replace with combined search implementation
    return results

def display_results(results, search_type="Search"):
    """
    Helper function to display search results in a readable format
    
    Args:
        results: Search results from ChromaDB
        search_type: Type of search for display purposes
    """
    print(f"\n=== {search_type} Results ===")
    
    # TODO: Implement result display
    # - Check if results exist
    # - Display book information
    # - Show similarity scores if available
    # - Format nicely for readability
    
    if not results:
        print("No results found")
        return
    
    # Handle both query results (with distances) and get results (without distances)
    # Your implementation here
    
    pass  # Replace with display logic

def run_advanced_search_examples(collection):
    """
    Exercise 5: Run comprehensive search examples
    
    TODO: Implement various search scenarios
    """
    print("🔍 Advanced Book Search Examples")
    print("=" * 50)
    
    # Example 1: Similarity search for magical fantasy adventure
    print("\n1. Searching for 'magical fantasy adventure':")
    # TODO: Call similarity_search() with appropriate query
    
    # Example 2: Filter books by genre (Fantasy or Science Fiction)
    print("\n2. Finding Fantasy or Science Fiction books:")
    # TODO: Call metadata_filter_search() with genre filters
    
    # Example 3: Filter books by high rating (4.0+)
    print("\n3. Finding highly-rated books (4.0+):")
    # TODO: Call metadata_filter_search() with rating filter
    
    # Example 4: Combined search - highly-rated dystopian books
    print("\n4. Finding highly-rated dystopian books about 'rebellion and freedom':")
    # TODO: Call combined_search() with both semantic and metadata filters
    
    # Bonus searches (implement if time allows)
    print("\n🏆 Bonus Searches:")
    
    # Bonus 1: Books from a specific decade
    print("\n5. Books from the 1960s:")
    # TODO: Implement decade-based search
    
    # Bonus 2: Books with similar page counts
    print("\n6. Books with 200-400 pages:")
    # TODO: Implement page count range search
    
    # Bonus 3: Books matching multiple themes
    print("\n7. Books about 'friendship' and 'courage':")
    # TODO: Implement multi-theme search

def main():
    """
    Main function to run the complete exercise
    """
    try:
        print("📚 Advanced Book Recommendation System")
        print("=" * 50)
        
        # Step 1: Create collection
        print("\n📖 Creating book collection...")
        collection = create_book_collection()
        
        if collection is None:
            print("❌ Collection creation failed. Please implement create_book_collection()")
            return
        
        # Step 2: Generate documents
        print("📝 Generating book documents...")
        documents = generate_book_documents(books)
        
        if not documents or all(doc == "" for doc in documents):
            print("❌ Document generation failed. Please implement generate_book_documents()")
            return
        
        # Step 3: Add books to collection
        print("📚 Adding books to collection...")
        add_books_to_collection(collection, books, documents)
        
        # Step 4: Verify data was added
        all_books = collection.get()
        if len(all_books['ids']) == 0:
            print("❌ No books found in collection. Please implement add_books_to_collection()")
            return
        
        print(f"✅ Successfully added {len(all_books['ids'])} books to collection")
        
        # Step 5: Run search examples
        run_advanced_search_examples(collection)
        
    except Exception as error:
        print(f"❌ Error occurred: {error}")
        print("\nDebugging tips:")
        print("- Check that ChromaDB client is properly initialized")
        print("- Verify embedding function is created correctly")
        print("- Ensure all TODO sections are implemented")
        print("- Make sure the required packages are installed")

if __name__ == "__main__":
    main()
