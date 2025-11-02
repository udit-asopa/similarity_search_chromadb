# Advanced Book Recommendation System - Complete Solution
# 
# This is the complete solution for the book recommendation exercise.
# It demonstrates all advanced search capabilities with ChromaDB.

import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB components
client = chromadb.Client()

# Create embedding function using SentenceTransformers
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Complete book dataset for the exercise
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
    Create a ChromaDB collection for book recommendations
    """
    try:
        collection = client.create_collection(
            name="book_recommendations",
            metadata={"description": "Advanced book recommendation system"},
            embedding_function=ef,
            configuration={
                "hnsw": {"space": "cosine"}
            }
        )
        print("✅ Collection 'book_recommendations' created successfully")
        return collection
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        return None

def generate_book_documents(books):
    """
    Create comprehensive text documents for semantic search
    
    Combines title, author, description, themes, and setting into rich searchable text
    """
    documents = []
    
    for book in books:
        # Create comprehensive document combining multiple fields
        document = f"{book['title']} by {book['author']}. "
        document += f"{book['description']} "
        document += f"Themes: {book['themes']}. "
        document += f"Setting: {book['setting']}. "
        document += f"Genre: {book['genre']}."
        
        documents.append(document)
    
    print(f"✅ Generated {len(documents)} book documents")
    return documents

def add_books_to_collection(collection, books, documents):
    """
    Add books to the collection with comprehensive metadata
    """
    try:
        # Prepare data for batch insertion
        ids = [book["id"] for book in books]
        
        # Prepare metadata (exclude 'id' field)
        metadatas = []
        for book in books:
            metadata = {k: v for k, v in book.items() if k != 'id'}
            metadatas.append(metadata)
        
        # Add to collection
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Successfully added {len(books)} books to collection")
    
    except Exception as e:
        print(f"❌ Error adding books: {e}")

def similarity_search(collection, query, n_results=5):
    """
    Perform similarity search using semantic embeddings
    """
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results['ids'][0]:
            print("No results found for similarity search")
            return None
            
        return results
    
    except Exception as e:
        print(f"❌ Error in similarity search: {e}")
        return None

def metadata_filter_search(collection, filters):
    """
    Filter books by metadata criteria
    """
    try:
        results = collection.get(where=filters)
        
        if not results['ids']:
            print("No results found for metadata filter")
            return None
            
        return results
    
    except Exception as e:
        print(f"❌ Error in metadata filtering: {e}")
        return None

def combined_search(collection, query, filters, n_results=5):
    """
    Combine similarity search with metadata filtering
    """
    try:
        results = collection.query(
            query_texts=[query],
            where=filters,
            n_results=n_results
        )
        
        if not results['ids'][0]:
            print("No results found for combined search")
            return None
            
        return results
    
    except Exception as e:
        print(f"❌ Error in combined search: {e}")
        return None

def display_results(results, search_type="Search"):
    """
    Display search results in a readable format
    """
    print(f"\n=== {search_type} Results ===")
    
    if not results:
        print("No results to display")
        return
    
    # Handle both query results (with distances) and get results (without distances)
    if 'distances' in results and results['distances'] and results['distances'][0]:
        # Query results with similarity scores
        for i in range(len(results['ids'][0])):
            book_id = results['ids'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            # Convert distance to similarity score (0-1, higher is better)
            similarity = 1 - (distance / 2)
            
            print(f"\n📖 {i+1}. {metadata['title']} by {metadata['author']}")
            print(f"   Genre: {metadata['genre']} | Rating: {metadata['rating']} | Year: {metadata['year']}")
            print(f"   Similarity: {similarity:.3f} | Pages: {metadata['pages']}")
            print(f"   Description: {metadata['description'][:100]}...")
    else:
        # Get results without similarity scores
        for i in range(len(results['ids'])):
            book_id = results['ids'][i]
            metadata = results['metadatas'][i]
            
            print(f"\n📖 {i+1}. {metadata['title']} by {metadata['author']}")
            print(f"   Genre: {metadata['genre']} | Rating: {metadata['rating']} | Year: {metadata['year']}")
            print(f"   Pages: {metadata['pages']}")
            print(f"   Description: {metadata['description'][:100]}...")

def run_advanced_search_examples(collection):
    """
    Run comprehensive search examples demonstrating various capabilities
    """
    print("🔍 Advanced Book Search Examples")
    print("=" * 50)
    
    # Example 1: Similarity search for magical fantasy adventure
    print("\n1. Searching for 'magical fantasy adventure':")
    results = similarity_search(collection, "magical fantasy adventure", n_results=3)
    display_results(results, "Similarity Search")
    
    # Example 2: Filter books by genre (Fantasy or Science Fiction)
    print("\n2. Finding Fantasy or Science Fiction books:")
    genre_filters = {"genre": {"$in": ["Fantasy", "Science Fiction"]}}
    results = metadata_filter_search(collection, genre_filters)
    display_results(results, "Genre Filter")
    
    # Example 3: Filter books by high rating (4.0+)
    print("\n3. Finding highly-rated books (4.0+):")
    rating_filters = {"rating": {"$gte": 4.0}}
    results = metadata_filter_search(collection, rating_filters)
    display_results(results, "High Rating Filter")
    
    # Example 4: Combined search - highly-rated dystopian books
    print("\n4. Finding highly-rated dystopian books about 'rebellion and freedom':")
    dystopian_filters = {
        "$and": [
            {"genre": "Dystopian"},
            {"rating": {"$gte": 4.0}}
        ]
    }
    results = combined_search(collection, "rebellion and freedom", dystopian_filters, n_results=3)
    display_results(results, "Combined Search")
    
    # Bonus searches
    print("\n🏆 Bonus Searches:")
    
    # Bonus 1: Books from a specific decade (1960s)
    print("\n5. Books from the 1960s:")
    decade_filters = {
        "$and": [
            {"year": {"$gte": 1960}},
            {"year": {"$lt": 1970}}
        ]
    }
    results = metadata_filter_search(collection, decade_filters)
    display_results(results, "1960s Books")
    
    # Bonus 2: Books with similar page counts (200-400 pages)
    print("\n6. Books with 200-400 pages:")
    page_filters = {
        "$and": [
            {"pages": {"$gte": 200}},
            {"pages": {"$lte": 400}}
        ]
    }
    results = metadata_filter_search(collection, page_filters)
    display_results(results, "Medium Length Books")
    
    # Bonus 3: Books about friendship and courage (semantic search)
    print("\n7. Books about 'friendship and courage':")
    results = similarity_search(collection, "friendship and courage", n_results=3)
    display_results(results, "Friendship & Courage Theme")
    
    # Bonus 4: Complex combined search - Fantasy books with high ratings and themes of good vs evil
    print("\n8. High-rated Fantasy books about 'good versus evil':")
    complex_filters = {
        "$and": [
            {"genre": "Fantasy"},
            {"rating": {"$gte": 4.3}}
        ]
    }
    results = combined_search(collection, "good versus evil", complex_filters, n_results=3)
    display_results(results, "Epic Fantasy Battle")

def demonstrate_search_analytics(collection):
    """
    Demonstrate search analytics and insights
    """
    print("\n📊 Search Analytics & Insights")
    print("=" * 40)
    
    # Get all books for analysis
    all_books = collection.get()
    
    if not all_books['ids']:
        print("No books found for analysis")
        return
    
    # Analyze collection composition
    genres = [meta['genre'] for meta in all_books['metadatas']]
    genre_counts = {}
    for genre in genres:
        genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    print(f"📚 Collection Overview:")
    print(f"   Total books: {len(all_books['ids'])}")
    print(f"   Genre distribution:")
    for genre, count in genre_counts.items():
        print(f"     - {genre}: {count} books")
    
    # Rating analysis
    ratings = [meta['rating'] for meta in all_books['metadatas']]
    avg_rating = sum(ratings) / len(ratings)
    max_rating = max(ratings)
    min_rating = min(ratings)
    
    print(f"\n⭐ Rating Analysis:")
    print(f"   Average rating: {avg_rating:.2f}")
    print(f"   Highest rating: {max_rating}")
    print(f"   Lowest rating: {min_rating}")
    
    # Publication year analysis
    years = [meta['year'] for meta in all_books['metadatas']]
    oldest_year = min(years)
    newest_year = max(years)
    
    print(f"\n📅 Publication Analysis:")
    print(f"   Year range: {oldest_year} - {newest_year}")
    print(f"   Span: {newest_year - oldest_year} years")

def main():
    """
    Main function to run the complete book recommendation system
    """
    try:
        print("📚 Advanced Book Recommendation System - Complete Solution")
        print("=" * 65)
        
        # Step 1: Create collection
        print("\n📖 Creating book collection...")
        collection = create_book_collection()
        
        if collection is None:
            print("❌ Failed to create collection. Exiting.")
            return
        
        # Step 2: Generate documents
        print("\n📝 Generating book documents...")
        documents = generate_book_documents(books)
        
        # Step 3: Add books to collection
        print("\n📚 Adding books to collection...")
        add_books_to_collection(collection, books, documents)
        
        # Step 4: Verify data was added
        all_books = collection.get()
        print(f"\n✅ Collection ready with {len(all_books['ids'])} books")
        
        # Step 5: Run search examples
        run_advanced_search_examples(collection)
        
        # Step 6: Show analytics
        demonstrate_search_analytics(collection)
        
        print("\n🎉 Book recommendation system demonstration complete!")
        print("\nKey takeaways:")
        print("- Semantic search finds books by meaning, not just keywords")
        print("- Metadata filtering enables precise criteria-based searches")
        print("- Combined search offers the best of both approaches")
        print("- ChromaDB makes it easy to build intelligent recommendation systems")
        
    except Exception as error:
        print(f"❌ Error occurred: {error}")
        print("\nIf you see import errors, make sure to run:")
        print("  pixi install")
        print("from the project root directory")

if __name__ == "__main__":
    main()
