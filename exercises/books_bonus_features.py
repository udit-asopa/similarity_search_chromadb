# Advanced Book Recommendation System - Bonus Features
#
# This file demonstrates advanced features and extensions beyond the basic exercise:
# - Recommendation scoring algorithms
# - Multi-criteria optimization
# - Book similarity clustering
# - Advanced analytics and insights
# - Interactive recommendation engine

import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from collections import defaultdict
import json

# Initialize ChromaDB with persistent storage for bonus features
client = chromadb.PersistentClient(path="./book_recommendations_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Extended book dataset with additional metadata
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
        "setting": "New York, 1920s",
        "reading_level": "High School",
        "awards": ["Modern Library's Top 100"],
        "popularity_score": 8.5
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
        "setting": "Alabama, 1930s",
        "reading_level": "High School",
        "awards": ["Pulitzer Prize", "Presidential Medal of Freedom"],
        "popularity_score": 9.2
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
        "setting": "Oceania, dystopian future",
        "reading_level": "College",
        "awards": ["Time's 100 Best Novels"],
        "popularity_score": 9.0
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
        "setting": "England, magical world",
        "reading_level": "Middle Grade",
        "awards": ["British Children's Book Award"],
        "popularity_score": 9.8
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
        "setting": "Middle-earth, fantasy realm",
        "reading_level": "College",
        "awards": ["International Fantasy Award"],
        "popularity_score": 9.5
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
        "setting": "Space, various planets",
        "reading_level": "High School",
        "awards": ["Hugo Award Nominee"],
        "popularity_score": 8.0
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
        "setting": "Arrakis, distant future",
        "reading_level": "College",
        "awards": ["Hugo Award", "Nebula Award"],
        "popularity_score": 8.8
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
        "setting": "Panem, dystopian future",
        "reading_level": "Young Adult",
        "awards": ["California Young Reader Medal"],
        "popularity_score": 8.7
    },
]

class AdvancedBookRecommendationSystem:
    """
    Advanced book recommendation system with multiple scoring algorithms
    and sophisticated search capabilities.
    """
    
    def __init__(self):
        self.collection = None
        self.setup_collection()
    
    def setup_collection(self):
        """Initialize or get existing collection with extended features"""
        try:
            self.collection = client.get_collection(
                name="advanced_book_recommendations",
                embedding_function=ef
            )
            print("✅ Retrieved existing advanced collection")
        except:
            self.collection = client.create_collection(
                name="advanced_book_recommendations",
                embedding_function=ef,
                metadata={"description": "Advanced book recommendation system with bonus features"}
            )
            self._populate_collection()
            print("✅ Created new advanced collection")
    
    def _populate_collection(self):
        """Populate collection with enhanced book data"""
        documents = []
        metadatas = []
        ids = []
        
        for book in books:
            # Create enhanced documents
            document = f"{book['title']} by {book['author']}. "
            document += f"{book['description']} "
            document += f"Themes: {book['themes']}. "
            document += f"Setting: {book['setting']}. "
            document += f"Genre: {book['genre']}. "
            document += f"Reading level: {book['reading_level']}. "
            document += f"Awards: {', '.join(book['awards']) if isinstance(book['awards'], list) else book['awards']}."
            
            documents.append(document)
            
            # Prepare metadata (convert lists to strings for ChromaDB compatibility)
            metadata = {}
            for k, v in book.items():
                if k != 'id':
                    if isinstance(v, list):
                        metadata[k] = ', '.join(v)  # Convert list to comma-separated string
                    else:
                        metadata[k] = v
            metadatas.append(metadata)
            ids.append(book['id'])
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"📚 Added {len(books)} books with enhanced metadata")
    
    def smart_recommendation_score(self, book_metadata, user_preferences=None):
        """
        Calculate a sophisticated recommendation score based on multiple factors
        
        Args:
            book_metadata: Book metadata dictionary
            user_preferences: User preference weights (optional)
        
        Returns:
            Float: Composite recommendation score (0-10)
        """
        if user_preferences is None:
            user_preferences = {
                'rating_weight': 0.3,
                'popularity_weight': 0.2,
                'recency_weight': 0.1,
                'award_weight': 0.2,
                'length_preference': 0.2  # Preference for book length
            }
        
        # Normalize rating (0-10 scale)
        rating_score = (book_metadata['rating'] / 5.0) * 10
        
        # Normalize popularity (assuming 0-10 scale)
        popularity_score = book_metadata['popularity_score']
        
        # Recency score (newer books score higher, but classics get bonus)
        current_year = 2024
        book_year = book_metadata['year']
        if current_year - book_year > 50:  # Classic bonus
            recency_score = 8.0
        else:
            recency_score = min(10, (book_year - 1900) / 12.4)  # Normalize to 0-10
        
        # Award score based on number of prestigious awards
        awards_str = book_metadata.get('awards', '')
        award_count = len(awards_str.split(', ')) if awards_str else 0
        award_score = min(10, award_count * 3)
        
        # Length preference score (medium length preferred)
        pages = book_metadata['pages']
        if 200 <= pages <= 400:
            length_score = 10
        elif 150 <= pages <= 600:
            length_score = 8
        else:
            length_score = 6
        
        # Calculate weighted score
        final_score = (
            rating_score * user_preferences['rating_weight'] +
            popularity_score * user_preferences['popularity_weight'] +
            recency_score * user_preferences['recency_weight'] +
            award_score * user_preferences['award_weight'] +
            length_score * user_preferences['length_preference']
        )
        
        return round(final_score, 2)
    
    def personalized_recommendations(self, query, user_profile=None, n_results=5):
        """
        Generate personalized recommendations based on user profile
        
        Args:
            query: Search query
            user_profile: Dictionary with user preferences and history
            n_results: Number of recommendations to return
        
        Returns:
            List of recommended books with scores
        """
        if user_profile is None:
            user_profile = {
                'preferred_genres': ['Fantasy', 'Science Fiction'],
                'rating_threshold': 4.0,
                'reading_level': 'High School',
                'max_pages': 500,
                'avoid_themes': ['horror', 'tragedy']
            }
        
        # Build dynamic filters based on user profile
        filters = {"$and": []}
        
        if user_profile.get('preferred_genres'):
            filters["$and"].append({
                "genre": {"$in": user_profile['preferred_genres']}
            })
        
        if user_profile.get('rating_threshold'):
            filters["$and"].append({
                "rating": {"$gte": user_profile['rating_threshold']}
            })
        
        if user_profile.get('max_pages'):
            filters["$and"].append({
                "pages": {"$lte": user_profile['max_pages']}
            })
        
        # Remove $and if no filters
        if not filters["$and"]:
            filters = None
        
        # Perform semantic search with filters
        results = self.collection.query(
            query_texts=[query],
            where=filters,
            n_results=n_results * 2  # Get more results for scoring
        )
        
        if not results['ids'][0]:
            return []
        
        # Calculate recommendation scores
        recommendations = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            similarity_distance = results['distances'][0][i]
            
            # Convert distance to similarity score
            semantic_score = max(0, 1 - (similarity_distance / 2))
            
            # Calculate smart recommendation score
            rec_score = self.smart_recommendation_score(metadata, user_profile.get('score_weights'))
            
            # Combine semantic and recommendation scores
            combined_score = (semantic_score * 0.4) + (rec_score / 10 * 0.6)
            
            recommendations.append({
                'id': results['ids'][0][i],
                'metadata': metadata,
                'semantic_score': semantic_score,
                'recommendation_score': rec_score,
                'combined_score': combined_score,
                'similarity_distance': similarity_distance
            })
        
        # Sort by combined score
        recommendations.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return recommendations[:n_results]
    
    def find_similar_books(self, book_id, n_results=3):
        """
        Find books similar to a given book using embedding similarity
        """
        # Get the target book
        target_book = self.collection.get(ids=[book_id])
        
        if not target_book['ids']:
            print(f"Book with ID '{book_id}' not found")
            return []
        
        target_document = target_book['documents'][0]
        
        # Search for similar books (excluding the target book)
        results = self.collection.query(
            query_texts=[target_document],
            n_results=n_results + 1  # +1 to account for the target book itself
        )
        
        # Filter out the target book
        similar_books = []
        for i, book_id_result in enumerate(results['ids'][0]):
            if book_id_result != book_id:  # Exclude the target book
                similar_books.append({
                    'id': book_id_result,
                    'metadata': results['metadatas'][0][i],
                    'similarity_distance': results['distances'][0][i],
                    'similarity_score': 1 - (results['distances'][0][i] / 2)
                })
        
        return similar_books[:n_results]
    
    def genre_diversity_analysis(self):
        """
        Analyze genre diversity and provide insights
        """
        all_books = self.collection.get()
        
        if not all_books['ids']:
            return {}
        
        # Genre analysis
        genres = [meta['genre'] for meta in all_books['metadatas']]
        genre_stats = defaultdict(list)
        
        for book in all_books['metadatas']:
            genre = book['genre']
            genre_stats[genre].append({
                'rating': book['rating'],
                'year': book['year'],
                'pages': book['pages'],
                'popularity': book['popularity_score']
            })
        
        # Calculate genre insights
        genre_insights = {}
        for genre, books_data in genre_stats.items():
            avg_rating = sum(book['rating'] for book in books_data) / len(books_data)
            avg_year = sum(book['year'] for book in books_data) / len(books_data)
            avg_pages = sum(book['pages'] for book in books_data) / len(books_data)
            avg_popularity = sum(book['popularity'] for book in books_data) / len(books_data)
            
            genre_insights[genre] = {
                'count': len(books_data),
                'avg_rating': round(avg_rating, 2),
                'avg_year': round(avg_year),
                'avg_pages': round(avg_pages),
                'avg_popularity': round(avg_popularity, 1)
            }
        
        return genre_insights
    
    def reading_time_estimator(self, book_id, reading_speed_wpm=200):
        """
        Estimate reading time for a book based on page count
        
        Args:
            book_id: ID of the book
            reading_speed_wpm: Words per minute reading speed (default: 200)
        
        Returns:
            Dictionary with reading time estimates
        """
        book = self.collection.get(ids=[book_id])
        
        if not book['ids']:
            return None
        
        metadata = book['metadatas'][0]
        pages = metadata['pages']
        
        # Estimate words (average ~300 words per page)
        estimated_words = pages * 300
        
        # Calculate reading times
        minutes = estimated_words / reading_speed_wpm
        hours = minutes / 60
        days_30_min = hours / 0.5  # 30 minutes per day
        days_60_min = hours / 1.0  # 60 minutes per day
        
        return {
            'book_title': metadata['title'],
            'pages': pages,
            'estimated_words': estimated_words,
            'reading_time': {
                'minutes': round(minutes),
                'hours': round(hours, 1),
                'days_30min_session': round(days_30_min, 1),
                'days_60min_session': round(days_60_min, 1)
            }
        }

def demonstrate_bonus_features():
    """
    Demonstrate all bonus features of the advanced recommendation system
    """
    print("🚀 Advanced Book Recommendation System - Bonus Features")
    print("=" * 70)
    
    # Initialize the advanced system
    system = AdvancedBookRecommendationSystem()
    
    # Feature 1: Personalized Recommendations
    print("\n🎯 Feature 1: Personalized Recommendations")
    print("-" * 45)
    
    user_profile = {
        'preferred_genres': ['Fantasy', 'Science Fiction'],
        'rating_threshold': 4.0,
        'reading_level': 'High School',
        'max_pages': 500,
        'score_weights': {
            'rating_weight': 0.4,
            'popularity_weight': 0.3,
            'recency_weight': 0.1,
            'award_weight': 0.1,
            'length_preference': 0.1
        }
    }
    
    recommendations = system.personalized_recommendations(
        "epic adventure with magic and friendship",
        user_profile=user_profile,
        n_results=3
    )
    
    print(f"Personalized recommendations for fantasy lover:")
    for i, rec in enumerate(recommendations, 1):
        metadata = rec['metadata']
        print(f"\n{i}. {metadata['title']} by {metadata['author']}")
        print(f"   Combined Score: {rec['combined_score']:.3f}")
        print(f"   Semantic Score: {rec['semantic_score']:.3f}")
        print(f"   Recommendation Score: {rec['recommendation_score']:.1f}/10")
        print(f"   Genre: {metadata['genre']} | Rating: {metadata['rating']}")
    
    # Feature 2: Similar Books Discovery
    print("\n\n📚 Feature 2: Similar Books Discovery")
    print("-" * 40)
    
    similar_books = system.find_similar_books("book_4", n_results=2)  # Harry Potter
    print("Books similar to 'Harry Potter and the Philosopher's Stone':")
    
    for i, book in enumerate(similar_books, 1):
        metadata = book['metadata']
        print(f"\n{i}. {metadata['title']} by {metadata['author']}")
        print(f"   Similarity Score: {book['similarity_score']:.3f}")
        print(f"   Genre: {metadata['genre']} | Rating: {metadata['rating']}")
        print(f"   Why similar: Shared themes in {metadata['themes'][:50]}...")
    
    # Feature 3: Genre Diversity Analysis
    print("\n\n📊 Feature 3: Genre Diversity Analysis")
    print("-" * 38)
    
    genre_insights = system.genre_diversity_analysis()
    print("Collection insights by genre:")
    
    for genre, stats in genre_insights.items():
        print(f"\n📖 {genre}:")
        print(f"   Books: {stats['count']}")
        print(f"   Avg Rating: {stats['avg_rating']}")
        print(f"   Avg Year: {stats['avg_year']}")
        print(f"   Avg Pages: {stats['avg_pages']}")
        print(f"   Avg Popularity: {stats['avg_popularity']}/10")
    
    # Feature 4: Reading Time Estimation
    print("\n\n⏱️  Feature 4: Reading Time Estimation")
    print("-" * 37)
    
    # Estimate for different books
    book_ids = ["book_4", "book_5", "book_7"]  # Harry Potter, LOTR, Dune
    
    for book_id in book_ids:
        reading_time = system.reading_time_estimator(book_id, reading_speed_wpm=250)
        if reading_time:
            print(f"\n📖 {reading_time['book_title']}:")
            print(f"   Pages: {reading_time['pages']}")
            print(f"   Estimated reading time: {reading_time['reading_time']['hours']} hours")
            print(f"   At 30 min/day: {reading_time['reading_time']['days_30min_session']} days")
            print(f"   At 60 min/day: {reading_time['reading_time']['days_60min_session']} days")
    
    # Feature 5: Advanced Multi-Criteria Search
    print("\n\n🔍 Feature 5: Advanced Multi-Criteria Search")
    print("-" * 43)
    
    # Complex search combining multiple factors
    complex_user_profile = {
        'preferred_genres': ['Classic', 'Dystopian'],
        'rating_threshold': 4.2,
        'max_pages': 400,
        'score_weights': {
            'rating_weight': 0.3,
            'popularity_weight': 0.2,
            'award_weight': 0.3,  # High weight on awards
            'recency_weight': 0.1,
            'length_preference': 0.1
        }
    }
    
    advanced_recs = system.personalized_recommendations(
        "thought-provoking literature about society and human nature",
        user_profile=complex_user_profile,
        n_results=3
    )
    
    print("Advanced search: Award-winning classics about society:")
    for i, rec in enumerate(advanced_recs, 1):
        metadata = rec['metadata']
        print(f"\n{i}. {metadata['title']} by {metadata['author']}")
        print(f"   Awards: {metadata['awards']}")
        print(f"   Combined Score: {rec['combined_score']:.3f}")
        print(f"   Themes: {metadata['themes'][:60]}...")

def interactive_recommendation_demo():
    """
    Interactive demonstration allowing user input for recommendations
    """
    print("\n\n🎮 Interactive Recommendation Demo")
    print("=" * 40)
    
    system = AdvancedBookRecommendationSystem()
    
    print("Answer a few questions to get personalized recommendations:")
    
    # Simple interactive questionnaire
    try:
        # Favorite genres
        print("\nWhat genres do you enjoy? (comma-separated)")
        print("Options: Classic, Fantasy, Science Fiction, Dystopian")
        genre_input = input("Genres: ").strip()
        
        if genre_input:
            preferred_genres = [g.strip() for g in genre_input.split(",")]
        else:
            preferred_genres = ["Fantasy", "Science Fiction"]  # Default
        
        # Rating preference
        print("\nMinimum rating preference? (1.0-5.0)")
        rating_input = input("Rating (default 4.0): ").strip()
        
        try:
            rating_threshold = float(rating_input) if rating_input else 4.0
        except ValueError:
            rating_threshold = 4.0
        
        # Page count preference
        print("\nMaximum book length in pages?")
        pages_input = input("Pages (default 500): ").strip()
        
        try:
            max_pages = int(pages_input) if pages_input else 500
        except ValueError:
            max_pages = 500
        
        # Search query
        print("\nWhat kind of story are you looking for?")
        query = input("Describe your ideal book: ").strip()
        
        if not query:
            query = "engaging story with interesting characters"
        
        # Build user profile
        user_profile = {
            'preferred_genres': preferred_genres,
            'rating_threshold': rating_threshold,
            'max_pages': max_pages
        }
        
        # Get recommendations
        recommendations = system.personalized_recommendations(
            query, user_profile=user_profile, n_results=3
        )
        
        print(f"\n🎯 Here are your personalized recommendations:")
        print("=" * 50)
        
        for i, rec in enumerate(recommendations, 1):
            metadata = rec['metadata']
            reading_time = system.reading_time_estimator(rec['id'])
            
            print(f"\n📖 {i}. {metadata['title']} by {metadata['author']}")
            print(f"   Genre: {metadata['genre']} | Rating: {metadata['rating']}/5")
            print(f"   Pages: {metadata['pages']} | Year: {metadata['year']}")
            print(f"   Match Score: {rec['combined_score']:.3f}")
            
            if reading_time:
                print(f"   Reading Time: ~{reading_time['reading_time']['hours']} hours")
            else:
                print(f"   Reading Time: Not available")
                
            print(f"   Description: {metadata['description']}")
            print(f"   Awards: {metadata['awards']}")
        
        # Suggest similar books to first recommendation
        if recommendations:
            print(f"\n💡 If you like '{recommendations[0]['metadata']['title']}', you might also enjoy:")
            similar = system.find_similar_books(recommendations[0]['id'], n_results=2)
            for book in similar:
                metadata = book['metadata']
                print(f"   - {metadata['title']} by {metadata['author']} (similarity: {book['similarity_score']:.3f})")
    
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError in interactive demo: {e}")

def main():
    """
    Run all bonus features demonstrations
    """
    try:
        # Main bonus features demo
        demonstrate_bonus_features()
        
        # Interactive demo
        print("\n" + "=" * 70)
        interactive_recommendation_demo()
        
        print("\n\n🎉 Bonus Features Demonstration Complete!")
        print("\nAdvanced features demonstrated:")
        print("✅ Personalized recommendation scoring")
        print("✅ Similar book discovery")
        print("✅ Genre diversity analysis")
        print("✅ Reading time estimation")
        print("✅ Multi-criteria search optimization")
        print("✅ Interactive recommendation engine")
        
    except Exception as error:
        print(f"❌ Error in bonus features: {error}")

if __name__ == "__main__":
    main()
