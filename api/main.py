from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
import uvicorn
from contextlib import asynccontextmanager

# Global variables for ChromaDB
client = None
collection = None
ef = None

# Pydantic models for request/response validation
class EmployeeSearchRequest(BaseModel):
    """Request model for employee search"""
    query: str = Field(..., description="Natural language search query", example="Python developer with web experience")
    n_results: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    
class MetadataFilter(BaseModel):
    """Model for metadata filtering"""
    department: Optional[str] = Field(None, example="Engineering")
    experience_min: Optional[int] = Field(None, ge=0, description="Minimum years of experience")
    experience_max: Optional[int] = Field(None, ge=0, description="Maximum years of experience")
    location: Optional[str] = Field(None, example="New York")
    employment_type: Optional[str] = Field(None, example="Full-time")

class AdvancedSearchRequest(BaseModel):
    """Request model for advanced search combining similarity and filters"""
    query: str = Field(..., description="Natural language search query")
    filters: Optional[MetadataFilter] = Field(None, description="Metadata filters")
    n_results: int = Field(default=5, ge=1, le=20, description="Number of results to return")

class EmployeeResponse(BaseModel):
    """Response model for employee data"""
    id: str
    name: str
    role: str
    department: str
    experience: int
    location: str
    employment_type: str
    similarity_score: float
    document: str

class SearchResponse(BaseModel):
    """Response model for search results"""
    query: str
    total_results: int
    results: List[EmployeeResponse]
    search_type: str

async def initialize_chromadb():
    """Initialize ChromaDB client and collection"""
    global client, collection, ef
    
    try:
        # Initialize embedding function
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create ChromaDB client
        client = chromadb.Client()
        
        # Try to get existing collection or create new one
        try:
            collection = client.get_collection(name="employee_collection", embedding_function=ef)
            print("✅ Connected to existing ChromaDB collection")
        except:
            # Create new collection if it doesn't exist
            collection = client.create_collection(
                name="employee_collection",
                metadata={"description": "A collection for storing employee data"},
                embedding_function=ef
            )
            
            # Add sample data
            await populate_sample_data()
            print("✅ Created new ChromaDB collection with sample data")
            
    except Exception as e:
        print(f"❌ Failed to initialize ChromaDB: {e}")
        raise

async def populate_sample_data():
    """Populate collection with sample employee data"""
    employees = [
        {
            "id": "employee_1",
            "name": "John Doe",
            "experience": 5,
            "department": "Engineering",
            "role": "Software Engineer",
            "skills": "Python, JavaScript, React, Node.js, databases",
            "location": "New York",
            "employment_type": "Full-time"
        },
        {
            "id": "employee_2",
            "name": "Jane Smith",
            "experience": 8,
            "department": "Marketing",
            "role": "Marketing Manager",
            "skills": "Digital marketing, SEO, content strategy, analytics, social media",
            "location": "Los Angeles",
            "employment_type": "Full-time"
        },
        {
            "id": "employee_3",
            "name": "Alice Johnson",
            "experience": 3,
            "department": "HR",
            "role": "HR Coordinator",
            "skills": "Recruitment, employee relations, HR policies, training programs",
            "location": "Chicago",
            "employment_type": "Full-time"
        },
        {
            "id": "employee_4",
            "name": "Michael Brown",
            "experience": 12,
            "department": "Engineering",
            "role": "Senior Software Engineer",
            "skills": "Java, Spring Boot, microservices, cloud architecture, DevOps",
            "location": "San Francisco",
            "employment_type": "Full-time"
        },
        {
            "id": "employee_5",
            "name": "Emily Wilson",
            "experience": 2,
            "department": "Marketing",
            "role": "Marketing Assistant",
            "skills": "Content creation, email marketing, market research, social media management",
            "location": "Austin",
            "employment_type": "Part-time"
        },
        {
            "id": "employee_6",
            "name": "David Lee",
            "experience": 15,
            "department": "Engineering",
            "role": "Engineering Manager",
            "skills": "Team leadership, project management, software architecture, mentoring",
            "location": "Seattle",
            "employment_type": "Full-time"
        },
        {
            "id": "employee_7",
            "name": "Sarah Clark",
            "experience": 8,
            "department": "HR",
            "role": "HR Manager",
            "skills": "Performance management, compensation planning, policy development, conflict resolution",
            "location": "Boston",
            "employment_type": "Full-time"
        },
        {
            "id": "employee_8",
            "name": "Chris Evans",
            "experience": 20,
            "department": "Engineering",
            "role": "Senior Architect",
            "skills": "System design, distributed systems, cloud platforms, technical strategy",
            "location": "New York",
            "employment_type": "Full-time"
        }
    ]
    
    # Create documents
    employee_documents = []
    for employee in employees:
        document = f"{employee['role']} with {employee['experience']} years of experience in {employee['department']}. "
        document += f"Skills: {employee['skills']}. Located in {employee['location']}. "
        document += f"Employment type: {employee['employment_type']}."
        employee_documents.append(document)
    
    # Add to collection
    collection.add(
        ids=[employee["id"] for employee in employees],
        documents=employee_documents,
        metadatas=[{
            "name": employee["name"],
            "department": employee["department"],
            "role": employee["role"],
            "experience": employee["experience"],
            "location": employee["location"],
            "employment_type": employee["employment_type"]
        } for employee in employees]
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("🚀 Starting Employee Search API...")
    await initialize_chromadb()
    print("✅ API ready to serve requests!")
    
    yield
    
    # Shutdown
    print("🔄 Shutting down API...")

# Create FastAPI app
app = FastAPI(
    title="Employee Similarity Search API",
    description="A powerful semantic search API for employee data using ChromaDB and SentenceTransformers",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def build_metadata_filter(filters: Optional[MetadataFilter]) -> Optional[Dict[str, Any]]:
    """Build ChromaDB metadata filter from request"""
    if not filters:
        return None
    
    conditions = []
    
    if filters.department:
        conditions.append({"department": filters.department})
    
    # Handle experience filtering with proper edge case handling
    if filters.experience_min is not None and filters.experience_max is not None:
        if filters.experience_min == filters.experience_max:
            # If min and max are the same, use exact match
            conditions.append({"experience": filters.experience_min})
        elif filters.experience_min < filters.experience_max:
            # Only add range if min < max
            conditions.append({
                "experience": {
                    "$gte": filters.experience_min,
                    "$lte": filters.experience_max
                }
            })
        # If min > max, ignore both (invalid range)
    elif filters.experience_min is not None and filters.experience_min > 0:
        conditions.append({"experience": {"$gte": filters.experience_min}})
    elif filters.experience_max is not None and filters.experience_max > 0:
        conditions.append({"experience": {"$lte": filters.experience_max}})
    
    if filters.location:
        conditions.append({"location": filters.location})
    
    if filters.employment_type:
        conditions.append({"employment_type": filters.employment_type})
    
    if len(conditions) == 0:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}

def format_search_results(results: Dict, query: str, search_type: str) -> SearchResponse:
    """Format ChromaDB results into API response"""
    employees = []
    
    for i in range(len(results['ids'][0]) if results['ids'] else 0):
        employee = EmployeeResponse(
            id=results['ids'][0][i],
            name=results['metadatas'][0][i]['name'],
            role=results['metadatas'][0][i]['role'],
            department=results['metadatas'][0][i]['department'],
            experience=results['metadatas'][0][i]['experience'],
            location=results['metadatas'][0][i]['location'],
            employment_type=results['metadatas'][0][i]['employment_type'],
            similarity_score=results['distances'][0][i] if results['distances'] else 0.0,
            document=results['documents'][0][i] if results['documents'] else ""
        )
        employees.append(employee)
    
    return SearchResponse(
        query=query,
        total_results=len(employees),
        results=employees,
        search_type=search_type
    )

# API Endpoints

@app.get("/", summary="API Health Check")
async def root():
    """Health check endpoint"""
    return {"message": "Employee Similarity Search API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test ChromaDB connection
        count = collection.count()
        return {
            "status": "healthy",
            "chromadb_status": "connected",
            "employee_count": count,
            "embedding_model": "all-MiniLM-L6-v2"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/search/similarity", response_model=SearchResponse, summary="Semantic similarity search")
async def similarity_search(request: EmployeeSearchRequest):
    """
    Perform semantic similarity search using natural language queries.
    
    Examples:
    - "Python developer with web experience"
    - "team leader with management skills"
    - "marketing professional with social media expertise"
    """
    try:
        results = collection.query(
            query_texts=[request.query],
            n_results=request.n_results
        )
        
        if not results['ids'] or len(results['ids'][0]) == 0:
            return SearchResponse(
                query=request.query,
                total_results=0,
                results=[],
                search_type="similarity"
            )
        
        return format_search_results(results, request.query, "similarity")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/search/filter", response_model=SearchResponse, summary="Metadata filtering search")
async def filter_search(
    department: Optional[str] = Query(None, description="Filter by department"),
    experience_min: Optional[int] = Query(None, description="Minimum years of experience"),
    experience_max: Optional[int] = Query(None, description="Maximum years of experience"),
    location: Optional[str] = Query(None, description="Filter by location"),
    employment_type: Optional[str] = Query(None, description="Filter by employment type"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results to return")
):
    """
    Filter employees by metadata criteria without semantic search.
    """
    try:
        filters = MetadataFilter(
            department=department,
            experience_min=experience_min,
            experience_max=experience_max,
            location=location,
            employment_type=employment_type
        )
        
        where_clause = build_metadata_filter(filters)
        
        if where_clause:
            results = collection.get(where=where_clause, limit=limit)
        else:
            results = collection.get(limit=limit)
        
        # Convert get() results to query() format for consistent formatting
        formatted_results = {
            'ids': [results['ids']],
            'documents': [results['documents']],
            'metadatas': [results['metadatas']],
            'distances': [[0.0] * len(results['ids'])]  # No distances for filter-only search
        }
        
        query_desc = f"Filter: {filters.dict(exclude_none=True)}"
        return format_search_results(formatted_results, query_desc, "filter")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Filter search failed: {str(e)}")

@app.post("/search/advanced", response_model=SearchResponse, summary="Combined similarity and metadata search")
async def advanced_search(request: AdvancedSearchRequest):
    """
    Perform advanced search combining semantic similarity with metadata filtering.
    
    This is the most powerful search option, allowing you to:
    1. Use natural language queries for semantic matching
    2. Apply metadata filters for precise criteria
    3. Get ranked results by relevance
    """
    try:
        where_clause = build_metadata_filter(request.filters)
        
        results = collection.query(
            query_texts=[request.query],
            n_results=request.n_results,
            where=where_clause
        )
        
        if not results['ids'] or len(results['ids'][0]) == 0:
            return SearchResponse(
                query=request.query,
                total_results=0,
                results=[],
                search_type="advanced"
            )
        
        return format_search_results(results, request.query, "advanced")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced search failed: {str(e)}")

@app.get("/employees", summary="List all employees")
async def list_employees(limit: int = Query(50, ge=1, le=100)):
    """Get all employees with optional limit"""
    try:
        results = collection.get(limit=limit)
        
        employees = []
        for i in range(len(results['ids'])):
            employee = {
                "id": results['ids'][i],
                "name": results['metadatas'][i]['name'],
                "role": results['metadatas'][i]['role'],
                "department": results['metadatas'][i]['department'],
                "experience": results['metadatas'][i]['experience'],
                "location": results['metadatas'][i]['location'],
                "employment_type": results['metadatas'][i]['employment_type']
            }
            employees.append(employee)
        
        return {
            "total_employees": len(employees),
            "employees": employees
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list employees: {str(e)}")

@app.get("/stats", summary="Get collection statistics")
async def get_stats():
    """Get statistics about the employee collection"""
    try:
        all_employees = collection.get()
        
        if not all_employees['metadatas']:
            return {"message": "No employees found"}
        
        # Calculate statistics
        departments = {}
        locations = {}
        employment_types = {}
        experience_levels = []
        
        for metadata in all_employees['metadatas']:
            # Department stats
            dept = metadata['department']
            departments[dept] = departments.get(dept, 0) + 1
            
            # Location stats
            loc = metadata['location']
            locations[loc] = locations.get(loc, 0) + 1
            
            # Employment type stats
            emp_type = metadata['employment_type']
            employment_types[emp_type] = employment_types.get(emp_type, 0) + 1
            
            # Experience stats
            experience_levels.append(metadata['experience'])
        
        return {
            "total_employees": len(all_employees['ids']),
            "departments": departments,
            "locations": locations,
            "employment_types": employment_types,
            "experience_stats": {
                "min": min(experience_levels) if experience_levels else 0,
                "max": max(experience_levels) if experience_levels else 0,
                "average": sum(experience_levels) / len(experience_levels) if experience_levels else 0
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )