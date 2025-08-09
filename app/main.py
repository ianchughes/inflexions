"""Main FastAPI application for the AI Editorial Engine."""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from .config import settings
from .models import (
    Puzzle, PuzzleGenerationRequest, PuzzleGenerationResponse,
    KnowledgeGraphEntity, EntityQuery, GenerationConstraints
)
from .database import KnowledgeGraphDB, VectorStore, CacheManager
from .agents import CreatorAgent, TricksterAgent, LinguistAgent, FactCheckerAgent, JudgeAgent
from .pipeline import PuzzlePipeline

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = structlog.get_logger(__name__)

# Global components
knowledge_graph: Optional[KnowledgeGraphDB] = None
vector_store: Optional[VectorStore] = None
cache_manager: Optional[CacheManager] = None
puzzle_pipeline: Optional[PuzzlePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    logger.info("Starting AI Editorial Engine...")
    
    global knowledge_graph, vector_store, cache_manager, puzzle_pipeline
    
    try:
        # Initialize components
        knowledge_graph = KnowledgeGraphDB()
        vector_store = VectorStore()
        cache_manager = CacheManager()
        
        # Initialize pipeline
        puzzle_pipeline = PuzzlePipeline(
            knowledge_graph=knowledge_graph,
            vector_store=vector_store,
            cache_manager=cache_manager
        )
        
        logger.info("AI Editorial Engine started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down AI Editorial Engine...")
        
        if knowledge_graph:
            knowledge_graph.close()
        if cache_manager:
            cache_manager.close()
        
        logger.info("AI Editorial Engine shut down")


# Create FastAPI app
app = FastAPI(
    title="AI Editorial Engine",
    description="Automated creation and curation system for UK Connections puzzles",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency injection
def get_knowledge_graph() -> KnowledgeGraphDB:
    """Get knowledge graph dependency."""
    if knowledge_graph is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")
    return knowledge_graph


def get_vector_store() -> VectorStore:
    """Get vector store dependency."""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not available")
    return vector_store


def get_cache_manager() -> CacheManager:
    """Get cache manager dependency."""
    if cache_manager is None:
        raise HTTPException(status_code=503, detail="Cache manager not available")
    return cache_manager


def get_puzzle_pipeline() -> PuzzlePipeline:
    """Get puzzle pipeline dependency."""
    if puzzle_pipeline is None:
        raise HTTPException(status_code=503, detail="Puzzle pipeline not available")
    return puzzle_pipeline


# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "AI Editorial Engine"}


@app.get("/health/detailed")
async def detailed_health_check(
    kg: KnowledgeGraphDB = Depends(get_knowledge_graph),
    cache: CacheManager = Depends(get_cache_manager)
):
    """Detailed health check with component status."""
    health_status = {
        "service": "AI Editorial Engine",
        "status": "healthy",
        "components": {
            "knowledge_graph": kg.health_check() if kg else False,
            "cache": cache.health_check() if cache else False,
            "vector_store": True  # VectorStore doesn't have external dependencies
        }
    }
    
    # Overall status
    all_healthy = all(health_status["components"].values())
    health_status["status"] = "healthy" if all_healthy else "degraded"
    
    status_code = 200 if all_healthy else 503
    return JSONResponse(content=health_status, status_code=status_code)


# Puzzle generation endpoints
@app.post("/api/v1/puzzles/generate", response_model=PuzzleGenerationResponse)
async def generate_puzzle(
    request: PuzzleGenerationRequest,
    pipeline: PuzzlePipeline = Depends(get_puzzle_pipeline)
):
    """Generate a new UK Connections puzzle."""
    try:
        logger.info("Generating new puzzle", constraints=request.constraints)
        
        # Run the full pipeline
        result = await pipeline.generate_puzzle(request.constraints)
        
        if result["success"]:
            return PuzzleGenerationResponse(
                puzzle=Puzzle(**result["puzzle"]),
                success=True,
                message="Puzzle generated successfully"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Puzzle generation failed: {result.get('error', 'Unknown error')}"
            )
    
    except Exception as e:
        logger.error(f"Error generating puzzle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/puzzles/{puzzle_id}")
async def get_puzzle(
    puzzle_id: str,
    cache: CacheManager = Depends(get_cache_manager)
):
    """Retrieve a specific puzzle by ID."""
    try:
        # Check cache first
        cached_puzzle = cache.get_cached_puzzle(puzzle_id)
        
        if cached_puzzle:
            return {"puzzle": cached_puzzle, "source": "cache"}
        else:
            raise HTTPException(status_code=404, detail="Puzzle not found")
    
    except Exception as e:
        logger.error(f"Error retrieving puzzle {puzzle_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/puzzles/{puzzle_id}/validate")
async def validate_puzzle(
    puzzle_id: str,
    cache: CacheManager = Depends(get_cache_manager)
):
    """Validate a puzzle's correctness and difficulty balance."""
    try:
        # Get puzzle from cache
        puzzle_data = cache.get_cached_puzzle(puzzle_id)
        
        if not puzzle_data:
            raise HTTPException(status_code=404, detail="Puzzle not found")
        
        puzzle = Puzzle(**puzzle_data)
        
        # Validate puzzle
        validation_results = {
            "puzzle_id": puzzle_id,
            "is_valid": True,
            "issues": []
        }
        
        # Check grid validity
        if not puzzle.validate_grid():
            validation_results["is_valid"] = False
            validation_results["issues"].append("Invalid grid: words don't match solution")
        
        # Check difficulty distribution
        difficulty_distribution = {}
        for category in puzzle.solution:
            if category.difficulty:
                difficulty_distribution[category.difficulty.value] = difficulty_distribution.get(category.difficulty.value, 0) + 1
        
        expected_distribution = {"YELLOW": 1, "GREEN": 1, "BLUE": 1, "PURPLE": 1}
        if difficulty_distribution != expected_distribution:
            validation_results["is_valid"] = False
            validation_results["issues"].append(f"Invalid difficulty distribution: {difficulty_distribution}")
        
        validation_results["difficulty_distribution"] = difficulty_distribution
        
        return validation_results
    
    except Exception as e:
        logger.error(f"Error validating puzzle {puzzle_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Knowledge graph endpoints
@app.get("/api/v1/knowledge-graph/entities")
async def query_entities(
    entity_types: Optional[str] = None,
    cultural_tiers: Optional[str] = None,
    search_text: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    kg: KnowledgeGraphDB = Depends(get_knowledge_graph)
):
    """Query entities from the knowledge graph."""
    try:
        # Parse query parameters
        from .models.entities import EntityType, CulturalSpecificityTier
        
        parsed_entity_types = None
        if entity_types:
            try:
                parsed_entity_types = [EntityType(t.strip()) for t in entity_types.split(",")]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid entity type: {e}")
        
        parsed_cultural_tiers = None
        if cultural_tiers:
            try:
                parsed_cultural_tiers = [CulturalSpecificityTier(int(t.strip())) for t in cultural_tiers.split(",")]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid cultural tier: {e}")
        
        parsed_tags = None
        if tags:
            parsed_tags = [tag.strip() for tag in tags.split(",")]
        
        # Create query object
        query = EntityQuery(
            entity_types=parsed_entity_types,
            cultural_specificity_tiers=parsed_cultural_tiers,
            search_text=search_text,
            tags=parsed_tags,
            limit=limit,
            offset=offset
        )
        
        # Execute query
        entities = kg.query_entities(query)
        
        return {
            "entities": [entity.dict() for entity in entities],
            "count": len(entities),
            "limit": limit,
            "offset": offset
        }
    
    except Exception as e:
        logger.error(f"Error querying entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/knowledge-graph/entities")
async def create_entity(
    entity: KnowledgeGraphEntity,
    kg: KnowledgeGraphDB = Depends(get_knowledge_graph),
    vs: VectorStore = Depends(get_vector_store)
):
    """Create a new entity in the knowledge graph."""
    try:
        # Create entity in knowledge graph
        success = kg.create_entity(entity)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create entity")
        
        # Add to vector store
        vs.add_entity(entity)
        
        return {"message": "Entity created successfully", "entity_id": entity.entity_id}
    
    except Exception as e:
        logger.error(f"Error creating entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/knowledge-graph/entities/{entity_id}")
async def get_entity(
    entity_id: str,
    kg: KnowledgeGraphDB = Depends(get_knowledge_graph)
):
    """Get a specific entity by ID."""
    try:
        entity = kg.get_entity(entity_id)
        
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        return {"entity": entity.dict()}
    
    except Exception as e:
        logger.error(f"Error retrieving entity {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/knowledge-graph/statistics")
async def get_knowledge_graph_stats(
    kg: KnowledgeGraphDB = Depends(get_knowledge_graph)
):
    """Get statistics about the knowledge graph."""
    try:
        stats = kg.get_statistics()
        return {"statistics": stats}
    
    except Exception as e:
        logger.error(f"Error getting knowledge graph statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Vector search endpoints
@app.get("/api/v1/search/semantic")
async def semantic_search(
    query: str,
    top_k: int = 10,
    threshold: float = 0.0,
    vs: VectorStore = Depends(get_vector_store)
):
    """Perform semantic search using vector similarity."""
    try:
        results = vs.search_similar(query, top_k=top_k, threshold=threshold)
        
        formatted_results = []
        for entity, similarity in results:
            formatted_results.append({
                "entity": entity.dict(),
                "similarity_score": similarity
            })
        
        return {
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results)
        }
    
    except Exception as e:
        logger.error(f"Error performing semantic search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Pipeline management endpoints
@app.get("/api/v1/pipeline/status")
async def get_pipeline_status(
    pipeline: PuzzlePipeline = Depends(get_puzzle_pipeline)
):
    """Get current pipeline status and metrics."""
    try:
        status = pipeline.get_status()
        return {"pipeline_status": status}
    
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cache management endpoints
@app.get("/api/v1/cache/stats")
async def get_cache_stats(
    cache: CacheManager = Depends(get_cache_manager)
):
    """Get cache statistics."""
    try:
        stats = cache.get_cache_stats()
        return {"cache_stats": stats}
    
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/cache/clear")
async def clear_cache(
    cache: CacheManager = Depends(get_cache_manager)
):
    """Clear all cached data."""
    try:
        success = cache.clear_all()
        
        if success:
            return {"message": "Cache cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear cache")
    
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development"
    )