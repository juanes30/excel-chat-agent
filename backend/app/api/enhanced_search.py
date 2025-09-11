"""Enhanced Search API Endpoints for Advanced Vector Store Features.

This module provides REST API endpoints that leverage the enhanced vector store
capabilities including hybrid search, analytics, and intelligent querying.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field, validator

from app.services.enhanced_vector_store import (
    AdvancedVectorStoreService,
    SearchQuery,
    SearchStrategy,
    RelevanceScoring,
    SearchResult
)
from app.services.vector_store_integration import VectorStoreIntegrator
from app.services.excel_processor import OptimizedExcelProcessor

logger = logging.getLogger(__name__)

# Router for enhanced search endpoints
router = APIRouter(prefix="/api/v2/search", tags=["Enhanced Search"])


# Request/Response Models
class EnhancedSearchRequest(BaseModel):
    """Enhanced search request with advanced options."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    strategy: Optional[SearchStrategy] = Field(SearchStrategy.ADAPTIVE, description="Search strategy")
    scoring: Optional[RelevanceScoring] = Field(RelevanceScoring.CONTEXT_AWARE, description="Relevance scoring method")
    max_results: Optional[int] = Field(10, ge=1, le=50, description="Maximum number of results")
    min_relevance: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Minimum relevance score")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Search filters")
    include_explanation: Optional[bool] = Field(False, description="Include result explanations")
    context_filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Context-aware filters")
    search_preferences: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User search preferences")


class FacetedSearchRequest(BaseModel):
    """Faceted search request."""
    query: str = Field(..., min_length=1, description="Search query")
    facets: List[str] = Field(..., description="Facets to aggregate")
    max_results: Optional[int] = Field(20, ge=1, le=100, description="Maximum results per facet")


class SimilarContentRequest(BaseModel):
    """Similar content search request."""
    reference_content: str = Field(..., min_length=10, description="Reference content for similarity")
    max_results: Optional[int] = Field(10, ge=1, le=20, description="Maximum similar results")
    exclude_self: Optional[bool] = Field(True, description="Exclude reference content from results")


class BatchProcessRequest(BaseModel):
    """Batch processing request."""
    directory_path: Optional[str] = Field(None, description="Directory path to process")
    analysis_mode: Optional[str] = Field("auto", description="Excel analysis mode")
    force_reprocess: Optional[bool] = Field(False, description="Force reprocessing of existing files")


class SearchResultResponse(BaseModel):
    """Enhanced search result response."""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    semantic_score: float
    keyword_score: float = 0.0
    quality_score: float = 0.0
    freshness_score: float = 0.0
    file_name: str = ""
    sheet_name: str = ""
    chunk_index: int = 0
    explanation: Optional[Dict[str, Any]] = None
    intelligent_context: Optional[Dict[str, Any]] = None


class EnhancedSearchResponse(BaseModel):
    """Enhanced search response."""
    results: List[SearchResultResponse]
    total_results: int
    search_strategy: str
    query_analysis: Dict[str, Any]
    faceted_insights: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None


class AnalyticsResponse(BaseModel):
    """Search analytics response."""
    performance_metrics: Dict[str, Any]
    recent_queries: List[Dict[str, Any]]
    cache_stats: Dict[str, Any]
    popular_queries: Dict[str, int]


class IntegrationStatsResponse(BaseModel):
    """Integration statistics response."""
    vector_store_stats: Dict[str, Any]
    excel_processor_stats: Dict[str, Any]
    search_analytics: Dict[str, Any]
    integration_health: Dict[str, Any]


# Dependency injection
async def get_enhanced_vector_store() -> AdvancedVectorStoreService:
    """Get enhanced vector store service instance."""
    # This would be injected from the main application
    # For now, we'll assume it's available globally
    from app.main import enhanced_vector_store
    if enhanced_vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Enhanced vector store service not available"
        )
    return enhanced_vector_store


async def get_vector_store_integrator() -> VectorStoreIntegrator:
    """Get vector store integrator instance."""
    from app.main import vector_store_integrator
    if vector_store_integrator is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store integrator not available"
        )
    return vector_store_integrator


# API Endpoints
@router.post("/enhanced", response_model=EnhancedSearchResponse)
async def enhanced_search(
    request: EnhancedSearchRequest,
    integrator: VectorStoreIntegrator = Depends(get_vector_store_integrator)
):
    """Perform enhanced semantic search with advanced features."""
    try:
        start_time = datetime.now()
        
        # Perform intelligent search
        result = await integrator.intelligent_search(
            query=request.query,
            context_filters=request.context_filters,
            search_preferences=request.search_preferences
        )
        
        # Convert results to response format
        search_results = []
        for r in result["primary_results"]:
            search_result = SearchResultResponse(
                content=r["content"],
                metadata=r["metadata"],
                relevance_score=r["relevance_score"],
                semantic_score=r.get("semantic_score", r["relevance_score"]),
                keyword_score=r.get("keyword_score", 0.0),
                quality_score=r.get("quality_score", 0.0),
                freshness_score=r.get("freshness_score", 0.0),
                file_name=r["file_name"],
                sheet_name=r["sheet_name"],
                chunk_index=r["chunk_index"],
                explanation=r.get("explanation"),
                intelligent_context=r.get("intelligent_context")
            )
            search_results.append(search_result)
        
        # Calculate performance metrics
        end_time = datetime.now()
        performance_metrics = {
            "query_time_ms": (end_time - start_time).total_seconds() * 1000,
            "timestamp": end_time.isoformat()
        }
        
        return EnhancedSearchResponse(
            results=search_results,
            total_results=result["total_results"],
            search_strategy=result["search_strategy"],
            query_analysis=result["query_analysis"],
            faceted_insights=result["faceted_insights"],
            performance_metrics=performance_metrics
        )
        
    except Exception as e:
        logger.error(f"Enhanced search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Enhanced search failed: {str(e)}"
        )


@router.post("/faceted", response_model=Dict[str, Any])
async def faceted_search(
    request: FacetedSearchRequest,
    vector_store: AdvancedVectorStoreService = Depends(get_enhanced_vector_store)
):
    """Perform faceted search with aggregations."""
    try:
        result = await vector_store.faceted_search(
            query=request.query,
            facets=request.facets,
            n_results=request.max_results
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Faceted search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Faceted search failed: {str(e)}"
        )


@router.post("/similar", response_model=List[SearchResultResponse])
async def similar_content_search(
    request: SimilarContentRequest,
    vector_store: AdvancedVectorStoreService = Depends(get_enhanced_vector_store)
):
    """Find content similar to reference content."""
    try:
        results = await vector_store.similar_content_search(
            reference_content=request.reference_content,
            n_results=request.max_results,
            exclude_self=request.exclude_self
        )
        
        # Convert results to response format
        search_results = []
        for r in results:
            search_result = SearchResultResponse(
                content=r.content,
                metadata=r.metadata,
                relevance_score=r.relevance_score,
                semantic_score=r.semantic_score,
                keyword_score=r.keyword_score,
                quality_score=r.quality_score,
                freshness_score=r.freshness_score,
                file_name=r.file_name,
                sheet_name=r.sheet_name,
                chunk_index=r.chunk_index,
                explanation=r.explanation
            )
            search_results.append(search_result)
        
        return search_results
        
    except Exception as e:
        logger.error(f"Similar content search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Similar content search failed: {str(e)}"
        )


@router.get("/analytics", response_model=AnalyticsResponse)
async def get_search_analytics(
    limit: int = Query(100, ge=1, le=1000, description="Limit for recent queries"),
    vector_store: AdvancedVectorStoreService = Depends(get_enhanced_vector_store)
):
    """Get search analytics and performance metrics."""
    try:
        analytics = vector_store.get_search_analytics(limit=limit)
        
        return AnalyticsResponse(
            performance_metrics=analytics["performance_metrics"],
            recent_queries=analytics["recent_queries"],
            cache_stats=analytics["cache_stats"],
            popular_queries=analytics["popular_queries"]
        )
        
    except Exception as e:
        logger.error(f"Get analytics error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get analytics: {str(e)}"
        )


@router.delete("/analytics")
async def clear_search_analytics(
    vector_store: AdvancedVectorStoreService = Depends(get_enhanced_vector_store)
):
    """Clear search analytics data."""
    try:
        vector_store.clear_analytics()
        return {"message": "Search analytics cleared successfully"}
        
    except Exception as e:
        logger.error(f"Clear analytics error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear analytics: {str(e)}"
        )


@router.delete("/cache")
async def clear_search_cache(
    vector_store: AdvancedVectorStoreService = Depends(get_enhanced_vector_store)
):
    """Clear search result cache."""
    try:
        vector_store.clear_cache()
        return {"message": "Search cache cleared successfully"}
        
    except Exception as e:
        logger.error(f"Clear cache error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.post("/process-and-index")
async def process_and_index_file(
    file_path: str,
    analysis_mode: str = "comprehensive",
    background_tasks: BackgroundTasks = None,
    integrator: VectorStoreIntegrator = Depends(get_vector_store_integrator)
):
    """Process and index a specific Excel file."""
    try:
        # Validate file path
        from pathlib import Path
        if not Path(file_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {file_path}"
            )
        
        # Process and index file
        result = await integrator.process_and_index_file(
            file_path=file_path,
            analysis_mode=analysis_mode
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Process and index error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process and index file: {str(e)}"
        )


@router.post("/batch-process")
async def batch_process_directory(
    request: BatchProcessRequest,
    background_tasks: BackgroundTasks,
    integrator: VectorStoreIntegrator = Depends(get_vector_store_integrator)
):
    """Batch process all Excel files in a directory."""
    try:
        # Validate directory
        if request.directory_path:
            from pathlib import Path
            if not Path(request.directory_path).exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Directory not found: {request.directory_path}"
                )
        
        # Start batch processing in background
        background_tasks.add_task(
            _batch_process_background,
            integrator,
            request.directory_path or "data/excel_files",
            request.analysis_mode,
            request.force_reprocess
        )
        
        return {
            "message": "Batch processing started",
            "directory": request.directory_path or "data/excel_files",
            "analysis_mode": request.analysis_mode
        }
        
    except Exception as e:
        logger.error(f"Batch process error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start batch processing: {str(e)}"
        )


@router.get("/integration-stats", response_model=IntegrationStatsResponse)
async def get_integration_stats(
    integrator: VectorStoreIntegrator = Depends(get_vector_store_integrator)
):
    """Get integration statistics and health metrics."""
    try:
        stats = integrator.get_integration_stats()
        
        return IntegrationStatsResponse(
            vector_store_stats=stats["vector_store_stats"],
            excel_processor_stats=stats["excel_processor_stats"],
            search_analytics=stats["search_analytics"],
            integration_health=stats["integration_health"]
        )
        
    except Exception as e:
        logger.error(f"Get integration stats error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get integration stats: {str(e)}"
        )


@router.get("/health")
async def health_check(
    vector_store: AdvancedVectorStoreService = Depends(get_enhanced_vector_store)
):
    """Health check for enhanced search services."""
    try:
        # Check vector store health
        health = vector_store.health_check()
        
        # Add enhanced service status
        enhanced_status = {
            "enhanced_search": "healthy",
            "analytics_enabled": vector_store.enable_analytics,
            "cache_size": len(vector_store.query_cache),
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "status": "healthy" if health["status"] == "healthy" else "degraded",
            "vector_store_health": health,
            "enhanced_features": enhanced_status
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Background task functions
async def _batch_process_background(
    integrator: VectorStoreIntegrator,
    directory_path: str,
    analysis_mode: str,
    force_reprocess: bool
):
    """Background task for batch processing."""
    try:
        logger.info(f"Starting batch processing: {directory_path}")
        
        result = await integrator.batch_process_directory(
            directory_path=directory_path,
            analysis_mode=analysis_mode
        )
        
        logger.info(f"Batch processing completed: {result['success_rate']:.1%} success rate")
        
    except Exception as e:
        logger.error(f"Batch processing background task error: {e}")


# Query validation utilities
def validate_search_query(query: str) -> str:
    """Validate and sanitize search query."""
    if not query or not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty"
        )
    
    # Basic sanitization
    query = query.strip()
    
    # Length validation
    if len(query) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Search query too long (max 1000 characters)"
        )
    
    return query


def validate_facets(facets: List[str]) -> List[str]:
    """Validate facet list."""
    if not facets:
        raise HTTPException(
            status_code=400,
            detail="At least one facet must be specified"
        )
    
    # Validate facet names
    valid_facets = {
        "file_name", "sheet_name", "chunk_type", "overall_quality",
        "has_patterns", "has_relationships", "completeness_score"
    }
    
    invalid_facets = [f for f in facets if f not in valid_facets]
    if invalid_facets:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid facets: {invalid_facets}. Valid facets: {list(valid_facets)}"
        )
    
    return facets