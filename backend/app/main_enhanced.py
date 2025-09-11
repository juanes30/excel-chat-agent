"""Enhanced FastAPI main application with Advanced Vector Store Integration.

This module extends the main application with enhanced vector store capabilities,
advanced search features, and intelligent integration services.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import enhanced services
from app.services.enhanced_vector_store import AdvancedVectorStoreService
from app.services.vector_store_integration import VectorStoreIntegrator
from app.services.excel_processor import OptimizedExcelProcessor
from app.services.llm_service import LangChainLLMService

# Import API routers
from app.api.enhanced_search import router as enhanced_search_router

# Import existing main components
from app.main import (
    ConnectionManager,
    app_start_time,
    logger
)

# Global enhanced services
enhanced_vector_store: Optional[AdvancedVectorStoreService] = None
vector_store_integrator: Optional[VectorStoreIntegrator] = None
excel_processor: Optional[OptimizedExcelProcessor] = None
llm_service: Optional[LangChainLLMService] = None


@asynccontextmanager
async def enhanced_lifespan(app: FastAPI):
    """Enhanced application lifespan with advanced services."""
    global enhanced_vector_store, vector_store_integrator, excel_processor, llm_service
    
    try:
        logger.info("🚀 Starting Enhanced Excel Chat Agent...")
        
        # Configuration
        data_directory = os.getenv("DATA_DIRECTORY", "data/excel_files")
        chroma_directory = os.getenv("CHROMA_DIRECTORY", "chroma_db")
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        # Initialize enhanced Excel processor
        logger.info("📊 Initializing Enhanced Excel Processor...")
        excel_processor = OptimizedExcelProcessor(
            data_directory=data_directory,
            max_file_size_mb=200,  # Increased for enhanced processor
            enable_parallel_processing=True,
            memory_limit_mb=1024
        )
        
        # Initialize enhanced vector store
        logger.info("🔍 Initializing Enhanced Vector Store...")
        enhanced_vector_store = AdvancedVectorStoreService(
            persist_directory=chroma_directory,
            collection_name="enhanced_excel_documents",
            embedding_model=embedding_model,
            enable_analytics=True,
            cache_ttl_minutes=60
        )
        
        # Initialize vector store integrator
        logger.info("🔗 Initializing Vector Store Integrator...")
        vector_store_integrator = VectorStoreIntegrator(
            vector_store=enhanced_vector_store,
            excel_processor=excel_processor
        )
        
        # Initialize LLM service
        logger.info("🤖 Initializing LLM Service...")
        llm_service = LangChainLLMService(
            model_name=os.getenv("OLLAMA_MODEL", "llama3"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            enable_streaming=True,
            context_window=4096
        )
        
        # Health checks
        logger.info("🏥 Performing health checks...")
        
        # Check enhanced vector store
        vector_health = enhanced_vector_store.health_check()
        if vector_health["status"] != "healthy":
            logger.warning(f"Vector store health check: {vector_health}")
        
        # Check Excel processor
        excel_stats = excel_processor.get_file_statistics()
        logger.info(f"Excel processor ready: {excel_stats['total_files']} files available")
        
        # Auto-index existing files if needed
        if excel_stats["total_files"] > 0:
            vector_stats = enhanced_vector_store.get_statistics()
            if vector_stats["total_documents"] == 0:
                logger.info("🔄 Auto-indexing existing Excel files...")
                try:
                    await _auto_index_files()
                except Exception as e:
                    logger.error(f"Auto-indexing failed: {e}")
        
        # Make services available globally for dependency injection
        app.state.enhanced_vector_store = enhanced_vector_store
        app.state.vector_store_integrator = vector_store_integrator
        app.state.excel_processor = excel_processor
        app.state.llm_service = llm_service
        
        logger.info("✅ Enhanced Excel Chat Agent started successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start enhanced services: {e}")
        raise
    finally:
        # Cleanup
        logger.info("🧹 Shutting down enhanced services...")
        
        if enhanced_vector_store:
            enhanced_vector_store.clear_cache()
        
        logger.info("👋 Enhanced Excel Chat Agent shutdown complete")


async def _auto_index_files():
    """Auto-index existing Excel files."""
    try:
        result = await vector_store_integrator.batch_process_directory(
            directory_path=excel_processor.data_directory,
            analysis_mode="auto"
        )
        
        logger.info(
            f"Auto-indexing completed: {result['successful_files']}/{result['total_files']} files indexed"
        )
        
    except Exception as e:
        logger.error(f"Auto-indexing error: {e}")
        raise


def create_enhanced_app() -> FastAPI:
    """Create enhanced FastAPI application."""
    
    app = FastAPI(
        title="Enhanced Excel Chat Agent API",
        description="Advanced Excel analysis with semantic search and AI chat capabilities",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=enhanced_lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include enhanced API routers
    app.include_router(enhanced_search_router)
    
    # Enhanced health check endpoint
    @app.get("/health/enhanced")
    async def enhanced_health_check():
        """Enhanced health check with detailed service status."""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - app_start_time).total_seconds(),
                "services": {}
            }
            
            # Check enhanced vector store
            if enhanced_vector_store:
                vector_health = enhanced_vector_store.health_check()
                vector_stats = enhanced_vector_store.get_statistics()
                health_status["services"]["enhanced_vector_store"] = {
                    "status": vector_health["status"],
                    "documents": vector_stats["total_documents"],
                    "files": vector_stats["unique_files"],
                    "cache_size": vector_stats["cache_size"]
                }
            
            # Check Excel processor
            if excel_processor:
                excel_stats = excel_processor.get_file_statistics()
                health_status["services"]["excel_processor"] = {
                    "status": "healthy",
                    "total_files": excel_stats["total_files"],
                    "total_sheets": excel_stats["total_sheets"],
                    "total_rows": excel_stats["total_rows"]
                }
            
            # Check vector store integrator
            if vector_store_integrator:
                integration_stats = vector_store_integrator.get_integration_stats()
                health_status["services"]["integration"] = integration_stats["integration_health"]
            
            # Check LLM service
            if llm_service:
                # Add LLM health check if available
                health_status["services"]["llm"] = {"status": "healthy"}
            
            # Overall status
            service_statuses = [
                service.get("status", "unknown") 
                for service in health_status["services"].values()
            ]
            
            if all(status == "healthy" for status in service_statuses):
                health_status["status"] = "healthy"
            elif any(status == "healthy" for status in service_statuses):
                health_status["status"] = "degraded"
            else:
                health_status["status"] = "unhealthy"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Enhanced health check error: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # Enhanced system stats endpoint
    @app.get("/stats/enhanced")
    async def enhanced_system_stats():
        """Get enhanced system statistics."""
        try:
            stats = {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - app_start_time).total_seconds(),
            }
            
            # Enhanced vector store stats
            if enhanced_vector_store:
                stats["vector_store"] = enhanced_vector_store.get_statistics()
                stats["search_analytics"] = enhanced_vector_store.get_search_analytics(limit=10)
            
            # Excel processor stats
            if excel_processor:
                stats["excel_processor"] = excel_processor.get_file_statistics()
            
            # Integration stats
            if vector_store_integrator:
                integration_stats = vector_store_integrator.get_integration_stats()
                stats["integration"] = integration_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Enhanced stats error: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # Benchmarking endpoint
    @app.post("/benchmark/search")
    async def benchmark_search_performance():
        """Benchmark search performance across different strategies."""
        try:
            if not enhanced_vector_store:
                raise HTTPException(status_code=503, detail="Enhanced vector store not available")
            
            test_queries = [
                "sales data analysis",
                "customer information patterns",
                "financial reporting trends",
                "product performance metrics",
                "operational efficiency data"
            ]
            
            benchmark_results = []
            
            for query_text in test_queries:
                strategy_results = {}
                
                # Test different strategies
                for strategy in [SearchStrategy.SEMANTIC_ONLY, SearchStrategy.HYBRID, SearchStrategy.ADAPTIVE]:
                    start_time = datetime.now()
                    
                    search_query = SearchQuery(
                        text=query_text,
                        strategy=strategy,
                        n_results=5
                    )
                    
                    results = await enhanced_vector_store.enhanced_search(search_query)
                    
                    end_time = datetime.now()
                    duration_ms = (end_time - start_time).total_seconds() * 1000
                    
                    strategy_results[strategy.value] = {
                        "duration_ms": duration_ms,
                        "results_count": len(results),
                        "avg_relevance": sum(r.relevance_score for r in results) / max(1, len(results))
                    }
                
                benchmark_results.append({
                    "query": query_text,
                    "strategies": strategy_results
                })
            
            return {
                "benchmark_results": benchmark_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Benchmark error: {e}")
            raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")
    
    return app


# Create the enhanced application instance
enhanced_app = create_enhanced_app()

# Export for use
__all__ = ["enhanced_app", "enhanced_vector_store", "vector_store_integrator"]


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main_enhanced:enhanced_app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )