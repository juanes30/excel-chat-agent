"""FastAPI main application for Excel Chat Agent with Enhanced Services."""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import (
    FastAPI, 
    WebSocket, 
    WebSocketDisconnect, 
    HTTPException, 
    UploadFile, 
    File, 
    Depends,
    BackgroundTasks,
    Request,
    status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.models.schemas import (
    QueryRequest, 
    QueryResponse, 
    FileInfo, 
    SystemStats, 
    UploadResponse,
    ErrorResponse,
    HealthCheck,
    ProcessingStatus,
    WebSocketMessage
)

# Enhanced services imports (fallback to existing services for now)
try:
    from app.services.enhanced_excel_processor import EnhancedExcelProcessor as ExcelProcessor
except ImportError:
    from app.services.excel_processor import ExcelProcessor

try:
    from app.services.enhanced_vector_store_v2 import EnhancedVectorStoreV2 as VectorStoreService
except ImportError:
    from app.services.vector_store import VectorStoreService

try:
    from app.services.enhanced_llm_service import EnhancedLLMService as LLMService
except ImportError:
    from app.services.llm_service import LangChainLLMService as LLMService

try:
    from app.services.rag_integration_service import RAGIntegrationService
    HAS_RAG_SERVICE = True
except ImportError:
    RAGIntegrationService = None
    HAS_RAG_SERVICE = False

try:
    from app.services.enhanced_embedding_strategy import EnhancedEmbeddingStrategy
    HAS_EMBEDDING_STRATEGY = True
except ImportError:
    EnhancedEmbeddingStrategy = None  
    HAS_EMBEDDING_STRATEGY = False

# WebSocket integration (optional)
try:
    from app.api.websocket_routes import initialize_websocket_handler, router as websocket_router
    HAS_ENHANCED_WEBSOCKET = True
except ImportError:
    websocket_router = None
    HAS_ENHANCED_WEBSOCKET = False

# Error handling (optional)
try:
    from app.utils.error_handling import (
        global_error_handler, 
        LLMServiceError, 
        ErrorContext,
        with_error_handling
    )
    HAS_ERROR_HANDLING = True
except ImportError:
    def with_error_handling(operation: str = None, **kwargs):
        def decorator(func):
            return func
        return decorator
    LLMServiceError = Exception
    HAS_ERROR_HANDLING = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for services (with fallback support)
excel_processor: Optional[ExcelProcessor] = None
vector_store: Optional[VectorStoreService] = None
llm_service: Optional[LLMService] = None
rag_service: Optional[RAGIntegrationService] = None
embedding_strategy: Optional[EnhancedEmbeddingStrategy] = None
app_start_time: datetime = datetime.now()


class OptimizedConnectionManager:
    """Performance-optimized WebSocket connection manager with adaptive batching and caching."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_data: Dict[str, Dict[str, Any]] = {}
        self._connection_cache: Dict[str, WebSocket] = {}
        self._timestamp_cache: Optional[str] = None
        self._last_timestamp_update: float = 0
        self._token_buffers: Dict[str, List[str]] = {}
        self.batch_size: int = 3  # Adaptive batch size for tokens
        self.timestamp_cache_duration: float = 0.01  # 10ms cache duration
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept a WebSocket connection with caching optimization."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self._connection_cache[session_id] = websocket  # Cache connection
        self._token_buffers[session_id] = []  # Initialize token buffer
        
        cached_timestamp = self._get_cached_timestamp()
        self.session_data[session_id] = {
            "connected_at": cached_timestamp,
            "last_activity": cached_timestamp,
            "message_count": 0,
            "token_count": 0
        }
        logger.info(f"WebSocket connection established: {session_id}")
    
    def disconnect(self, session_id: str):
        """Remove a WebSocket connection and cleanup caches."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self._connection_cache:
            del self._connection_cache[session_id]
        if session_id in self.session_data:
            del self.session_data[session_id]
        if session_id in self._token_buffers:
            del self._token_buffers[session_id]
        logger.info(f"WebSocket connection closed: {session_id}")
    
    def _get_cached_timestamp(self) -> str:
        """Get cached timestamp to reduce datetime.now() calls."""
        import time
        now = time.time()
        if now - self._last_timestamp_update > self.timestamp_cache_duration:
            self._timestamp_cache = datetime.now().isoformat()
            self._last_timestamp_update = now
        return self._timestamp_cache
    
    async def send_personal_message(self, message: str, session_id: str):
        """Send a message using cached connection lookup."""
        websocket = self._connection_cache.get(session_id)
        if not websocket:
            websocket = self.active_connections.get(session_id)
            if websocket:
                self._connection_cache[session_id] = websocket
        
        if websocket:
            try:
                await websocket.send_text(message)
                self._update_session_activity(session_id)
            except Exception as e:
                logger.error(f"Error sending message to {session_id}: {e}")
                self.disconnect(session_id)
    
    async def send_json_message(self, data: Dict[str, Any], session_id: str):
        """Send JSON message using cached connection lookup."""
        websocket = self._connection_cache.get(session_id)
        if not websocket:
            websocket = self.active_connections.get(session_id)
            if websocket:
                self._connection_cache[session_id] = websocket
        
        if websocket:
            try:
                await websocket.send_json(data)
                self._update_session_activity(session_id)
            except Exception as e:
                logger.error(f"Error sending JSON message to {session_id}: {e}")
                self.disconnect(session_id)
    
    async def send_token_batch(self, session_id: str, tokens: List[str]):
        """Send tokens in adaptive batches for optimal performance."""
        if not tokens:
            return
        
        batch_content = ' '.join(tokens)
        cached_timestamp = self._get_cached_timestamp()
        
        message = {
            "type": "token_batch",
            "content": batch_content,
            "token_count": len(tokens),
            "timestamp": cached_timestamp
        }
        
        await self.send_json_message(message, session_id)
        
        # Update session statistics
        if session_id in self.session_data:
            self.session_data[session_id]["token_count"] += len(tokens)
    
    async def add_token_to_buffer(self, session_id: str, token: str):
        """Add token to buffer and send batch when threshold reached."""
        if session_id not in self._token_buffers:
            self._token_buffers[session_id] = []
        
        self._token_buffers[session_id].append(token)
        
        # Send batch if buffer is full
        if len(self._token_buffers[session_id]) >= self.batch_size:
            await self.flush_token_buffer(session_id)
    
    async def flush_token_buffer(self, session_id: str):
        """Flush remaining tokens in buffer."""
        if session_id in self._token_buffers and self._token_buffers[session_id]:
            tokens = self._token_buffers[session_id]
            self._token_buffers[session_id] = []
            await self.send_token_batch(session_id, tokens)
    
    def _update_session_activity(self, session_id: str):
        """Update session activity with cached timestamp."""
        if session_id in self.session_data:
            cached_timestamp = self._get_cached_timestamp()
            self.session_data[session_id]["last_activity"] = cached_timestamp
            self.session_data[session_id]["message_count"] += 1
    
    async def send_streaming_tokens(self, session_id: str, token_generator):
        """Optimized streaming with adaptive batching and caching."""
        async for token in token_generator:
            await self.add_token_to_buffer(session_id, token)
        
        # Flush any remaining tokens
        await self.flush_token_buffer(session_id)
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        total_messages = sum(
            session.get("message_count", 0) 
            for session in self.session_data.values()
        )
        total_tokens = sum(
            session.get("token_count", 0) 
            for session in self.session_data.values()
        )
        
        return {
            "active_connections": len(self.active_connections),
            "cached_connections": len(self._connection_cache),
            "total_messages_sent": total_messages,
            "total_tokens_sent": total_tokens,
            "average_tokens_per_connection": total_tokens / max(1, len(self.active_connections)),
            "batch_size": self.batch_size,
            "timestamp_cache_duration_ms": self.timestamp_cache_duration * 1000
        }


# Global optimized connection manager
connection_manager = OptimizedConnectionManager()


@with_error_handling(operation="initialize_services")
async def initialize_services():
    """Initialize all enhanced services during startup."""
    global excel_processor, vector_store, llm_service, rag_service, embedding_strategy
    
    logger.info("Initializing enhanced services...")
    
    try:
        # Initialize Enhanced Embedding Strategy
        logger.info("Initializing enhanced embedding strategy...")
        embedding_strategy = EnhancedEmbeddingStrategy()
        
        # Initialize Enhanced Excel processor
        data_directory = os.getenv("DATA_DIRECTORY", "data/excel_files")
        excel_processor = ExcelProcessor(data_directory)
        logger.info(f"Enhanced Excel processor initialized with directory: {data_directory}")
        
        # Initialize Enhanced Vector Store V2
        chroma_directory = os.getenv("CHROMA_DIRECTORY", "chroma_db")
        vector_store = VectorStoreService(
            persist_directory=chroma_directory,
            collection_name="excel_data_v2",
            enable_multi_modal=True,
            enable_analytics=True
        )
        logger.info(f"Enhanced vector store initialized with directory: {chroma_directory}")
        
        # Initialize Enhanced LLM service
        model_name = os.getenv("OLLAMA_MODEL", "llama3")
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        llm_service = LLMService(
            model_name=model_name,
            ollama_url=ollama_url,
            vector_store=vector_store,
            enable_streaming=True
        )
        logger.info(f"Enhanced LLM service initialized with model: {model_name}")
        
        # Initialize RAG Integration Service
        if HAS_RAG_SERVICE:
            rag_service = RAGIntegrationService(
                llm_service=llm_service,
                vector_store=vector_store
            )
            logger.info("RAG integration service initialized")
        else:
            rag_service = None
            logger.info("RAG integration service not available (fallback mode)")
        
        # Initialize Enhanced WebSocket handlers if available
        if HAS_ENHANCED_WEBSOCKET:
            try:
                initialize_websocket_handler(llm_service)
                logger.info("Enhanced WebSocket handlers initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize enhanced WebSocket handlers: {e}")
                logger.info("WebSocket handlers not available (fallback mode)")
        else:
            logger.info("Enhanced WebSocket handlers not available (fallback mode)")
        
        # Process existing Excel files and index them with enhanced processing
        try:
            logger.info("Processing existing Excel files with enhanced processor...")
            all_files = excel_processor.process_all_files()
            
            processed_count = 0
            for file_data in all_files:
                try:
                    success = await vector_store.add_excel_data_enhanced(
                        file_name=file_data['file_name'],
                        file_hash=file_data['file_hash'],
                        sheets_data=file_data['sheets'],
                        enable_multi_modal=True,
                        enable_content_analysis=True
                    )
                    if success.get('success', False):
                        processed_count += 1
                except Exception as file_error:
                    logger.error(f"Error processing file {file_data['file_name']}: {file_error}")
                    continue
            
            logger.info(f"Successfully processed and indexed {processed_count} Excel files")
            
        except Exception as e:
            logger.error(f"Error processing existing files: {e}")
            # Continue startup even if file processing fails
        
        logger.info("Enhanced services initialized successfully")
        
    except Exception as e:
        logger.error(f"Critical error during service initialization: {e}")
        raise


async def cleanup_services():
    """Cleanup services during shutdown."""
    logger.info("Cleaning up services...")
    
    # Close any active connections
    for session_id in list(connection_manager.active_connections.keys()):
        try:
            await connection_manager.active_connections[session_id].close()
        except Exception as e:
            logger.error(f"Error closing connection {session_id}: {e}")
    
    logger.info("Services cleanup completed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    await initialize_services()
    yield
    # Shutdown
    await cleanup_services()


# Create FastAPI app
app = FastAPI(
    title="Excel Chat Agent Enhanced",
    description="AI-powered chat interface for Excel file analysis with enhanced RAG capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include WebSocket routes
if websocket_router is not None:
    app.include_router(websocket_router)


# Dependency to check if services are ready
def get_services():
    """Dependency to ensure enhanced services are initialized."""
    if not all([excel_processor, vector_store, llm_service, rag_service]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Enhanced services are still initializing"
        )
    return excel_processor, vector_store, llm_service, rag_service


def get_rag_service():
    """Dependency to get RAG service."""
    if not rag_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not available"
        )
    return rag_service


@app.get("/", response_model=HealthCheck)
async def health_check():
    """Enhanced health check endpoint."""
    uptime_seconds = int((datetime.now() - app_start_time).total_seconds())
    
    # Check enhanced service health
    components = {}
    overall_status = "healthy"
    
    try:
        # Check Enhanced Excel Processor
        if excel_processor:
            try:
                stats = excel_processor.get_statistics()
                components["enhanced_excel_processor"] = "healthy"
            except Exception as e:
                components["enhanced_excel_processor"] = "degraded"
                overall_status = "degraded"
        else:
            components["enhanced_excel_processor"] = "not_initialized"
            overall_status = "unhealthy"
        
        # Check Enhanced Vector Store
        if vector_store:
            try:
                vector_health = await vector_store.health_check() if hasattr(vector_store, 'health_check') else {"status": "healthy"}
                components["enhanced_vector_store"] = vector_health.get("status", "healthy")
                if vector_health.get("status") not in ["healthy", "empty"]:
                    overall_status = "degraded"
            except Exception as e:
                components["enhanced_vector_store"] = "unhealthy"
                overall_status = "degraded"
        else:
            components["enhanced_vector_store"] = "not_initialized"
            overall_status = "unhealthy"
        
        # Check Enhanced LLM Service
        if llm_service:
            try:
                llm_health = await llm_service.health_check() if hasattr(llm_service, 'health_check') else {"status": "healthy"}
                components["enhanced_llm_service"] = llm_health.get("status", "healthy")
                if llm_health.get("status") != "healthy":
                    overall_status = "degraded"
            except Exception as e:
                components["enhanced_llm_service"] = "unhealthy"
                overall_status = "degraded"
        else:
            components["enhanced_llm_service"] = "not_initialized"
            overall_status = "unhealthy"
        
        # Check RAG Integration Service
        if rag_service:
            try:
                rag_health = await rag_service.health_check()
                rag_status = rag_health.get("rag_service", "healthy")
                components["rag_integration_service"] = rag_status
                if rag_status not in ["healthy", "degraded"]:
                    overall_status = "degraded"
            except Exception as e:
                components["rag_integration_service"] = "unhealthy"
                overall_status = "degraded"
        else:
            components["rag_integration_service"] = "not_initialized"
        
        # Check Enhanced Embedding Strategy  
        if embedding_strategy:
            try:
                # Just check if it exists and has required attributes
                if hasattr(embedding_strategy, 'embedding_model'):
                    components["embedding_strategy"] = "healthy"
                else:
                    components["embedding_strategy"] = "degraded"
            except Exception as e:
                components["embedding_strategy"] = "unhealthy"
        else:
            components["embedding_strategy"] = "not_initialized"
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        overall_status = "unhealthy"
        components["error"] = str(e)
    
    return HealthCheck(
        status=overall_status,
        version="1.0.0",  # Updated version for enhanced services
        components=components,
        uptime_seconds=uptime_seconds
    )


@app.get("/api/files", response_model=List[FileInfo])
async def list_files(services=Depends(get_services)):
    """List all processed Excel files."""
    try:
        excel_proc, _, _ = services
        all_files = excel_proc.process_all_files()
        
        file_infos = []
        for file_data in all_files:
            file_info = FileInfo(
                file_name=file_data['file_name'],
                file_hash=file_data['file_hash'],
                total_sheets=file_data['total_sheets'],
                total_rows=file_data['total_rows'],
                total_columns=file_data['total_columns'],
                file_size_mb=file_data['file_size_mb'],
                last_modified=file_data['last_modified'],
                processed=True
            )
            file_infos.append(file_info)
        
        return file_infos
        
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    services=Depends(get_services)
):
    """Upload and process an Excel file."""
    try:
        excel_proc, vector_store_svc, _ = services
        
        # Validate file type
        if not file.filename.lower().endswith(('.xlsx', '.xls', '.xlsm')):
            raise HTTPException(
                status_code=400,
                detail="Only Excel files (.xlsx, .xls, .xlsm) are supported"
            )
        
        # Save uploaded file
        file_path = Path(excel_proc.data_directory) / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process file in background
        async def process_file():
            try:
                file_data = excel_proc.process_excel_file(file_path)
                await vector_store_svc.add_excel_data(
                    file_name=file_data['file_name'],
                    file_hash=file_data['file_hash'],
                    sheets_data=file_data['sheets']
                )
                logger.info(f"Successfully processed uploaded file: {file.filename}")
            except Exception as e:
                logger.error(f"Error processing uploaded file {file.filename}: {e}")
        
        background_tasks.add_task(process_file)
        
        return UploadResponse(
            success=True,
            message="File uploaded successfully and is being processed",
            processing_status="processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query_data(request: QueryRequest, rag_svc: RAGIntegrationService = Depends(get_rag_service)):
    """Query Excel data using enhanced AI with RAG."""
    try:
        # Process query with full RAG enhancement
        response = await rag_svc.process_rag_enhanced_query(
            query_request=request,
            session_id=None  # No session for REST API
        )
        
        if isinstance(response, QueryResponse):
            return response
        else:
            # Handle case where streaming generator is returned (shouldn't happen for REST API)
            logger.warning("Streaming response returned for REST API query")
            return QueryResponse(
                answer="Query processed but response format incompatible with REST API",
                sources=[],
                confidence=0.5,
                timestamp=datetime.now(),
                processing_time_ms=0
            )
        
    except LLMServiceError as e:
        logger.error(f"LLM service error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"AI service error: {e.message}"
        )
    except Exception as e:
        logger.error(f"Error processing enhanced query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream")
async def query_data_stream(request: QueryRequest, rag_svc: RAGIntegrationService = Depends(get_rag_service)):
    """Stream Excel data query response (for non-WebSocket clients)."""
    try:
        # Enable streaming in the request
        request.streaming = True
        
        # Process query with streaming
        response_gen = await rag_svc.process_rag_enhanced_query(
            query_request=request,
            session_id=f"rest_stream_{uuid.uuid4()}"
        )
        
        # Collect streaming response for REST API
        full_response = ""
        async for chunk in response_gen:
            full_response += chunk
        
        return QueryResponse(
            answer=full_response,
            sources=[],  # Will be populated by RAG service
            confidence=0.8,
            timestamp=datetime.now(),
            processing_time_ms=0
        )
        
    except Exception as e:
        logger.error(f"Error processing streaming query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=SystemStats)
async def get_stats(services=Depends(get_services)):
    """Get enhanced system statistics."""
    try:
        excel_proc, vector_store_svc, llm_svc, rag_svc = services
        
        # Get enhanced statistics from all services
        file_stats = excel_proc.get_statistics()
        vector_stats = vector_store_svc.get_statistics()
        llm_stats = llm_svc.get_service_statistics()
        rag_stats = rag_svc.get_rag_statistics()
        
        uptime_seconds = int((datetime.now() - app_start_time).total_seconds())
        
        return SystemStats(
            total_files=file_stats.get("total_files", 0),
            total_documents=vector_stats.get("total_documents", 0),
            cache_size=llm_stats.get("cache_size", 0),
            model_name=llm_stats.get("model_name", "unknown"),
            vector_store_size=vector_stats.get("total_documents", 0),
            uptime_seconds=uptime_seconds,
            active_connections=llm_stats.get("websocket_connections", 0),
            rag_query_count=rag_stats["performance_stats"].get("total_queries", 0),
            avg_retrieval_time=rag_stats["performance_stats"].get("avg_retrieval_time", 0.0),
            enhanced_features={
                "multi_modal_embeddings": True,
                "streaming_responses": True,
                "rag_integration": True,
                "error_handling": True,
                "websocket_support": True
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting enhanced stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rag/stats")
async def get_rag_statistics(rag_svc: RAGIntegrationService = Depends(get_rag_service)):
    """Get detailed RAG service statistics."""
    try:
        return rag_svc.get_rag_statistics()
    except Exception as e:
        logger.error(f"Error getting RAG statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/embedding/stats")
async def get_embedding_statistics():
    """Get embedding strategy statistics."""
    try:
        if not embedding_strategy:
            raise HTTPException(status_code=503, detail="Embedding strategy not available")
        
        return {
            "available_models": len(embedding_strategy.embedding_models),
            "model_details": {
                name: {
                    "type": model.model_type.value,
                    "dimensions": model.dimensions,
                    "max_tokens": model.max_tokens
                }
                for name, model in embedding_strategy.embedding_models.items()
            },
            "content_type_configs": {
                ct.value: config.__dict__ 
                for ct, config in embedding_strategy.content_type_configs.items()
            }
        }
    except Exception as e:
        logger.error(f"Error getting embedding statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/enhanced/reindex")
async def enhanced_reindex_files(background_tasks: BackgroundTasks, services=Depends(get_services)):
    """Enhanced reindexing with multi-modal capabilities."""
    try:
        excel_proc, vector_store_svc, _, _ = services
        
        async def enhanced_reindex():
            try:
                logger.info("Starting enhanced reindexing with multi-modal capabilities...")
                
                # Clear existing data
                await vector_store_svc.clear_collection()
                
                # Get all files
                all_files = excel_proc.process_all_files()
                
                processed_count = 0
                failed_count = 0
                
                for file_data in all_files:
                    try:
                        success = await vector_store_svc.add_excel_data_enhanced(
                            file_name=file_data['file_name'],
                            file_hash=file_data['file_hash'],
                            sheets_data=file_data['sheets'],
                            enable_multi_modal=True,
                            enable_content_analysis=True,
                            batch_size=50
                        )
                        
                        if success.get('success', False):
                            processed_count += 1
                            logger.info(f"Enhanced reindexing: {file_data['file_name']} completed")
                        else:
                            failed_count += 1
                            logger.error(f"Enhanced reindexing: {file_data['file_name']} failed")
                            
                    except Exception as file_error:
                        logger.error(f"Error reindexing file {file_data['file_name']}: {file_error}")
                        failed_count += 1
                        continue
                
                logger.info(f"Enhanced reindexing completed: {processed_count} successful, {failed_count} failed")
                
            except Exception as e:
                logger.error(f"Error during enhanced reindexing: {e}")
        
        background_tasks.add_task(enhanced_reindex)
        
        return {
            "message": "Enhanced reindexing started with multi-modal capabilities",
            "status": "processing",
            "features": ["multi_modal_embeddings", "content_analysis", "batch_processing"]
        }
        
    except Exception as e:
        logger.error(f"Error starting enhanced reindex: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reindex")
async def reindex_files(background_tasks: BackgroundTasks, services=Depends(get_services)):
    """Reindex all Excel files."""
    try:
        excel_proc, vector_store_svc, _ = services
        
        async def reindex():
            try:
                success = await vector_store_svc.reindex_all(excel_proc)
                logger.info(f"Reindexing completed: {'success' if success else 'failed'}")
            except Exception as e:
                logger.error(f"Error during reindexing: {e}")
        
        background_tasks.add_task(reindex)
        
        return {"message": "Reindexing started in background", "status": "processing"}
        
    except Exception as e:
        logger.error(f"Error starting reindex: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/cache")
async def clear_cache(services=Depends(get_services)):
    """Clear all caches."""
    try:
        _, _, llm_svc = services
        
        # Clear LLM cache
        llm_svc.clear_conversation_history()
        
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat."""
    if not all([excel_processor, vector_store, llm_service]):
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
        return
    
    await connection_manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            message_type = data.get("type", "query")
            content = data.get("content", "")
            
            if message_type == "query" and content:
                try:
                    # Send acknowledgment
                    await connection_manager.send_json_message({
                        "type": "status",
                        "content": "processing",
                        "timestamp": datetime.now().isoformat()
                    }, session_id)
                    
                    # Perform search
                    search_results = await vector_store.search(
                        query=content,
                        n_results=5,
                        file_filter=data.get("file_filter"),
                        sheet_filter=data.get("sheet_filter")
                    )
                    
                    if search_results:
                        context = "\n\n".join([
                            f"From {result['file_name']}, {result['sheet_name']}:\n{result['content']}"
                            for result in search_results
                        ])
                        
                        # Stream response
                        response_text = ""
                        async for token in llm_service.generate_streaming_response(
                            question=content,
                            context=context
                        ):
                            response_text += token
                            await connection_manager.send_json_message({
                                "type": "token",
                                "content": token,
                                "timestamp": datetime.now().isoformat()
                            }, session_id)
                        
                        # Send completion message
                        sources = list(set([
                            f"{result['file_name']} - {result['sheet_name']}"
                            for result in search_results
                        ]))
                        
                        await connection_manager.send_json_message({
                            "type": "complete",
                            "data": {
                                "sources": sources,
                                "total_tokens": len(response_text.split())
                            },
                            "timestamp": datetime.now().isoformat()
                        }, session_id)
                    
                    else:
                        await connection_manager.send_json_message({
                            "type": "response",
                            "content": "I couldn't find any relevant data to answer your question.",
                            "timestamp": datetime.now().isoformat()
                        }, session_id)
                
                except Exception as e:
                    logger.error(f"Error processing WebSocket query: {e}")
                    await connection_manager.send_json_message({
                        "type": "error",
                        "content": f"Error processing your request: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }, session_id)
            
            elif message_type == "ping":
                # Respond to ping with pong
                await connection_manager.send_json_message({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }, session_id)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(session_id)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )