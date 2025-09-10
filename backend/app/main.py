"""FastAPI main application for Excel Chat Agent."""

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
from app.services.excel_processor import ExcelProcessor
from app.services.vector_store import VectorStoreService
from app.services.llm_service import LangChainLLMService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for services
excel_processor: Optional[ExcelProcessor] = None
vector_store: Optional[VectorStoreService] = None
llm_service: Optional[LangChainLLMService] = None
app_start_time: datetime = datetime.now()


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_data: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept a WebSocket connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_data[session_id] = {
            "connected_at": datetime.now(),
            "last_activity": datetime.now()
        }
        logger.info(f"WebSocket connection established: {session_id}")
    
    def disconnect(self, session_id: str):
        """Remove a WebSocket connection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.session_data:
            del self.session_data[session_id]
        logger.info(f"WebSocket connection closed: {session_id}")
    
    async def send_personal_message(self, message: str, session_id: str):
        """Send a message to a specific WebSocket connection."""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_text(message)
                self.session_data[session_id]["last_activity"] = datetime.now()
            except Exception as e:
                logger.error(f"Error sending message to {session_id}: {e}")
                self.disconnect(session_id)
    
    async def send_json_message(self, data: Dict[str, Any], session_id: str):
        """Send a JSON message to a specific WebSocket connection."""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_json(data)
                self.session_data[session_id]["last_activity"] = datetime.now()
            except Exception as e:
                logger.error(f"Error sending JSON message to {session_id}: {e}")
                self.disconnect(session_id)
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)


# Global connection manager
connection_manager = ConnectionManager()


async def initialize_services():
    """Initialize all services during startup."""
    global excel_processor, vector_store, llm_service
    
    logger.info("Initializing services...")
    
    # Initialize Excel processor
    data_directory = os.getenv("DATA_DIRECTORY", "data/excel_files")
    excel_processor = ExcelProcessor(data_directory)
    
    # Initialize vector store
    chroma_directory = os.getenv("CHROMA_DIRECTORY", "chroma_db")
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    vector_store = VectorStoreService(chroma_directory, embedding_model=embedding_model)
    
    # Initialize LLM service
    model_name = os.getenv("OLLAMA_MODEL", "llama3")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    llm_service = LangChainLLMService(model_name=model_name, ollama_url=ollama_url)
    
    # Process existing Excel files and index them
    try:
        logger.info("Processing existing Excel files...")
        all_files = excel_processor.process_all_files()
        
        for file_data in all_files:
            await vector_store.add_excel_data(
                file_name=file_data['file_name'],
                file_hash=file_data['file_hash'],
                sheets_data=file_data['sheets']
            )
        
        logger.info(f"Processed and indexed {len(all_files)} Excel files")
        
    except Exception as e:
        logger.error(f"Error processing existing files: {e}")
    
    logger.info("Services initialized successfully")


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
    title="Excel Chat Agent",
    description="AI-powered chat interface for Excel file analysis",
    version="0.1.0",
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


# Dependency to check if services are ready
def get_services():
    """Dependency to ensure services are initialized."""
    if not all([excel_processor, vector_store, llm_service]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Services are still initializing"
        )
    return excel_processor, vector_store, llm_service


@app.get("/", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    uptime_seconds = int((datetime.now() - app_start_time).total_seconds())
    
    # Check service health
    components = {}
    overall_status = "healthy"
    
    try:
        if excel_processor:
            components["excel_processor"] = "healthy"
        else:
            components["excel_processor"] = "not_initialized"
            overall_status = "unhealthy"
        
        if vector_store:
            vector_health = vector_store.health_check()
            components["vector_store"] = vector_health["status"]
            if vector_health["status"] != "healthy":
                overall_status = "degraded"
        else:
            components["vector_store"] = "not_initialized"
            overall_status = "unhealthy"
        
        if llm_service:
            llm_health = llm_service.health_check()
            components["llm_service"] = llm_health["status"]
            if llm_health["status"] != "healthy":
                overall_status = "degraded"
        else:
            components["llm_service"] = "not_initialized"
            overall_status = "unhealthy"
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        overall_status = "unhealthy"
        components["error"] = str(e)
    
    return HealthCheck(
        status=overall_status,
        version="0.1.0",
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
async def query_data(request: QueryRequest, services=Depends(get_services)):
    """Query Excel data using AI."""
    try:
        start_time = time.time()
        excel_proc, vector_store_svc, llm_svc = services
        
        # Perform vector search to get relevant context
        search_results = await vector_store_svc.search(
            query=request.question,
            n_results=request.max_results,
            file_filter=request.file_filter,
            sheet_filter=request.sheet_filter
        )
        
        if not search_results:
            return QueryResponse(
                answer="I couldn't find any relevant data to answer your question. Please make sure your Excel files are properly uploaded and indexed.",
                sources=[],
                confidence=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Prepare context for LLM
        context = "\n\n".join([
            f"From {result['file_name']}, {result['sheet_name']}:\n{result['content']}"
            for result in search_results
        ])
        
        # Analyze request to determine intent
        analysis = await llm_svc.analyze_data_request(request.question)
        
        # Generate response
        llm_response = await llm_svc.generate_response(
            question=request.question,
            context=context,
            intent=analysis["intent"]
        )
        
        # Prepare sources
        sources = list(set([
            f"{result['file_name']} - {result['sheet_name']}"
            for result in search_results
        ]))
        
        file_sources = list(set([result['file_name'] for result in search_results]))
        sheet_sources = list(set([result['sheet_name'] for result in search_results]))
        
        # Calculate confidence based on search results relevance
        avg_relevance = sum(result['relevance_score'] for result in search_results) / len(search_results)
        confidence = min(avg_relevance * 1.2, 1.0)  # Boost confidence slightly
        
        # Generate chart recommendation if applicable
        chart_data = None
        if any(keyword in request.question.lower() for keyword in ['chart', 'graph', 'plot', 'visualize']):
            column_info = []
            for result in search_results:
                metadata = result.get('metadata', {})
                if 'columns' in metadata:
                    column_info.extend([col['name'] for col in metadata['columns']])
            
            if column_info:
                chart_data = await llm_svc.recommend_chart(
                    question=request.question,
                    data_description=context[:500],  # Truncate for chart analysis
                    column_info=list(set(column_info))
                )
        
        return QueryResponse(
            answer=llm_response["answer"],
            sources=sources,
            confidence=confidence,
            chart_data=chart_data,
            processing_time_ms=llm_response.get("processing_time_ms", 0),
            tokens_used=llm_response.get("tokens_used", 0),
            file_sources=file_sources,
            sheet_sources=sheet_sources
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=SystemStats)
async def get_stats(services=Depends(get_services)):
    """Get system statistics."""
    try:
        excel_proc, vector_store_svc, llm_svc = services
        
        # Get file statistics
        file_stats = excel_proc.get_file_statistics()
        
        # Get vector store statistics
        vector_stats = vector_store_svc.get_statistics()
        
        # Get LLM service statistics
        llm_stats = llm_svc.get_statistics()
        
        uptime_seconds = int((datetime.now() - app_start_time).total_seconds())
        
        return SystemStats(
            total_files=file_stats["total_files"],
            total_documents=vector_stats["total_documents"],
            cache_size=llm_stats["cache_size"],
            model_name=llm_stats["model_name"],
            vector_store_size=vector_stats["total_documents"],
            uptime_seconds=uptime_seconds,
            active_connections=connection_manager.get_connection_count()
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
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