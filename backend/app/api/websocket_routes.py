"""FastAPI WebSocket routes for LLM streaming communication."""

import asyncio
import json
import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.websocket.llm_websocket import LLMWebSocketHandler, connection_manager
from app.services.enhanced_llm_service import EnhancedLLMService
from app.services.enhanced_vector_store_v2 import EnhancedVectorStoreV2

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])

# Global handler instance (will be initialized in main.py)
llm_websocket_handler: Optional[LLMWebSocketHandler] = None


def get_llm_websocket_handler() -> LLMWebSocketHandler:
    """Dependency to get the LLM WebSocket handler."""
    global llm_websocket_handler
    if not llm_websocket_handler:
        raise HTTPException(status_code=503, detail="LLM WebSocket handler not initialized")
    return llm_websocket_handler


def initialize_websocket_handler(enhanced_llm_service: EnhancedLLMService):
    """Initialize the global WebSocket handler."""
    global llm_websocket_handler
    llm_websocket_handler = LLMWebSocketHandler(enhanced_llm_service)
    connection_manager.register_handler("llm", llm_websocket_handler)
    logger.info("LLM WebSocket handler initialized")


@router.websocket("/chat")
async def websocket_chat_endpoint(websocket: WebSocket, session_id: Optional[str] = None):
    """WebSocket endpoint for chat communication with LLM streaming.
    
    This endpoint handles real-time chat communication with the enhanced LLM service,
    supporting streaming responses and various message types.
    
    Message Types:
    - query: Send question to LLM for processing
    - ping: Keep connection alive
    - clear_conversation: Clear conversation history
    - get_stats: Get service statistics
    - set_streaming_mode: Change streaming mode
    """
    handler = get_llm_websocket_handler()
    actual_session_id = None
    
    try:
        # Accept connection and register session
        actual_session_id = await handler.connect(websocket, session_id)
        logger.info(f"WebSocket chat connection established: {actual_session_id}")
        
        # Main message loop
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle message
                await handler.handle_message(actual_session_id, message)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket chat disconnected: {actual_session_id}")
                break
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON received from {actual_session_id}: {e}")
                await handler.send_message(actual_session_id, {
                    "type": "error",
                    "message": f"Invalid JSON format: {str(e)}",
                    "timestamp": None
                })
            except Exception as e:
                logger.error(f"Error in WebSocket chat loop for {actual_session_id}: {e}")
                await handler.send_message(actual_session_id, {
                    "type": "error",
                    "message": f"Server error: {str(e)}",
                    "timestamp": None
                })
                # Continue the loop to maintain connection
    
    except Exception as e:
        logger.error(f"Fatal error in WebSocket chat endpoint: {e}")
    
    finally:
        # Cleanup connection
        if actual_session_id:
            await handler.disconnect(actual_session_id)


@router.websocket("/admin")
async def websocket_admin_endpoint(websocket: WebSocket):
    """WebSocket endpoint for administrative functions.
    
    This endpoint provides administrative capabilities including:
    - Broadcasting messages to all connected clients
    - Monitoring connection health
    - Service statistics
    - System commands
    """
    handler = get_llm_websocket_handler()
    admin_session_id = None
    
    try:
        # Accept admin connection
        admin_session_id = await handler.connect(websocket, f"admin_{id(websocket)}")
        logger.info(f"WebSocket admin connection established: {admin_session_id}")
        
        # Send admin welcome message
        await handler.send_message(admin_session_id, {
            "type": "admin_connected",
            "message": "Admin WebSocket connection established",
            "capabilities": [
                "broadcast",
                "health_check",
                "connection_stats",
                "service_stats",
                "shutdown_connections"
            ]
        })
        
        while True:
            try:
                # Receive admin command
                data = await websocket.receive_text()
                message = json.loads(data)
                
                await _handle_admin_message(handler, admin_session_id, message)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket admin disconnected: {admin_session_id}")
                break
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON from admin {admin_session_id}: {e}")
                await handler.send_message(admin_session_id, {
                    "type": "error",
                    "message": f"Invalid JSON format: {str(e)}"
                })
            except Exception as e:
                logger.error(f"Error in admin WebSocket loop: {e}")
                await handler.send_message(admin_session_id, {
                    "type": "error",
                    "message": f"Admin error: {str(e)}"
                })
    
    except Exception as e:
        logger.error(f"Fatal error in admin WebSocket endpoint: {e}")
    
    finally:
        if admin_session_id:
            await handler.disconnect(admin_session_id)


async def _handle_admin_message(handler: LLMWebSocketHandler, admin_session_id: str, message: Dict[str, Any]):
    """Handle administrative WebSocket messages."""
    message_type = message.get("type", "unknown")
    
    try:
        if message_type == "broadcast":
            # Broadcast message to all connections
            broadcast_data = message.get("data", {})
            await handler.broadcast_message({
                "type": "admin_broadcast",
                "data": broadcast_data,
                "from": "admin"
            }, exclude_sessions=[admin_session_id])
            
            await handler.send_message(admin_session_id, {
                "type": "broadcast_sent",
                "message": f"Message broadcast to {len(handler.active_connections) - 1} connections"
            })
        
        elif message_type == "health_check":
            # Perform health check
            health_info = await handler.health_check()
            await handler.send_message(admin_session_id, {
                "type": "health_check_result",
                "data": health_info
            })
        
        elif message_type == "connection_stats":
            # Get connection statistics
            connections_info = handler.get_all_connections_info()
            await handler.send_message(admin_session_id, {
                "type": "connection_stats",
                "data": {
                    "total_connections": len(connections_info),
                    "connections": connections_info
                }
            })
        
        elif message_type == "service_stats":
            # Get service statistics
            service_stats = handler.llm_service.get_service_statistics()
            await handler.send_message(admin_session_id, {
                "type": "service_stats",
                "data": service_stats
            })
        
        elif message_type == "disconnect_session":
            # Disconnect specific session
            target_session_id = message.get("data", {}).get("session_id")
            if target_session_id and target_session_id in handler.active_connections:
                await handler.disconnect(target_session_id)
                await handler.send_message(admin_session_id, {
                    "type": "session_disconnected",
                    "message": f"Session {target_session_id} disconnected"
                })
            else:
                await handler.send_message(admin_session_id, {
                    "type": "error",
                    "message": f"Session not found: {target_session_id}"
                })
        
        elif message_type == "shutdown_all":
            # Shutdown all connections (except admin)
            session_ids = list(handler.active_connections.keys())
            session_ids.remove(admin_session_id)  # Don't disconnect admin
            
            tasks = []
            for session_id in session_ids:
                tasks.append(handler.disconnect(session_id))
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            await handler.send_message(admin_session_id, {
                "type": "shutdown_complete",
                "message": f"Disconnected {len(session_ids)} sessions"
            })
        
        else:
            await handler.send_message(admin_session_id, {
                "type": "error",
                "message": f"Unknown admin command: {message_type}"
            })
    
    except Exception as e:
        logger.error(f"Error handling admin message: {e}")
        await handler.send_message(admin_session_id, {
            "type": "error",
            "message": f"Error executing admin command: {str(e)}"
        })


# REST endpoints for WebSocket management
@router.get("/status")
async def get_websocket_status():
    """Get WebSocket service status."""
    try:
        handler = get_llm_websocket_handler()
        
        # Get basic statistics
        connections_info = handler.get_all_connections_info()
        health_info = await handler.health_check()
        
        return JSONResponse(content={
            "status": "active",
            "total_connections": len(connections_info),
            "healthy_connections": health_info["healthy_connections"],
            "unhealthy_connections": health_info["unhealthy_connections"],
            "connection_details": connections_info,
            "handler_initialized": True
        })
    
    except HTTPException:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "message": "WebSocket handler not initialized",
                "handler_initialized": False
            }
        )
    except Exception as e:
        logger.error(f"Error getting WebSocket status: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "handler_initialized": False
            }
        )


@router.post("/broadcast")
async def broadcast_message(message: Dict[str, Any]):
    """Broadcast message to all WebSocket connections."""
    try:
        handler = get_llm_websocket_handler()
        
        # Add timestamp and source
        broadcast_message = {
            "type": "api_broadcast",
            "data": message,
            "timestamp": None,
            "source": "api"
        }
        
        await handler.broadcast_message(broadcast_message)
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Message broadcast to {len(handler.active_connections)} connections",
            "recipient_count": len(handler.active_connections)
        })
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error broadcasting message: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )


@router.delete("/connections/{session_id}")
async def disconnect_session(session_id: str):
    """Disconnect a specific WebSocket session."""
    try:
        handler = get_llm_websocket_handler()
        
        if session_id not in handler.active_connections:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "not_found",
                    "message": f"Session not found: {session_id}"
                }
            )
        
        await handler.disconnect(session_id)
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Session {session_id} disconnected"
        })
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error disconnecting session {session_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )


@router.get("/connections")
async def get_connections():
    """Get information about all active WebSocket connections."""
    try:
        handler = get_llm_websocket_handler()
        connections_info = handler.get_all_connections_info()
        
        return JSONResponse(content={
            "status": "success",
            "total_connections": len(connections_info),
            "connections": connections_info
        })
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting connections info: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )


@router.get("/health")
async def websocket_health_check():
    """Perform comprehensive WebSocket health check."""
    try:
        handler = get_llm_websocket_handler()
        health_info = await handler.health_check()
        
        # Also check LLM service health
        llm_health = await handler.llm_service.health_check()
        
        overall_status = "healthy"
        if health_info["unhealthy_connections"] > 0:
            overall_status = "degraded"
        if llm_health.get("status") != "healthy":
            overall_status = "unhealthy"
        
        return JSONResponse(content={
            "status": overall_status,
            "websocket_health": health_info,
            "llm_service_health": llm_health,
            "timestamp": None
        })
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in WebSocket health check: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": None
            }
        )