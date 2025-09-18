"""WebSocket handlers for LLM streaming communication."""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

import websockets
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from fastapi.websockets import WebSocketState

from app.models.schemas import QueryRequest, WebSocketMessage
from app.services.enhanced_llm_service import EnhancedLLMService, StreamingMode

logger = logging.getLogger(__name__)


class LLMWebSocketHandler:
    """Handles WebSocket connections for LLM streaming."""
    
    def __init__(self, enhanced_llm_service: EnhancedLLMService):
        self.llm_service = enhanced_llm_service
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str = None) -> str:
        """Accept WebSocket connection and register session."""
        await websocket.accept()
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Register connection
        self.active_connections[session_id] = websocket
        self.connection_metadata[session_id] = {
            "connected_at": datetime.now(),
            "last_activity": datetime.now(),
            "message_count": 0,
            "client_info": websocket.client
        }
        
        # Register with LLM service WebSocket manager
        await self.llm_service.websocket_manager.register_connection(websocket, session_id)
        
        logger.info(f"WebSocket connected: {session_id}")
        
        # Send welcome message
        welcome_message = {
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "message": "Connected to Excel Chat Agent LLM Service"
        }
        await self.send_message(session_id, welcome_message)
        
        return session_id
    
    async def disconnect(self, session_id: str):
        """Handle WebSocket disconnection."""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            
            # Close WebSocket if still open
            if websocket.client_state != WebSocketState.DISCONNECTED:
                try:
                    await websocket.close()
                except Exception as e:
                    logger.warning(f"Error closing WebSocket for {session_id}: {e}")
            
            # Unregister connection
            del self.active_connections[session_id]
            if session_id in self.connection_metadata:
                del self.connection_metadata[session_id]
            
            # Unregister from LLM service
            await self.llm_service.websocket_manager.unregister_connection(session_id)
            
            logger.info(f"WebSocket disconnected: {session_id}")
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Send message to specific WebSocket connection."""
        if session_id not in self.active_connections:
            logger.warning(f"Attempted to send message to non-existent session: {session_id}")
            return False
        
        websocket = self.active_connections[session_id]
        
        try:
            # Update last activity
            if session_id in self.connection_metadata:
                self.connection_metadata[session_id]["last_activity"] = datetime.now()
                self.connection_metadata[session_id]["message_count"] += 1
            
            # Send message
            await websocket.send_text(json.dumps(message, default=str))
            return True
            
        except WebSocketDisconnect:
            await self.disconnect(session_id)
            return False
        except Exception as e:
            logger.error(f"Error sending message to {session_id}: {e}")
            await self.disconnect(session_id)
            return False
    
    async def handle_message(self, session_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket message."""
        try:
            message_type = message.get("type", "unknown")
            
            if message_type == "query":
                await self._handle_query_message(session_id, message)
            elif message_type == "ping":
                await self._handle_ping_message(session_id)
            elif message_type == "clear_conversation":
                await self._handle_clear_conversation(session_id)
            elif message_type == "get_stats":
                await self._handle_get_stats(session_id)
            elif message_type == "set_streaming_mode":
                await self._handle_set_streaming_mode(session_id, message)
            else:
                await self.send_message(session_id, {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}",
                    "timestamp": datetime.now().isoformat()
                })
        
        except Exception as e:
            logger.error(f"Error handling message from {session_id}: {e}")
            await self.send_message(session_id, {
                "type": "error",
                "message": f"Error processing message: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
    
    async def _handle_query_message(self, session_id: str, message: Dict[str, Any]):
        """Handle query message for LLM processing."""
        try:
            # Extract query parameters
            data = message.get("data", {})
            question = data.get("question")
            
            if not question:
                await self.send_message(session_id, {
                    "type": "error",
                    "message": "Question is required",
                    "timestamp": datetime.now().isoformat()
                })
                return
            
            # Create query request
            query_request = QueryRequest(
                question=question,
                file_filter=data.get("file_filter"),
                sheet_filter=data.get("sheet_filter"),
                max_results=data.get("max_results", 5),
                include_statistics=data.get("include_statistics", False),
                streaming=data.get("streaming", True)
            )
            
            # Get streaming mode from session metadata
            streaming_mode_str = self.connection_metadata.get(session_id, {}).get("streaming_mode", "token_by_token")
            try:
                streaming_mode = StreamingMode(streaming_mode_str)
            except ValueError:
                streaming_mode = StreamingMode.TOKEN_BY_TOKEN
            
            # Send query acknowledgment
            await self.send_message(session_id, {
                "type": "query_received",
                "message": "Processing your question...",
                "query_id": message.get("query_id"),
                "timestamp": datetime.now().isoformat()
            })
            
            # Generate streaming response
            if query_request.streaming:
                async for response_chunk in self.llm_service.generate_enhanced_response(
                    query_request, session_id, streaming_mode
                ):
                    # Response chunks are sent automatically through WebSocket manager
                    # This generator ensures the streaming continues
                    pass
            else:
                # Generate standard response
                response = await self.llm_service.generate_enhanced_response(
                    query_request, session_id
                )
                
                # Send complete response
                await self.send_message(session_id, {
                    "type": "query_response",
                    "data": response.dict() if hasattr(response, 'dict') else response,
                    "query_id": message.get("query_id"),
                    "timestamp": datetime.now().isoformat()
                })
        
        except Exception as e:
            logger.error(f"Error processing query from {session_id}: {e}")
            await self.send_message(session_id, {
                "type": "query_error",
                "message": f"Error processing query: {str(e)}",
                "query_id": message.get("query_id"),
                "timestamp": datetime.now().isoformat()
            })
    
    async def _handle_ping_message(self, session_id: str):
        """Handle ping message for connection keepalive."""
        await self.send_message(session_id, {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })
    
    async def _handle_clear_conversation(self, session_id: str):
        """Handle clear conversation request."""
        try:
            self.llm_service.clear_conversation_memory(session_id)
            await self.send_message(session_id, {
                "type": "conversation_cleared",
                "message": "Conversation history cleared",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error clearing conversation for {session_id}: {e}")
            await self.send_message(session_id, {
                "type": "error",
                "message": f"Error clearing conversation: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
    
    async def _handle_get_stats(self, session_id: str):
        """Handle statistics request."""
        try:
            stats = self.llm_service.get_service_statistics()
            
            # Add WebSocket-specific stats
            session_stats = {
                "connected_at": self.connection_metadata.get(session_id, {}).get("connected_at"),
                "message_count": self.connection_metadata.get(session_id, {}).get("message_count", 0),
                "last_activity": self.connection_metadata.get(session_id, {}).get("last_activity")
            }
            
            await self.send_message(session_id, {
                "type": "stats_response",
                "data": {
                    "service_stats": stats,
                    "session_stats": session_stats,
                    "total_connections": len(self.active_connections)
                },
                "timestamp": datetime.now().isoformat()
            })
        
        except Exception as e:
            logger.error(f"Error getting stats for {session_id}: {e}")
            await self.send_message(session_id, {
                "type": "error",
                "message": f"Error retrieving statistics: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
    
    async def _handle_set_streaming_mode(self, session_id: str, message: Dict[str, Any]):
        """Handle streaming mode change request."""
        try:
            new_mode = message.get("data", {}).get("streaming_mode", "token_by_token")
            
            # Validate streaming mode
            try:
                StreamingMode(new_mode)
            except ValueError:
                await self.send_message(session_id, {
                    "type": "error",
                    "message": f"Invalid streaming mode: {new_mode}",
                    "timestamp": datetime.now().isoformat()
                })
                return
            
            # Update session metadata
            if session_id in self.connection_metadata:
                self.connection_metadata[session_id]["streaming_mode"] = new_mode
            
            await self.send_message(session_id, {
                "type": "streaming_mode_updated",
                "data": {"streaming_mode": new_mode},
                "message": f"Streaming mode updated to: {new_mode}",
                "timestamp": datetime.now().isoformat()
            })
        
        except Exception as e:
            logger.error(f"Error setting streaming mode for {session_id}: {e}")
            await self.send_message(session_id, {
                "type": "error",
                "message": f"Error setting streaming mode: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
    
    async def broadcast_message(self, message: Dict[str, Any], exclude_sessions: Optional[list] = None):
        """Broadcast message to all connected clients."""
        exclude_sessions = exclude_sessions or []
        
        tasks = []
        for session_id in self.active_connections:
            if session_id not in exclude_sessions:
                tasks.append(self.send_message(session_id, message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_connection_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get connection information for a session."""
        if session_id not in self.connection_metadata:
            return None
        
        metadata = self.connection_metadata[session_id].copy()
        metadata["is_connected"] = session_id in self.active_connections
        metadata["streaming_mode"] = metadata.get("streaming_mode", "token_by_token")
        
        return metadata
    
    def get_all_connections_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active connections."""
        return {
            session_id: self.get_connection_info(session_id)
            for session_id in self.active_connections
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on WebSocket connections."""
        healthy_connections = 0
        total_connections = len(self.active_connections)
        
        # Test each connection with a ping
        for session_id, websocket in list(self.active_connections.items()):
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    healthy_connections += 1
                else:
                    await self.disconnect(session_id)
            except Exception as e:
                logger.warning(f"Unhealthy connection detected: {session_id}, {e}")
                await self.disconnect(session_id)
        
        return {
            "total_connections": total_connections,
            "healthy_connections": healthy_connections,
            "unhealthy_connections": total_connections - healthy_connections,
            "connection_details": self.get_all_connections_info()
        }


class ConnectionManager:
    """Global connection manager for WebSocket connections."""
    
    def __init__(self):
        self.handlers: Dict[str, LLMWebSocketHandler] = {}
        self.default_handler: Optional[LLMWebSocketHandler] = None
    
    def register_handler(self, handler_name: str, handler: LLMWebSocketHandler):
        """Register a WebSocket handler."""
        self.handlers[handler_name] = handler
        if not self.default_handler:
            self.default_handler = handler
        logger.info(f"Registered WebSocket handler: {handler_name}")
    
    def get_handler(self, handler_name: str = None) -> Optional[LLMWebSocketHandler]:
        """Get a WebSocket handler by name or return default."""
        if handler_name and handler_name in self.handlers:
            return self.handlers[handler_name]
        return self.default_handler
    
    async def broadcast_to_all_handlers(self, message: Dict[str, Any]):
        """Broadcast message to all handlers."""
        tasks = []
        for handler in self.handlers.values():
            tasks.append(handler.broadcast_message(message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def shutdown(self):
        """Shutdown all handlers and connections."""
        tasks = []
        for handler in self.handlers.values():
            for session_id in list(handler.active_connections.keys()):
                tasks.append(handler.disconnect(session_id))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("All WebSocket handlers shut down")


# Global connection manager instance
connection_manager = ConnectionManager()