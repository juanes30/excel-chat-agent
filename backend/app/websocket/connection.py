"""WebSocket connection management for Excel Chat Agent."""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketConnection:
    """Individual WebSocket connection wrapper."""
    
    def __init__(self, websocket: WebSocket, session_id: str):
        self.websocket = websocket
        self.session_id = session_id
        self.connected_at = datetime.now()
        self.last_activity = datetime.now()
        self.message_count = 0
        self.is_active = True
        self.metadata: Dict[str, Any] = {}
    
    async def send_message(self, message_type: str, content: str = "", data: Optional[Dict[str, Any]] = None):
        """Send a structured message through the WebSocket."""
        try:
            message = {
                "type": message_type,
                "content": content,
                "data": data or {},
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id
            }
            
            await self.websocket.send_json(message)
            self.last_activity = datetime.now()
            self.message_count += 1
            
        except Exception as e:
            logger.error(f"Error sending message to {self.session_id}: {e}")
            self.is_active = False
            raise
    
    async def send_streaming_token(self, token: str):
        """Send a single token for streaming responses."""
        await self.send_message("token", content=token)
    
    async def send_error(self, error_message: str, error_code: Optional[str] = None):
        """Send an error message."""
        await self.send_message(
            "error",
            content=error_message,
            data={"error_code": error_code} if error_code else {}
        )
    
    async def send_status(self, status: str, message: str = ""):
        """Send a status update."""
        await self.send_message("status", content=message, data={"status": status})
    
    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity = datetime.now()
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about this connection."""
        return {
            "session_id": self.session_id,
            "connected_at": self.connected_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "message_count": self.message_count,
            "is_active": self.is_active,
            "connection_duration_seconds": int((datetime.now() - self.connected_at).total_seconds()),
            "metadata": self.metadata
        }


class ConnectionManager:
    """Advanced WebSocket connection manager with additional features."""
    
    def __init__(self, heartbeat_interval: int = 30, max_connections: int = 100):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.heartbeat_interval = heartbeat_interval
        self.max_connections = max_connections
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.total_connections = 0
        self.total_messages_sent = 0
        self.start_time = datetime.now()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background tasks for heartbeat and cleanup."""
        if self.heartbeat_task is None or self.heartbeat_task.done():
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def connect(self, websocket: WebSocket, session_id: Optional[str] = None) -> str:
        """Accept a new WebSocket connection."""
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Check connection limit
        if len(self.connections) >= self.max_connections:
            await websocket.close(code=1008, reason="Connection limit reached")
            raise Exception("Connection limit reached")
        
        # Accept the connection
        await websocket.accept()
        
        # Create connection wrapper
        connection = WebSocketConnection(websocket, session_id)
        self.connections[session_id] = connection
        
        # Update statistics
        self.total_connections += 1
        
        logger.info(f"WebSocket connection established: {session_id} "
                   f"(Total active: {len(self.connections)})")
        
        # Send welcome message
        await connection.send_message("connected", "Connection established", {
            "session_id": session_id,
            "server_time": datetime.now().isoformat()
        })
        
        return session_id
    
    def disconnect(self, session_id: str):
        """Disconnect a WebSocket connection."""
        if session_id in self.connections:
            connection = self.connections[session_id]
            connection.is_active = False
            del self.connections[session_id]
            
            logger.info(f"WebSocket connection closed: {session_id} "
                       f"(Total active: {len(self.connections)})")
    
    async def send_to_session(self, session_id: str, message_type: str, 
                            content: str = "", data: Optional[Dict[str, Any]] = None) -> bool:
        """Send a message to a specific session."""
        if session_id not in self.connections:
            return False
        
        try:
            connection = self.connections[session_id]
            await connection.send_message(message_type, content, data)
            self.total_messages_sent += 1
            return True
        
        except Exception as e:
            logger.error(f"Error sending message to {session_id}: {e}")
            self.disconnect(session_id)
            return False
    
    async def broadcast(self, message_type: str, content: str = "", 
                       data: Optional[Dict[str, Any]] = None, 
                       exclude_sessions: Optional[Set[str]] = None):
        """Broadcast a message to all connected sessions."""
        exclude_sessions = exclude_sessions or set()
        
        disconnected_sessions = []
        
        for session_id, connection in self.connections.items():
            if session_id in exclude_sessions:
                continue
            
            try:
                await connection.send_message(message_type, content, data)
                self.total_messages_sent += 1
            except Exception as e:
                logger.error(f"Error broadcasting to {session_id}: {e}")
                disconnected_sessions.append(session_id)
        
        # Clean up disconnected sessions
        for session_id in disconnected_sessions:
            self.disconnect(session_id)
    
    async def send_streaming_response(self, session_id: str, response_generator):
        """Send a streaming response token by token."""
        if session_id not in self.connections:
            return False
        
        try:
            connection = self.connections[session_id]
            
            # Send start of stream
            await connection.send_status("streaming", "Starting response stream")
            
            # Stream tokens
            async for token in response_generator:
                await connection.send_streaming_token(token)
            
            # Send end of stream
            await connection.send_status("stream_complete", "Response stream completed")
            
            return True
            
        except Exception as e:
            logger.error(f"Error streaming to {session_id}: {e}")
            self.disconnect(session_id)
            return False
    
    def get_connection(self, session_id: str) -> Optional[WebSocketConnection]:
        """Get a specific connection."""
        return self.connections.get(session_id)
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.connections.keys())
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.connections)
    
    def get_connection_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific connection."""
        connection = self.connections.get(session_id)
        return connection.get_connection_info() if connection else None
    
    def get_all_connections_info(self) -> List[Dict[str, Any]]:
        """Get information about all connections."""
        return [conn.get_connection_info() for conn in self.connections.values()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get connection manager statistics."""
        now = datetime.now()
        uptime = now - self.start_time
        
        # Calculate activity statistics
        active_connections = len(self.connections)
        total_messages = sum(conn.message_count for conn in self.connections.values())
        
        # Find oldest and newest connections
        if self.connections:
            oldest_connection = min(self.connections.values(), key=lambda c: c.connected_at)
            newest_connection = max(self.connections.values(), key=lambda c: c.connected_at)
            
            oldest_duration = int((now - oldest_connection.connected_at).total_seconds())
            newest_duration = int((now - newest_connection.connected_at).total_seconds())
        else:
            oldest_duration = newest_duration = 0
        
        return {
            "active_connections": active_connections,
            "total_connections_ever": self.total_connections,
            "total_messages_sent": self.total_messages_sent,
            "total_messages_received": total_messages,
            "uptime_seconds": int(uptime.total_seconds()),
            "longest_connection_duration": oldest_duration,
            "newest_connection_duration": newest_duration,
            "average_messages_per_connection": round(total_messages / max(active_connections, 1), 2),
            "messages_per_second": round(self.total_messages_sent / max(uptime.total_seconds(), 1), 4)
        }
    
    async def _heartbeat_loop(self):
        """Background task to send heartbeat pings."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if self.connections:
                    disconnected = []
                    
                    for session_id, connection in self.connections.items():
                        try:
                            await connection.send_message("ping", "", {"server_time": datetime.now().isoformat()})
                        except Exception as e:
                            logger.warning(f"Heartbeat failed for {session_id}: {e}")
                            disconnected.append(session_id)
                    
                    # Clean up failed connections
                    for session_id in disconnected:
                        self.disconnect(session_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
    
    async def _cleanup_loop(self):
        """Background task to clean up inactive connections."""
        cleanup_interval = 300  # 5 minutes
        inactive_threshold = timedelta(minutes=10)
        
        while True:
            try:
                await asyncio.sleep(cleanup_interval)
                
                now = datetime.now()
                inactive_sessions = []
                
                for session_id, connection in self.connections.items():
                    if (now - connection.last_activity) > inactive_threshold:
                        inactive_sessions.append(session_id)
                
                # Close inactive connections
                for session_id in inactive_sessions:
                    logger.info(f"Closing inactive connection: {session_id}")
                    try:
                        connection = self.connections[session_id]
                        await connection.websocket.close(code=1000, reason="Inactive connection")
                    except Exception as e:
                        logger.warning(f"Error closing inactive connection {session_id}: {e}")
                    finally:
                        self.disconnect(session_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def shutdown(self):
        """Shutdown the connection manager and close all connections."""
        logger.info("Shutting down connection manager...")
        
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Close all connections
        for session_id in list(self.connections.keys()):
            try:
                connection = self.connections[session_id]
                await connection.send_message("server_shutdown", "Server is shutting down")
                await connection.websocket.close(code=1001, reason="Server shutdown")
            except Exception as e:
                logger.warning(f"Error closing connection during shutdown {session_id}: {e}")
            finally:
                self.disconnect(session_id)
        
        logger.info("Connection manager shutdown complete")


# Global connection manager instance
connection_manager = ConnectionManager()