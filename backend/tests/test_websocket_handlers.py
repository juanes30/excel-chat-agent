"""Test suite for WebSocket handlers and connection management."""

import asyncio
import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.websocket.llm_websocket import LLMWebSocketHandler, ConnectionManager
from app.services.enhanced_llm_service import EnhancedLLMService
from app.models.schemas import QueryRequest, QueryResponse
from fastapi import WebSocket
from fastapi.websockets import WebSocketState


class TestLLMWebSocketHandler:
    """Test suite for LLM WebSocket Handler."""
    
    @pytest.fixture
    def mock_enhanced_llm_service(self):
        """Create a mock enhanced LLM service."""
        mock_service = AsyncMock(spec=EnhancedLLMService)
        mock_service.websocket_manager = AsyncMock()
        mock_service.generate_enhanced_response = AsyncMock()
        mock_service.clear_conversation_memory = AsyncMock()
        mock_service.get_service_statistics = MagicMock(return_value={
            "model_name": "llama3",
            "active_sessions": 1,
            "cache_size": 0
        })
        return mock_service
    
    @pytest.fixture
    def websocket_handler(self, mock_enhanced_llm_service):
        """Create a WebSocket handler instance."""
        return LLMWebSocketHandler(mock_enhanced_llm_service)
    
    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket connection."""
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws.client_state = WebSocketState.CONNECTED
        mock_ws.client = MagicMock()
        return mock_ws
    
    @pytest.mark.asyncio
    async def test_connect_websocket(self, websocket_handler, mock_websocket):
        """Test WebSocket connection establishment."""
        session_id = await websocket_handler.connect(mock_websocket)
        
        # Verify connection was accepted
        mock_websocket.accept.assert_called_once()
        
        # Verify session was registered
        assert session_id in websocket_handler.active_connections
        assert session_id in websocket_handler.connection_metadata
        assert websocket_handler.active_connections[session_id] == mock_websocket
        
        # Verify welcome message was sent
        mock_websocket.send_text.assert_called()
        sent_message = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_message["type"] == "connection_established"
        assert sent_message["session_id"] == session_id
    
    @pytest.mark.asyncio
    async def test_connect_with_custom_session_id(self, websocket_handler, mock_websocket):
        """Test WebSocket connection with custom session ID."""
        custom_session_id = "custom_session_123"
        
        session_id = await websocket_handler.connect(mock_websocket, custom_session_id)
        
        assert session_id == custom_session_id
        assert custom_session_id in websocket_handler.active_connections
    
    @pytest.mark.asyncio
    async def test_disconnect_websocket(self, websocket_handler, mock_websocket):
        """Test WebSocket disconnection."""
        # First connect
        session_id = await websocket_handler.connect(mock_websocket)
        assert session_id in websocket_handler.active_connections
        
        # Then disconnect
        await websocket_handler.disconnect(session_id)
        
        # Verify cleanup
        assert session_id not in websocket_handler.active_connections
        assert session_id not in websocket_handler.connection_metadata
        mock_websocket.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_message_success(self, websocket_handler, mock_websocket):
        """Test sending message to WebSocket connection."""
        session_id = await websocket_handler.connect(mock_websocket)
        
        message = {"type": "test", "content": "Hello"}
        result = await websocket_handler.send_message(session_id, message)
        
        assert result is True
        mock_websocket.send_text.assert_called_with(json.dumps(message, default=str))
        
        # Verify metadata was updated
        assert websocket_handler.connection_metadata[session_id]["message_count"] == 2  # +1 from welcome
    
    @pytest.mark.asyncio
    async def test_send_message_to_nonexistent_session(self, websocket_handler):
        """Test sending message to non-existent session."""
        result = await websocket_handler.send_message("nonexistent", {"type": "test"})
        assert result is False
    
    @pytest.mark.asyncio
    async def test_handle_query_message(self, websocket_handler, mock_websocket, mock_enhanced_llm_service):
        """Test handling query message."""
        session_id = await websocket_handler.connect(mock_websocket)
        
        # Configure mock response
        mock_response = QueryResponse(
            answer="Test response",
            sources=["test.xlsx"],
            confidence=0.8,
            timestamp=datetime.now(),
            processing_time_ms=100
        )
        mock_enhanced_llm_service.generate_enhanced_response.return_value = mock_response
        
        query_message = {
            "type": "query",
            "data": {
                "question": "What are the sales figures?",
                "file_filter": "sales.xlsx",
                "streaming": False
            },
            "query_id": "test_query_123"
        }
        
        await websocket_handler.handle_message(session_id, query_message)
        
        # Verify LLM service was called
        mock_enhanced_llm_service.generate_enhanced_response.assert_called_once()
        
        # Verify response messages were sent
        assert mock_websocket.send_text.call_count >= 2  # Welcome + query_received + query_response
    
    @pytest.mark.asyncio
    async def test_handle_query_message_streaming(self, websocket_handler, mock_websocket, mock_enhanced_llm_service):
        """Test handling streaming query message."""
        session_id = await websocket_handler.connect(mock_websocket)
        
        # Configure mock streaming response
        async def mock_streaming_generator():
            yield "Part 1 "
            yield "Part 2 "
            yield "Part 3"
        
        mock_enhanced_llm_service.generate_enhanced_response.return_value = mock_streaming_generator()
        
        query_message = {
            "type": "query",
            "data": {
                "question": "Stream me a response",
                "streaming": True
            }
        }
        
        await websocket_handler.handle_message(session_id, query_message)
        
        # Verify streaming was called
        mock_enhanced_llm_service.generate_enhanced_response.assert_called_once()
        call_args = mock_enhanced_llm_service.generate_enhanced_response.call_args
        assert call_args[0][0].streaming is True  # QueryRequest.streaming
    
    @pytest.mark.asyncio
    async def test_handle_ping_message(self, websocket_handler, mock_websocket):
        """Test handling ping message."""
        session_id = await websocket_handler.connect(mock_websocket)
        
        ping_message = {"type": "ping"}
        await websocket_handler.handle_message(session_id, ping_message)
        
        # Check that pong was sent (welcome message + pong)
        assert mock_websocket.send_text.call_count == 2
        pong_call = mock_websocket.send_text.call_args_list[1]
        sent_message = json.loads(pong_call[0][0])
        assert sent_message["type"] == "pong"
    
    @pytest.mark.asyncio
    async def test_handle_clear_conversation(self, websocket_handler, mock_websocket, mock_enhanced_llm_service):
        """Test handling clear conversation message."""
        session_id = await websocket_handler.connect(mock_websocket)
        
        clear_message = {"type": "clear_conversation"}
        await websocket_handler.handle_message(session_id, clear_message)
        
        # Verify conversation was cleared
        mock_enhanced_llm_service.clear_conversation_memory.assert_called_once_with(session_id)
        
        # Verify confirmation was sent
        assert mock_websocket.send_text.call_count == 2  # welcome + confirmation
    
    @pytest.mark.asyncio
    async def test_handle_get_stats(self, websocket_handler, mock_websocket, mock_enhanced_llm_service):
        """Test handling get stats message."""
        session_id = await websocket_handler.connect(mock_websocket)
        
        stats_message = {"type": "get_stats"}
        await websocket_handler.handle_message(session_id, stats_message)
        
        # Verify stats were requested
        mock_enhanced_llm_service.get_service_statistics.assert_called_once()
        
        # Verify stats response was sent
        assert mock_websocket.send_text.call_count == 2
        stats_call = mock_websocket.send_text.call_args_list[1]
        sent_message = json.loads(stats_call[0][0])
        assert sent_message["type"] == "stats_response"
        assert "service_stats" in sent_message["data"]
    
    @pytest.mark.asyncio
    async def test_handle_set_streaming_mode(self, websocket_handler, mock_websocket):
        """Test handling set streaming mode message."""
        session_id = await websocket_handler.connect(mock_websocket)
        
        mode_message = {
            "type": "set_streaming_mode",
            "data": {"streaming_mode": "sentence_by_sentence"}
        }
        await websocket_handler.handle_message(session_id, mode_message)
        
        # Verify streaming mode was updated in metadata
        assert websocket_handler.connection_metadata[session_id]["streaming_mode"] == "sentence_by_sentence"
        
        # Verify confirmation was sent
        assert mock_websocket.send_text.call_count == 2
    
    @pytest.mark.asyncio
    async def test_handle_invalid_streaming_mode(self, websocket_handler, mock_websocket):
        """Test handling invalid streaming mode."""
        session_id = await websocket_handler.connect(mock_websocket)
        
        mode_message = {
            "type": "set_streaming_mode",
            "data": {"streaming_mode": "invalid_mode"}
        }
        await websocket_handler.handle_message(session_id, mode_message)
        
        # Verify error was sent
        error_call = mock_websocket.send_text.call_args_list[1]
        sent_message = json.loads(error_call[0][0])
        assert sent_message["type"] == "error"
        assert "invalid" in sent_message["message"].lower()
    
    @pytest.mark.asyncio
    async def test_handle_unknown_message_type(self, websocket_handler, mock_websocket):
        """Test handling unknown message type."""
        session_id = await websocket_handler.connect(mock_websocket)
        
        unknown_message = {"type": "unknown_type"}
        await websocket_handler.handle_message(session_id, unknown_message)
        
        # Verify error was sent
        error_call = mock_websocket.send_text.call_args_list[1]
        sent_message = json.loads(error_call[0][0])
        assert sent_message["type"] == "error"
        assert "unknown message type" in sent_message["message"].lower()
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self, websocket_handler):
        """Test broadcasting message to all connections."""
        # Connect multiple WebSockets
        websockets = []
        session_ids = []
        
        for i in range(3):
            mock_ws = AsyncMock(spec=WebSocket)
            mock_ws.accept = AsyncMock()
            mock_ws.send_text = AsyncMock()
            mock_ws.client_state = WebSocketState.CONNECTED
            mock_ws.client = MagicMock()
            
            session_id = await websocket_handler.connect(mock_ws)
            websockets.append(mock_ws)
            session_ids.append(session_id)
        
        # Broadcast message
        broadcast_message = {"type": "broadcast", "content": "Hello everyone"}
        await websocket_handler.broadcast_message(broadcast_message)
        
        # Verify all connections received the message
        for mock_ws in websockets:
            calls = mock_ws.send_text.call_args_list
            # Should have welcome message + broadcast message
            assert len(calls) == 2
            broadcast_call = calls[1]
            sent_message = json.loads(broadcast_call[0][0])
            assert sent_message == broadcast_message
    
    @pytest.mark.asyncio
    async def test_broadcast_with_exclusions(self, websocket_handler):
        """Test broadcasting with excluded sessions."""
        # Connect two WebSockets
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws1.accept = AsyncMock()
        mock_ws1.send_text = AsyncMock()
        mock_ws1.client_state = WebSocketState.CONNECTED
        mock_ws1.client = MagicMock()
        
        mock_ws2 = AsyncMock(spec=WebSocket)
        mock_ws2.accept = AsyncMock()
        mock_ws2.send_text = AsyncMock()
        mock_ws2.client_state = WebSocketState.CONNECTED
        mock_ws2.client = MagicMock()
        
        session_id1 = await websocket_handler.connect(mock_ws1)
        session_id2 = await websocket_handler.connect(mock_ws2)
        
        # Broadcast excluding session 1
        broadcast_message = {"type": "broadcast", "content": "Hello"}
        await websocket_handler.broadcast_message(broadcast_message, exclude_sessions=[session_id1])
        
        # Verify only session 2 received the broadcast
        assert mock_ws1.send_text.call_count == 1  # Only welcome message
        assert mock_ws2.send_text.call_count == 2  # Welcome + broadcast
    
    def test_get_connection_info(self, websocket_handler, mock_websocket):
        """Test getting connection information."""
        # Test non-existent session
        info = websocket_handler.get_connection_info("nonexistent")
        assert info is None
        
        # Test existing session
        asyncio.run(websocket_handler.connect(mock_websocket, "test_session"))
        
        info = websocket_handler.get_connection_info("test_session")
        assert info is not None
        assert info["is_connected"] is True
        assert "connected_at" in info
        assert "streaming_mode" in info
    
    def test_get_all_connections_info(self, websocket_handler):
        """Test getting all connections information."""
        # Initially empty
        all_info = websocket_handler.get_all_connections_info()
        assert len(all_info) == 0
        
        # Add some test data
        websocket_handler.active_connections["session1"] = AsyncMock()
        websocket_handler.connection_metadata["session1"] = {
            "connected_at": datetime.now(),
            "message_count": 5
        }
        
        all_info = websocket_handler.get_all_connections_info()
        assert len(all_info) == 1
        assert "session1" in all_info
    
    @pytest.mark.asyncio
    async def test_health_check(self, websocket_handler, mock_websocket):
        """Test WebSocket handler health check."""
        # Connect a healthy WebSocket
        session_id = await websocket_handler.connect(mock_websocket)
        
        health = await websocket_handler.health_check()
        
        assert health["total_connections"] == 1
        assert health["healthy_connections"] == 1
        assert health["unhealthy_connections"] == 0
        assert "connection_details" in health
    
    @pytest.mark.asyncio
    async def test_health_check_with_unhealthy_connection(self, websocket_handler):
        """Test health check with unhealthy connections."""
        # Add a connection with disconnected state
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.client_state = WebSocketState.DISCONNECTED
        mock_ws.close = AsyncMock()
        
        websocket_handler.active_connections["unhealthy"] = mock_ws
        websocket_handler.connection_metadata["unhealthy"] = {}
        
        health = await websocket_handler.health_check()
        
        # Should clean up the unhealthy connection
        assert "unhealthy" not in websocket_handler.active_connections


class TestConnectionManager:
    """Test suite for Connection Manager."""
    
    @pytest.fixture
    def connection_manager(self):
        """Create a connection manager instance."""
        return ConnectionManager()
    
    @pytest.fixture
    def mock_handler(self):
        """Create a mock WebSocket handler."""
        return MagicMock(spec=LLMWebSocketHandler)
    
    def test_register_handler(self, connection_manager, mock_handler):
        """Test registering a WebSocket handler."""
        connection_manager.register_handler("test_handler", mock_handler)
        
        assert "test_handler" in connection_manager.handlers
        assert connection_manager.handlers["test_handler"] == mock_handler
        assert connection_manager.default_handler == mock_handler
    
    def test_get_handler_by_name(self, connection_manager, mock_handler):
        """Test getting handler by name."""
        connection_manager.register_handler("specific_handler", mock_handler)
        
        retrieved_handler = connection_manager.get_handler("specific_handler")
        assert retrieved_handler == mock_handler
    
    def test_get_default_handler(self, connection_manager, mock_handler):
        """Test getting default handler."""
        connection_manager.register_handler("default", mock_handler)
        
        # Get handler without specifying name
        retrieved_handler = connection_manager.get_handler()
        assert retrieved_handler == mock_handler
    
    def test_get_nonexistent_handler(self, connection_manager, mock_handler):
        """Test getting non-existent handler returns default."""
        connection_manager.register_handler("default", mock_handler)
        
        retrieved_handler = connection_manager.get_handler("nonexistent")
        assert retrieved_handler == mock_handler  # Should return default
    
    @pytest.mark.asyncio
    async def test_broadcast_to_all_handlers(self, connection_manager):
        """Test broadcasting to all handlers."""
        # Register multiple handlers
        handlers = {}
        for i in range(3):
            handler = AsyncMock(spec=LLMWebSocketHandler)
            handler.broadcast_message = AsyncMock()
            handler_name = f"handler_{i}"
            handlers[handler_name] = handler
            connection_manager.register_handler(handler_name, handler)
        
        message = {"type": "global_broadcast", "content": "Hello all"}
        await connection_manager.broadcast_to_all_handlers(message)
        
        # Verify all handlers received the broadcast
        for handler in handlers.values():
            handler.broadcast_message.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_shutdown(self, connection_manager):
        """Test connection manager shutdown."""
        # Create mock handlers with connections
        handler1 = AsyncMock(spec=LLMWebSocketHandler)
        handler1.active_connections = {"session1": MagicMock(), "session2": MagicMock()}
        handler1.disconnect = AsyncMock()
        
        handler2 = AsyncMock(spec=LLMWebSocketHandler)
        handler2.active_connections = {"session3": MagicMock()}
        handler2.disconnect = AsyncMock()
        
        connection_manager.register_handler("handler1", handler1)
        connection_manager.register_handler("handler2", handler2)
        
        await connection_manager.shutdown()
        
        # Verify all connections were disconnected
        assert handler1.disconnect.call_count == 2  # session1 and session2
        assert handler2.disconnect.call_count == 1  # session3


@pytest.mark.asyncio
async def test_websocket_error_handling():
    """Test error handling in WebSocket operations."""
    mock_llm_service = AsyncMock()
    handler = LLMWebSocketHandler(mock_llm_service)
    
    # Test handling malformed message
    mock_websocket = AsyncMock()
    mock_websocket.accept = AsyncMock()
    mock_websocket.send_text = AsyncMock()
    mock_websocket.client_state = WebSocketState.CONNECTED
    mock_websocket.client = MagicMock()
    
    session_id = await handler.connect(mock_websocket)
    
    # Send malformed message (missing required fields)
    malformed_message = {"type": "query"}  # Missing data field
    
    await handler.handle_message(session_id, malformed_message)
    
    # Should send error response
    error_call = mock_websocket.send_text.call_args_list[1]  # Skip welcome message
    sent_message = json.loads(error_call[0][0])
    assert sent_message["type"] == "error"


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()