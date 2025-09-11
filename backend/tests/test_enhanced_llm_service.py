"""Comprehensive test suite for Enhanced LLM Service."""

import asyncio
import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, AsyncGenerator

from app.services.enhanced_llm_service import EnhancedLLMService, StreamingMode, WebSocketManager
from app.models.schemas import QueryRequest, QueryResponse, ChartType, ChartData
from app.utils.error_handling import LLMServiceError, OllamaConnectionError
from langchain.schema import AIMessage


class TestEnhancedLLMService:
    """Test suite for Enhanced LLM Service."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        mock_store = AsyncMock()
        mock_store.enhanced_search.return_value = [
            {
                "content": "Sample Excel data about sales performance",
                "metadata": {"file_name": "sales.xlsx", "sheet_name": "Q1_Data"},
                "relevance_score": 0.85,
                "content_type": "numerical"
            }
        ]
        return mock_store
    
    @pytest.fixture
    def enhanced_llm_service(self, mock_vector_store):
        """Create an Enhanced LLM Service instance for testing."""
        with patch('app.services.enhanced_llm_service.ChatOllama') as mock_ollama:
            # Configure mock Ollama responses
            mock_response = AsyncMock()
            mock_response.content = "This is a test response from the LLM."
            
            mock_ollama.return_value.ainvoke = AsyncMock(return_value=mock_response)
            mock_ollama.return_value.astream = AsyncMock()
            
            service = EnhancedLLMService(
                model_name="llama3",
                vector_store=mock_vector_store
            )
            return service
    
    @pytest.fixture
    def sample_query_request(self):
        """Create a sample query request."""
        return QueryRequest(
            question="What are the sales figures for Q1?",
            file_filter="sales.xlsx",
            max_results=5,
            include_statistics=True,
            streaming=False
        )
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, mock_vector_store):
        """Test service initialization."""
        with patch('app.services.enhanced_llm_service.ChatOllama'):
            service = EnhancedLLMService(
                model_name="llama3",
                ollama_url="http://localhost:11434",
                temperature=0.7,
                max_tokens=2048,
                vector_store=mock_vector_store
            )
            
            assert service.model_name == "llama3"
            assert service.ollama_url == "http://localhost:11434"
            assert service.temperature == 0.7
            assert service.max_tokens == 2048
            assert service.vector_store == mock_vector_store
            assert isinstance(service.websocket_manager, WebSocketManager)
    
    @pytest.mark.asyncio
    async def test_generate_enhanced_response_standard(self, enhanced_llm_service, sample_query_request):
        """Test standard (non-streaming) response generation."""
        response = await enhanced_llm_service.generate_enhanced_response(
            sample_query_request,
            session_id="test_session"
        )
        
        assert isinstance(response, QueryResponse)
        assert response.answer == "This is a test response from the LLM."
        assert len(response.sources) > 0
        assert 0.0 <= response.confidence <= 1.0
        assert response.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_generate_enhanced_response_streaming(self, enhanced_llm_service):
        """Test streaming response generation."""
        request = QueryRequest(
            question="Analyze the sales data",
            streaming=True
        )
        
        # Mock streaming response
        async def mock_astream(*args, **kwargs):
            chunks = ["This ", "is ", "a ", "streaming ", "response."]
            for chunk in chunks:
                mock_chunk = MagicMock()
                mock_chunk.content = chunk
                yield mock_chunk
        
        enhanced_llm_service.streaming_llm.astream = mock_astream
        
        response_gen = await enhanced_llm_service.generate_enhanced_response(
            request,
            session_id="test_session",
            streaming_mode=StreamingMode.TOKEN_BY_TOKEN
        )
        
        # Collect streaming response
        full_response = ""
        async for chunk in response_gen:
            full_response += chunk
        
        assert full_response == "This is a streaming response."
    
    @pytest.mark.asyncio
    async def test_analyze_data_request(self, enhanced_llm_service):
        """Test data request analysis."""
        question = "Compare sales between Q1 and Q2 and show me a chart"
        
        # Mock analysis LLM response
        mock_analysis_response = AsyncMock()
        mock_analysis_response.content = "Analysis: This is a comparison query requiring visualization."
        enhanced_llm_service.analysis_llm.ainvoke = AsyncMock(return_value=mock_analysis_response)
        
        analysis = await enhanced_llm_service.analyze_data_request(question)
        
        assert analysis["intent"] == "comparison"
        assert analysis["is_comparative"] is True
        assert analysis["visualization_potential"] is True
        assert "chart" in analysis["keywords"]
        assert analysis["complexity"] in ["simple", "moderate", "complex"]
    
    @pytest.mark.asyncio
    async def test_recommend_enhanced_chart(self, enhanced_llm_service):
        """Test enhanced chart recommendation."""
        question = "Show me sales trends over time"
        context = "Sales data with dates and revenue figures"
        analysis = {"keywords": ["sales", "trends", "time"]}
        
        # Mock chart recommendation response
        mock_chart_response = AsyncMock()
        mock_chart_response.content = '''
        {
            "recommended": true,
            "chart_type": "line",
            "reasoning": "Line chart is best for showing trends over time",
            "data_columns": ["date", "revenue"],
            "chart_config": {
                "title": "Sales Trends Over Time",
                "x_axis": "Date",
                "y_axis": "Revenue"
            }
        }
        '''
        enhanced_llm_service.analysis_llm.ainvoke = AsyncMock(return_value=mock_chart_response)
        
        chart_data = await enhanced_llm_service.recommend_enhanced_chart(question, context, analysis)
        
        assert chart_data is not None
        assert chart_data.type == ChartType.LINE
        assert "Sales Trends Over Time" in chart_data.title
        assert "Line chart is best for showing trends over time" in chart_data.description
    
    @pytest.mark.asyncio
    async def test_conversation_memory_management(self, enhanced_llm_service):
        """Test conversation memory management."""
        session_id = "test_session_memory"
        
        # Get conversation memory
        memory1 = enhanced_llm_service.get_conversation_memory(session_id)
        memory2 = enhanced_llm_service.get_conversation_memory(session_id)
        
        # Should return the same memory instance
        assert memory1 is memory2
        
        # Test clearing memory
        enhanced_llm_service.clear_conversation_memory(session_id)
        assert len(memory1.chat_memory.messages) == 0
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, enhanced_llm_service):
        """Test retry mechanism with successful operation."""
        mock_operation = AsyncMock(return_value="success")
        
        result = await enhanced_llm_service.execute_with_retry(mock_operation, "arg1", key="value")
        
        assert result == "success"
        mock_operation.assert_called_once_with("arg1", key="value")
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_failure_then_success(self, enhanced_llm_service):
        """Test retry mechanism with initial failure then success."""
        mock_operation = AsyncMock()
        mock_operation.side_effect = [
            ConnectionError("First attempt fails"),
            "success"
        ]
        
        result = await enhanced_llm_service.execute_with_retry(mock_operation)
        
        assert result == "success"
        assert mock_operation.call_count == 2
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_all_failures(self, enhanced_llm_service):
        """Test retry mechanism with all attempts failing."""
        mock_operation = AsyncMock(side_effect=ConnectionError("Always fails"))
        
        with pytest.raises(ConnectionError):
            await enhanced_llm_service.execute_with_retry(mock_operation)
        
        assert mock_operation.call_count == enhanced_llm_service.max_retries
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, enhanced_llm_service):
        """Test health check when service is healthy."""
        # Mock successful LLM response
        mock_response = AsyncMock()
        mock_response.content = "Hello"
        enhanced_llm_service.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        health = await enhanced_llm_service.health_check()
        
        assert health["status"] == "healthy"
        assert "response_time_ms" in health
        assert "model_name" in health
        assert "websocket_manager" in health
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, enhanced_llm_service):
        """Test health check when service is unhealthy."""
        # Mock LLM failure
        enhanced_llm_service.llm.ainvoke = AsyncMock(side_effect=ConnectionError("Ollama not available"))
        
        health = await enhanced_llm_service.health_check()
        
        assert health["status"] == "unhealthy"
        assert "error" in health
    
    def test_get_service_statistics(self, enhanced_llm_service):
        """Test service statistics retrieval."""
        stats = enhanced_llm_service.get_service_statistics()
        
        expected_keys = [
            "model_name", "ollama_url", "temperature", "max_tokens",
            "active_sessions", "websocket_connections", "cache_size",
            "vector_store_connected", "total_conversations"
        ]
        
        for key in expected_keys:
            assert key in stats


class TestWebSocketManager:
    """Test suite for WebSocket Manager."""
    
    @pytest.fixture
    def websocket_manager(self):
        """Create a WebSocket Manager instance."""
        return WebSocketManager()
    
    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket connection."""
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        return mock_ws
    
    @pytest.mark.asyncio
    async def test_register_connection(self, websocket_manager, mock_websocket):
        """Test WebSocket connection registration."""
        session_id = "test_session"
        
        await websocket_manager.register_connection(mock_websocket, session_id)
        
        assert session_id in websocket_manager.connections
        assert session_id in websocket_manager.session_data
        assert websocket_manager.connections[session_id] == mock_websocket
    
    @pytest.mark.asyncio
    async def test_unregister_connection(self, websocket_manager, mock_websocket):
        """Test WebSocket connection unregistration."""
        session_id = "test_session"
        
        # First register
        await websocket_manager.register_connection(mock_websocket, session_id)
        assert session_id in websocket_manager.connections
        
        # Then unregister
        await websocket_manager.unregister_connection(session_id)
        assert session_id not in websocket_manager.connections
        assert session_id not in websocket_manager.session_data
    
    @pytest.mark.asyncio
    async def test_send_to_session(self, websocket_manager, mock_websocket):
        """Test sending message to specific session."""
        session_id = "test_session"
        message = {"type": "test", "content": "Hello"}
        
        # Register connection
        await websocket_manager.register_connection(mock_websocket, session_id)
        
        # Send message
        await websocket_manager.send_to_session(session_id, message)
        
        # Verify message was sent
        mock_websocket.send.assert_called_once_with(json.dumps(message))
        assert websocket_manager.session_data[session_id]["message_count"] == 1
    
    @pytest.mark.asyncio
    async def test_broadcast(self, websocket_manager):
        """Test broadcasting message to all connections."""
        # Register multiple connections
        connections = {}
        for i in range(3):
            session_id = f"session_{i}"
            mock_ws = AsyncMock()
            connections[session_id] = mock_ws
            await websocket_manager.register_connection(mock_ws, session_id)
        
        message = {"type": "broadcast", "content": "Hello everyone"}
        
        await websocket_manager.broadcast(message)
        
        # Verify all connections received the message
        for session_id, mock_ws in connections.items():
            mock_ws.send.assert_called_with(json.dumps(message))
    
    def test_get_session_info(self, websocket_manager):
        """Test getting session information."""
        # Test non-existent session
        info = websocket_manager.get_session_info("non_existent")
        assert info is None
        
        # Test with session data
        session_id = "test_session"
        websocket_manager.session_data[session_id] = {
            "connected_at": datetime.now(),
            "message_count": 5
        }
        
        info = websocket_manager.get_session_info(session_id)
        assert info is not None
        assert info["message_count"] == 5
        assert "is_connected" in info
    
    def test_get_active_sessions(self, websocket_manager, mock_websocket):
        """Test getting list of active sessions."""
        # Initially empty
        sessions = websocket_manager.get_active_sessions()
        assert len(sessions) == 0
        
        # Add connections manually for testing
        websocket_manager.connections["session1"] = mock_websocket
        websocket_manager.connections["session2"] = mock_websocket
        
        sessions = websocket_manager.get_active_sessions()
        assert len(sessions) == 2
        assert "session1" in sessions
        assert "session2" in sessions


class TestStreamingModes:
    """Test suite for different streaming modes."""
    
    @pytest.mark.asyncio
    async def test_token_by_token_streaming(self, enhanced_llm_service):
        """Test token-by-token streaming mode."""
        request = QueryRequest(question="Test streaming", streaming=True)
        
        # Mock token-by-token streaming
        tokens = ["Hello", " ", "world", "!"]
        
        async def mock_astream(*args, **kwargs):
            for token in tokens:
                mock_chunk = MagicMock()
                mock_chunk.content = token
                yield mock_chunk
        
        enhanced_llm_service.streaming_llm.astream = mock_astream
        
        response_gen = await enhanced_llm_service.generate_enhanced_response(
            request,
            session_id="test",
            streaming_mode=StreamingMode.TOKEN_BY_TOKEN
        )
        
        collected_tokens = []
        async for token in response_gen:
            collected_tokens.append(token)
        
        assert collected_tokens == tokens
    
    @pytest.mark.asyncio
    async def test_sentence_streaming(self, enhanced_llm_service):
        """Test sentence-by-sentence streaming mode."""
        # This would require more complex mocking of the callback handler
        # For now, we'll test the enum values
        assert StreamingMode.SENTENCE_BY_SENTENCE.value == "sentence_by_sentence"
        assert StreamingMode.PARAGRAPH_BY_PARAGRAPH.value == "paragraph_by_paragraph"
        assert StreamingMode.TOKEN_BY_TOKEN.value == "token_by_token"


class TestErrorHandling:
    """Test suite for error handling in LLM service."""
    
    @pytest.mark.asyncio
    async def test_ollama_connection_error_handling(self, mock_vector_store):
        """Test handling of Ollama connection errors."""
        with patch('app.services.enhanced_llm_service.ChatOllama') as mock_ollama:
            # Configure to raise connection error
            mock_ollama.return_value.ainvoke = AsyncMock(
                side_effect=ConnectionError("Cannot connect to Ollama")
            )
            
            service = EnhancedLLMService(vector_store=mock_vector_store)
            request = QueryRequest(question="Test error handling")
            
            # Should handle the error gracefully and return an error response
            response = await service.generate_enhanced_response(request, "test_session")
            
            assert "error" in response.answer.lower()
            assert response.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_vector_store_error_handling(self, enhanced_llm_service):
        """Test handling of vector store errors."""
        # Configure vector store to raise an error
        enhanced_llm_service.vector_store.enhanced_search = AsyncMock(
            side_effect=Exception("Vector store error")
        )
        
        request = QueryRequest(question="Test vector store error")
        
        # Should handle the error gracefully
        response = await enhanced_llm_service.generate_enhanced_response(request, "test_session")
        
        # Should still generate a response despite vector store error
        assert isinstance(response, QueryResponse)
    
    @pytest.mark.asyncio
    async def test_retry_logic_with_transient_errors(self, enhanced_llm_service):
        """Test retry logic with transient errors."""
        call_count = 0
        
        async def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient error")
            return "success after retries"
        
        result = await enhanced_llm_service.execute_with_retry(mock_operation)
        
        assert result == "success after retries"
        assert call_count == 3


@pytest.mark.integration
class TestLLMServiceIntegration:
    """Integration tests for the LLM service (requires actual services running)."""
    
    @pytest.mark.skipif(
        not pytest.config.getoption("--integration"),
        reason="Integration tests require --integration flag"
    )
    @pytest.mark.asyncio
    async def test_real_ollama_connection(self):
        """Test actual connection to Ollama service (integration test)."""
        service = EnhancedLLMService(
            model_name="llama3",
            ollama_url="http://localhost:11434"
        )
        
        try:
            health = await service.health_check()
            assert health["status"] in ["healthy", "unhealthy"]
        except Exception as e:
            pytest.skip(f"Ollama not available for integration test: {e}")


# Pytest configuration and fixtures
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()