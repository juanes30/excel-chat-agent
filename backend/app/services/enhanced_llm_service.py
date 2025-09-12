"""Enhanced LLM Service with advanced streaming capabilities and WebSocket integration."""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema.output_parser import OutputParserException
from pydantic import BaseModel, Field

from app.models.schemas import ChartType, ChartData
from app.services.llm_service import LangChainLLMService

logger = logging.getLogger(__name__)


class EnhancedStreamingCallback(BaseCallbackHandler):
    """Enhanced callback handler for streaming LLM responses with WebSocket support."""
    
    def __init__(self):
        self.tokens = []
        self.is_streaming = False
        self.websocket_callbacks = []
    
    def add_websocket_callback(self, callback):
        """Add a WebSocket callback for real-time streaming."""
        self.websocket_callbacks.append(callback)
    
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new token from LLM with WebSocket broadcasting."""
        self.tokens.append(token)
        self.is_streaming = True
        
        # Broadcast to WebSocket connections
        for callback in self.websocket_callbacks:
            try:
                await callback({"type": "token", "content": token})
            except Exception as e:
                logger.warning(f"WebSocket callback failed: {e}")
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Handle LLM completion."""
        self.is_streaming = False


class EnhancedLLMService(LangChainLLMService):
    """Enhanced LLM Service with advanced features and proper async support."""
    
    def __init__(self, 
                 model_name: str = "llama3",
                 ollama_url: str = "http://localhost:11434",
                 temperature: float = 0.7,
                 max_tokens: int = 2048,
                 vector_store=None,
                 enable_streaming: bool = True):
        """Initialize the Enhanced LLM Service.
        
        Args:
            model_name: Name of the Ollama model to use
            ollama_url: URL of the Ollama server
            temperature: Temperature for response generation
            max_tokens: Maximum tokens per response
            vector_store: Optional vector store for RAG functionality
            enable_streaming: Whether to enable streaming responses
        """
        # Initialize base service
        super().__init__(model_name, ollama_url, temperature, max_tokens)
        
        # Enhanced features
        self.vector_store = vector_store
        self.enable_streaming = enable_streaming
        self.streaming_callback = EnhancedStreamingCallback()
        self.conversation_history = {}
        self.performance_metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "average_response_time": 0,
            "error_count": 0
        }
        
        logger.info(f"Enhanced LLM Service initialized with model: {model_name}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Async health check for the Enhanced LLM service.
        
        Returns:
            Dictionary with health status and metrics
        """
        try:
            # Test LLM availability
            test_start = time.time()
            test_message = HumanMessage(content="Hello")
            
            # Use async invoke instead of sync
            response = await self.llm.ainvoke([test_message])
            response_time = int((time.time() - test_start) * 1000)
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "model_name": self.model_name,
                "ollama_url": self.ollama_url,
                "streaming_enabled": self.enable_streaming,
                "vector_store_connected": self.vector_store is not None,
                "total_requests": self.performance_metrics["total_requests"],
                "error_count": self.performance_metrics["error_count"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Enhanced LLM service health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_name": self.model_name,
                "ollama_url": self.ollama_url,
                "timestamp": datetime.now().isoformat()
            }
    
    async def generate_streaming_response(self, 
                                        messages: List[BaseMessage], 
                                        session_id: str = None) -> AsyncGenerator[str, None]:
        """Generate streaming response with enhanced features.
        
        Args:
            messages: List of messages for the conversation
            session_id: Optional session ID for conversation tracking
            
        Yields:
            Tokens as they are generated
        """
        try:
            self.performance_metrics["total_requests"] += 1
            start_time = time.time()
            
            # Add RAG context if vector store is available
            if self.vector_store and len(messages) > 0:
                last_message = messages[-1].content
                try:
                    # Search for relevant context
                    context_results = await self.vector_store.asearch(last_message, k=3)
                    if context_results:
                        context = "\n".join([doc.page_content for doc in context_results])
                        enhanced_message = f"Context: {context}\n\nUser Query: {last_message}"
                        messages[-1].content = enhanced_message
                except Exception as e:
                    logger.warning(f"RAG context retrieval failed: {e}")
            
            # Use streaming callback
            self.streaming_callback.tokens = []
            
            # Generate response with streaming
            async for token in self.llm.astream(messages):
                token_content = token.content if hasattr(token, 'content') else str(token)
                self.streaming_callback.tokens.append(token_content)
                yield token_content
            
            # Update metrics
            response_time = time.time() - start_time
            self.performance_metrics["total_tokens"] += len(self.streaming_callback.tokens)
            self.performance_metrics["average_response_time"] = (
                (self.performance_metrics["average_response_time"] * (self.performance_metrics["total_requests"] - 1) + response_time) / 
                self.performance_metrics["total_requests"]
            )
            
            # Store conversation history
            if session_id:
                if session_id not in self.conversation_history:
                    self.conversation_history[session_id] = []
                self.conversation_history[session_id].extend(messages)
                self.conversation_history[session_id].append(
                    AIMessage(content="".join(self.streaming_callback.tokens))
                )
                
        except Exception as e:
            self.performance_metrics["error_count"] += 1
            logger.error(f"Enhanced streaming response failed: {e}")
            yield f"Error: {str(e)}"
    
    async def analyze_excel_query(self, 
                                query: str, 
                                excel_context: Dict[str, Any] = None,
                                session_id: str = None) -> Dict[str, Any]:
        """Analyze Excel-related query with enhanced context understanding.
        
        Args:
            query: User query about Excel data
            excel_context: Context from Excel processor
            session_id: Optional session ID
            
        Returns:
            Analysis results with recommendations
        """
        try:
            # Build enhanced prompt with Excel context
            excel_prompt = self._build_excel_analysis_prompt(query, excel_context)
            messages = [HumanMessage(content=excel_prompt)]
            
            # Generate response
            response_tokens = []
            async for token in self.generate_streaming_response(messages, session_id):
                response_tokens.append(token)
            
            response_text = "".join(response_tokens)
            
            return {
                "query": query,
                "response": response_text,
                "context_used": excel_context is not None,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Excel query analysis failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _build_excel_analysis_prompt(self, query: str, excel_context: Dict[str, Any] = None) -> str:
        """Build enhanced prompt for Excel data analysis."""
        base_prompt = f"""You are an expert Excel data analyst. Please analyze the following query and provide detailed insights.

User Query: {query}"""
        
        if excel_context:
            context_info = f"""
            
Available Excel Data Context:
- Files: {excel_context.get('files', [])}
- Total rows: {excel_context.get('total_rows', 0)}
- Total columns: {excel_context.get('total_columns', 0)}
- Column names: {excel_context.get('columns', [])}
- Data types: {excel_context.get('data_types', {})}
- Sample data: {excel_context.get('sample_data', {})}"""
            
            base_prompt += context_info
        
        base_prompt += """

Please provide:
1. A clear analysis of the query
2. Specific insights based on the available data
3. Any recommendations for further analysis
4. Suggest visualizations if appropriate

Be concise but comprehensive in your response."""
        
        return base_prompt
    
    def get_conversation_history(self, session_id: str) -> List[BaseMessage]:
        """Get conversation history for a session."""
        return self.conversation_history.get(session_id, [])
    
    def clear_conversation_history(self, session_id: str = None):
        """Clear conversation history for a session or all sessions."""
        if session_id:
            self.conversation_history.pop(session_id, None)
        else:
            self.conversation_history.clear()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the service."""
        return {
            **self.performance_metrics,
            "active_conversations": len(self.conversation_history),
            "uptime_hours": (datetime.now() - datetime.now()).total_seconds() / 3600  # Placeholder
        }
    
    async def shutdown(self):
        """Gracefully shutdown the enhanced service."""
        logger.info("Shutting down Enhanced LLM Service...")
        self.clear_conversation_history()
        # Additional cleanup if needed