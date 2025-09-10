"""LangChain LLM Service with Ollama integration for Excel data analysis."""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema.output_parser import OutputParserException
from pydantic import BaseModel, Field

from app.models.schemas import ChartType, ChartData

logger = logging.getLogger(__name__)


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses."""
    
    def __init__(self):
        self.tokens = []
        self.is_streaming = False
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new token from LLM."""
        self.tokens.append(token)
        self.is_streaming = True
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Handle LLM completion."""
        self.is_streaming = False


class ChartRecommendation(BaseModel):
    """Structured output for chart recommendations."""
    recommended: bool = Field(description="Whether a chart is recommended")
    chart_type: Optional[str] = Field(description="Type of chart recommended")
    reasoning: str = Field(description="Reasoning for the recommendation")
    data_columns: List[str] = Field(default_factory=list, description="Columns to use for the chart")


class LangChainLLMService:
    """Service for LLM interactions using LangChain with Ollama."""

    def __init__(self, 
                 model_name: str = "llama3",
                 ollama_url: str = "http://localhost:11434",
                 temperature: float = 0.7,
                 max_tokens: int = 2048):
        """Initialize the LangChain LLM Service.
        
        Args:
            model_name: Name of the Ollama model to use
            ollama_url: URL of the Ollama server
            temperature: Temperature for text generation
            max_tokens: Maximum tokens for generation
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize Ollama LLM
        self.llm = ChatOllama(
            model=model_name,
            base_url=ollama_url,
            temperature=temperature,
            num_predict=max_tokens,
            verbose=True
        )
        
        # Initialize streaming LLM for real-time responses
        self.streaming_llm = ChatOllama(
            model=model_name,
            base_url=ollama_url,
            temperature=temperature,
            num_predict=max_tokens,
            streaming=True,
            verbose=True
        )
        
        # Conversation memory with window
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Response cache with TTL
        self._response_cache = {}
        self._cache_ttl = timedelta(minutes=30)
        
        # Initialize prompt templates
        self._setup_prompt_templates()
        
        logger.info(f"Initialized LangChain LLM Service with model: {model_name}")

    def _setup_prompt_templates(self):
        """Set up various prompt templates for different use cases."""
        
        # Data analysis prompt
        self.data_analysis_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert data analyst helping users understand Excel data. 
            You have access to processed Excel file information and should provide clear, accurate analysis.
            
            Guidelines:
            - Be specific and reference actual data from the files
            - Provide actionable insights
            - Mention data sources (file and sheet names)
            - Suggest visualizations when appropriate
            - If data is unclear, ask clarifying questions
            
            Available data context: {context}"""),
            ("human", "{question}")
        ])
        
        # SQL query generation prompt
        self.sql_query_template = PromptTemplate(
            input_variables=["question", "schema_info"],
            template="""Based on the following Excel data schema, generate a SQL-like query to answer the question.
            
            Schema Information:
            {schema_info}
            
            Question: {question}
            
            Provide a clear explanation of what data would be needed to answer this question.
            If the question cannot be answered with the available data, explain what additional data would be required.
            
            Query/Analysis:"""
        )
        
        # Summary generation prompt
        self.summary_template = PromptTemplate(
            input_variables=["data_info", "focus"],
            template="""Summarize the following Excel data information with a focus on {focus}:
            
            Data Information:
            {data_info}
            
            Provide a concise summary highlighting:
            1. Key characteristics of the data
            2. Notable patterns or trends
            3. Data quality observations
            4. Potential insights or areas of interest
            
            Summary:"""
        )
        
        # Chart recommendation prompt
        self.chart_template = PromptTemplate(
            input_variables=["question", "data_description", "column_info"],
            template="""Based on the user's question and data description, determine if a chart would be helpful and recommend the best chart type.
            
            Question: {question}
            
            Data Description: {data_description}
            
            Available Columns: {column_info}
            
            Consider the following chart types: line, bar, pie, scatter, heatmap, histogram, box
            
            Respond with JSON in this format:
            {{
                "recommended": true/false,
                "chart_type": "chart_type_name",
                "reasoning": "explanation of why this chart is recommended",
                "data_columns": ["column1", "column2"]
            }}
            
            Response:"""
        )
        
        # Comparison prompt
        self.comparison_template = ChatPromptTemplate.from_messages([
            ("system", """You are helping users compare different aspects of their Excel data.
            Focus on quantitative comparisons and highlight significant differences or similarities.
            
            Available data: {context}"""),
            ("human", "Compare the following: {comparison_request}")
        ])

    def _get_cache_key(self, query: str, context: str) -> str:
        """Generate cache key for a query and context."""
        combined = f"{query}_{context}"
        return str(hash(combined))

    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        if 'timestamp' not in cache_entry:
            return False
        
        cache_time = cache_entry['timestamp']
        return datetime.now() - cache_time < self._cache_ttl

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if valid."""
        if cache_key in self._response_cache:
            entry = self._response_cache[cache_key]
            if self._is_cache_valid(entry):
                logger.debug("Cache hit for query")
                return entry['response']
            else:
                # Remove expired cache entry
                del self._response_cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: str):
        """Cache a response."""
        self._response_cache[cache_key] = {
            'response': response,
            'timestamp': datetime.now()
        }
        
        # Clean up old cache entries (keep only last 100)
        if len(self._response_cache) > 100:
            oldest_key = min(self._response_cache.keys(), 
                           key=lambda k: self._response_cache[k]['timestamp'])
            del self._response_cache[oldest_key]

    async def generate_response(self, 
                              question: str,
                              context: str,
                              intent: str = "data_analysis",
                              use_cache: bool = True) -> Dict[str, Any]:
        """Generate a response using the appropriate prompt template.
        
        Args:
            question: User's question
            context: Data context from vector search
            intent: Type of query (data_analysis, summary, comparison, etc.)
            use_cache: Whether to use response caching
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(question, context) if use_cache else None
            if cache_key:
                cached_response = self._get_cached_response(cache_key)
                if cached_response:
                    return {
                        "answer": cached_response,
                        "processing_time_ms": int((time.time() - start_time) * 1000),
                        "cached": True,
                        "intent": intent
                    }
            
            # Select appropriate prompt template
            if intent == "data_analysis":
                prompt = self.data_analysis_template.format_messages(
                    context=context,
                    question=question
                )
            elif intent == "summary":
                prompt = self.summary_template.format(
                    data_info=context,
                    focus=question
                )
                prompt = [HumanMessage(content=prompt)]
            elif intent == "comparison":
                prompt = self.comparison_template.format_messages(
                    context=context,
                    comparison_request=question
                )
            else:
                # Default to data analysis
                prompt = self.data_analysis_template.format_messages(
                    context=context,
                    question=question
                )
            
            # Generate response
            response = await self.llm.ainvoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Cache the response
            if cache_key:
                self._cache_response(cache_key, answer)
            
            # Add to conversation memory
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(answer)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                "answer": answer,
                "processing_time_ms": processing_time,
                "cached": False,
                "intent": intent,
                "tokens_used": len(answer.split())  # Rough estimation
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "answer": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "cached": False,
                "intent": intent,
                "error": True
            }

    async def generate_streaming_response(self, 
                                        question: str,
                                        context: str,
                                        intent: str = "data_analysis") -> AsyncGenerator[str, None]:
        """Generate a streaming response.
        
        Args:
            question: User's question
            context: Data context from vector search
            intent: Type of query
            
        Yields:
            Token strings as they are generated
        """
        try:
            # Setup callback handler for streaming
            callback_handler = StreamingCallbackHandler()
            
            # Select appropriate prompt template
            if intent == "data_analysis":
                prompt = self.data_analysis_template.format_messages(
                    context=context,
                    question=question
                )
            else:
                # Use default template for other intents in streaming
                prompt = self.data_analysis_template.format_messages(
                    context=context,
                    question=question
                )
            
            # Generate streaming response
            response_text = ""
            async for chunk in self.streaming_llm.astream(prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    response_text += chunk.content
                    yield chunk.content
            
            # Add to conversation memory after streaming is complete
            if response_text:
                self.memory.chat_memory.add_user_message(question)
                self.memory.chat_memory.add_ai_message(response_text)
            
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield f"Error: {str(e)}"

    async def recommend_chart(self, 
                            question: str,
                            data_description: str,
                            column_info: List[str]) -> Optional[ChartData]:
        """Recommend a chart based on the question and data.
        
        Args:
            question: User's question
            data_description: Description of the data
            column_info: List of available columns
            
        Returns:
            ChartData if a chart is recommended, None otherwise
        """
        try:
            # Generate chart recommendation
            prompt = self.chart_template.format(
                question=question,
                data_description=data_description,
                column_info=", ".join(column_info)
            )
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse JSON response
            try:
                # Extract JSON from response if it's embedded in text
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    recommendation = json.loads(json_str)
                    
                    if recommendation.get('recommended', False):
                        chart_type = recommendation.get('chart_type', 'bar')
                        
                        # Validate chart type
                        try:
                            validated_chart_type = ChartType(chart_type)
                        except ValueError:
                            validated_chart_type = ChartType.BAR
                        
                        return ChartData(
                            type=validated_chart_type,
                            data=[],  # Will be populated by the calling function
                            title=f"Chart for: {question}",
                            description=recommendation.get('reasoning', ''),
                            config={
                                "recommended_columns": recommendation.get('data_columns', [])
                            }
                        )
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse chart recommendation JSON: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating chart recommendation: {e}")
            return None

    async def analyze_data_request(self, question: str) -> Dict[str, Any]:
        """Analyze the user's request to determine intent and requirements.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with analysis results
        """
        try:
            analysis_prompt = f"""Analyze the following question about Excel data and determine:
            1. The main intent (data_analysis, summary, comparison, calculation, etc.)
            2. What type of data would be needed to answer it
            3. Whether it's asking for specific values or general insights
            4. Keywords that would be useful for searching the data
            
            Question: {question}
            
            Respond with a brief analysis focusing on these aspects."""
            
            response = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
            analysis_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract keywords from the original question (simple approach)
            keywords = []
            question_lower = question.lower()
            
            # Common data-related keywords
            data_keywords = ['sales', 'revenue', 'profit', 'cost', 'price', 'quantity', 
                           'date', 'month', 'year', 'customer', 'product', 'category',
                           'sum', 'average', 'total', 'count', 'maximum', 'minimum']
            
            for keyword in data_keywords:
                if keyword in question_lower:
                    keywords.append(keyword)
            
            # Determine intent based on question patterns
            intent = "data_analysis"  # default
            if any(word in question_lower for word in ['compare', 'vs', 'versus', 'difference']):
                intent = "comparison"
            elif any(word in question_lower for word in ['summary', 'overview', 'describe']):
                intent = "summary"
            elif any(word in question_lower for word in ['calculate', 'compute', 'total', 'sum']):
                intent = "calculation"
            
            return {
                "intent": intent,
                "analysis": analysis_text,
                "keywords": keywords,
                "requires_specific_data": any(word in question_lower for word in 
                                            ['which', 'what', 'how many', 'how much', 'when']),
                "is_comparative": 'compare' in question_lower or 'vs' in question_lower
            }
            
        except Exception as e:
            logger.error(f"Error analyzing data request: {e}")
            return {
                "intent": "data_analysis",
                "analysis": "Could not analyze the request",
                "keywords": [],
                "requires_specific_data": True,
                "is_comparative": False
            }

    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history.
        
        Args:
            limit: Maximum number of messages to return
            
        Returns:
            List of conversation messages
        """
        try:
            messages = self.memory.chat_memory.messages[-limit*2:]  # *2 because of user/ai pairs
            
            formatted_messages = []
            for i, message in enumerate(messages):
                if isinstance(message, HumanMessage):
                    role = "user"
                elif isinstance(message, AIMessage):
                    role = "assistant"
                elif isinstance(message, SystemMessage):
                    role = "system"
                else:
                    role = "unknown"
                
                formatted_messages.append({
                    "role": role,
                    "content": message.content,
                    "timestamp": datetime.now(),  # We don't store timestamps in this simple implementation
                    "index": i
                })
            
            return formatted_messages
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

    def clear_conversation_history(self):
        """Clear the conversation memory."""
        self.memory.clear()
        logger.info("Cleared conversation history")

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the LLM service.
        
        Returns:
            Dictionary with health status
        """
        try:
            # Try to generate a simple response
            test_start = time.time()
            asyncio.run(self.llm.ainvoke([HumanMessage(content="Hello")]))
            response_time = int((time.time() - test_start) * 1000)
            
            return {
                "status": "healthy",
                "model_name": self.model_name,
                "ollama_url": self.ollama_url,
                "response_time_ms": response_time,
                "cache_size": len(self._response_cache),
                "conversation_length": len(self.memory.chat_memory.messages)
            }
            
        except Exception as e:
            logger.error(f"LLM service health check failed: {e}")
            return {
                "status": "unhealthy",
                "model_name": self.model_name,
                "ollama_url": self.ollama_url,
                "error": str(e),
                "cache_size": len(self._response_cache),
                "conversation_length": 0
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics.
        
        Returns:
            Dictionary with service statistics
        """
        return {
            "model_name": self.model_name,
            "ollama_url": self.ollama_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "cache_size": len(self._response_cache),
            "cache_ttl_minutes": int(self._cache_ttl.total_seconds() / 60),
            "memory_window_size": self.memory.k,
            "conversation_length": len(self.memory.chat_memory.messages)
        }