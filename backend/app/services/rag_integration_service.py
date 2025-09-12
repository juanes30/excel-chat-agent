"""RAG Integration Service connecting Enhanced LLM Service with Enhanced Vector Store V2."""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

from app.services.enhanced_llm_service import EnhancedLLMService
from langchain.schema import HumanMessage
from app.services.enhanced_vector_store_v2 import EnhancedVectorStoreV2, MultiModalSearchResult
from app.services.enhanced_embedding_strategy import ContentType, EmbeddingModel
from app.models.schemas import QueryRequest, QueryResponse, ChartData
from app.utils.error_handling import (
    with_error_handling, 
    ErrorContext, 
    LLMServiceError, 
    VectorStoreError,
    VECTOR_STORE_RETRY_CONFIG,
    OLLAMA_RETRY_CONFIG
)

logger = logging.getLogger(__name__)


class RAGStrategy(str, Enum):
    """RAG retrieval strategies."""
    STANDARD = "standard"
    MULTI_MODAL = "multi_modal"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"
    BUSINESS_CONTEXT = "business_context"


class ContextRelevanceLevel(str, Enum):
    """Context relevance levels for filtering."""
    HIGH = "high"        # > 0.8
    MEDIUM = "medium"    # 0.5 - 0.8
    LOW = "low"          # 0.3 - 0.5
    MINIMAL = "minimal"  # < 0.3


@dataclass
class RAGContext:
    """Enhanced context for RAG operations."""
    search_results: List[MultiModalSearchResult]
    content_types: Dict[str, int]
    business_domains: List[str]
    hierarchical_levels: List[str]
    confidence_scores: List[float]
    processing_time_ms: float
    search_strategy: RAGStrategy
    total_results: int
    filtered_results: int


@dataclass
class ContextualizedPrompt:
    """Contextualized prompt with enhanced metadata."""
    prompt: str
    context_summary: str
    source_attribution: List[str]
    content_types: List[str]
    confidence_level: float
    token_count: int


class RAGIntegrationService:
    """Service integrating Enhanced LLM with Enhanced Vector Store for sophisticated RAG."""
    
    def __init__(self, 
                 llm_service: EnhancedLLMService,
                 vector_store: EnhancedVectorStoreV2,
                 default_strategy: RAGStrategy = RAGStrategy.ADAPTIVE,
                 relevance_threshold: float = 0.3,
                 max_context_tokens: int = 4000):
        """Initialize RAG Integration Service.
        
        Args:
            llm_service: Enhanced LLM service instance
            vector_store: Enhanced vector store instance
            default_strategy: Default RAG strategy
            relevance_threshold: Minimum relevance score for context inclusion
            max_context_tokens: Maximum tokens for context window
        """
        self.llm_service = llm_service
        self.vector_store = vector_store
        self.default_strategy = default_strategy
        self.relevance_threshold = relevance_threshold
        self.max_context_tokens = max_context_tokens
        
        # Performance tracking
        self.query_stats = {
            "total_queries": 0,
            "avg_retrieval_time": 0.0,
            "avg_generation_time": 0.0,
            "cache_hits": 0,
            "context_overflow_count": 0
        }
        
        # Strategy configurations
        self.strategy_configs = {
            RAGStrategy.STANDARD: {
                "search_type": "semantic",
                "include_statistics": False,
                "content_type_weights": {"textual": 1.0, "numerical": 0.7, "hierarchical": 0.5}
            },
            RAGStrategy.MULTI_MODAL: {
                "search_type": "multi_modal",
                "include_statistics": True,
                "content_type_weights": {"textual": 1.0, "numerical": 1.0, "hierarchical": 0.8}
            },
            RAGStrategy.ADAPTIVE: {
                "search_type": "adaptive",
                "include_statistics": True,
                "content_type_weights": {"textual": 1.0, "numerical": 0.9, "hierarchical": 0.7}
            },
            RAGStrategy.HIERARCHICAL: {
                "search_type": "hierarchical",
                "include_statistics": True,
                "content_type_weights": {"hierarchical": 1.0, "textual": 0.8, "numerical": 0.6}
            },
            RAGStrategy.BUSINESS_CONTEXT: {
                "search_type": "business",
                "include_statistics": True,
                "content_type_weights": {"textual": 1.0, "numerical": 0.9, "business": 1.0}
            }
        }
        
        logger.info("RAG Integration Service initialized")

    @with_error_handling(operation="determine_rag_strategy")
    async def determine_optimal_strategy(self, 
                                       query_request: QueryRequest,
                                       analysis: Dict[str, Any]) -> RAGStrategy:
        """Determine optimal RAG strategy based on query analysis.
        
        Args:
            query_request: Original query request
            analysis: Query analysis from LLM service
            
        Returns:
            Optimal RAG strategy
        """
        intent = analysis.get("intent", "data_analysis")
        complexity = analysis.get("complexity", "moderate")
        business_domain = analysis.get("business_domain", "general")
        keywords = analysis.get("keywords", [])
        
        # Strategy selection logic
        if intent == "comparison" and complexity == "complex":
            return RAGStrategy.MULTI_MODAL
        
        elif "hierarchical" in keywords or "structure" in query_request.question.lower():
            return RAGStrategy.HIERARCHICAL
        
        elif business_domain != "general" or any(biz in keywords for biz in ["sales", "finance", "operations"]):
            return RAGStrategy.BUSINESS_CONTEXT
        
        elif complexity == "complex" or query_request.include_statistics:
            return RAGStrategy.MULTI_MODAL
        
        else:
            return RAGStrategy.ADAPTIVE

    @with_error_handling(
        operation="enhanced_retrieval", 
        retry_config=VECTOR_STORE_RETRY_CONFIG
    )
    async def enhanced_retrieval(self, 
                               query_request: QueryRequest,
                               strategy: RAGStrategy,
                               analysis: Dict[str, Any]) -> RAGContext:
        """Perform enhanced retrieval using specified strategy.
        
        Args:
            query_request: Query request
            strategy: RAG strategy to use
            analysis: Query analysis
            
        Returns:
            RAG context with retrieval results
        """
        start_time = time.time()
        
        try:
            # Get strategy configuration
            config = self.strategy_configs.get(strategy, self.strategy_configs[RAGStrategy.ADAPTIVE])
            
            # Enhanced vector search
            search_results = await self.vector_store.enhanced_search(
                query=query_request.question,
                n_results=query_request.max_results * 2,  # Get more for filtering
                file_filter=query_request.file_filter,
                sheet_filter=query_request.sheet_filter,
                search_type=config["search_type"],
                include_statistics=config["include_statistics"]
            )
            
            # Convert to MultiModalSearchResult if needed
            enhanced_results = []
            for result in search_results:
                if isinstance(result, dict):
                    enhanced_result = MultiModalSearchResult(
                        content=result.get("content", ""),
                        metadata=result.get("metadata", {}),
                        relevance_score=result.get("relevance_score", 0.0),
                        content_type=self._extract_content_type(result),
                        embedding_strategy=result.get("embedding_strategy"),
                        numerical_features=result.get("numerical_features"),
                        business_context=result.get("business_context"),
                        hierarchy_features=result.get("hierarchy_features")
                    )
                else:
                    enhanced_result = result
                
                enhanced_results.append(enhanced_result)
            
            # Filter by relevance and content type weights
            filtered_results = await self._filter_and_weight_results(
                enhanced_results, 
                config["content_type_weights"],
                analysis
            )
            
            # Limit to max results after filtering
            final_results = filtered_results[:query_request.max_results]
            
            # Analyze retrieved context
            content_types = {}
            business_domains = []
            hierarchical_levels = []
            confidence_scores = []
            
            for result in final_results:
                # Content type analysis
                content_type = result.content_type.value if result.content_type else "unknown"
                content_types[content_type] = content_types.get(content_type, 0) + 1
                
                # Business context extraction
                if result.business_context:
                    domain = result.business_context.get("domain", "general")
                    if domain not in business_domains:
                        business_domains.append(domain)
                
                # Hierarchical level extraction
                if result.hierarchy_features:
                    level = result.hierarchy_features.get("level", "cell")
                    if level not in hierarchical_levels:
                        hierarchical_levels.append(level)
                
                confidence_scores.append(result.relevance_score)
            
            processing_time = (time.time() - start_time) * 1000
            
            return RAGContext(
                search_results=final_results,
                content_types=content_types,
                business_domains=business_domains,
                hierarchical_levels=hierarchical_levels,
                confidence_scores=confidence_scores,
                processing_time_ms=processing_time,
                search_strategy=strategy,
                total_results=len(enhanced_results),
                filtered_results=len(final_results)
            )
            
        except Exception as e:
            logger.error(f"Enhanced retrieval failed: {e}")
            raise VectorStoreError(
                f"Failed to retrieve enhanced context: {str(e)}",
                context=ErrorContext(
                    operation="enhanced_retrieval",
                    additional_data={"strategy": strategy.value, "query": query_request.question}
                )
            )

    async def _filter_and_weight_results(self, 
                                       results: List[MultiModalSearchResult],
                                       content_weights: Dict[str, float],
                                       analysis: Dict[str, Any]) -> List[MultiModalSearchResult]:
        """Filter and weight results based on strategy and query analysis."""
        
        weighted_results = []
        
        for result in results:
            # Base relevance score
            score = result.relevance_score
            
            # Apply content type weighting
            content_type = result.content_type.value if result.content_type else "textual"
            weight = content_weights.get(content_type, 0.5)
            weighted_score = score * weight
            
            # Boost score based on query analysis
            if analysis.get("requires_specific_data", False):
                if result.numerical_features or "data" in result.content.lower():
                    weighted_score *= 1.2
            
            if analysis.get("is_comparative", False):
                if "compare" in result.content.lower() or len(result.metadata.get("columns", [])) > 1:
                    weighted_score *= 1.1
            
            # Business domain matching
            business_domain = analysis.get("business_domain", "general")
            if result.business_context and result.business_context.get("domain") == business_domain:
                weighted_score *= 1.15
            
            # Only include results above threshold
            if weighted_score >= self.relevance_threshold:
                # Update the relevance score with weighted version
                result.relevance_score = weighted_score
                weighted_results.append(result)
        
        # Sort by weighted relevance score
        weighted_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return weighted_results

    def _extract_content_type(self, result: Dict[str, Any]) -> Optional[ContentType]:
        """Extract content type from search result."""
        metadata = result.get("metadata", {})
        content_type_str = metadata.get("content_type")
        
        if content_type_str:
            try:
                return ContentType(content_type_str)
            except ValueError:
                pass
        
        # Infer from content
        content = result.get("content", "").lower()
        if any(indicator in content for indicator in ["number", "value", "amount", "count"]):
            return ContentType.NUMERICAL
        elif any(indicator in content for indicator in ["file", "sheet", "column", "row"]):
            return ContentType.HIERARCHICAL
        else:
            return ContentType.TEXTUAL

    @with_error_handling(operation="contextualize_prompt")
    async def contextualize_prompt(self, 
                                 query_request: QueryRequest,
                                 rag_context: RAGContext,
                                 analysis: Dict[str, Any]) -> ContextualizedPrompt:
        """Create contextualized prompt from RAG context.
        
        Args:
            query_request: Original query request
            rag_context: Retrieved RAG context
            analysis: Query analysis
            
        Returns:
            Contextualized prompt ready for LLM
        """
        try:
            # Build context summary
            context_parts = []
            source_attribution = []
            content_types = []
            
            token_count = 0
            max_tokens = self.max_context_tokens
            
            for idx, result in enumerate(rag_context.search_results):
                if token_count >= max_tokens:
                    break
                
                # Estimate token count (rough approximation)
                content_tokens = len(result.content.split()) * 1.3  # Account for tokenization
                if token_count + content_tokens > max_tokens:
                    # Truncate content to fit
                    remaining_tokens = max_tokens - token_count
                    words_to_take = int(remaining_tokens / 1.3)
                    truncated_content = " ".join(result.content.split()[:words_to_take]) + "..."
                    content_tokens = remaining_tokens
                else:
                    truncated_content = result.content
                
                # Build context entry
                metadata = result.metadata
                source_info = f"{metadata.get('file_name', 'Unknown')} → {metadata.get('sheet_name', 'Unknown')}"
                
                context_entry = f"[Source {idx+1}: {source_info}]\n{truncated_content}"
                
                # Add content type and relevance information
                if result.content_type:
                    context_entry += f"\n(Content Type: {result.content_type.value}, Relevance: {result.relevance_score:.2f})"
                
                # Add numerical features if available
                if result.numerical_features:
                    stats = []
                    for key, value in result.numerical_features.items():
                        if isinstance(value, (int, float)):
                            stats.append(f"{key}: {value:.2f}")
                    if stats:
                        context_entry += f"\n(Statistics: {', '.join(stats)})"
                
                context_parts.append(context_entry)
                source_attribution.append(source_info)
                
                if result.content_type and result.content_type.value not in content_types:
                    content_types.append(result.content_type.value)
                
                token_count += content_tokens
            
            # Build comprehensive context summary
            context_summary = "\n\n".join(context_parts)
            
            # Calculate confidence level
            if rag_context.confidence_scores:
                avg_confidence = sum(rag_context.confidence_scores) / len(rag_context.confidence_scores)
            else:
                avg_confidence = 0.0
            
            # Create enhanced prompt based on intent
            intent = analysis.get("intent", "data_analysis")
            
            if intent == "comparison":
                prompt_template = """Based on the following Excel data context, provide a detailed comparison analysis:

Context Data:
{context}

Question: {question}

Please provide:
1. Direct comparison with specific data points
2. Key differences and similarities
3. Statistical insights if available
4. Business implications
5. Visualization recommendations if appropriate

Focus on quantitative analysis and cite specific sources."""
            
            elif intent == "summary":
                prompt_template = """Based on the following Excel data context, provide a comprehensive summary:

Context Data:
{context}

Question: {question}

Please provide:
1. Key highlights and main findings
2. Data patterns and trends
3. Notable statistics
4. Areas requiring attention
5. Summary insights

Be concise but thorough, citing specific data sources."""
            
            else:  # Default data analysis
                prompt_template = """Based on the following Excel data context, analyze and answer the question:

Context Data:
{context}

Question: {question}

Please provide:
1. Direct answer based on the available data
2. Supporting evidence from the context
3. Relevant insights and patterns
4. Any limitations of the analysis
5. Recommendations for further analysis if needed

Be specific and reference the data sources."""
            
            # Format the prompt
            formatted_prompt = prompt_template.format(
                context=context_summary,
                question=query_request.question
            )
            
            return ContextualizedPrompt(
                prompt=formatted_prompt,
                context_summary=context_summary,
                source_attribution=source_attribution,
                content_types=content_types,
                confidence_level=avg_confidence,
                token_count=token_count
            )
            
        except Exception as e:
            logger.error(f"Prompt contextualization failed: {e}")
            # Return basic prompt as fallback
            return ContextualizedPrompt(
                prompt=f"Question: {query_request.question}\n\nPlease provide an analysis based on available Excel data.",
                context_summary="Context retrieval failed",
                source_attribution=[],
                content_types=[],
                confidence_level=0.0,
                token_count=50
            )

    @with_error_handling(
        operation="rag_enhanced_query",
        retry_config=OLLAMA_RETRY_CONFIG
    )
    async def process_rag_enhanced_query(self, 
                                       query_request: QueryRequest,
                                       session_id: Optional[str] = None,
                                       streaming_mode: StreamingMode = StreamingMode.TOKEN_BY_TOKEN) -> Union[QueryResponse, AsyncGenerator[str, None]]:
        """Process query with full RAG enhancement.
        
        Args:
            query_request: Original query request
            session_id: Session identifier for conversation context
            streaming_mode: Streaming mode for response
            
        Returns:
            Enhanced query response or streaming generator
        """
        start_time = time.time()
        self.query_stats["total_queries"] += 1
        
        try:
            # Step 1: Analyze the query
            analysis = await self.llm_service.analyze_data_request(query_request.question)
            logger.info(f"Query analysis: {analysis['intent']} ({analysis['complexity']})")
            
            # Step 2: Determine optimal RAG strategy
            strategy = await self.determine_optimal_strategy(query_request, analysis)
            logger.info(f"Selected RAG strategy: {strategy.value}")
            
            # Step 3: Enhanced retrieval
            retrieval_start = time.time()
            rag_context = await self.enhanced_retrieval(query_request, strategy, analysis)
            retrieval_time = (time.time() - retrieval_start) * 1000
            
            logger.info(f"Retrieved {len(rag_context.search_results)} results in {retrieval_time:.0f}ms")
            
            # Step 4: Contextualize prompt
            contextualized_prompt = await self.contextualize_prompt(query_request, rag_context, analysis)
            
            # Check for context overflow
            if contextualized_prompt.token_count >= self.max_context_tokens:
                self.query_stats["context_overflow_count"] += 1
                logger.warning(f"Context overflow: {contextualized_prompt.token_count} tokens")
            
            # Step 5: Generate enhanced response
            generation_start = time.time()
            
            # Create enhanced query request for LLM service
            enhanced_request = QueryRequest(
                question=contextualized_prompt.prompt,
                file_filter=query_request.file_filter,
                sheet_filter=query_request.sheet_filter,
                max_results=query_request.max_results,
                include_statistics=query_request.include_statistics,
                streaming=query_request.streaming
            )
            
            if query_request.streaming and session_id:
                # Streaming response
                response_gen = await self.llm_service.generate_enhanced_response(
                    enhanced_request, session_id, streaming_mode
                )
                
                # Wrap generator to add RAG metadata
                async def enhanced_streaming_generator():
                    full_response = ""
                    async for chunk in response_gen:
                        full_response += chunk
                        yield chunk
                    
                    # Update statistics
                    generation_time = (time.time() - generation_start) * 1000
                    self.query_stats["avg_retrieval_time"] = (
                        (self.query_stats["avg_retrieval_time"] * (self.query_stats["total_queries"] - 1) + retrieval_time) 
                        / self.query_stats["total_queries"]
                    )
                    self.query_stats["avg_generation_time"] = (
                        (self.query_stats["avg_generation_time"] * (self.query_stats["total_queries"] - 1) + generation_time) 
                        / self.query_stats["total_queries"]
                    )
                
                return enhanced_streaming_generator()
            
            else:
                # Standard response
                response = await self.llm_service.generate_enhanced_response(
                    enhanced_request, session_id
                )
                
                generation_time = (time.time() - generation_start) * 1000
                total_time = (time.time() - start_time) * 1000
                
                # Enhance response with RAG metadata
                if isinstance(response, QueryResponse):
                    response.sources = contextualized_prompt.source_attribution[:5]  # Top 5 sources
                    response.confidence = contextualized_prompt.confidence_level
                    response.processing_time_ms = int(total_time)
                    
                    # Add RAG-specific metadata to a custom field if available
                    if hasattr(response, 'metadata'):
                        response.metadata = {
                            "rag_strategy": strategy.value,
                            "content_types": contextualized_prompt.content_types,
                            "context_token_count": contextualized_prompt.token_count,
                            "retrieval_time_ms": retrieval_time,
                            "generation_time_ms": generation_time,
                            "filtered_results": rag_context.filtered_results,
                            "total_candidates": rag_context.total_results
                        }
                
                # Update statistics
                self.query_stats["avg_retrieval_time"] = (
                    (self.query_stats["avg_retrieval_time"] * (self.query_stats["total_queries"] - 1) + retrieval_time) 
                    / self.query_stats["total_queries"]
                )
                self.query_stats["avg_generation_time"] = (
                    (self.query_stats["avg_generation_time"] * (self.query_stats["total_queries"] - 1) + generation_time) 
                    / self.query_stats["total_queries"]
                )
                
                logger.info(f"RAG query completed in {total_time:.0f}ms (retrieval: {retrieval_time:.0f}ms, generation: {generation_time:.0f}ms)")
                
                return response
                
        except Exception as e:
            logger.error(f"RAG enhanced query failed: {e}")
            
            # Fallback to basic LLM service
            try:
                fallback_response = await self.llm_service.generate_enhanced_response(
                    query_request, session_id, streaming_mode
                )
                
                if isinstance(fallback_response, QueryResponse):
                    fallback_response.sources = ["Fallback - RAG unavailable"]
                    fallback_response.confidence = 0.3
                
                return fallback_response
                
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise LLMServiceError(
                    f"Both RAG and fallback processing failed: {str(e)} | {str(fallback_error)}",
                    context=ErrorContext(operation="rag_enhanced_query")
                )

    def get_rag_statistics(self) -> Dict[str, Any]:
        """Get RAG service statistics."""
        return {
            "service_info": {
                "default_strategy": self.default_strategy.value,
                "relevance_threshold": self.relevance_threshold,
                "max_context_tokens": self.max_context_tokens
            },
            "performance_stats": self.query_stats.copy(),
            "strategy_configs": {
                strategy.value: config for strategy, config in self.strategy_configs.items()
            },
            "llm_service_stats": self.llm_service.get_service_statistics(),
            "vector_store_stats": self.vector_store.get_statistics() if hasattr(self.vector_store, 'get_statistics') else {}
        }

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for RAG service."""
        health_status = {
            "rag_service": "healthy",
            "components": {}
        }
        
        try:
            # Check LLM service
            llm_health = await self.llm_service.health_check()
            health_status["components"]["llm_service"] = llm_health
            
            # Check vector store
            if hasattr(self.vector_store, 'health_check'):
                vector_health = await self.vector_store.health_check()
                health_status["components"]["vector_store"] = vector_health
            else:
                # Basic vector store test
                try:
                    stats = self.vector_store.get_statistics()
                    health_status["components"]["vector_store"] = {
                        "status": "healthy" if stats.get("total_documents", 0) > 0 else "empty",
                        "document_count": stats.get("total_documents", 0)
                    }
                except Exception as e:
                    health_status["components"]["vector_store"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
            
            # Determine overall health
            component_statuses = [
                comp.get("status", "unknown") for comp in health_status["components"].values()
            ]
            
            if all(status == "healthy" for status in component_statuses):
                health_status["rag_service"] = "healthy"
            elif any(status == "unhealthy" for status in component_statuses):
                health_status["rag_service"] = "degraded"
            else:
                health_status["rag_service"] = "unknown"
            
            health_status["query_stats"] = self.query_stats
            
            return health_status
            
        except Exception as e:
            logger.error(f"RAG health check failed: {e}")
            return {
                "rag_service": "unhealthy",
                "error": str(e),
                "components": health_status.get("components", {})
            }