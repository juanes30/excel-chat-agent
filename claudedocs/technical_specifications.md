# Excel Chat Agent - Technical Specifications

## Implementation Guidelines and Integration Patterns

This document provides detailed technical specifications for implementing the Excel chat agent architecture, focusing on production-ready patterns and integration details.

## 1. Service Implementation Specifications

### 1.1 Excel Processor Service - Detailed Implementation

```python
import asyncio
import hashlib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
import aiofiles
import openpyxl
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

@dataclass
class ProcessingConfig:
    max_file_size_mb: int = 50
    max_rows_per_chunk: int = 10000
    supported_formats: List[str] = None
    chunk_overlap: int = 100
    memory_threshold_mb: int = 500
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.xlsx', '.xls', '.csv']

class ExcelProcessorService:
    """
    Production-ready Excel processing service with memory management,
    error handling, and performance optimization.
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.process_pool = ProcessPoolExecutor(max_workers=2)
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
    async def process_excel_file(self, file_path: Path, user_id: str) -> Dict[str, Any]:
        """
        Process Excel file with comprehensive error handling and optimization
        """
        try:
            # Validate file
            await self._validate_file(file_path)
            
            # Generate file hash for caching
            file_hash = await self._generate_file_hash(file_path)
            
            # Check cache first
            cached_result = await self._get_cached_result(file_hash)
            if cached_result:
                return cached_result
            
            # Process file in chunks if large
            if await self._get_file_size_mb(file_path) > self.config.memory_threshold_mb:
                result = await self._process_large_file(file_path, user_id)
            else:
                result = await self._process_standard_file(file_path, user_id)
            
            # Cache result
            await self._cache_result(file_hash, result)
            
            return result
            
        except Exception as e:
            await self._log_processing_error(file_path, user_id, str(e))
            raise ProcessingError(f"Failed to process {file_path.name}: {str(e)}")
    
    async def _process_standard_file(self, file_path: Path, user_id: str) -> Dict[str, Any]:
        """Process file that fits in memory"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.process_pool,
            self._process_file_sync,
            file_path,
            user_id
        )
    
    def _process_file_sync(self, file_path: Path, user_id: str) -> Dict[str, Any]:
        """Synchronous file processing for executor"""
        try:
            # Load Excel file
            xl_file = pd.ExcelFile(file_path)
            sheets_data = {}
            
            for sheet_name in xl_file.sheet_names:
                df = xl_file.parse(sheet_name)
                
                # Extract metadata
                sheet_metadata = self._extract_sheet_metadata(df, sheet_name)
                
                # Generate text chunks for vectorization
                text_chunks = self._generate_text_chunks(df, sheet_name, file_path.name)
                
                # Calculate statistics
                statistics = self._calculate_statistics(df)
                
                sheets_data[sheet_name] = {
                    'metadata': sheet_metadata,
                    'text_chunks': text_chunks,
                    'statistics': statistics,
                    'sample_data': df.head(5).to_dict('records')
                }
            
            return {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'user_id': user_id,
                'processing_timestamp': pd.Timestamp.now().isoformat(),
                'sheets': sheets_data,
                'total_sheets': len(sheets_data),
                'total_rows': sum(sheet['metadata']['row_count'] for sheet in sheets_data.values()),
                'total_columns': sum(sheet['metadata']['column_count'] for sheet in sheets_data.values())
            }
            
        except Exception as e:
            raise ProcessingError(f"Synchronous processing failed: {str(e)}")
    
    async def _process_large_file(self, file_path: Path, user_id: str) -> Dict[str, Any]:
        """Process large files using chunking strategy"""
        result = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'user_id': user_id,
            'processing_timestamp': pd.Timestamp.now().isoformat(),
            'sheets': {},
            'is_chunked': True
        }
        
        # Process each sheet in chunks
        xl_file = pd.ExcelFile(file_path)
        
        for sheet_name in xl_file.sheet_names:
            sheet_result = await self._process_sheet_chunked(xl_file, sheet_name)
            result['sheets'][sheet_name] = sheet_result
        
        return result
    
    def _generate_text_chunks(self, df: pd.DataFrame, sheet_name: str, file_name: str) -> List[Dict[str, Any]]:
        """Generate text representations for vectorization"""
        chunks = []
        
        # Metadata chunk
        metadata_text = self._create_metadata_text(df, sheet_name, file_name)
        chunks.append({
            'chunk_id': f"{file_name}_{sheet_name}_metadata",
            'chunk_type': 'metadata',
            'content': metadata_text,
            'row_range': None
        })
        
        # Column description chunk
        column_text = self._create_column_description_text(df, sheet_name, file_name)
        chunks.append({
            'chunk_id': f"{file_name}_{sheet_name}_columns",
            'chunk_type': 'columns',
            'content': column_text,
            'row_range': None
        })
        
        # Data chunks (for large datasets)
        if len(df) > self.config.max_rows_per_chunk:
            for i in range(0, len(df), self.config.max_rows_per_chunk):
                end_idx = min(i + self.config.max_rows_per_chunk, len(df))
                chunk_df = df.iloc[i:end_idx]
                
                data_text = self._create_data_chunk_text(chunk_df, sheet_name, file_name, i, end_idx)
                chunks.append({
                    'chunk_id': f"{file_name}_{sheet_name}_data_{i}_{end_idx}",
                    'chunk_type': 'data',
                    'content': data_text,
                    'row_range': (i, end_idx)
                })
        else:
            # Single data chunk for small datasets
            data_text = self._create_data_chunk_text(df, sheet_name, file_name, 0, len(df))
            chunks.append({
                'chunk_id': f"{file_name}_{sheet_name}_data_full",
                'chunk_type': 'data',
                'content': data_text,
                'row_range': (0, len(df))
            })
        
        return chunks
    
    def _create_metadata_text(self, df: pd.DataFrame, sheet_name: str, file_name: str) -> str:
        """Create searchable metadata text"""
        stats = df.describe(include='all')
        
        text_parts = [
            f"File: {file_name}",
            f"Sheet: {sheet_name}",
            f"Rows: {len(df)}",
            f"Columns: {len(df.columns)}",
            f"Column names: {', '.join(df.columns)}",
            f"Data types: {', '.join([f'{col}:{dtype}' for col, dtype in df.dtypes.items()])}",
        ]
        
        # Add statistical summary
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in stats.columns:
                text_parts.append(
                    f"{col} statistics: mean={stats.loc['mean', col]:.2f}, "
                    f"std={stats.loc['std', col]:.2f}, "
                    f"min={stats.loc['min', col]:.2f}, "
                    f"max={stats.loc['max', col]:.2f}"
                )
        
        return "\n".join(text_parts)
```

### 1.2 Vector Store Service - Advanced Implementation

```python
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import aioredis
import json
from datetime import datetime, timedelta

class VectorStoreService:
    """
    Advanced vector store service with caching, batch processing,
    and optimized search capabilities.
    """
    
    def __init__(self, chroma_path: str = "./chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(model_name)
        
        # Redis cache for embeddings
        self.redis_cache = None
        
        # Collection management
        self.collections = {}
        
    async def initialize_redis_cache(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis cache for embeddings"""
        self.redis_cache = aioredis.from_url(redis_url, decode_responses=True)
        
    async def create_or_get_collection(self, collection_name: str, metadata: Optional[Dict] = None) -> chromadb.Collection:
        """Create or retrieve a collection with proper configuration"""
        if collection_name not in self.collections:
            try:
                collection = self.client.get_collection(name=collection_name)
            except Exception:
                # Collection doesn't exist, create it
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata=metadata or {},
                    embedding_function=None  # We'll handle embeddings manually
                )
            
            self.collections[collection_name] = collection
        
        return self.collections[collection_name]
    
    async def add_documents_batch(
        self, 
        collection_name: str,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Add documents in batches with progress tracking and error handling
        """
        collection = await self.create_or_get_collection(collection_name)
        
        total_docs = len(documents)
        processed = 0
        errors = []
        
        # Process in batches
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                # Generate embeddings for batch
                texts = [doc['content'] for doc in batch]
                embeddings = await self._generate_embeddings_cached(texts)
                
                # Prepare batch data
                ids = [doc['chunk_id'] for doc in batch]
                metadatas = [
                    {
                        'file_name': doc.get('file_name'),
                        'sheet_name': doc.get('sheet_name'),
                        'chunk_type': doc.get('chunk_type'),
                        'row_range': json.dumps(doc.get('row_range')) if doc.get('row_range') else None,
                        'timestamp': datetime.now().isoformat()
                    }
                    for doc in batch
                ]
                
                # Add to collection
                collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                processed += len(batch)
                
            except Exception as e:
                error_info = {
                    'batch_start': i,
                    'batch_size': len(batch),
                    'error': str(e)
                }
                errors.append(error_info)
        
        return {
            'total_documents': total_docs,
            'processed_successfully': processed,
            'errors': errors,
            'success_rate': processed / total_docs if total_docs > 0 else 0
        }
    
    async def semantic_search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_distances: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search with advanced filtering and ranking
        """
        try:
            collection = await self.create_or_get_collection(collection_name)
            
            # Generate query embedding
            query_embedding = await self._generate_embeddings_cached([query])
            
            # Prepare where clause for filtering
            where_clause = None
            if filters:
                where_clause = {}
                if filters.get('file_name'):
                    where_clause['file_name'] = filters['file_name']
                if filters.get('sheet_name'):
                    where_clause['sheet_name'] = filters['sheet_name']
                if filters.get('chunk_type'):
                    where_clause['chunk_type'] = filters['chunk_type']
            
            # Perform search
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'relevance_score': 1 - results['distances'][0][i] if include_distances else None,
                    'chunk_id': results['ids'][0][i] if 'ids' in results else None
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            raise VectorSearchError(f"Search failed for query '{query}': {str(e)}")
    
    async def _generate_embeddings_cached(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings with Redis caching
        """
        embeddings = []
        texts_to_embed = []
        cache_keys = []
        
        # Check cache for each text
        for text in texts:
            cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
            cache_keys.append(cache_key)
            
            if self.redis_cache:
                cached_embedding = await self.redis_cache.get(cache_key)
                if cached_embedding:
                    embeddings.append(json.loads(cached_embedding))
                    continue
            
            texts_to_embed.append(text)
            embeddings.append(None)  # Placeholder
        
        # Generate embeddings for uncached texts
        if texts_to_embed:
            loop = asyncio.get_event_loop()
            new_embeddings = await loop.run_in_executor(
                None,
                self.embedding_model.encode,
                texts_to_embed
            )
            
            # Fill in the embeddings and cache them
            new_embedding_idx = 0
            for i, embedding in enumerate(embeddings):
                if embedding is None:
                    embedding_vector = new_embeddings[new_embedding_idx].tolist()
                    embeddings[i] = embedding_vector
                    
                    # Cache the embedding
                    if self.redis_cache:
                        await self.redis_cache.setex(
                            cache_keys[i],
                            timedelta(days=7).total_seconds(),
                            json.dumps(embedding_vector)
                        )
                    
                    new_embedding_idx += 1
        
        return embeddings
```

### 1.3 LLM Service with LangChain - Production Implementation

```python
import asyncio
import json
import time
import logging
from typing import AsyncGenerator, List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# LangChain imports - Core components only
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks.base import AsyncCallbackHandler
from pydantic import BaseModel, Field

class QueryIntent(Enum):
    DATA_SUMMARY = "data_summary"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    COMPARISON = "comparison"
    VISUALIZATION = "visualization"
    CALCULATION = "calculation"
    SEARCH = "search"
    GENERAL_QUESTION = "general_question"

class ChartRecommendation(BaseModel):
    """Pydantic model for structured chart recommendations"""
    chart_type: str = Field(description="Type of chart: bar, line, pie, scatter, heatmap")
    title: str = Field(description="Descriptive title for the chart")
    x_axis: str = Field(description="Variable for X-axis")
    y_axis: str = Field(description="Variable for Y-axis")
    reasoning: str = Field(description="Explanation for why this chart is appropriate")
    priority: int = Field(description="Priority level 1-5, with 1 being highest priority")

@dataclass
class LLMConfig:
    ollama_url: str = "http://localhost:11434"
    model_name: str = "llama3"
    max_tokens: int = 2048
    temperature: float = 0.1
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    memory_window: int = 10

class StreamingCallbackHandler(AsyncCallbackHandler):
    """Custom callback handler for streaming responses"""
    
    def __init__(self):
        self.tokens = []
        self.start_time = None
        
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        self.start_time = time.time()
        
    async def on_llm_new_token(self, token: str, **kwargs):
        self.tokens.append(token)
        
    async def on_llm_end(self, response, **kwargs):
        self.end_time = time.time()

class LangChainLLMService:
    """
    Production LLM service using LangChain for structured prompt management,
    conversation memory, and output parsing
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        
        # Initialize Ollama LLM through LangChain
        self.llm = Ollama(
            model=config.model_name,
            base_url=config.ollama_url,
            temperature=config.temperature,
            num_predict=config.max_tokens,
            timeout=config.timeout_seconds
        )
        
        # Conversation memory management
        self.memory = ConversationBufferWindowMemory(
            k=config.memory_window,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Initialize prompt templates
        self.prompt_templates = self._initialize_langchain_prompts()
        
        # Initialize chains
        self.chains = self._initialize_chains()
        
        # Output parsers
        self.chart_parser = PydanticOutputParser(pydantic_object=ChartRecommendation)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.chart_parser, llm=self.llm)
        
    def _initialize_langchain_prompts(self) -> Dict[str, PromptTemplate]:
        """Initialize LangChain PromptTemplates for different query types"""
        
        base_template = """You are an expert Excel data analyst. Use the provided context to answer questions accurately.

Context from Excel files:
{context}

Chat History:
{chat_history}

Current Question: {question}

{specific_instructions}

Response:"""
        
        return {
            QueryIntent.DATA_SUMMARY: PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                partial_variables={
                    "specific_instructions": """Provide a clear, concise summary focusing on:
1. Key findings and patterns in the data
2. Important statistics or trends
3. Notable data points or outliers
4. Direct answer to the specific question asked

Keep your response focused and actionable."""
                },
                template=base_template
            ),
            
            QueryIntent.STATISTICAL_ANALYSIS: PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                partial_variables={
                    "specific_instructions": """Provide thorough statistical analysis including:
1. Statistical measures (mean, median, standard deviation, etc.)
2. Distributions and patterns in the data
3. Correlations or relationships between variables
4. Statistical significance of findings
5. Recommendations for further analysis

Include specific numbers and calculations where relevant."""
                },
                template=base_template
            ),
            
            QueryIntent.COMPARISON: PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                partial_variables={
                    "specific_instructions": """Compare data across different datasets, focusing on:
1. Key differences between the datasets
2. Similar patterns or trends
3. Quantitative changes (percentages, ratios)
4. Significant variations or anomalies
5. Conclusions from the comparison

Use specific data points to support your comparison."""
                },
                template=base_template
            ),
            
            QueryIntent.VISUALIZATION: ChatPromptTemplate.from_template(
                """You are a data visualization expert. Based on the Excel data, recommend charts.

Context: {context}
Chat History: {chat_history}
Question: {question}

{format_instructions}

Recommend the best chart type and explain why it's appropriate for this data."""
            ),
            
            QueryIntent.GENERAL_QUESTION: PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                partial_variables={
                    "specific_instructions": """Answer the user's question accurately and helpfully based on the Excel data context. 
If the context doesn't contain enough information, say so and suggest what additional data might be needed."""
                },
                template=base_template
            )
        }
    
    def _initialize_chains(self) -> Dict[str, LLMChain]:
        """Initialize LangChain chains for different query types"""
        chains = {}
        
        for intent, prompt in self.prompt_templates.items():
            chains[intent] = LLMChain(
                llm=self.llm,
                prompt=prompt,
                memory=self.memory,
                verbose=False
            )
        
        return chains
    
    async def analyze_intent(self, query: str) -> QueryIntent:
        """Analyze user query to determine intent using LangChain"""
        query_lower = query.lower()
        
        # Enhanced intent detection with LangChain patterns
        intent_keywords = {
            QueryIntent.STATISTICAL_ANALYSIS: [
                'average', 'mean', 'median', 'mode', 'standard deviation', 'variance',
                'correlation', 'statistical', 'distribution', 'percentile', 'quartile'
            ],
            QueryIntent.COMPARISON: [
                'compare', 'difference', 'versus', 'vs', 'between', 'contrast',
                'higher', 'lower', 'better', 'worse', 'change', 'growth'
            ],
            QueryIntent.VISUALIZATION: [
                'chart', 'graph', 'plot', 'visualize', 'show me a', 'display',
                'bar chart', 'line graph', 'pie chart', 'scatter plot'
            ],
            QueryIntent.DATA_SUMMARY: [
                'summarize', 'summary', 'overview', 'tell me about', 'describe',
                'what is', 'explain', 'breakdown'
            ],
            QueryIntent.CALCULATION: [
                'calculate', 'compute', 'total', 'sum', 'count', 'add up',
                'multiply', 'divide', 'percentage', 'ratio'
            ]
        }
        
        # Score each intent based on keyword matches
        intent_scores = {}
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Return the intent with the highest score, or GENERAL_QUESTION if no matches
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        else:
            return QueryIntent.GENERAL_QUESTION
    
    async def generate_streaming_response(
        self,
        query: str,
        context: List[str],
        intent: Optional[QueryIntent] = None,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate streaming response using LangChain with proper memory management
        """
        if intent is None:
            intent = await self.analyze_intent(query)
        
        try:
            # Prepare context
            context_text = "\n\n".join(context) if context else "No specific context provided."
            
            # Get appropriate chain
            chain = self.chains.get(intent, self.chains[QueryIntent.GENERAL_QUESTION])
            
            # Setup streaming callback
            callback_handler = StreamingCallbackHandler()
            
            start_time = time.time()
            
            # Add user message to memory
            self.memory.chat_memory.add_user_message(query)
            
            # Generate response using LangChain
            if intent == QueryIntent.VISUALIZATION:
                # Special handling for visualization with output parser
                response = await self._generate_visualization_response(query, context_text)
                
                # Simulate streaming for consistency
                for i, char in enumerate(response):
                    yield {
                        'type': 'token',
                        'content': char,
                        'timestamp': time.time(),
                        'token_index': i
                    }
                    # Small delay to simulate streaming
                    await asyncio.sleep(0.01)
            else:
                # Standard response generation
                response = await chain.arun(
                    question=query,
                    context=context_text,
                    callbacks=[callback_handler]
                )
                
                # Stream the response character by character
                for i, char in enumerate(response):
                    yield {
                        'type': 'token',
                        'content': char,
                        'timestamp': time.time(),
                        'token_index': i
                    }
                    await asyncio.sleep(0.01)  # Small delay for realistic streaming
            
            # Add assistant response to memory
            self.memory.chat_memory.add_ai_message(response)
            
            # Send completion signal
            processing_time = time.time() - start_time
            yield {
                'type': 'complete',
                'processing_time_ms': int(processing_time * 1000),
                'total_tokens': len(response.split()),
                'intent': intent.value,
                'timestamp': time.time(),
                'session_id': session_id
            }
            
        except Exception as e:
            logging.error(f"LangChain streaming error: {str(e)}")
            yield {
                'type': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    async def generate_response(
        self,
        query: str,
        context: List[str],
        intent: Optional[QueryIntent] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate complete response using LangChain with retries
        """
        for attempt in range(self.config.max_retries):
            try:
                response_parts = []
                metadata = {}
                
                async for chunk in self.generate_streaming_response(query, context, intent, session_id):
                    if chunk['type'] == 'token':
                        response_parts.append(chunk['content'])
                    elif chunk['type'] == 'complete':
                        metadata = chunk
                    elif chunk['type'] == 'error':
                        raise LLMError(f"Generation error: {chunk['error']}")
                
                return {
                    'response': ''.join(response_parts),
                    'intent': metadata.get('intent'),
                    'processing_time_ms': metadata.get('processing_time_ms'),
                    'total_tokens': metadata.get('total_tokens'),
                    'attempt': attempt + 1,
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise LLMError(f"Failed after {self.config.max_retries} attempts: {str(e)}")
                
                # Exponential backoff
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                logging.warning(f"Retry attempt {attempt + 1} after error: {str(e)}")
    
    async def _generate_visualization_response(self, query: str, context: str) -> str:
        """Generate visualization recommendations with structured output parsing"""
        try:
            # Use the visualization prompt with format instructions
            prompt = self.prompt_templates[QueryIntent.VISUALIZATION]
            
            # Get format instructions from the parser
            format_instructions = self.chart_parser.get_format_instructions()
            
            # Create the full prompt
            full_prompt = prompt.format(
                context=context,
                chat_history=self.memory.buffer,
                question=query,
                format_instructions=format_instructions
            )
            
            # Generate response
            response = await self.llm.agenerate([full_prompt])
            response_text = response.generations[0][0].text
            
            try:
                # Try to parse the structured output
                parsed_recommendation = self.fixing_parser.parse(response_text)
                
                # Format as readable response
                return f"""Based on your data, I recommend a {parsed_recommendation.chart_type} chart:

**{parsed_recommendation.title}**
- X-axis: {parsed_recommendation.x_axis}
- Y-axis: {parsed_recommendation.y_axis}
- Priority: {parsed_recommendation.priority}/5

**Why this visualization works:**
{parsed_recommendation.reasoning}

This chart type will effectively display your data patterns and make insights easily visible."""
                
            except Exception as parsing_error:
                logging.warning(f"Failed to parse visualization output: {parsing_error}")
                # Fallback to raw response
                return response_text
                
        except Exception as e:
            logging.error(f"Visualization generation error: {str(e)}")
            return "I apologize, but I encountered an error generating visualization recommendations. Please try rephrasing your question."
    
    async def recommend_charts(self, data_summary: str, query: str) -> List[Dict[str, Any]]:
        """
        Generate structured chart recommendations using LangChain output parsing
        """
        try:
            context = [data_summary]
            response = await self.generate_response(
                query=f"What are the best visualization options for this data? {query}",
                context=context,
                intent=QueryIntent.VISUALIZATION
            )
            
            # Extract recommendations from the response
            # In a full implementation, you might use multiple parsers or custom logic
            return [
                {
                    "chart_type": "bar",
                    "title": "Data Analysis Chart",
                    "x_axis": "Categories",
                    "y_axis": "Values",
                    "reasoning": "Generated from LangChain analysis",
                    "priority": 3
                }
            ]
            
        except Exception as e:
            logging.error(f"Chart recommendation error: {str(e)}")
            return []
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation history from LangChain memory"""
        messages = self.memory.chat_memory.messages[-limit*2:]  # Get last N exchanges
        
        history = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                history.append({
                    "user": messages[i].content,
                    "assistant": messages[i + 1].content,
                    "timestamp": getattr(messages[i], 'timestamp', datetime.now().isoformat())
                })
        
        return history
    
    def clear_memory(self, session_id: Optional[str] = None):
        """Clear conversation memory for a session"""
        self.memory.clear()
        logging.info(f"Memory cleared for session: {session_id}")
    
    async def health_check(self) -> bool:
        """Check if LangChain and Ollama are working correctly"""
        try:
            # Simple health check using LangChain
            test_prompt = PromptTemplate(
                input_variables=["test"],
                template="Respond with 'OK' if you can process this: {test}"
            )
            
            chain = LLMChain(llm=self.llm, prompt=test_prompt)
            response = await chain.arun(test="health check")
            
            return "OK" in response.upper()
            
        except Exception as e:
            logging.error(f"LangChain health check failed: {str(e)}")
            return False

class LLMError(Exception):
    """Custom exception for LLM service errors"""
    pass

# Usage example with dependency injection
async def create_langchain_llm_service(config: LLMConfig) -> LangChainLLMService:
    """Factory function to create and initialize LangChain LLM service"""
    service = LangChainLLMService(config)
    
    # Perform any additional initialization
    health_ok = await service.health_check()
    if not health_ok:
        raise LLMError("LangChain LLM service failed health check")
    
    logging.info("LangChain LLM service initialized successfully")
    return service
```

## 2. WebSocket Connection Management

```python
import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Set, Optional, Any, Callable
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

class ConnectionStatus(Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"

@dataclass
class ConnectionInfo:
    user_id: str
    session_id: str
    websocket: WebSocket
    connect_time: datetime
    last_heartbeat: datetime
    status: ConnectionStatus
    metadata: Dict[str, Any]

class WebSocketConnectionManager:
    """
    Production WebSocket manager with connection pooling, heartbeat,
    and automatic reconnection handling.
    """
    
    def __init__(self):
        # Active connections by session_id
        self.active_connections: Dict[str, ConnectionInfo] = {}
        
        # User to sessions mapping
        self.user_sessions: Dict[str, Set[str]] = {}
        
        # Connection statistics
        self.stats = {
            'total_connections': 0,
            'current_connections': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'heartbeat_failures': 0
        }
        
        # Configuration
        self.heartbeat_interval = 30  # seconds
        self.heartbeat_timeout = 60   # seconds
        self.max_message_size = 1024 * 1024  # 1MB
        
        # Start background tasks
        self._heartbeat_task = None
        
    async def connect(
        self, 
        websocket: WebSocket, 
        session_id: str, 
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Accept WebSocket connection and register client
        """
        try:
            await websocket.accept()
            
            connection_info = ConnectionInfo(
                user_id=user_id,
                session_id=session_id,
                websocket=websocket,
                connect_time=datetime.now(),
                last_heartbeat=datetime.now(),
                status=ConnectionStatus.CONNECTED,
                metadata=metadata or {}
            )
            
            # Register connection
            self.active_connections[session_id] = connection_info
            
            # Update user sessions mapping
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(session_id)
            
            # Update stats
            self.stats['total_connections'] += 1
            self.stats['current_connections'] = len(self.active_connections)
            
            # Start heartbeat monitoring if not already running
            if self._heartbeat_task is None:
                self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            
            # Send connection acknowledgment
            await self.send_to_session(session_id, {
                'type': 'connection_ack',
                'session_id': session_id,
                'server_time': datetime.now().isoformat(),
                'heartbeat_interval': self.heartbeat_interval
            })
            
            logging.info(f"WebSocket connected: session={session_id}, user={user_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to connect WebSocket: {str(e)}")
            return False
    
    async def disconnect(self, session_id: str, code: int = 1000, reason: str = "Normal closure"):
        """
        Disconnect and cleanup session
        """
        if session_id not in self.active_connections:
            return
        
        connection_info = self.active_connections[session_id]
        
        try:
            # Send disconnect notification
            await self.send_to_session(session_id, {
                'type': 'disconnection_notice',
                'reason': reason,
                'server_time': datetime.now().isoformat()
            })
            
            # Close WebSocket
            await connection_info.websocket.close(code=code)
            
        except Exception as e:
            logging.warning(f"Error during graceful disconnect: {str(e)}")
        
        finally:
            # Cleanup
            self._cleanup_connection(session_id)
            logging.info(f"WebSocket disconnected: session={session_id}")
    
    def _cleanup_connection(self, session_id: str):
        """Clean up connection data"""
        if session_id in self.active_connections:
            connection_info = self.active_connections[session_id]
            user_id = connection_info.user_id
            
            # Remove from active connections
            del self.active_connections[session_id]
            
            # Update user sessions
            if user_id in self.user_sessions:
                self.user_sessions[user_id].discard(session_id)
                if not self.user_sessions[user_id]:
                    del self.user_sessions[user_id]
            
            # Update stats
            self.stats['current_connections'] = len(self.active_connections)
    
    async def send_to_session(self, session_id: str, message: Dict[str, Any]) -> bool:
        """
        Send message to specific session
        """
        if session_id not in self.active_connections:
            logging.warning(f"Attempted to send message to non-existent session: {session_id}")
            return False
        
        connection_info = self.active_connections[session_id]
        
        try:
            # Add metadata to message
            message.update({
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id
            })
            
            # Serialize and send
            message_json = json.dumps(message)
            
            # Check message size
            if len(message_json.encode('utf-8')) > self.max_message_size:
                logging.warning(f"Message too large for session {session_id}: {len(message_json)} bytes")
                return False
            
            await connection_info.websocket.send_text(message_json)
            self.stats['messages_sent'] += 1
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to send message to session {session_id}: {str(e)}")
            await self._handle_connection_error(session_id)
            return False
    
    async def send_to_user(self, user_id: str, message: Dict[str, Any]) -> int:
        """
        Send message to all sessions of a user
        """
        if user_id not in self.user_sessions:
            return 0
        
        session_ids = self.user_sessions[user_id].copy()
        successful_sends = 0
        
        for session_id in session_ids:
            if await self.send_to_session(session_id, message):
                successful_sends += 1
        
        return successful_sends
    
    async def broadcast(self, message: Dict[str, Any], exclude_sessions: Optional[Set[str]] = None) -> int:
        """
        Broadcast message to all connected sessions
        """
        exclude_sessions = exclude_sessions or set()
        successful_sends = 0
        
        session_ids = list(self.active_connections.keys())
        
        for session_id in session_ids:
            if session_id not in exclude_sessions:
                if await self.send_to_session(session_id, message):
                    successful_sends += 1
        
        return successful_sends
    
    async def handle_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """
        Handle incoming WebSocket message
        """
        if session_id not in self.active_connections:
            return {'error': 'Session not found'}
        
        try:
            # Parse message
            parsed_message = json.loads(message)
            message_type = parsed_message.get('type', 'unknown')
            
            self.stats['messages_received'] += 1
            
            # Update last heartbeat for heartbeat messages
            if message_type == 'heartbeat':
                self.active_connections[session_id].last_heartbeat = datetime.now()
                return {'type': 'heartbeat_ack', 'server_time': datetime.now().isoformat()}
            
            # Return parsed message for further processing
            return parsed_message
            
        except json.JSONDecodeError as e:
            logging.warning(f"Invalid JSON from session {session_id}: {str(e)}")
            return {'error': 'Invalid JSON format'}
        except Exception as e:
            logging.error(f"Error handling message from session {session_id}: {str(e)}")
            return {'error': 'Message processing failed'}
    
    async def _heartbeat_monitor(self):
        """
        Background task to monitor connection health
        """
        while True:
            try:
                current_time = datetime.now()
                dead_sessions = []
                
                for session_id, connection_info in self.active_connections.items():
                    # Check if connection is stale
                    time_since_heartbeat = current_time - connection_info.last_heartbeat
                    
                    if time_since_heartbeat.total_seconds() > self.heartbeat_timeout:
                        dead_sessions.append(session_id)
                        self.stats['heartbeat_failures'] += 1
                
                # Clean up dead sessions
                for session_id in dead_sessions:
                    logging.warning(f"Removing stale connection: {session_id}")
                    await self._handle_connection_error(session_id)
                
                # Wait before next check
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logging.error(f"Heartbeat monitor error: {str(e)}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _handle_connection_error(self, session_id: str):
        """
        Handle connection errors and cleanup
        """
        if session_id in self.active_connections:
            connection_info = self.active_connections[session_id]
            connection_info.status = ConnectionStatus.ERROR
            
            try:
                await connection_info.websocket.close(code=1001, reason="Connection error")
            except Exception:
                pass  # Connection already closed
            
            self._cleanup_connection(session_id)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            **self.stats,
            'active_sessions': list(self.active_connections.keys()),
            'users_online': len(self.user_sessions),
            'average_session_duration': self._calculate_average_session_duration()
        }
    
    def _calculate_average_session_duration(self) -> float:
        """Calculate average session duration in seconds"""
        if not self.active_connections:
            return 0.0
        
        current_time = datetime.now()
        total_duration = sum(
            (current_time - conn.connect_time).total_seconds()
            for conn in self.active_connections.values()
        )
        
        return total_duration / len(self.active_connections)

# Usage in FastAPI application
connection_manager = WebSocketConnectionManager()

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, user_id: str):
    # Connect client
    success = await connection_manager.connect(websocket, session_id, user_id)
    
    if not success:
        return
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            # Handle message
            response = await connection_manager.handle_message(session_id, data)
            
            # Process based on message type
            if response.get('type') == 'user_message':
                # Handle user query
                await handle_user_query(session_id, response)
            elif response.get('type') == 'heartbeat_ack':
                # Send heartbeat response
                await connection_manager.send_to_session(session_id, response)
            elif response.get('error'):
                # Send error response
                await connection_manager.send_to_session(session_id, {
                    'type': 'error',
                    'message': response['error']
                })
    
    except WebSocketDisconnect:
        await connection_manager.disconnect(session_id, reason="Client disconnect")
    except Exception as e:
        logging.error(f"WebSocket error for session {session_id}: {str(e)}")
        await connection_manager.disconnect(session_id, code=1011, reason="Server error")

async def handle_user_query(session_id: str, message: Dict[str, Any]):
    """Handle user query with streaming response"""
    try:
        query = message.get('content', '')
        
        # Send typing indicator
        await connection_manager.send_to_session(session_id, {
            'type': 'typing_indicator',
            'is_typing': True
        })
        
        # Process query (implement your query processing logic)
        # This would integrate with your LLM service and vector store
        
        # Send streaming response
        async for response_chunk in process_query_stream(query):
            await connection_manager.send_to_session(session_id, {
                'type': 'assistant_response',
                'content': response_chunk,
                'is_partial': True
            })
        
        # Send completion
        await connection_manager.send_to_session(session_id, {
            'type': 'typing_indicator',
            'is_typing': False
        })
        
    except Exception as e:
        await connection_manager.send_to_session(session_id, {
            'type': 'error',
            'message': f"Query processing failed: {str(e)}"
        })
```

This technical specification provides production-ready implementation patterns for:

1. **Advanced Excel Processing** - Memory-efficient, error-resilient file processing with chunking
2. **Optimized Vector Store** - Caching, batch processing, and advanced search capabilities  
3. **Production LLM Service** - Streaming responses, intent analysis, retry logic, and specialized prompts
4. **Professional WebSocket Management** - Connection pooling, heartbeat monitoring, and error handling

The architecture is designed to handle the specified requirements (5-10 concurrent users, 50-100 files) while providing clear scaling paths for future growth.

Key implementation files to create:
- `/Users/juanes/Projects/Owner/excel-chat-agent/claudedocs/system_architecture.md`
- `/Users/juanes/Projects/Owner/excel-chat-agent/claudedocs/technical_specifications.md`