"""Pydantic models and schemas for the Excel Chat Agent API."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class ChartType(str, Enum):
    """Supported chart types for data visualization."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX = "box"


class QueryRequest(BaseModel):
    """Request model for querying Excel data."""
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask about the Excel data")
    file_filter: Optional[str] = Field(None, description="Filter by file name (partial match)")
    sheet_filter: Optional[str] = Field(None, description="Filter by sheet name (partial match)")
    max_results: Optional[int] = Field(default=5, ge=1, le=20, description="Maximum number of results to return")
    include_statistics: bool = Field(default=False, description="Include statistical analysis in the response")
    streaming: bool = Field(default=True, description="Enable streaming response")
    context_window: Optional[int] = Field(default=10, ge=1, le=50, description="Number of previous messages to include for context")


class ChartData(BaseModel):
    """Data structure for chart recommendations."""
    type: ChartType
    data: List[Dict[str, Any]] = Field(..., description="Chart data points")
    config: Dict[str, Any] = Field(default_factory=dict, description="Chart configuration options")
    title: Optional[str] = Field(None, description="Chart title")
    description: Optional[str] = Field(None, description="Chart description")
    x_axis: Optional[str] = Field(None, description="X-axis label")
    y_axis: Optional[str] = Field(None, description="Y-axis label")


class QueryResponse(BaseModel):
    """Response model for Excel data queries."""
    answer: str = Field(..., description="The AI-generated answer")
    sources: List[str] = Field(default_factory=list, description="Sources used to generate the answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the answer")
    chart_data: Optional[ChartData] = Field(None, description="Chart recommendation if applicable")
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used")
    file_sources: List[str] = Field(default_factory=list, description="Source files used")
    sheet_sources: List[str] = Field(default_factory=list, description="Source sheets used")


class FileInfo(BaseModel):
    """Information about an uploaded Excel file."""
    file_name: str = Field(..., description="Name of the file")
    file_hash: str = Field(..., description="MD5 hash of the file")
    total_sheets: int = Field(..., ge=0, description="Total number of sheets")
    total_rows: int = Field(..., ge=0, description="Total number of rows across all sheets")
    total_columns: int = Field(..., ge=0, description="Total number of unique columns")
    file_size_mb: float = Field(..., ge=0, description="File size in megabytes")
    last_modified: datetime = Field(..., description="Last modification time")
    uploaded_at: Optional[datetime] = Field(default_factory=datetime.now)
    processed: bool = Field(default=False, description="Whether the file has been processed")
    
    @validator('file_size_mb')
    def validate_file_size(cls, v):
        """Validate file size doesn't exceed limit."""
        if v > 100:
            raise ValueError('File size exceeds 100MB limit')
        return round(v, 2)


class SheetInfo(BaseModel):
    """Information about an Excel sheet."""
    sheet_name: str = Field(..., description="Name of the sheet")
    num_rows: int = Field(..., ge=0, description="Number of rows in the sheet")
    num_cols: int = Field(..., ge=0, description="Number of columns in the sheet")
    columns: List[str] = Field(..., description="List of column names")
    data_types: Dict[str, str] = Field(default_factory=dict, description="Data types of columns")
    sample_data: List[Dict[str, Any]] = Field(default_factory=list, description="Sample data rows")


class ColumnInfo(BaseModel):
    """Information about a column in an Excel sheet."""
    name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="Data type (numeric, text, datetime)")
    dtype: str = Field(..., description="Pandas dtype")
    non_null_count: int = Field(..., ge=0, description="Number of non-null values")
    null_count: int = Field(..., ge=0, description="Number of null values")
    stats: Dict[str, Any] = Field(default_factory=dict, description="Statistical information")


class WebSocketMessage(BaseModel):
    """WebSocket message structure."""
    type: str = Field(..., description="Message type (query, response, error, status)")
    content: Optional[str] = Field(None, description="Message content")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data payload")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    session_id: Optional[str] = Field(None, description="Session identifier")
    message_id: Optional[str] = Field(None, description="Unique message identifier")


class SystemStats(BaseModel):
    """System statistics and health information."""
    total_files: int = Field(..., ge=0, description="Total number of processed files")
    total_documents: int = Field(..., ge=0, description="Total number of documents in vector store")
    cache_size: int = Field(..., ge=0, description="Cache size in number of entries")
    model_name: str = Field(..., description="Currently active LLM model")
    vector_store_size: int = Field(..., ge=0, description="Vector store size in number of vectors")
    uptime_seconds: int = Field(..., ge=0, description="System uptime in seconds")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    disk_usage_mb: Optional[float] = Field(None, description="Disk usage in MB")
    active_connections: int = Field(default=0, ge=0, description="Number of active WebSocket connections")


class UploadResponse(BaseModel):
    """Response for file upload operations."""
    success: bool = Field(..., description="Whether the upload was successful")
    message: str = Field(..., description="Response message")
    file_info: Optional[FileInfo] = Field(None, description="Information about the uploaded file")
    processing_status: str = Field(default="pending", description="Processing status")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")


class HealthCheck(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Health status (healthy, unhealthy, degraded)")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(default="0.1.0", description="Application version")
    components: Dict[str, str] = Field(default_factory=dict, description="Component health status")
    uptime_seconds: int = Field(..., ge=0, description="Application uptime")


class ProcessingStatus(BaseModel):
    """Status of file processing operations."""
    file_name: str = Field(..., description="Name of the file being processed")
    status: str = Field(..., description="Processing status (pending, processing, completed, failed)")
    progress_percentage: float = Field(..., ge=0, le=100, description="Processing progress percentage")
    current_step: str = Field(..., description="Current processing step")
    estimated_time_remaining: Optional[int] = Field(None, description="Estimated time remaining in seconds")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    started_at: datetime = Field(..., description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing completion time")


class CacheStats(BaseModel):
    """Cache statistics and information."""
    total_entries: int = Field(..., ge=0, description="Total number of cache entries")
    hit_rate: float = Field(..., ge=0, le=1, description="Cache hit rate")
    miss_rate: float = Field(..., ge=0, le=1, description="Cache miss rate")
    size_mb: float = Field(..., ge=0, description="Cache size in megabytes")
    oldest_entry: Optional[datetime] = Field(None, description="Timestamp of oldest cache entry")
    newest_entry: Optional[datetime] = Field(None, description="Timestamp of newest cache entry")


class ConversationMessage(BaseModel):
    """Individual message in a conversation."""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    message_id: str = Field(..., description="Unique message identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")


class ConversationHistory(BaseModel):
    """Complete conversation history."""
    session_id: str = Field(..., description="Session identifier")
    messages: List[ConversationMessage] = Field(default_factory=list, description="List of messages")
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    total_messages: int = Field(..., ge=0, description="Total number of messages")


class SearchResult(BaseModel):
    """Result from vector search operations."""
    content: str = Field(..., description="Matching content")
    file_name: str = Field(..., description="Source file name")
    sheet_name: str = Field(..., description="Source sheet name")
    score: float = Field(..., ge=0, le=1, description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class IndexingProgress(BaseModel):
    """Progress information for indexing operations."""
    total_files: int = Field(..., ge=0, description="Total number of files to index")
    processed_files: int = Field(..., ge=0, description="Number of files processed")
    current_file: Optional[str] = Field(None, description="Currently processing file")
    progress_percentage: float = Field(..., ge=0, le=100, description="Indexing progress percentage")
    estimated_time_remaining: Optional[int] = Field(None, description="Estimated time remaining in seconds")
    errors: List[str] = Field(default_factory=list, description="List of errors encountered")
    started_at: datetime = Field(..., description="Indexing start time")
    
    @validator('processed_files')
    def validate_processed_files(cls, v, values):
        """Ensure processed files doesn't exceed total files."""
        if 'total_files' in values and v > values['total_files']:
            raise ValueError('Processed files cannot exceed total files')
        return v