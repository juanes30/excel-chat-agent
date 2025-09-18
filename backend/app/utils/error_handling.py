"""Comprehensive error handling and retry mechanisms for the LLM service."""

import asyncio
import functools
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for classification."""
    NETWORK = "network"
    LLM_SERVICE = "llm_service"
    VECTOR_STORE = "vector_store"
    WEBSOCKET = "websocket"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    EXTERNAL_SERVICE = "external_service"
    UNKNOWN = "unknown"


class RetryStrategy(str, Enum):
    """Retry strategies for different error types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    NO_RETRY = "no_retry"
    IMMEDIATE_RETRY = "immediate_retry"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retriable_exceptions: List[Type[Exception]] = field(default_factory=list)
    non_retriable_exceptions: List[Type[Exception]] = field(default_factory=list)


class LLMServiceError(Exception):
    """Base exception for LLM service errors."""
    
    def __init__(self, 
                 message: str,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[ErrorContext] = None,
                 original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext(operation="unknown")
        self.original_exception = original_exception
        self.timestamp = datetime.now()


class OllamaConnectionError(LLMServiceError):
    """Error connecting to Ollama service."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message,
            category=ErrorCategory.LLM_SERVICE,
            severity=ErrorSeverity.HIGH,
            context=context
        )


class VectorStoreError(LLMServiceError):
    """Error with vector store operations."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message,
            category=ErrorCategory.VECTOR_STORE,
            severity=ErrorSeverity.MEDIUM,
            context=context
        )


class WebSocketError(LLMServiceError):
    """Error with WebSocket operations."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message,
            category=ErrorCategory.WEBSOCKET,
            severity=ErrorSeverity.MEDIUM,
            context=context
        )


class RateLimitError(LLMServiceError):
    """Rate limit exceeded error."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, context: Optional[ErrorContext] = None):
        super().__init__(
            message,
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            context=context
        )
        self.retry_after = retry_after


class ValidationError(LLMServiceError):
    """Input validation error."""
    
    def __init__(self, message: str, field_errors: Optional[Dict[str, str]] = None, context: Optional[ErrorContext] = None):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            context=context
        )
        self.field_errors = field_errors or {}


class CircuitBreakerError(LLMServiceError):
    """Circuit breaker is open error."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            context=context
        )


class ErrorHandler:
    """Comprehensive error handling and retry mechanism."""
    
    def __init__(self):
        self.error_stats: Dict[str, Dict[str, Any]] = {}
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        self.default_retry_config = RetryConfig()
        
        # Configure default retriable exceptions
        self.default_retry_config.retriable_exceptions = [
            ConnectionError,
            TimeoutError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RequestError,
            OllamaConnectionError
        ]
        
        # Configure non-retriable exceptions
        self.default_retry_config.non_retriable_exceptions = [
            ValidationError,
            ValueError,
            TypeError,
            KeyError
        ]
    
    def categorize_error(self, exception: Exception, context: Optional[ErrorContext] = None) -> ErrorCategory:
        """Categorize an error based on its type and context."""
        
        if isinstance(exception, LLMServiceError):
            return exception.category
        
        # Network-related errors
        if isinstance(exception, (ConnectionError, httpx.ConnectError, httpx.ConnectTimeout)):
            return ErrorCategory.NETWORK
        
        # Timeout errors
        if isinstance(exception, (TimeoutError, httpx.ReadTimeout, asyncio.TimeoutError)):
            return ErrorCategory.NETWORK
        
        # HTTP errors
        if isinstance(exception, httpx.HTTPStatusError):
            status_code = exception.response.status_code
            if status_code == 429:
                return ErrorCategory.RATE_LIMIT
            elif 500 <= status_code < 600:
                return ErrorCategory.EXTERNAL_SERVICE
            elif 400 <= status_code < 500:
                return ErrorCategory.VALIDATION
        
        # Validation errors
        if isinstance(exception, (ValueError, TypeError, KeyError)):
            return ErrorCategory.VALIDATION
        
        # WebSocket errors
        if "websocket" in str(type(exception)).lower():
            return ErrorCategory.WEBSOCKET
        
        return ErrorCategory.UNKNOWN
    
    def assess_severity(self, exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Assess the severity of an error."""
        
        if isinstance(exception, LLMServiceError):
            return exception.severity
        
        # Critical errors that affect core functionality
        if isinstance(exception, (SystemExit, MemoryError, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if category in [ErrorCategory.LLM_SERVICE, ErrorCategory.EXTERNAL_SERVICE]:
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if category in [ErrorCategory.NETWORK, ErrorCategory.VECTOR_STORE, ErrorCategory.RATE_LIMIT]:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        if category in [ErrorCategory.VALIDATION, ErrorCategory.WEBSOCKET]:
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    def should_retry(self, exception: Exception, attempt: int, retry_config: RetryConfig) -> bool:
        """Determine if an operation should be retried."""
        
        # Check if we've exceeded max attempts
        if attempt >= retry_config.max_attempts:
            return False
        
        # Check non-retriable exceptions first
        if any(isinstance(exception, exc_type) for exc_type in retry_config.non_retriable_exceptions):
            return False
        
        # Check retriable exceptions
        if any(isinstance(exception, exc_type) for exc_type in retry_config.retriable_exceptions):
            return True
        
        # Category-based retry logic
        category = self.categorize_error(exception)
        
        # Always retry network and external service errors
        if category in [ErrorCategory.NETWORK, ErrorCategory.EXTERNAL_SERVICE]:
            return True
        
        # Never retry validation errors
        if category == ErrorCategory.VALIDATION:
            return False
        
        # Retry rate limit errors with special handling
        if category == ErrorCategory.RATE_LIMIT:
            return True
        
        # Default to retry for unknown errors
        return True
    
    def calculate_delay(self, attempt: int, retry_config: RetryConfig, exception: Optional[Exception] = None) -> float:
        """Calculate delay before next retry attempt."""
        
        # Handle rate limit errors specially
        if isinstance(exception, RateLimitError) and exception.retry_after:
            return float(exception.retry_after)
        
        base_delay = retry_config.base_delay
        
        if retry_config.strategy == RetryStrategy.NO_RETRY:
            return 0.0
        
        elif retry_config.strategy == RetryStrategy.IMMEDIATE_RETRY:
            return 0.0
        
        elif retry_config.strategy == RetryStrategy.FIXED_DELAY:
            delay = base_delay
        
        elif retry_config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = base_delay * attempt
        
        elif retry_config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = base_delay * (retry_config.backoff_multiplier ** (attempt - 1))
        
        else:
            delay = base_delay
        
        # Apply maximum delay limit
        delay = min(delay, retry_config.max_delay)
        
        # Add jitter to prevent thundering herd
        if retry_config.jitter:
            import random
            delay = delay * (0.5 + 0.5 * random.random())
        
        return delay
    
    def record_error(self, exception: Exception, context: Optional[ErrorContext] = None):
        """Record error statistics for monitoring."""
        category = self.categorize_error(exception, context)
        severity = self.assess_severity(exception, category)
        
        error_key = f"{category.value}_{type(exception).__name__}"
        
        if error_key not in self.error_stats:
            self.error_stats[error_key] = {
                "count": 0,
                "first_seen": datetime.now(),
                "last_seen": datetime.now(),
                "category": category.value,
                "severity": severity.value,
                "exception_type": type(exception).__name__
            }
        
        self.error_stats[error_key]["count"] += 1
        self.error_stats[error_key]["last_seen"] = datetime.now()
        
        # Log error based on severity
        error_msg = f"Error recorded: {category.value} - {type(exception).__name__}: {str(exception)}"
        
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(error_msg)
        elif severity == ErrorSeverity.HIGH:
            logger.error(error_msg)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(error_msg)
        else:
            logger.info(error_msg)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        total_errors = sum(stats["count"] for stats in self.error_stats.values())
        
        # Group by category
        category_stats = {}
        for error_key, stats in self.error_stats.items():
            category = stats["category"]
            if category not in category_stats:
                category_stats[category] = {"count": 0, "types": []}
            
            category_stats[category]["count"] += stats["count"]
            category_stats[category]["types"].append({
                "exception_type": stats["exception_type"],
                "count": stats["count"],
                "last_seen": stats["last_seen"]
            })
        
        return {
            "total_errors": total_errors,
            "error_types": len(self.error_stats),
            "category_breakdown": category_stats,
            "detailed_stats": self.error_stats
        }
    
    def clear_error_statistics(self):
        """Clear error statistics."""
        self.error_stats.clear()
        logger.info("Error statistics cleared")


class CircuitBreaker:
    """Circuit breaker pattern for external service calls."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
    
    def can_execute(self) -> bool:
        """Check if the circuit breaker allows execution."""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)):
                self.state = "half-open"
                return True
            return False
        
        # half-open state allows one attempt
        return True
    
    def record_success(self):
        """Record successful execution."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self, exception: Exception):
        """Record failed execution."""
        if isinstance(exception, self.expected_exception):
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
        
        if self.state == "half-open":
            self.state = "open"


def with_error_handling(operation: str = None, 
                       retry_config: Optional[RetryConfig] = None,
                       circuit_breaker: Optional[CircuitBreaker] = None,
                       context: Optional[ErrorContext] = None):
    """Decorator for adding comprehensive error handling and retry logic."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            error_handler = ErrorHandler()
            config = retry_config or error_handler.default_retry_config
            op_name = operation or func.__name__
            
            # Create context if not provided
            func_context = context or ErrorContext(operation=op_name)
            
            attempt = 0
            last_exception = None
            
            while attempt < config.max_attempts:
                attempt += 1
                
                # Check circuit breaker
                if circuit_breaker and not circuit_breaker.can_execute():
                    raise CircuitBreakerError(
                        f"Circuit breaker is open for operation: {op_name}",
                        context=func_context
                    )
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Record success
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    
                    return result
                
                except Exception as e:
                    last_exception = e
                    
                    # Record error
                    error_handler.record_error(e, func_context)
                    
                    # Record circuit breaker failure
                    if circuit_breaker:
                        circuit_breaker.record_failure(e)
                    
                    # Check if should retry
                    if not error_handler.should_retry(e, attempt, config):
                        break
                    
                    # Calculate delay before retry
                    if attempt < config.max_attempts:
                        delay = error_handler.calculate_delay(attempt, config, e)
                        if delay > 0:
                            logger.info(f"Retrying {op_name} in {delay:.2f}s (attempt {attempt}/{config.max_attempts})")
                            await asyncio.sleep(delay)
            
            # All retries exhausted, wrap and raise the last exception
            if isinstance(last_exception, LLMServiceError):
                raise last_exception
            else:
                category = error_handler.categorize_error(last_exception, func_context)
                severity = error_handler.assess_severity(last_exception, category)
                
                raise LLMServiceError(
                    f"Operation '{op_name}' failed after {config.max_attempts} attempts: {str(last_exception)}",
                    category=category,
                    severity=severity,
                    context=func_context,
                    original_exception=last_exception
                )
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # For synchronous functions, create a simple error handling wrapper
            error_handler = ErrorHandler()
            op_name = operation or func.__name__
            func_context = context or ErrorContext(operation=op_name)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.record_error(e, func_context)
                
                if isinstance(e, LLMServiceError):
                    raise e
                else:
                    category = error_handler.categorize_error(e, func_context)
                    severity = error_handler.assess_severity(e, category)
                    
                    raise LLMServiceError(
                        f"Operation '{op_name}' failed: {str(e)}",
                        category=category,
                        severity=severity,
                        context=func_context,
                        original_exception=e
                    )
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global error handler instance
global_error_handler = ErrorHandler()


def create_ollama_circuit_breaker() -> CircuitBreaker:
    """Create a circuit breaker specifically for Ollama service calls."""
    return CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=30.0,
        expected_exception=OllamaConnectionError
    )


def create_vector_store_circuit_breaker() -> CircuitBreaker:
    """Create a circuit breaker for vector store operations."""
    return CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=15.0,
        expected_exception=VectorStoreError
    )


# Predefined retry configurations
OLLAMA_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    base_delay=1.0,
    max_delay=30.0,
    backoff_multiplier=2.0,
    retriable_exceptions=[OllamaConnectionError, ConnectionError, TimeoutError]
)

VECTOR_STORE_RETRY_CONFIG = RetryConfig(
    max_attempts=2,
    strategy=RetryStrategy.LINEAR_BACKOFF,
    base_delay=0.5,
    max_delay=5.0,
    retriable_exceptions=[VectorStoreError, ConnectionError]
)

WEBSOCKET_RETRY_CONFIG = RetryConfig(
    max_attempts=1,  # Don't retry WebSocket operations
    strategy=RetryStrategy.NO_RETRY,
    non_retriable_exceptions=[WebSocketError]
)