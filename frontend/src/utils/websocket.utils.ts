/**
 * WebSocket utility functions for connection management and message handling
 */

import { v4 as uuidv4 } from 'uuid';
import {
  WebSocketMessage,
  WebSocketState,
  WebSocketError,
  ConnectionError,
  MessageError
} from '../types/chat.types';

/**
 * Generate a unique session ID for WebSocket connection
 */
export const generateSessionId = (): string => {
  return `session_${uuidv4()}`;
};

/**
 * Create WebSocket URL with session ID
 */
export const createWebSocketUrl = (baseUrl: string, sessionId: string): string => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  let url = baseUrl.replace(/^https?:/, protocol);

  // Ensure URL ends properly for session ID
  if (!url.endsWith('/ws')) {
    url = url.replace(/\/ws.*$/, '/ws');
  }

  return `${url}/${sessionId}`;
};

/**
 * Validate WebSocket message format
 */
export const validateMessage = (message: any): message is WebSocketMessage => {
  if (!message || typeof message !== 'object') {
    return false;
  }

  return (
    typeof message.type === 'string' &&
    (message.content === undefined || typeof message.content === 'string') &&
    (message.data === undefined || typeof message.data === 'object') &&
    (message.timestamp === undefined || typeof message.timestamp === 'string')
  );
};

/**
 * Parse WebSocket message safely
 */
export const parseWebSocketMessage = (data: string): WebSocketMessage => {
  try {
    const parsed = JSON.parse(data);

    if (!validateMessage(parsed)) {
      throw new MessageError('Invalid message format');
    }

    return parsed;
  } catch (error) {
    if (error instanceof MessageError) {
      throw error;
    }
    throw new MessageError(`Failed to parse message: ${error}`);
  }
};

/**
 * Serialize message for WebSocket transmission
 */
export const serializeMessage = (message: WebSocketMessage): string => {
  try {
    // Add timestamp if not present
    const messageWithTimestamp = {
      ...message,
      timestamp: message.timestamp || new Date().toISOString()
    };

    return JSON.stringify(messageWithTimestamp);
  } catch (error) {
    throw new MessageError(`Failed to serialize message: ${error}`);
  }
};

/**
 * Get WebSocket ready state as enum
 */
export const getWebSocketState = (readyState: number): WebSocketState => {
  switch (readyState) {
    case WebSocket.CONNECTING:
      return WebSocketState.CONNECTING;
    case WebSocket.OPEN:
      return WebSocketState.CONNECTED;
    case WebSocket.CLOSING:
      return WebSocketState.DISCONNECTING;
    case WebSocket.CLOSED:
      return WebSocketState.DISCONNECTED;
    default:
      return WebSocketState.ERROR;
  }
};

/**
 * Check if WebSocket error is recoverable
 */
export const isRecoverableError = (error: Event | Error): boolean => {
  if (error instanceof WebSocketError) {
    return error.recoverable;
  }

  // Network errors are usually recoverable
  if (error instanceof ErrorEvent) {
    return !error.error || error.error.code !== 'ECONNREFUSED';
  }

  return true; // Default to recoverable
};

/**
 * Calculate exponential backoff delay
 */
export const calculateBackoffDelay = (
  attempt: number,
  baseDelay: number = 1000,
  maxDelay: number = 30000
): number => {
  const delay = Math.min(baseDelay * Math.pow(2, attempt), maxDelay);

  // Add jitter to prevent thundering herd
  const jitter = Math.random() * 0.1 * delay;

  return delay + jitter;
};

/**
 * Create a query message
 */
export const createQueryMessage = (
  question: string,
  options: {
    fileFilter?: string;
    sheetFilter?: string;
    maxResults?: number;
    includeStatistics?: boolean;
    streaming?: boolean;
    queryId?: string;
  } = {}
): WebSocketMessage => {
  return {
    type: 'query',
    data: {
      question,
      file_filter: options.fileFilter,
      sheet_filter: options.sheetFilter,
      max_results: options.maxResults || 5,
      include_statistics: options.includeStatistics || false,
      streaming: options.streaming !== false // Default to true
    },
    query_id: options.queryId || uuidv4(),
    timestamp: new Date().toISOString()
  };
};

/**
 * Create a ping message for heartbeat
 */
export const createPingMessage = (): WebSocketMessage => {
  return {
    type: 'ping',
    timestamp: new Date().toISOString()
  };
};

/**
 * Check if message is a streaming token
 */
export const isStreamingToken = (message: WebSocketMessage): boolean => {
  return message.type === 'token' || message.type === 'token_batch';
};

/**
 * Check if message indicates completion
 */
export const isCompletionMessage = (message: WebSocketMessage): boolean => {
  return message.type === 'complete';
};

/**
 * Check if message is an error
 */
export const isErrorMessage = (message: WebSocketMessage): boolean => {
  return message.type === 'error' || message.type === 'query_error';
};

/**
 * Extract token content from streaming message
 */
export const extractTokenContent = (message: WebSocketMessage): string => {
  if (message.type === 'token' || message.type === 'token_batch') {
    return message.content || '';
  }
  return '';
};

/**
 * Create connection error from WebSocket close event
 */
export const createConnectionError = (event: CloseEvent): ConnectionError => {
  const reason = event.reason || 'Connection closed unexpectedly';
  const code = event.code;

  let message = `WebSocket connection closed: ${reason}`;

  if (code === 1006) {
    message = 'Connection lost - network error or server unavailable';
  } else if (code === 1011) {
    message = 'Server error - please try again later';
  } else if (code === 1012) {
    message = 'Server restarting - reconnecting...';
  }

  return new ConnectionError(message);
};

/**
 * Debounce function for limiting message sending
 */
export const debounce = <T extends (...args: any[]) => void>(
  func: T,
  delay: number
): ((...args: Parameters<T>) => void) => {
  let timeoutId: NodeJS.Timeout;

  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func(...args), delay);
  };
};

/**
 * Throttle function for rate limiting
 */
export const throttle = <T extends (...args: any[]) => void>(
  func: T,
  limit: number
): ((...args: Parameters<T>) => void) => {
  let inThrottle: boolean;

  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
};

/**
 * Safe WebSocket send with error handling
 */
export const safeWebSocketSend = (
  socket: WebSocket,
  message: WebSocketMessage
): Promise<void> => {
  return new Promise((resolve, reject) => {
    if (socket.readyState !== WebSocket.OPEN) {
      reject(new ConnectionError('WebSocket is not connected'));
      return;
    }

    try {
      const serialized = serializeMessage(message);
      socket.send(serialized);
      resolve();
    } catch (error) {
      reject(new MessageError(`Failed to send message: ${error}`));
    }
  });
};

/**
 * Performance monitoring utilities
 */
export class PerformanceMonitor {
  private messageCount = 0;
  private tokenCount = 0;
  private startTime = Date.now();
  private latencySum = 0;
  private latencyCount = 0;

  recordMessage(): void {
    this.messageCount++;
  }

  recordTokens(count: number): void {
    this.tokenCount += count;
  }

  recordLatency(latency: number): void {
    this.latencySum += latency;
    this.latencyCount++;
  }

  getMetrics() {
    const uptime = Date.now() - this.startTime;
    const uptimeSeconds = uptime / 1000;

    return {
      messagesPerSecond: this.messageCount / uptimeSeconds,
      averageLatency: this.latencyCount > 0 ? this.latencySum / this.latencyCount : 0,
      connectionUptime: uptimeSeconds,
      totalMessages: this.messageCount,
      totalTokens: this.tokenCount
    };
  }

  reset(): void {
    this.messageCount = 0;
    this.tokenCount = 0;
    this.startTime = Date.now();
    this.latencySum = 0;
    this.latencyCount = 0;
  }
}