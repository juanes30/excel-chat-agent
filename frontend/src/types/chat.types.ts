/**
 * TypeScript interfaces for Excel Chat Agent WebSocket communication
 * Based on backend FastAPI WebSocket message structure
 */

// WebSocket Connection States
export enum WebSocketState {
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  DISCONNECTING = 'disconnecting',
  DISCONNECTED = 'disconnected',
  ERROR = 'error',
  RECONNECTING = 'reconnecting'
}

// WebSocket Message Types (matching backend implementation)
export type WebSocketMessageType =
  | 'query'           // Send question to LLM
  | 'token'           // Individual token streaming (legacy)
  | 'token_batch'     // Optimized batch streaming
  | 'complete'        // Response completion with metadata
  | 'error'           // Error messages
  | 'status'          // Processing status updates
  | 'ping'            // Keep connection alive
  | 'pong'            // Response to ping
  | 'response'        // Simple response message
  | 'connection_established'  // Welcome message
  | 'query_received'  // Query acknowledgment
  | 'query_response'  // Complete query response
  | 'query_error';    // Query-specific errors

// Base WebSocket Message Interface
export interface WebSocketMessage {
  type: WebSocketMessageType;
  content?: string;
  data?: Record<string, any>;
  timestamp?: string;
  session_id?: string;
  query_id?: string;
}

// Specific Message Interfaces
export interface QueryMessage extends WebSocketMessage {
  type: 'query';
  data: {
    question: string;
    file_filter?: string;
    sheet_filter?: string;
    max_results?: number;
    include_statistics?: boolean;
    streaming?: boolean;
  };
}

export interface TokenMessage extends WebSocketMessage {
  type: 'token';
  content: string;
  timestamp: string;
}

export interface TokenBatchMessage extends WebSocketMessage {
  type: 'token_batch';
  content: string;
  token_count: number;
  timestamp: string;
}

export interface CompleteMessage extends WebSocketMessage {
  type: 'complete';
  data: {
    sources: string[];
    total_tokens: number;
    processing_time_ms?: number;
  };
  timestamp: string;
}

export interface ErrorMessage extends WebSocketMessage {
  type: 'error';
  content: string;
  timestamp: string;
}

export interface StatusMessage extends WebSocketMessage {
  type: 'status';
  content: 'processing' | 'completed' | 'failed';
  timestamp: string;
}

// Chat Message Interfaces for UI
export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant' | 'system' | 'error';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  sources?: string[];
  metadata?: {
    processing_time_ms?: number;
    total_tokens?: number;
    file_filter?: string;
    sheet_filter?: string;
  };
}

// Chat State Management
export interface ChatState {
  messages: ChatMessage[];
  isConnected: boolean;
  connectionState: WebSocketState;
  isTyping: boolean;
  currentMessage: string;
  sessionId: string;
  lastError: string | null;
  reconnectAttempts: number;
  maxReconnectAttempts: number;
}

// WebSocket Configuration
export interface WebSocketConfig {
  url: string;
  reconnectInterval: number;
  maxReconnectAttempts: number;
  heartbeatInterval: number;
  autoReconnect: boolean;
  protocols?: string[];
}

// Excel-specific interfaces
export interface ExcelFileInfo {
  name: string;
  file_hash: string;
  total_sheets: number;
  total_rows: number;
  total_columns: number;
  file_size_mb: number;
  last_modified: Date;
  sheets: string[];
  lastModified?: Date;
  size?: number;
}

export interface QueryFilters {
  fileFilter?: string;
  sheetFilter?: string;
  maxResults?: number;
  includeStatistics?: boolean;
}

// Hook Return Types
export interface UseWebSocketReturn {
  connectionState: WebSocketState;
  isConnected: boolean;
  sendMessage: (message: WebSocketMessage) => void;
  lastMessage: WebSocketMessage | null;
  reconnect: () => void;
  disconnect: () => void;
}

export interface UseChatReturn {
  messages: ChatMessage[];
  sendQuery: (query: string, filters?: QueryFilters) => void;
  isTyping: boolean;
  isConnected: boolean;
  connectionState: WebSocketState;
  clearMessages: () => void;
  reconnect: () => void;
  lastError: string | null;
  retryLastQuery: () => void;
}

// Error Types
export class WebSocketError extends Error {
  constructor(
    message: string,
    public code?: string,
    public recoverable: boolean = true
  ) {
    super(message);
    this.name = 'WebSocketError';
  }
}

export class ConnectionError extends WebSocketError {
  constructor(message: string) {
    super(message, 'CONNECTION_ERROR', true);
    this.name = 'ConnectionError';
  }
}

export class MessageError extends WebSocketError {
  constructor(message: string) {
    super(message, 'MESSAGE_ERROR', false);
    this.name = 'MessageError';
  }
}

// Utility Types
export type MessageHandler = (message: WebSocketMessage) => void;
export type ErrorHandler = (error: WebSocketError) => void;
export type ConnectionStateHandler = (state: WebSocketState) => void;

// Performance Monitoring Types
export interface PerformanceMetrics {
  messagesPerSecond: number;
  averageLatency: number;
  connectionUptime: number;
  totalMessages: number;
  totalTokens: number;
  reconnectionCount: number;
}

// Component Props Types
export interface ChatContainerProps {
  className?: string;
  websocketUrl?: string;
  autoConnect?: boolean;
  maxMessages?: number;
  enableFileUpload?: boolean;
}

export interface MessageListProps {
  messages: ChatMessage[];
  isTyping?: boolean;
  className?: string;
}

export interface MessageInputProps {
  onSendMessage: (message: string, filters?: QueryFilters) => void;
  disabled?: boolean;
  placeholder?: string;
  maxLength?: number;
  className?: string;
}

export interface ConnectionStatusProps {
  connectionState: WebSocketState;
  reconnectAttempts: number;
  maxReconnectAttempts: number;
  onReconnect?: () => void;
  className?: string;
}

// Theme and Styling
export interface ChatTheme {
  primary: string;
  secondary: string;
  background: string;
  text: string;
  border: string;
  userMessage: string;
  assistantMessage: string;
  systemMessage: string;
  errorMessage: string;
}