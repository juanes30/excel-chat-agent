/**
 * Custom hook for WebSocket connection management with auto-reconnection
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import {
  WebSocketState,
  WebSocketMessage,
  UseWebSocketReturn,
  WebSocketError,
  ConnectionError
} from '../types/chat.types';
import {
  createWebSocketUrl,
  generateSessionId,
  parseWebSocketMessage,
  safeWebSocketSend,
  calculateBackoffDelay,
  isRecoverableError,
  createPingMessage,
  PerformanceMonitor
} from '../utils/websocket.utils';

interface UseWebSocketOptions {
  url: string;
  autoConnect?: boolean;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  protocols?: string[];
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (error: WebSocketError) => void;
  onMessage?: (message: WebSocketMessage) => void;
}

export const useWebSocket = (options: UseWebSocketOptions): UseWebSocketReturn => {
  const {
    url,
    autoConnect = true,
    maxReconnectAttempts = 5,
    heartbeatInterval = 30000,
    protocols,
    onOpen,
    onClose,
    onError,
    onMessage
  } = options;

  const [connectionState, setConnectionState] = useState<WebSocketState>(WebSocketState.DISCONNECTED);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  const websocketRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string>(generateSessionId());
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const performanceMonitorRef = useRef(new PerformanceMonitor());
  const isReconnectingRef = useRef(false);

  const isConnected = connectionState === WebSocketState.CONNECTED;

  const cleanup = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (heartbeatTimeoutRef.current) {
      clearTimeout(heartbeatTimeoutRef.current);
      heartbeatTimeoutRef.current = null;
    }
  }, []);

  const startHeartbeat = useCallback(() => {
    if (heartbeatTimeoutRef.current) {
      clearTimeout(heartbeatTimeoutRef.current);
    }

    heartbeatTimeoutRef.current = setTimeout(() => {
      if (websocketRef.current?.readyState === WebSocket.OPEN) {
        const pingMessage = createPingMessage();
        safeWebSocketSend(websocketRef.current, pingMessage).catch((error) => {
          console.warn('Failed to send heartbeat:', error);
        });
        startHeartbeat(); // Schedule next heartbeat
      }
    }, heartbeatInterval);
  }, [heartbeatInterval]);

  const connect = useCallback(() => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    try {
      cleanup();
      setConnectionState(WebSocketState.CONNECTING);

      const wsUrl = createWebSocketUrl(url, sessionIdRef.current);
      const socket = new WebSocket(wsUrl, protocols);

      socket.onopen = (event) => {
        console.log('✅ WebSocket connected');
        setConnectionState(WebSocketState.CONNECTED);
        setReconnectAttempts(0);
        isReconnectingRef.current = false;
        performanceMonitorRef.current.reset();
        startHeartbeat();
        onOpen?.(event);
      };

      socket.onclose = (event) => {
        console.log(`❌ WebSocket closed: code=${event.code}, reason="${event.reason}", wasClean=${event.wasClean}`);
        setConnectionState(WebSocketState.DISCONNECTED);
        cleanup();

        // DISABLED: Let external reconnection logic handle this
        // Internal reconnection logic removed to prevent conflicts

        onClose?.(event);
      };

      socket.onerror = (event) => {
        console.log('🚨 WebSocket error:', event);
        setConnectionState(WebSocketState.ERROR);
        const error = new WebSocketError('WebSocket connection error');
        onError?.(error);
      };

      socket.onmessage = (event) => {
        try {
          const message = parseWebSocketMessage(event.data);
          setLastMessage(message);
          performanceMonitorRef.current.recordMessage();

          // Handle pong responses
          if (message.type === 'pong') {
            return; // Just acknowledge, don't pass to handlers
          }

          onMessage?.(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
          const wsError = new WebSocketError('Invalid message format');
          onError?.(wsError);
        }
      };

      websocketRef.current = socket;
    } catch (error) {
      setConnectionState(WebSocketState.ERROR);
      const wsError = new ConnectionError(`Failed to create WebSocket connection: ${error}`);
      onError?.(wsError);
    }
  }, [url, protocols, reconnectAttempts, maxReconnectAttempts, onOpen, onClose, onError, onMessage, cleanup, startHeartbeat]);

  const disconnect = useCallback(() => {
    cleanup();
    isReconnectingRef.current = false;

    if (websocketRef.current) {
      websocketRef.current.close(1000, 'User disconnected');
      websocketRef.current = null;
    }

    setConnectionState(WebSocketState.DISCONNECTED);
    setReconnectAttempts(0);
  }, [cleanup]);

  const reconnect = useCallback(() => {
    disconnect();
    sessionIdRef.current = generateSessionId(); // Generate new session
    setReconnectAttempts(0);
    connect();
  }, [disconnect, connect]);

  const sendMessage = useCallback(async (message: WebSocketMessage) => {
    if (!websocketRef.current || websocketRef.current.readyState !== WebSocket.OPEN) {
      throw new ConnectionError('WebSocket is not connected');
    }

    try {
      await safeWebSocketSend(websocketRef.current, message);
    } catch (error) {
      const wsError = error instanceof WebSocketError ? error :
        new WebSocketError(`Failed to send message: ${error}`);
      onError?.(wsError);
      throw wsError;
    }
  }, [onError]);

  // Auto-connect on mount - FIXED: removed connect from dependencies
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    // Cleanup on unmount
    return () => {
      cleanup();
      if (websocketRef.current) {
        websocketRef.current.close();
        websocketRef.current = null;
      }
    };
  }, [autoConnect]); // Only depend on autoConnect, not connect function

  return {
    connectionState,
    isConnected,
    sendMessage,
    lastMessage,
    reconnect,
    disconnect
  };
};