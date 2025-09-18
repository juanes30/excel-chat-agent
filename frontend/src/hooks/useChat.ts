/**
 * Custom hook for chat functionality with WebSocket integration
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import {
  ChatMessage,
  QueryFilters,
  UseChatReturn,
  WebSocketMessage,
  WebSocketState,
  WebSocketError
} from '../types/chat.types';
import {
  createUserMessage,
  createAssistantMessage,
  websocketToChatMessage,
  getLastStreamingMessage,
  cleanupSystemMessages,
  completeStreamingMessage
} from '../utils/message.utils';
import {
  createQueryMessage,
  isStreamingToken,
  isCompletionMessage,
  isErrorMessage
} from '../utils/websocket.utils';
import { useWebSocket } from './useWebSocket';

interface UseChatOptions {
  websocketUrl: string;
  maxMessages?: number;
  autoCleanupInterval?: number;
  onError?: (error: WebSocketError) => void;
}

export const useChat = (options: UseChatOptions): UseChatReturn => {
  const {
    websocketUrl,
    maxMessages = 100,
    autoCleanupInterval = 5 * 60 * 1000, // 5 minutes
    onError
  } = options;

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [lastError, setLastError] = useState<string | null>(null);

  const currentQueryRef = useRef<string | null>(null);
  const streamingMessageIdRef = useRef<string | null>(null);
  const cleanupIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const handleWebSocketMessage = useCallback((wsMessage: WebSocketMessage) => {
    try {
      setLastError(null);

      // Handle different message types
      if (isStreamingToken(wsMessage)) {
        setMessages(prevMessages => {
          const lastStreamingMessage = getLastStreamingMessage(prevMessages);

          if (lastStreamingMessage && lastStreamingMessage.isStreaming) {
            // Update existing streaming message
            const updatedMessage = websocketToChatMessage(wsMessage, lastStreamingMessage);
            if (updatedMessage) {
              streamingMessageIdRef.current = updatedMessage.id;
              return prevMessages.map(msg =>
                msg.id === lastStreamingMessage.id ? updatedMessage : msg
              );
            }
          } else {
            // Create new streaming message
            const newMessage = websocketToChatMessage(wsMessage);
            if (newMessage) {
              streamingMessageIdRef.current = newMessage.id;
              return [...prevMessages, newMessage];
            }
          }

          return prevMessages;
        });
      } else if (isCompletionMessage(wsMessage)) {
        setIsTyping(false);

        setMessages(prevMessages => {
          const lastStreamingMessage = getLastStreamingMessage(prevMessages);

          if (lastStreamingMessage) {
            const completedMessage = websocketToChatMessage(wsMessage, lastStreamingMessage);
            if (completedMessage) {
              streamingMessageIdRef.current = null;
              return prevMessages.map(msg =>
                msg.id === lastStreamingMessage.id ? completedMessage : msg
              );
            }
          }

          return prevMessages;
        });
      } else if (isErrorMessage(wsMessage)) {
        setIsTyping(false);
        const errorMessage = websocketToChatMessage(wsMessage);
        if (errorMessage) {
          setMessages(prev => [...prev, errorMessage]);
          setLastError(wsMessage.content || 'An error occurred');
        }
      } else if (wsMessage.type === 'status') {
        if (wsMessage.content === 'processing') {
          setIsTyping(true);
        }

        const statusMessage = websocketToChatMessage(wsMessage);
        if (statusMessage) {
          setMessages(prev => [...prev, statusMessage]);
        }
      } else if (wsMessage.type === 'connection_established') {
        const welcomeMessage = websocketToChatMessage(wsMessage);
        if (welcomeMessage) {
          setMessages(prev => [...prev, welcomeMessage]);
        }
      }
    } catch (error) {
      console.error('Error handling WebSocket message:', error);
      setLastError('Failed to process message');
    }
  }, []);

  const handleWebSocketError = useCallback((error: WebSocketError) => {
    setIsTyping(false);
    setLastError(error.message);
    onError?.(error);
  }, [onError]);

  const { connectionState, isConnected, sendMessage, reconnect, disconnect } = useWebSocket({
    url: websocketUrl,
    autoConnect: true,
    maxReconnectAttempts: 5,
    heartbeatInterval: 30000,
    onMessage: handleWebSocketMessage,
    onError: handleWebSocketError,
    onOpen: () => {
      setLastError(null);
    },
    onClose: (event) => {
      setIsTyping(false);
      if (!event.wasClean) {
        setLastError('Connection lost');
      }
    }
  });

  const sendQuery = useCallback(async (query: string, filters?: QueryFilters) => {
    if (!isConnected) {
      setLastError('Not connected to server');
      return;
    }

    if (!query.trim()) {
      setLastError('Query cannot be empty');
      return;
    }

    try {
      // Add user message to chat
      const userMessage = createUserMessage(query);
      setMessages(prev => [...prev, userMessage]);

      // Store current query
      currentQueryRef.current = query;

      // Create and send WebSocket query message
      const queryMessage = createQueryMessage(query, {
        fileFilter: filters?.fileFilter,
        sheetFilter: filters?.sheetFilter,
        maxResults: filters?.maxResults,
        includeStatistics: filters?.includeStatistics,
        streaming: true
      });

      await sendMessage(queryMessage);
      setIsTyping(true);
      setLastError(null);

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Failed to send query';
      setLastError(errorMsg);
      setIsTyping(false);
    }
  }, [isConnected, sendMessage]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setIsTyping(false);
    setLastError(null);
    streamingMessageIdRef.current = null;
    currentQueryRef.current = null;
  }, []);

  const retryLastQuery = useCallback(() => {
    if (currentQueryRef.current) {
      sendQuery(currentQueryRef.current);
    }
  }, [sendQuery]);

  // Auto-cleanup old system messages
  useEffect(() => {
    if (autoCleanupInterval > 0) {
      cleanupIntervalRef.current = setInterval(() => {
        setMessages(prev => cleanupSystemMessages(prev, 5)); // 5 minutes
      }, autoCleanupInterval);

      return () => {
        if (cleanupIntervalRef.current) {
          clearInterval(cleanupIntervalRef.current);
        }
      };
    }
  }, [autoCleanupInterval]);

  // Limit message history
  useEffect(() => {
    if (messages.length > maxMessages) {
      setMessages(prev => prev.slice(-maxMessages));
    }
  }, [messages.length, maxMessages]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (cleanupIntervalRef.current) {
        clearInterval(cleanupIntervalRef.current);
      }
    };
  }, []);

  return {
    messages,
    sendQuery,
    isTyping,
    isConnected,
    connectionState,
    clearMessages,
    reconnect,
    lastError,
    retryLastQuery
  };
};