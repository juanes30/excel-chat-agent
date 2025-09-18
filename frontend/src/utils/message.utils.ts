/**
 * Message utility functions for chat functionality
 */

import { v4 as uuidv4 } from 'uuid';
import {
  ChatMessage,
  WebSocketMessage,
  TokenMessage,
  TokenBatchMessage,
  CompleteMessage,
  ErrorMessage
} from '../types/chat.types';

/**
 * Create a new chat message
 */
export const createChatMessage = (
  type: ChatMessage['type'],
  content: string,
  options: {
    sources?: string[];
    metadata?: ChatMessage['metadata'];
    isStreaming?: boolean;
  } = {}
): ChatMessage => {
  return {
    id: uuidv4(),
    type,
    content,
    timestamp: new Date(),
    isStreaming: options.isStreaming || false,
    sources: options.sources,
    metadata: options.metadata
  };
};

/**
 * Create a user message
 */
export const createUserMessage = (content: string): ChatMessage => {
  return createChatMessage('user', content);
};

/**
 * Create an assistant message (initially empty for streaming)
 */
export const createAssistantMessage = (content: string = '', isStreaming: boolean = true): ChatMessage => {
  return createChatMessage('assistant', content, { isStreaming });
};

/**
 * Create a system message
 */
export const createSystemMessage = (content: string): ChatMessage => {
  return createChatMessage('system', content);
};

/**
 * Create an error message
 */
export const createErrorMessage = (content: string): ChatMessage => {
  return createChatMessage('error', content);
};

/**
 * Update a streaming message with new content
 */
export const updateStreamingMessage = (
  message: ChatMessage,
  newContent: string,
  append: boolean = true
): ChatMessage => {
  return {
    ...message,
    content: append ? message.content + newContent : newContent,
    timestamp: new Date() // Update timestamp for latest activity
  };
};

/**
 * Complete a streaming message
 */
export const completeStreamingMessage = (
  message: ChatMessage,
  sources?: string[],
  metadata?: ChatMessage['metadata']
): ChatMessage => {
  return {
    ...message,
    isStreaming: false,
    sources: sources || message.sources,
    metadata: { ...message.metadata, ...metadata }
  };
};

/**
 * Convert WebSocket message to chat message
 */
export const websocketToChatMessage = (
  wsMessage: WebSocketMessage,
  existingMessage?: ChatMessage
): ChatMessage | null => {
  switch (wsMessage.type) {
    case 'token':
    case 'token_batch':
      const tokenContent = wsMessage.content || '';

      if (existingMessage && existingMessage.isStreaming) {
        // Append to existing streaming message
        return updateStreamingMessage(existingMessage, tokenContent, true);
      } else {
        // Create new streaming message
        return createAssistantMessage(tokenContent, true);
      }

    case 'complete':
      if (existingMessage) {
        const completeData = (wsMessage as CompleteMessage).data;
        return completeStreamingMessage(
          existingMessage,
          completeData?.sources,
          {
            processing_time_ms: completeData?.processing_time_ms,
            total_tokens: completeData?.total_tokens
          }
        );
      }
      return null;

    case 'error':
    case 'query_error':
      return createErrorMessage(wsMessage.content || 'An error occurred');

    case 'status':
      if (wsMessage.content === 'processing') {
        return createSystemMessage('Processing your request...');
      }
      return null;

    case 'response':
      return createAssistantMessage(wsMessage.content || '', false);

    case 'connection_established':
      return createSystemMessage('Connected to Excel Chat Agent');

    default:
      return null;
  }
};

/**
 * Format message timestamp for display
 */
export const formatMessageTime = (timestamp: Date): string => {
  const now = new Date();
  const diff = now.getTime() - timestamp.getTime();

  // Less than 1 minute
  if (diff < 60000) {
    return 'just now';
  }

  // Less than 1 hour
  if (diff < 3600000) {
    const minutes = Math.floor(diff / 60000);
    return `${minutes} min ago`;
  }

  // Less than 1 day
  if (diff < 86400000) {
    const hours = Math.floor(diff / 3600000);
    return `${hours} hour${hours > 1 ? 's' : ''} ago`;
  }

  // More than 1 day
  return timestamp.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
};

/**
 * Get message display text for different types
 */
export const getMessageDisplayText = (message: ChatMessage): string => {
  if (message.type === 'system' && message.content === 'Processing your request...') {
    return 'Processing your request...';
  }

  if (message.type === 'error') {
    return `Error: ${message.content}`;
  }

  return message.content;
};

/**
 * Check if message is from user
 */
export const isUserMessage = (message: ChatMessage): boolean => {
  return message.type === 'user';
};

/**
 * Check if message is from assistant
 */
export const isAssistantMessage = (message: ChatMessage): boolean => {
  return message.type === 'assistant';
};

/**
 * Check if message is a system message
 */
export const isSystemMessage = (message: ChatMessage): boolean => {
  return message.type === 'system';
};

/**
 * Check if message is an error
 */
export const isErrorMessage = (message: ChatMessage): boolean => {
  return message.type === 'error';
};

/**
 * Check if message is currently streaming
 */
export const isStreamingMessage = (message: ChatMessage): boolean => {
  return message.isStreaming === true;
};

/**
 * Get the last message of a specific type
 */
export const getLastMessageOfType = (
  messages: ChatMessage[],
  type: ChatMessage['type']
): ChatMessage | null => {
  const filtered = messages.filter(msg => msg.type === type);
  return filtered.length > 0 ? filtered[filtered.length - 1] : null;
};

/**
 * Get the last streaming message
 */
export const getLastStreamingMessage = (messages: ChatMessage[]): ChatMessage | null => {
  const streamingMessages = messages.filter(msg => msg.isStreaming);
  return streamingMessages.length > 0 ? streamingMessages[streamingMessages.length - 1] : null;
};

/**
 * Remove system messages older than specified minutes
 */
export const cleanupSystemMessages = (
  messages: ChatMessage[],
  maxAgeMinutes: number = 5
): ChatMessage[] => {
  const cutoff = new Date(Date.now() - maxAgeMinutes * 60 * 1000);

  return messages.filter(message => {
    if (message.type === 'system' && message.timestamp < cutoff) {
      return false;
    }
    return true;
  });
};

/**
 * Group messages by conversation (user query + assistant response)
 */
export const groupMessagesByConversation = (messages: ChatMessage[]): ChatMessage[][] => {
  const conversations: ChatMessage[][] = [];
  let currentConversation: ChatMessage[] = [];

  for (const message of messages) {
    if (message.type === 'user') {
      // Start new conversation
      if (currentConversation.length > 0) {
        conversations.push([...currentConversation]);
      }
      currentConversation = [message];
    } else if (message.type === 'assistant' || message.type === 'error') {
      // Add to current conversation
      currentConversation.push(message);
    }
    // Skip system messages for conversation grouping
  }

  // Add final conversation if it exists
  if (currentConversation.length > 0) {
    conversations.push(currentConversation);
  }

  return conversations;
};

/**
 * Calculate conversation statistics
 */
export const calculateConversationStats = (messages: ChatMessage[]) => {
  const userMessages = messages.filter(isUserMessage);
  const assistantMessages = messages.filter(isAssistantMessage);
  const errorMessages = messages.filter(isErrorMessage);

  const totalTokens = assistantMessages.reduce((sum, msg) => {
    return sum + (msg.metadata?.total_tokens || 0);
  }, 0);

  const averageResponseTime = assistantMessages.reduce((sum, msg) => {
    return sum + (msg.metadata?.processing_time_ms || 0);
  }, 0) / assistantMessages.length;

  return {
    totalQueries: userMessages.length,
    totalResponses: assistantMessages.length,
    totalErrors: errorMessages.length,
    totalTokens,
    averageResponseTime: averageResponseTime || 0,
    conversationLength: messages.length
  };
};

/**
 * Search messages by content
 */
export const searchMessages = (
  messages: ChatMessage[],
  query: string,
  options: {
    caseSensitive?: boolean;
    includeSystemMessages?: boolean;
  } = {}
): ChatMessage[] => {
  const searchQuery = options.caseSensitive ? query : query.toLowerCase();

  return messages.filter(message => {
    if (!options.includeSystemMessages && message.type === 'system') {
      return false;
    }

    const content = options.caseSensitive ? message.content : message.content.toLowerCase();
    return content.includes(searchQuery);
  });
};

/**
 * Export messages to JSON
 */
export const exportMessagesToJSON = (messages: ChatMessage[]): string => {
  const exportData = {
    exported_at: new Date().toISOString(),
    message_count: messages.length,
    messages: messages.map(msg => ({
      ...msg,
      timestamp: msg.timestamp.toISOString()
    }))
  };

  return JSON.stringify(exportData, null, 2);
};

/**
 * Sanitize message content for display (prevent XSS)
 */
export const sanitizeMessageContent = (content: string): string => {
  // Basic HTML escaping
  return content
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
    .replace(/\//g, '&#x2F;');
};

/**
 * Truncate long messages for preview
 */
export const truncateMessage = (content: string, maxLength: number = 100): string => {
  if (content.length <= maxLength) {
    return content;
  }

  return content.substring(0, maxLength) + '...';
};