/**
 * MessageList component for displaying chat messages with streaming support
 */

import React, { useEffect, useRef, useState, useMemo } from 'react';
import { ChatMessage } from '../types/chat.types';
import {
  isUserMessage,
  isAssistantMessage,
  isSystemMessage,
  isErrorMessage,
  isStreamingMessage,
  formatMessageTime,
  truncateMessage,
  sanitizeMessageContent
} from '../utils/message.utils';
import { User, Bot, AlertCircle, Info, Clock, CheckCircle, Loader2 } from 'lucide-react';

interface MessageListProps {
  messages: ChatMessage[];
  isTyping?: boolean;
  className?: string;
  autoScroll?: boolean;
  showTimestamps?: boolean;
  maxDisplayLength?: number;
}

interface MessageItemProps {
  message: ChatMessage;
  showTimestamp: boolean;
  maxDisplayLength: number;
}

const MessageItem: React.FC<MessageItemProps> = ({ message, showTimestamp, maxDisplayLength }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const isLongMessage = message.content.length > maxDisplayLength;
  const displayContent = isLongMessage && !isExpanded
    ? truncateMessage(message.content, maxDisplayLength)
    : message.content;

  const MessageIcon = () => {
    if (isUserMessage(message)) {
      return <User className="w-6 h-6 text-blue-600" />;
    } else if (isAssistantMessage(message)) {
      return <Bot className="w-6 h-6 text-green-600" />;
    } else if (isErrorMessage(message)) {
      return <AlertCircle className="w-6 h-6 text-red-600" />;
    } else {
      return <Info className="w-6 h-6 text-gray-600" />;
    }
  };

  const getMessageStyles = () => {
    const baseStyles = "flex gap-3 p-4 rounded-lg shadow-sm border";

    if (isUserMessage(message)) {
      return `${baseStyles} bg-blue-50 border-blue-200 ml-8`;
    } else if (isAssistantMessage(message)) {
      return `${baseStyles} bg-green-50 border-green-200 mr-8`;
    } else if (isErrorMessage(message)) {
      return `${baseStyles} bg-red-50 border-red-200`;
    } else {
      return `${baseStyles} bg-gray-50 border-gray-200`;
    }
  };

  const getContentStyles = () => {
    if (isErrorMessage(message)) {
      return "text-red-800";
    } else if (isSystemMessage(message)) {
      return "text-gray-700 italic";
    }
    return "text-gray-900";
  };

  return (
    <div className={getMessageStyles()}>
      <div className="flex-shrink-0">
        <MessageIcon />
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="font-medium text-sm">
              {isUserMessage(message) ? 'You' :
               isAssistantMessage(message) ? 'Assistant' :
               isErrorMessage(message) ? 'Error' : 'System'}
            </span>

            {isStreamingMessage(message) && (
              <div className="flex items-center gap-1 text-xs text-gray-500">
                <Loader2 className="w-3 h-3 animate-spin" />
                <span>Streaming...</span>
              </div>
            )}

            {message.metadata?.total_tokens && (
              <span className="text-xs text-gray-500 bg-gray-200 px-2 py-1 rounded">
                {message.metadata.total_tokens} tokens
              </span>
            )}
          </div>

          {showTimestamp && (
            <div className="flex items-center gap-1 text-xs text-gray-500">
              <Clock className="w-3 h-3" />
              <span>{formatMessageTime(message.timestamp)}</span>
            </div>
          )}
        </div>

        <div className={`${getContentStyles()} whitespace-pre-wrap break-words`}>
          <div dangerouslySetInnerHTML={{
            __html: sanitizeMessageContent(displayContent)
          }} />

          {isLongMessage && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="mt-2 text-sm text-blue-600 hover:text-blue-800 underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded"
            >
              {isExpanded ? 'Show less' : 'Show more'}
            </button>
          )}
        </div>

        {message.sources && message.sources.length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-200">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Sources:</h4>
            <div className="flex flex-wrap gap-2">
              {message.sources.map((source, index) => (
                <span
                  key={index}
                  className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full"
                >
                  {source}
                </span>
              ))}
            </div>
          </div>
        )}

        {message.metadata && (message.metadata.processing_time_ms || message.metadata.total_tokens) && (
          <div className="mt-2 flex items-center gap-4 text-xs text-gray-500">
            {message.metadata.processing_time_ms && (
              <div className="flex items-center gap-1">
                <CheckCircle className="w-3 h-3" />
                <span>{message.metadata.processing_time_ms}ms</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const TypingIndicator: React.FC = () => (
  <div className="flex gap-3 p-4 rounded-lg bg-green-50 border border-green-200 mr-8">
    <div className="flex-shrink-0">
      <Bot className="w-6 h-6 text-green-600" />
    </div>
    <div className="flex-1">
      <div className="flex items-center gap-2 mb-2">
        <span className="font-medium text-sm">Assistant</span>
        <div className="flex items-center gap-1 text-xs text-gray-500">
          <Loader2 className="w-3 h-3 animate-spin" />
          <span>Typing...</span>
        </div>
      </div>
      <div className="flex gap-1">
        <div className="w-2 h-2 bg-green-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
        <div className="w-2 h-2 bg-green-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
        <div className="w-2 h-2 bg-green-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
      </div>
    </div>
  </div>
);

export const MessageList: React.FC<MessageListProps> = ({
  messages,
  isTyping = false,
  className = '',
  autoScroll = true,
  showTimestamps = true,
  maxDisplayLength = 1000
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [shouldAutoScroll, setShouldAutoScroll] = useState(autoScroll);
  const [lastMessageCount, setLastMessageCount] = useState(messages.length);

  // Memoize filtered messages to avoid unnecessary re-renders
  const displayMessages = useMemo(() => {
    return messages.filter(msg => msg.content.trim() !== '');
  }, [messages]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (shouldAutoScroll && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({
        behavior: messages.length > lastMessageCount ? 'smooth' : 'auto'
      });
    }
    setLastMessageCount(messages.length);
  }, [messages.length, shouldAutoScroll, lastMessageCount]);

  // Handle scroll events to detect if user has scrolled up
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !autoScroll) return;

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
      setShouldAutoScroll(isNearBottom);
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    return () => container.removeEventListener('scroll', handleScroll);
  }, [autoScroll]);

  // Reset auto-scroll when typing starts
  useEffect(() => {
    if (isTyping && autoScroll) {
      setShouldAutoScroll(true);
    }
  }, [isTyping, autoScroll]);

  if (displayMessages.length === 0 && !isTyping) {
    return (
      <div className={`flex items-center justify-center h-full text-gray-500 ${className}`}>
        <div className="text-center">
          <Bot className="w-16 h-16 mx-auto mb-4 text-gray-400" />
          <h3 className="text-lg font-medium mb-2">Welcome to Excel Chat Agent</h3>
          <p className="text-sm">Ask questions about your Excel files and get instant answers.</p>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={`flex-1 overflow-y-auto space-y-4 p-4 ${className}`}
      style={{ scrollBehavior: 'smooth' }}
    >
      {displayMessages.map((message) => (
        <MessageItem
          key={message.id}
          message={message}
          showTimestamp={showTimestamps}
          maxDisplayLength={maxDisplayLength}
        />
      ))}

      {isTyping && <TypingIndicator />}

      <div ref={messagesEndRef} className="h-1" />

      {!shouldAutoScroll && autoScroll && (
        <button
          onClick={() => {
            setShouldAutoScroll(true);
            messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
          }}
          className="fixed bottom-24 right-8 bg-blue-600 text-white p-3 rounded-full shadow-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors"
          aria-label="Scroll to bottom"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
        </button>
      )}
    </div>
  );
};