/**
 * Chat Context for centralized state management and WebSocket communication
 */

import React, { createContext, useContext, useReducer, useCallback, useEffect, ReactNode } from 'react';
import {
  ChatMessage,
  ChatState,
  WebSocketState,
  QueryFilters,
  ExcelFileInfo,
  PerformanceMetrics
} from '../types/chat.types';
import { useChat } from '../hooks/useChat';
import { useReconnection } from '../hooks/useReconnection';

// Action types for the chat reducer
type ChatAction =
  | { type: 'SET_MESSAGES'; payload: ChatMessage[] }
  | { type: 'ADD_MESSAGE'; payload: ChatMessage }
  | { type: 'UPDATE_MESSAGE'; payload: { id: string; updates: Partial<ChatMessage> } }
  | { type: 'CLEAR_MESSAGES' }
  | { type: 'SET_CONNECTION_STATE'; payload: WebSocketState }
  | { type: 'SET_TYPING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_CURRENT_MESSAGE'; payload: string }
  | { type: 'SET_SESSION_ID'; payload: string }
  | { type: 'INCREMENT_RECONNECT_ATTEMPTS' }
  | { type: 'RESET_RECONNECT_ATTEMPTS' }
  | { type: 'SET_FILES'; payload: ExcelFileInfo[] }
  | { type: 'SET_PERFORMANCE_METRICS'; payload: PerformanceMetrics };

// Extended state that includes additional UI and application state
interface ExtendedChatState extends ChatState {
  files: ExcelFileInfo[];
  selectedFile: string | null;
  selectedSheet: string | null;
  performanceMetrics: PerformanceMetrics | null;
  isUploading: boolean;
  uploadProgress: number;
  systemStats: any;
}

interface ChatContextValue {
  state: ExtendedChatState;
  sendQuery: (query: string, filters?: QueryFilters) => Promise<void>;
  clearMessages: () => void;
  reconnect: () => void;
  setCurrentMessage: (message: string) => void;
  setSelectedFile: (fileName: string | null) => void;
  setSelectedSheet: (sheetName: string | null) => void;
  uploadFile: (file: File) => Promise<void>;
  refreshFiles: () => Promise<void>;
  retryLastQuery: () => void;
}

const initialState: ExtendedChatState = {
  messages: [],
  isConnected: false,
  connectionState: WebSocketState.DISCONNECTED,
  isTyping: false,
  currentMessage: '',
  sessionId: '',
  lastError: null,
  reconnectAttempts: 0,
  maxReconnectAttempts: 10,
  files: [],
  selectedFile: null,
  selectedSheet: null,
  performanceMetrics: null,
  isUploading: false,
  uploadProgress: 0,
  systemStats: null
};

function chatReducer(state: ExtendedChatState, action: ChatAction): ExtendedChatState {
  switch (action.type) {
    case 'SET_MESSAGES':
      return {
        ...state,
        messages: action.payload
      };

    case 'ADD_MESSAGE':
      return {
        ...state,
        messages: [...state.messages, action.payload]
      };

    case 'UPDATE_MESSAGE': {
      const { id, updates } = action.payload;
      return {
        ...state,
        messages: state.messages.map(msg =>
          msg.id === id ? { ...msg, ...updates } : msg
        )
      };
    }

    case 'CLEAR_MESSAGES':
      return {
        ...state,
        messages: [],
        lastError: null
      };

    case 'SET_CONNECTION_STATE':
      return {
        ...state,
        connectionState: action.payload,
        isConnected: action.payload === WebSocketState.CONNECTED
      };

    case 'SET_TYPING':
      return {
        ...state,
        isTyping: action.payload
      };

    case 'SET_ERROR':
      return {
        ...state,
        lastError: action.payload
      };

    case 'SET_CURRENT_MESSAGE':
      return {
        ...state,
        currentMessage: action.payload
      };

    case 'SET_SESSION_ID':
      return {
        ...state,
        sessionId: action.payload
      };

    case 'INCREMENT_RECONNECT_ATTEMPTS':
      return {
        ...state,
        reconnectAttempts: state.reconnectAttempts + 1
      };

    case 'RESET_RECONNECT_ATTEMPTS':
      return {
        ...state,
        reconnectAttempts: 0
      };

    case 'SET_FILES':
      return {
        ...state,
        files: action.payload
      };

    case 'SET_PERFORMANCE_METRICS':
      return {
        ...state,
        performanceMetrics: action.payload
      };

    default:
      return state;
  }
}

const ChatContext = createContext<ChatContextValue | undefined>(undefined);

interface ChatProviderProps {
  children: ReactNode;
  websocketUrl?: string;
  apiBaseUrl?: string;
}

export const ChatProvider: React.FC<ChatProviderProps> = ({
  children,
  websocketUrl = 'ws://localhost:8005/ws/chat',
  apiBaseUrl = 'http://localhost:8005'
}) => {
  const [state, dispatch] = useReducer(chatReducer, initialState);

  const {
    messages,
    sendQuery: chatSendQuery,
    isTyping,
    isConnected,
    connectionState,
    clearMessages: chatClearMessages,
    reconnect: chatReconnect,
    lastError,
    retryLastQuery
  } = useChat({
    websocketUrl,
    maxMessages: 200,
    onError: (error) => {
      dispatch({ type: 'SET_ERROR', payload: error.message });
    }
  });

  const {
    reconnectAttempts,
    isReconnecting,
    shouldReconnect,
    startReconnection,
    resetReconnection
  } = useReconnection({
    strategy: {
      maxAttempts: 10,
      baseDelay: 1000,
      maxDelay: 30000
    },
    onReconnectAttempt: (attempt, delay) => {
      console.log(`🔄 Reconnection attempt ${attempt} in ${delay}ms - Connection state: ${connectionState}`);
    },
    onReconnectSuccess: () => {
      dispatch({ type: 'RESET_RECONNECT_ATTEMPTS' });
      dispatch({ type: 'SET_ERROR', payload: null });
    },
    onReconnectFailure: (error) => {
      dispatch({ type: 'SET_ERROR', payload: error.message });
    }
  });

  // Sync chat hook state with context state
  useEffect(() => {
    dispatch({ type: 'SET_MESSAGES', payload: messages });
  }, [messages]);

  useEffect(() => {
    dispatch({ type: 'SET_CONNECTION_STATE', payload: connectionState });
  }, [connectionState]);

  useEffect(() => {
    dispatch({ type: 'SET_TYPING', payload: isTyping });
  }, [isTyping]);

  useEffect(() => {
    if (lastError) {
      dispatch({ type: 'SET_ERROR', payload: lastError });
    }
  }, [lastError]);

  // Auto-reconnection logic - FIXED: prevent reconnection on clean disconnects
  useEffect(() => {
    // Only reconnect on ERROR state, not on clean DISCONNECTED
    if (connectionState === WebSocketState.ERROR && !isReconnecting) {
      console.log(`🔍 Starting reconnection for ERROR state`);
      startReconnection(chatReconnect);
    }
  }, [connectionState, isReconnecting]); // Only state dependencies

  const sendQuery = useCallback(async (query: string, filters?: QueryFilters) => {
    dispatch({ type: 'SET_ERROR', payload: null });

    const queryFilters: QueryFilters = {
      fileFilter: filters?.fileFilter || state.selectedFile || undefined,
      sheetFilter: filters?.sheetFilter || state.selectedSheet || undefined,
      maxResults: filters?.maxResults || 5,
      includeStatistics: filters?.includeStatistics || false
    };

    await chatSendQuery(query, queryFilters);
  }, [chatSendQuery, state.selectedFile, state.selectedSheet]);

  const clearMessages = useCallback(() => {
    chatClearMessages();
    dispatch({ type: 'CLEAR_MESSAGES' });
  }, [chatClearMessages]);

  const reconnect = useCallback(() => {
    resetReconnection();
    chatReconnect();
  }, [resetReconnection, chatReconnect]);

  const setCurrentMessage = useCallback((message: string) => {
    dispatch({ type: 'SET_CURRENT_MESSAGE', payload: message });
  }, []);

  const setSelectedFile = useCallback((fileName: string | null) => {
    dispatch({ type: 'SET_FILES', payload: state.files.map(f => ({ ...f, selected: f.name === fileName })) });
  }, [state.files]);

  const setSelectedSheet = useCallback((sheetName: string | null) => {
    // Update selected sheet logic would go here
    console.log('Selected sheet:', sheetName);
  }, []);

  const uploadFile = useCallback(async (file: File): Promise<void> => {
    if (!file.name.match(/\.(xlsx?|csv)$/i)) {
      throw new Error('Please select an Excel file (.xlsx, .xls) or CSV file');
    }

    dispatch({ type: 'SET_ERROR', payload: null });

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${apiBaseUrl}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Upload failed: ${response.statusText}`);
      }

      const result = await response.json();

      // Refresh file list after successful upload
      await refreshFiles();

      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Upload failed';
      dispatch({ type: 'SET_ERROR', payload: errorMessage });
      throw error;
    }
  }, [apiBaseUrl]);

  const refreshFiles = useCallback(async (): Promise<void> => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/files`);

      if (!response.ok) {
        throw new Error(`Failed to fetch files: ${response.statusText}`);
      }

      const files = await response.json();
      dispatch({ type: 'SET_FILES', payload: files });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load files';
      dispatch({ type: 'SET_ERROR', payload: errorMessage });
    }
  }, [apiBaseUrl]);

  // Load files on mount
  useEffect(() => {
    refreshFiles();
  }, [refreshFiles]);

  const contextValue: ChatContextValue = {
    state,
    sendQuery,
    clearMessages,
    reconnect,
    setCurrentMessage,
    setSelectedFile,
    setSelectedSheet,
    uploadFile,
    refreshFiles,
    retryLastQuery
  };

  return (
    <ChatContext.Provider value={contextValue}>
      {children}
    </ChatContext.Provider>
  );
};

export const useChatContext = (): ChatContextValue => {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error('useChatContext must be used within a ChatProvider');
  }
  return context;
};