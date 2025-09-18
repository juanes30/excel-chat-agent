/**
 * Main Application component with error boundaries and connection resilience
 */

import React, { useCallback } from 'react';
import { ChatProvider, useChatContext } from './context/ChatContext';
import { ChatContainer, ErrorBoundary, ConnectionResilience } from './components';
import './App.css';

// Configuration for the chat application
const WEBSOCKET_URL = process.env.REACT_APP_WEBSOCKET_URL || 'ws://localhost:8000/ws';
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

// Enhanced Chat Container with Connection Resilience
const EnhancedChatContainer: React.FC = () => {
  const { state, reconnect } = useChatContext();

  const handleRetry = useCallback(() => {
    reconnect();
  }, [reconnect]);

  const handleClearError = useCallback(() => {
    // Clear error logic would be implemented in the context
    console.log('Clearing error');
  }, []);

  return (
    <ConnectionResilience
      connectionState={state.connectionState}
      reconnectAttempts={state.reconnectAttempts}
      maxReconnectAttempts={state.maxReconnectAttempts}
      isOnline={navigator.onLine}
      lastError={state.lastError}
      onRetry={handleRetry}
      onClearError={handleClearError}
      enableAdvancedMonitoring={true}
      enablePredictiveReconnection={true}
    >
      <ChatContainer
        enableFileUpload={true}
        showSidebar={true}
        theme="light"
      />
    </ConnectionResilience>
  );
};

// Error reporting function
const handleError = (error: Error, errorInfo: React.ErrorInfo) => {
  // Log to console for development
  console.error('Application Error:', error);
  console.error('Error Info:', errorInfo);

  // In production, you would send this to an error reporting service
  // Example: Sentry.captureException(error, { contexts: { errorInfo } });
};

function App() {
  return (
    <div className="App">
      <ErrorBoundary
        onError={handleError}
        maxRetries={3}
        showDetails={process.env.NODE_ENV === 'development'}
        enableReporting={process.env.NODE_ENV === 'production'}
      >
        <ChatProvider
          websocketUrl={WEBSOCKET_URL}
          apiBaseUrl={API_BASE_URL}
        >
          <EnhancedChatContainer />
        </ChatProvider>
      </ErrorBoundary>
    </div>
  );
}

export default App;