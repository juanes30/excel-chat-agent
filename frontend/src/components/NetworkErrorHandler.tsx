/**
 * NetworkErrorHandler component for handling network-specific errors and connection issues
 */

import React, { useState, useEffect, useCallback } from 'react';
import { WebSocketState, WebSocketError } from '../types/chat.types';
import {
  WifiOff,
  Wifi,
  AlertCircle,
  RefreshCw,
  Clock,
  Shield,
  Zap,
  Server
} from 'lucide-react';

interface NetworkError {
  type: 'connection' | 'timeout' | 'server' | 'authentication' | 'rate_limit' | 'unknown';
  message: string;
  timestamp: Date;
  recoverable: boolean;
  retryAfter?: number;
}

interface NetworkErrorHandlerProps {
  connectionState: WebSocketState;
  lastError?: string | null;
  onRetry: () => void;
  onClearError: () => void;
  className?: string;
  autoRetry?: boolean;
  retryDelay?: number;
}

const getErrorDetails = (error: string): NetworkError => {
  const errorLower = error.toLowerCase();
  const timestamp = new Date();

  if (errorLower.includes('network') || errorLower.includes('connection')) {
    return {
      type: 'connection',
      message: 'Network connection lost. Please check your internet connection.',
      timestamp,
      recoverable: true
    };
  }

  if (errorLower.includes('timeout')) {
    return {
      type: 'timeout',
      message: 'Request timed out. The server may be busy or temporarily unavailable.',
      timestamp,
      recoverable: true,
      retryAfter: 5000
    };
  }

  if (errorLower.includes('server') || errorLower.includes('5')) {
    return {
      type: 'server',
      message: 'Server error occurred. Please try again in a few moments.',
      timestamp,
      recoverable: true,
      retryAfter: 10000
    };
  }

  if (errorLower.includes('unauthorized') || errorLower.includes('401') || errorLower.includes('403')) {
    return {
      type: 'authentication',
      message: 'Authentication failed. Please refresh the page to reconnect.',
      timestamp,
      recoverable: false
    };
  }

  if (errorLower.includes('rate') || errorLower.includes('429')) {
    return {
      type: 'rate_limit',
      message: 'Too many requests. Please wait a moment before trying again.',
      timestamp,
      recoverable: true,
      retryAfter: 30000
    };
  }

  return {
    type: 'unknown',
    message: error || 'An unexpected error occurred.',
    timestamp,
    recoverable: true
  };
};

const getErrorIcon = (type: NetworkError['type']) => {
  switch (type) {
    case 'connection':
      return WifiOff;
    case 'timeout':
      return Clock;
    case 'server':
      return Server;
    case 'authentication':
      return Shield;
    case 'rate_limit':
      return Zap;
    default:
      return AlertCircle;
  }
};

const getErrorColor = (type: NetworkError['type']) => {
  switch (type) {
    case 'connection':
      return 'text-red-600 bg-red-50 border-red-200';
    case 'timeout':
      return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    case 'server':
      return 'text-purple-600 bg-purple-50 border-purple-200';
    case 'authentication':
      return 'text-orange-600 bg-orange-50 border-orange-200';
    case 'rate_limit':
      return 'text-blue-600 bg-blue-50 border-blue-200';
    default:
      return 'text-gray-600 bg-gray-50 border-gray-200';
  }
};

export const NetworkErrorHandler: React.FC<NetworkErrorHandlerProps> = ({
  connectionState,
  lastError,
  onRetry,
  onClearError,
  className = '',
  autoRetry = true,
  retryDelay = 5000
}) => {
  const [isRetrying, setIsRetrying] = useState(false);
  const [retryCountdown, setRetryCountdown] = useState(0);
  const [networkStatus, setNetworkStatus] = useState({
    isOnline: navigator.onLine,
    lastChecked: new Date()
  });

  const errorDetails = lastError ? getErrorDetails(lastError) : null;
  const ErrorIcon = errorDetails ? getErrorIcon(errorDetails.type) : AlertCircle;
  const errorColors = errorDetails ? getErrorColor(errorDetails.type) : 'text-gray-600 bg-gray-50 border-gray-200';

  // Monitor network status
  useEffect(() => {
    const updateNetworkStatus = () => {
      setNetworkStatus({
        isOnline: navigator.onLine,
        lastChecked: new Date()
      });
    };

    const handleOnline = () => {
      updateNetworkStatus();
      const details = lastError ? getErrorDetails(lastError) : null;
      if (lastError && details?.type === 'connection') {
        // Auto-retry when connection is restored
        handleRetry();
      }
    };

    const handleOffline = () => {
      updateNetworkStatus();
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [lastError]);

  // Auto-retry logic
  useEffect(() => {
    if (!autoRetry || !errorDetails?.recoverable || !networkStatus.isOnline) {
      return;
    }

    const delay = errorDetails.retryAfter || retryDelay;
    let countdownInterval: NodeJS.Timeout;
    let retryTimeout: NodeJS.Timeout;

    // Start countdown
    setRetryCountdown(Math.ceil(delay / 1000));

    countdownInterval = setInterval(() => {
      setRetryCountdown(prev => {
        if (prev <= 1) {
          clearInterval(countdownInterval);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    // Schedule retry
    retryTimeout = setTimeout(() => {
      if (networkStatus.isOnline) {
        handleRetry();
      }
    }, delay);

    return () => {
      clearInterval(countdownInterval);
      clearTimeout(retryTimeout);
    };
  }, [lastError, autoRetry, retryDelay, networkStatus.isOnline]);

  const handleRetry = useCallback(async () => {
    if (isRetrying) return;

    setIsRetrying(true);
    setRetryCountdown(0);

    try {
      await onRetry();
      onClearError();
    } catch (error) {
      console.error('Retry failed:', error);
    } finally {
      setIsRetrying(false);
    }
  }, [isRetrying, onRetry, onClearError]);

  const handleDismiss = useCallback(() => {
    onClearError();
    setRetryCountdown(0);
  }, [onClearError]);

  // Don't show if no error or if connected
  if (!lastError || connectionState === WebSocketState.CONNECTED) {
    return null;
  }

  return (
    <div className={`rounded-lg border p-4 ${errorColors} ${className}`}>
      <div className="flex items-start gap-3">
        <ErrorIcon className="w-5 h-5 flex-shrink-0 mt-0.5" />

        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1">
              <h3 className="font-medium text-sm">
                Connection {errorDetails?.type === 'connection' ? 'Lost' : 'Error'}
              </h3>
              <p className="text-sm opacity-90 mt-1">
                {errorDetails?.message}
              </p>

              {/* Network Status */}
              <div className="flex items-center gap-4 mt-3 text-xs opacity-75">
                <div className="flex items-center gap-1">
                  {networkStatus.isOnline ? (
                    <Wifi className="w-3 h-3" />
                  ) : (
                    <WifiOff className="w-3 h-3" />
                  )}
                  <span>
                    {networkStatus.isOnline ? 'Online' : 'Offline'}
                  </span>
                </div>
                <span>
                  Last checked: {networkStatus.lastChecked.toLocaleTimeString()}
                </span>
              </div>

              {/* Auto-retry countdown */}
              {autoRetry && retryCountdown > 0 && errorDetails?.recoverable && (
                <div className="flex items-center gap-1 mt-2 text-xs opacity-75">
                  <RefreshCw className="w-3 h-3" />
                  <span>Auto-retry in {retryCountdown}s</span>
                </div>
              )}
            </div>

            {/* Action Buttons */}
            <div className="flex items-center gap-2 flex-shrink-0">
              {errorDetails?.recoverable && networkStatus.isOnline && (
                <button
                  onClick={handleRetry}
                  disabled={isRetrying}
                  className="px-3 py-1 bg-white bg-opacity-20 hover:bg-opacity-30 rounded text-xs font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
                >
                  {isRetrying ? (
                    <>
                      <RefreshCw className="w-3 h-3 animate-spin" />
                      <span>Retrying...</span>
                    </>
                  ) : (
                    <>
                      <RefreshCw className="w-3 h-3" />
                      <span>Retry</span>
                    </>
                  )}
                </button>
              )}

              <button
                onClick={handleDismiss}
                className="px-3 py-1 bg-white bg-opacity-20 hover:bg-opacity-30 rounded text-xs font-medium transition-colors"
              >
                Dismiss
              </button>
            </div>
          </div>

          {/* Recovery Suggestions */}
          {!networkStatus.isOnline && (
            <div className="mt-3 p-2 bg-white bg-opacity-20 rounded text-xs">
              <p className="font-medium mb-1">Troubleshooting Tips:</p>
              <ul className="space-y-1 text-xs opacity-90">
                <li>• Check your internet connection</li>
                <li>• Try switching between WiFi and mobile data</li>
                <li>• Restart your router if using WiFi</li>
                <li>• Check if other websites are working</li>
              </ul>
            </div>
          )}

          {errorDetails?.type === 'authentication' && (
            <div className="mt-3 p-2 bg-white bg-opacity-20 rounded text-xs">
              <p className="font-medium mb-1">Authentication Error:</p>
              <p className="text-xs opacity-90">
                Please refresh the page to re-establish your connection.
              </p>
              <button
                onClick={() => window.location.reload()}
                className="mt-2 px-2 py-1 bg-white bg-opacity-30 hover:bg-opacity-40 rounded text-xs font-medium transition-colors"
              >
                Refresh Page
              </button>
            </div>
          )}

          {errorDetails?.type === 'rate_limit' && (
            <div className="mt-3 p-2 bg-white bg-opacity-20 rounded text-xs">
              <p className="font-medium mb-1">Rate Limit Exceeded:</p>
              <p className="text-xs opacity-90">
                Please wait before sending more requests. The system will automatically retry.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};