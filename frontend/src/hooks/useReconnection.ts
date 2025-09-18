/**
 * Custom hook for advanced WebSocket reconnection strategies and network monitoring
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { WebSocketState } from '../types/chat.types';
import { calculateBackoffDelay } from '../utils/websocket.utils';

interface ReconnectionStrategy {
  maxAttempts: number;
  baseDelay: number;
  maxDelay: number;
  backoffMultiplier: number;
  enableJitter: boolean;
}

interface NetworkStatus {
  isOnline: boolean;
  effectiveType?: string;
  downlink?: number;
  rtt?: number;
}

interface UseReconnectionOptions {
  strategy?: Partial<ReconnectionStrategy>;
  onReconnectAttempt?: (attempt: number, delay: number) => void;
  onReconnectSuccess?: () => void;
  onReconnectFailure?: (error: Error) => void;
  onNetworkChange?: (status: NetworkStatus) => void;
  enableNetworkMonitoring?: boolean;
}

interface UseReconnectionReturn {
  reconnectAttempts: number;
  isReconnecting: boolean;
  nextReconnectDelay: number;
  networkStatus: NetworkStatus;
  shouldReconnect: (connectionState: WebSocketState) => boolean;
  startReconnection: (reconnectFn: () => Promise<void> | void) => void;
  stopReconnection: () => void;
  resetReconnection: () => void;
  getReconnectionDelay: (attempt: number) => number;
}

const defaultStrategy: ReconnectionStrategy = {
  maxAttempts: 10,
  baseDelay: 1000,
  maxDelay: 30000,
  backoffMultiplier: 2,
  enableJitter: true
};

export const useReconnection = (options: UseReconnectionOptions = {}): UseReconnectionReturn => {
  const {
    strategy: userStrategy = {},
    onReconnectAttempt,
    onReconnectSuccess,
    onReconnectFailure,
    onNetworkChange,
    enableNetworkMonitoring = true
  } = options;

  const strategy = { ...defaultStrategy, ...userStrategy };

  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [isReconnecting, setIsReconnecting] = useState(false);
  const [nextReconnectDelay, setNextReconnectDelay] = useState(0);
  const [networkStatus, setNetworkStatus] = useState<NetworkStatus>({
    isOnline: navigator.onLine
  });

  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectFunctionRef = useRef<(() => Promise<void> | void) | null>(null);
  const isReconnectionActiveRef = useRef(false);

  const getReconnectionDelay = useCallback((attempt: number): number => {
    const { baseDelay, maxDelay, backoffMultiplier, enableJitter } = strategy;

    let delay = Math.min(baseDelay * Math.pow(backoffMultiplier, attempt), maxDelay);

    if (enableJitter) {
      // Add jitter to prevent thundering herd effect
      const jitterRange = delay * 0.1;
      const jitter = (Math.random() - 0.5) * jitterRange;
      delay += jitter;
    }

    return Math.max(delay, 0);
  }, [strategy]);

  const updateNetworkStatus = useCallback(() => {
    const connection = (navigator as any).connection || (navigator as any).mozConnection || (navigator as any).webkitConnection;

    const status: NetworkStatus = {
      isOnline: navigator.onLine
    };

    if (connection) {
      status.effectiveType = connection.effectiveType;
      status.downlink = connection.downlink;
      status.rtt = connection.rtt;
    }

    setNetworkStatus(prev => {
      const hasChanged = prev.isOnline !== status.isOnline ||
                        prev.effectiveType !== status.effectiveType;

      if (hasChanged) {
        onNetworkChange?.(status);
      }

      return status;
    });
  }, []); // Remove onNetworkChange dependency to prevent loops

  const shouldReconnect = useCallback((connectionState: WebSocketState): boolean => {
    if (!networkStatus.isOnline) {
      return false; // Don't reconnect if offline
    }

    if (reconnectAttempts >= strategy.maxAttempts) {
      return false; // Max attempts reached
    }

    return connectionState === WebSocketState.DISCONNECTED ||
           connectionState === WebSocketState.ERROR;
  }, [networkStatus.isOnline, reconnectAttempts, strategy.maxAttempts]);

  const stopReconnection = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    setIsReconnecting(false);
    setNextReconnectDelay(0);
    isReconnectionActiveRef.current = false;
  }, []);

  const resetReconnection = useCallback(() => {
    stopReconnection();
    setReconnectAttempts(0);
  }, [stopReconnection]);

  const performReconnection = useCallback(async () => {
    if (!reconnectFunctionRef.current || !isReconnectionActiveRef.current) {
      return;
    }

    const currentAttempt = reconnectAttempts;

    try {
      await reconnectFunctionRef.current();

      // If we get here, reconnection was successful
      setIsReconnecting(false);
      setReconnectAttempts(0);
      setNextReconnectDelay(0);
      isReconnectionActiveRef.current = false;
      onReconnectSuccess?.(
      );
    } catch (error) {
      const nextAttempt = currentAttempt + 1;
      setReconnectAttempts(nextAttempt);

      if (nextAttempt >= strategy.maxAttempts) {
        setIsReconnecting(false);
        isReconnectionActiveRef.current = false;
        onReconnectFailure?.(new Error(`Max reconnection attempts (${strategy.maxAttempts}) reached`));
        return;
      }

      // Schedule next reconnection attempt
      const delay = getReconnectionDelay(nextAttempt);
      setNextReconnectDelay(delay);

      reconnectTimeoutRef.current = setTimeout(() => {
        if (isReconnectionActiveRef.current && networkStatus.isOnline) {
          performReconnection();
        }
      }, delay);

      onReconnectFailure?.(error instanceof Error ? error : new Error('Reconnection failed'));
    }
  }, [reconnectAttempts, strategy.maxAttempts, getReconnectionDelay, networkStatus.isOnline, onReconnectSuccess, onReconnectFailure]);

  const startReconnection = useCallback((reconnectFn: () => Promise<void> | void) => {
    if (isReconnectionActiveRef.current) {
      return; // Already reconnecting
    }

    if (!networkStatus.isOnline) {
      return; // Don't start if offline
    }

    reconnectFunctionRef.current = reconnectFn;
    isReconnectionActiveRef.current = true;
    setIsReconnecting(true);

    const delay = getReconnectionDelay(reconnectAttempts);
    setNextReconnectDelay(delay);

    onReconnectAttempt?.(reconnectAttempts, delay);

    if (delay > 0) {
      reconnectTimeoutRef.current = setTimeout(() => {
        performReconnection();
      }, delay);
    } else {
      performReconnection();
    }
  }, [reconnectAttempts, networkStatus.isOnline, getReconnectionDelay, onReconnectAttempt, performReconnection]);

  // Network monitoring
  useEffect(() => {
    if (!enableNetworkMonitoring) {
      return;
    }

    updateNetworkStatus();

    const handleOnline = () => {
      updateNetworkStatus();
      // If we come back online and have a reconnect function, attempt reconnection
      if (reconnectFunctionRef.current && !isReconnectionActiveRef.current) {
        startReconnection(reconnectFunctionRef.current);
      }
    };

    const handleOffline = () => {
      updateNetworkStatus();
      stopReconnection(); // Stop reconnection attempts when offline
    };

    const handleConnectionChange = () => {
      updateNetworkStatus();
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Listen for connection changes if supported
    const connection = (navigator as any).connection || (navigator as any).mozConnection || (navigator as any).webkitConnection;
    if (connection) {
      connection.addEventListener('change', handleConnectionChange);
    }

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);

      if (connection) {
        connection.removeEventListener('change', handleConnectionChange);
      }
    };
  }, [enableNetworkMonitoring]); // Remove function dependencies to prevent loops

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    };
  }, []); // Remove stopReconnection dependency

  return {
    reconnectAttempts,
    isReconnecting,
    nextReconnectDelay,
    networkStatus,
    shouldReconnect,
    startReconnection,
    stopReconnection,
    resetReconnection,
    getReconnectionDelay
  };
};