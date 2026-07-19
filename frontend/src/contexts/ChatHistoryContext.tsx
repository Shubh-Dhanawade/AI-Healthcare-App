'use client';

/**
 * ChatHistoryContext
 *
 * Persists chat messages per user in localStorage so conversation history
 * survives tab switches, back navigation, and page refreshes.
 *
 * Key format: `chat_history_{userId}`
 */

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  ReactNode,
} from 'react';
import { useAuth } from '@/contexts/AuthContext';

// ── Types ──────────────────────────────────────────────────────────────────────

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  /** Document filenames that were referenced in this assistant response */
  sources?: string[];
  timestamp?: number;
}

interface ChatHistoryContextType {
  /** Full message list for the Conversational AI page */
  conversationalMessages: ChatMessage[];
  addConversationalMessage: (message: ChatMessage) => void;
  updateLastConversationalMessage: (content: string, sources?: string[]) => void;
  clearConversationalHistory: () => void;
}

// ── Context ────────────────────────────────────────────────────────────────────

const ChatHistoryContext = createContext<ChatHistoryContextType | null>(null);

// ── Helpers ────────────────────────────────────────────────────────────────────

function getStorageKey(userId: string, chatType: string): string {
  return `chat_history_${chatType}_${userId}`;
}

function loadFromStorage(key: string): ChatMessage[] {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveToStorage(key: string, messages: ChatMessage[]): void {
  try {
    // Keep only the last 200 messages to avoid localStorage bloat
    const trimmed = messages.slice(-200);
    localStorage.setItem(key, JSON.stringify(trimmed));
  } catch {
    // Silently fail on storage quota exceeded
  }
}

// ── Provider ───────────────────────────────────────────────────────────────────

export function ChatHistoryProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const userId = user?.id ?? 'anonymous';

  // ── Conversational AI messages ──────────────────────────────────────────────
  const convKey = getStorageKey(userId, 'conversational');
  const [conversationalMessages, setConversationalMessages] = useState<ChatMessage[]>([]);

  // Load from localStorage once we have a userId
  useEffect(() => {
    if (userId === 'anonymous') return;
    const stored = loadFromStorage(convKey);
    setConversationalMessages(stored);
  }, [convKey, userId]);

  // Save to localStorage whenever messages change
  useEffect(() => {
    if (userId === 'anonymous' || conversationalMessages.length === 0) return;
    saveToStorage(convKey, conversationalMessages);
  }, [convKey, conversationalMessages, userId]);

  const addConversationalMessage = useCallback((message: ChatMessage) => {
    setConversationalMessages(prev => [
      ...prev,
      { ...message, timestamp: message.timestamp ?? Date.now() },
    ]);
  }, []);

  const updateLastConversationalMessage = useCallback(
    (content: string, sources?: string[]) => {
      setConversationalMessages(prev => {
        if (prev.length === 0) return prev;
        const updated = [...prev];
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          content,
          ...(sources !== undefined ? { sources } : {}),
        };
        return updated;
      });
    },
    []
  );

  const clearConversationalHistory = useCallback(() => {
    setConversationalMessages([]);
    localStorage.removeItem(convKey);
  }, [convKey]);

  return (
    <ChatHistoryContext.Provider
      value={{
        conversationalMessages,
        addConversationalMessage,
        updateLastConversationalMessage,
        clearConversationalHistory,
      }}
    >
      {children}
    </ChatHistoryContext.Provider>
  );
}

// ── Hook ───────────────────────────────────────────────────────────────────────

export function useChatHistory(): ChatHistoryContextType {
  const ctx = useContext(ChatHistoryContext);
  if (!ctx) throw new Error('useChatHistory must be used within ChatHistoryProvider');
  return ctx;
}
