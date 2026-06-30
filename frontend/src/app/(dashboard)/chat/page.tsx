'use client';

import React, { useState, useEffect, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import { documentsApi, aiApi } from '@/lib/apiHelpers';
import { Document } from '@/types';
import {
  MessageSquare, Send, Sparkles, Check, ChevronDown, Loader2, RefreshCw, FileText, Globe, Plus, Trash2
} from 'lucide-react';
import toast from 'react-hot-toast';

// ── Response Cleaner ──────────────────────────────────────────────────────────
function cleanResponse(raw: string): string {
  return raw
    .replace(/\[SOURCES:[^\]]*\]/g, '')           // strip the sources tag — rendered separately
    .replace(/\s*\bIn\s+[a-zA-Z0-9_\-\.]+\.(?:pdf|docx)\s*-\s*Page\s*\d+(?:\s*context)?\.?/gi, '') // strip any inline citations
    .replace(/\n*\b(?:Reference|Source)s?:[\s\S]*$/gi, '') // strip any trailing inline references/sources
    .replace(/^\s*(ASSISTANT|USER|SYSTEM)\s*:\s*/gim, '')
    .replace(/^context:\s*.+$/gim, '')
    .replace(/^(source|doc|document):\s*.+$/gim, '')
    .replace(/^[\u0600-\u06FF\s]+(?=\n|[A-Z])/g, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

// ── Parse [SOURCES:...] tag from raw accumulated content ──────────────────────
function parseSources(raw: string): string[] {
  const match = raw.match(/\[SOURCES:([^\]]+)\]/);
  if (!match) return [];
  return match[1].split('|').map(s => s.trim()).filter(Boolean);
}

// ── Markdown-Style Renderer ───────────────────────────────────────────────────
function FormattedMessage({ content }: { content: string }) {
  const cleaned = cleanResponse(content);
  if (!cleaned) return null;

  const lines = cleaned.split('\n');

  const renderInline = (text: string) => {
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={i} className="font-semibold text-white">{part.slice(2, -2)}</strong>;
      }
      const italicParts = part.split(/(\*[^*]+\*|_[^_]+_)/g);
      return italicParts.map((p, j) => {
        if ((p.startsWith('*') && p.endsWith('*')) || (p.startsWith('_') && p.endsWith('_'))) {
          return <em key={`${i}-${j}`} className="italic text-slate-300">{p.slice(1, -1)}</em>;
        }
        return <span key={`${i}-${j}`}>{p}</span>;
      });
    });
  };

  const elements: React.ReactNode[] = [];
  let i = 0;
  let bulletBuffer: string[] = [];
  let orderedBuffer: { num: string; text: string }[] = [];

  const flushBullets = () => {
    if (bulletBuffer.length > 0) {
      elements.push(
        <ul key={`ul-${i}`} className="list-none space-y-1.5 my-2 pl-1">
          {bulletBuffer.map((item, idx) => (
            <li key={idx} className="flex items-start gap-2 text-slate-200">
              <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-teal-400 flex-shrink-0" />
              <span>{renderInline(item)}</span>
            </li>
          ))}
        </ul>
      );
      bulletBuffer = [];
    }
  };

  const flushOrdered = () => {
    if (orderedBuffer.length > 0) {
      elements.push(
        <ol key={`ol-${i}`} className="space-y-1.5 my-2 pl-1">
          {orderedBuffer.map((item, idx) => (
            <li key={idx} className="flex items-start gap-2 text-slate-200">
              <span className="font-semibold text-teal-400 flex-shrink-0 text-xs mt-0.5">{item.num}.</span>
              <span>{renderInline(item.text)}</span>
            </li>
          ))}
        </ol>
      );
      orderedBuffer = [];
    }
  };

  for (; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    const bulletMatch = trimmed.match(/^([•\-\*])\s+(.+)/);
    if (bulletMatch) {
      flushOrdered();
      bulletBuffer.push(bulletMatch[2]);
      continue;
    }

    const orderedMatch = trimmed.match(/^(Step\s*)?(\d+)[.:\)]\s+(.+)/i);
    if (orderedMatch) {
      flushBullets();
      orderedBuffer.push({ num: orderedMatch[2], text: orderedMatch[3] });
      continue;
    }

    const h3Match = trimmed.match(/^###\s+(.+)/);
    if (h3Match) { flushBullets(); flushOrdered(); elements.push(<p key={i} className="font-semibold text-teal-300 mt-3 mb-1 text-sm">{h3Match[1]}</p>); continue; }
    const h2Match = trimmed.match(/^##\s+(.+)/);
    if (h2Match) { flushBullets(); flushOrdered(); elements.push(<p key={i} className="font-bold text-white mt-3 mb-1">{h2Match[1]}</p>); continue; }

    if (trimmed === '') {
      flushBullets();
      flushOrdered();
      if (elements.length > 0) elements.push(<div key={`gap-${i}`} className="h-2" />);
      continue;
    }

    flushBullets();
    flushOrdered();
    elements.push(
      <p key={i} className="text-slate-200 leading-relaxed">
        {renderInline(trimmed)}
      </p>
    );
  }

  flushBullets();
  flushOrdered();

  return <div className="space-y-1">{elements}</div>;
}

// ── Source Badges ─────────────────────────────────────────────────────────────
function SourceBadges({ sources }: { sources: string[] }) {
  if (!sources || sources.length === 0) return null;
  return (
    <div className="mt-3 pt-2.5 border-t border-slate-700/50 flex flex-wrap gap-1.5">
      <span className="text-xs text-slate-500 mr-0.5 self-center">📄 Referenced:</span>
      {sources.map((src, i) => (
        <span
          key={i}
          className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium"
          style={{
            background: 'rgba(20,184,166,0.12)',
            border: '1px solid rgba(20,184,166,0.25)',
            color: '#5eead4'
          }}
        >
          <FileText className="w-2.5 h-2.5" />
          {src}
        </span>
      ))}
    </div>
  );
}

// ── Types ─────────────────────────────────────────────────────────────────────
interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
}

interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

const SUGGESTED_PROMPTS = [
  "What is my co-pay for room rent?",
  "Are pre-existing conditions covered?",
  "What is the maternity benefit waiting period?",
  "List the major exclusions of this policy."
];

// ── Main Component ────────────────────────────────────────────────────────────
export default function ChatPage() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [isLoadingSessions, setIsLoadingSessions] = useState(true);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Fetch user's documents
  const { data: documents = [], isLoading: isLoadingDocs } = useQuery<Document[]>({
    queryKey: ['documents'],
    queryFn: documentsApi.list,
  });

  const activeDocs = documents.filter(d =>
    ['completed', 'summarized', 'text_extracted'].includes(d.status)
  );

  // Fetch chat sessions from database
  const fetchSessionsList = async (selectFirst: boolean = false) => {
    try {
      const data = await aiApi.getSessions();
      setSessions(data);
      if (selectFirst && data.length > 0 && !activeSessionId) {
        setActiveSessionId(data[0].id);
      }
    } catch (err) {
      console.error("Failed to load chat sessions:", err);
    } finally {
      setIsLoadingSessions(false);
    }
  };

  // Load sessions list on component mount
  useEffect(() => {
    fetchSessionsList(true);
  }, []);

  // Fetch messages when active session changes
  useEffect(() => {
    const loadSessionMessages = async () => {
      if (!activeSessionId) {
        setMessages([]);
        return;
      }
      setIsLoadingMessages(true);
      try {
        const msgData = await aiApi.getSessionMessages(activeSessionId);
        setMessages(msgData);
      } catch (err) {
        console.error("Failed to load session messages:", err);
        toast.error("Failed to load chat history.");
      } finally {
        setIsLoadingMessages(false);
      }
    };
    loadSessionMessages();
  }, [activeSessionId]);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isSending]);

  // Close dropdown on click outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleToggleDoc = (docId: string) => {
    setSelectedDocIds(prev =>
      prev.includes(docId) ? prev.filter(id => id !== docId) : [...prev, docId]
    );
  };

  const handleSend = async (textToSend: string) => {
    if (!textToSend.trim() || isSending) return;

    const userMessage: Message = { role: 'user', content: textToSend };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsSending(true);
    setIsThinking(true);

    // Add empty assistant message slot for streaming
    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

    const token = localStorage.getItem('access_token');
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

    try {
      const response = await fetch(`${API_URL}/ai/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          query: textToSend,
          document_ids: selectedDocIds.length > 0 ? selectedDocIds : undefined,
          session_id: activeSessionId || undefined
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to initialize stream: ${response.statusText}`);
      }

      setIsThinking(false);

      // Read custom session ID header if returned
      const newSessionId = response.headers.get('X-Chat-Session-Id');
      if (newSessionId && newSessionId !== activeSessionId) {
        setActiveSessionId(newSessionId);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder('utf-8');
      let accumulatedContent = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          accumulatedContent += chunk;

          const sources = parseSources(accumulatedContent);
          setMessages(prev => {
            if (prev.length === 0) return prev;
            const updated = [...prev];
            updated[updated.length - 1] = {
              role: 'assistant',
              content: accumulatedContent,
              sources: sources.length > 0 ? sources : undefined
            };
            return updated;
          });
        }
      }

      // Refresh list to pull updated titles
      await fetchSessionsList();

    } catch (error: any) {
      console.error(error);
      const errorMsg = error.message || "Failed to communicate with AI.";
      toast.error(errorMsg);
      const fallback = "❌ Sorry, I encountered an error. Please make sure the local Ollama server is running, or check if the backend is configured properly.";
      setMessages(prev => {
        if (prev.length === 0) return prev;
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: fallback
        };
        return updated;
      });
    } finally {
      setIsSending(false);
      setIsThinking(false);
    }
  };

  const handleNewChat = () => {
    setActiveSessionId(null);
    setMessages([]);
    toast.success("Ready for a new conversation!");
  };

  const handleDeleteSession = async (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation();
    if (!confirm("Are you sure you want to delete this chat conversation?")) return;

    try {
      await aiApi.deleteSession(sessionId);
      toast.success("Conversation deleted");
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
        setMessages([]);
      }
      fetchSessionsList();
    } catch (err) {
      console.error("Failed to delete chat session:", err);
      toast.error("Failed to delete chat conversation.");
    }
  };

  const getSelectedDocsLabel = () => {
    if (selectedDocIds.length === 0) return "Query All Policies (Default)";
    if (selectedDocIds.length === 1) {
      const doc = activeDocs.find(d => d.id === selectedDocIds[0]);
      return doc ? doc.original_filename : "1 policy selected";
    }
    return `${selectedDocIds.length} policies selected`;
  };

  // Active query mode indicator
  const queryModeInfo = selectedDocIds.length === 0
    ? { icon: Globe, label: 'All your documents', color: '#60a5fa', bg: 'rgba(59,130,246,0.12)', border: 'rgba(59,130,246,0.25)' }
    : {
        icon: FileText,
        label: `${selectedDocIds.length} document${selectedDocIds.length > 1 ? 's' : ''} selected`,
        color: '#5eead4',
        bg: 'rgba(20,184,166,0.12)',
        border: 'rgba(20,184,166,0.25)'
      };

  const QIcon = queryModeInfo.icon;

  return (
    <div className="flex h-[calc(100vh-100px)] gap-4 fade-in relative">
      {/* Left Sidebar - Chat History */}
      <div className="w-64 glass-card hidden md:flex flex-col flex-shrink-0 p-4 border border-slate-800/80 rounded-2xl bg-slate-950/20 backdrop-blur-md">
        {/* "+ New Chat" Button */}
        <button
          onClick={handleNewChat}
          className="flex items-center justify-center gap-2 w-full py-2.5 px-4 rounded-xl border border-teal-500/30 bg-teal-500/5 text-teal-400 font-semibold text-sm hover:bg-teal-500/10 hover:border-teal-500/50 transition-all duration-300 shadow-md shadow-teal-500/5 mb-4 group"
        >
          <Plus className="w-4 h-4 transition-transform group-hover:rotate-90" />
          New Chat
        </button>

        {/* History List */}
        <div className="flex-1 flex flex-col min-h-0">
          <div className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-2.5 px-1">
            Recent Chats
          </div>
          
          <div className="flex-1 overflow-y-auto space-y-1.5 scrollbar pr-1">
            {isLoadingSessions ? (
              <div className="flex items-center justify-center p-6 gap-2 text-xs text-slate-500">
                <Loader2 className="w-3.5 h-3.5 animate-spin text-teal-400" />
                Loading history...
              </div>
            ) : sessions.length === 0 ? (
              <div className="p-4 text-center text-xs text-slate-600 leading-relaxed">
                No past conversations. Start querying to save your chat logs.
              </div>
            ) : (
              sessions.map((session) => {
                const isActive = session.id === activeSessionId;
                return (
                  <button
                    key={session.id}
                    onClick={() => setActiveSessionId(session.id)}
                    className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-left transition-all duration-200 group relative border ${
                      isActive
                        ? 'bg-slate-900 border-slate-700/50 text-white font-medium shadow-lg'
                        : 'text-slate-400 hover:bg-slate-900/60 hover:text-white border-transparent'
                    }`}
                  >
                    <div className="flex items-center min-w-0 mr-6">
                      <MessageSquare className={`w-3.5 h-3.5 mr-2 flex-shrink-0 transition-colors ${
                        isActive ? 'text-teal-400' : 'text-slate-500 group-hover:text-teal-400'
                      }`} />
                      <span className="truncate text-xs">{session.title}</span>
                    </div>
                    
                    <button
                      onClick={(e) => handleDeleteSession(e, session.id)}
                      className="opacity-0 group-hover:opacity-100 focus:opacity-100 p-1 hover:text-red-400 rounded transition-all duration-150 absolute right-2 hover:bg-slate-800"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </button>
                );
              })
            )}
          </div>
        </div>

        {/* Database Pulse Footer */}
        <div className="pt-3 border-t border-slate-800/80 flex items-center gap-2 text-[10px] text-slate-500">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
          <span>Synced with SQLite Database</span>
        </div>
      </div>

      {/* Right Area - Active Chat Interface */}
      <div className="flex-1 flex flex-col gap-4 overflow-hidden">
        {/* Top Selector Ribbon */}
        <div className="glass-card p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 flex-shrink-0">
          <div className="flex flex-col gap-1 w-full sm:w-auto">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Select Policies to Query</label>

            <div className="flex items-center gap-3 mt-1 flex-wrap">
              {/* Custom Dropdown */}
              <div className="relative" ref={dropdownRef}>
                <button
                  onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                  className="flex items-center justify-between w-full sm:w-72 px-4 py-2 bg-slate-900/60 border border-slate-700/50 rounded-xl text-sm font-medium text-white transition-all hover:bg-slate-800/60 focus:border-blue-500/60 outline-none"
                >
                  <span className="truncate pr-2">{getSelectedDocsLabel()}</span>
                  <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform duration-200 ${isDropdownOpen ? 'rotate-180' : ''}`} />
                </button>

                {isDropdownOpen && (
                  <div className="absolute left-0 mt-2 w-full sm:w-96 bg-[#111827] border border-slate-700/60 rounded-xl shadow-2xl z-50 py-2 max-h-72 overflow-y-auto backdrop-blur-xl">
                    <div className="px-3 py-1 border-b border-slate-800 text-xs font-medium text-slate-500 pb-2">
                      Active Health Insurance Policies ({activeDocs.length})
                    </div>
                    {isLoadingDocs ? (
                      <div className="flex items-center justify-center p-6 gap-2 text-sm text-slate-400">
                        <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
                        Loading documents...
                      </div>
                    ) : activeDocs.length === 0 ? (
                      <div className="p-4 text-center text-sm text-slate-500">
                        No active policies. Upload policy documents and wait for analysis to complete.
                      </div>
                    ) : (
                      <div className="p-1 space-y-1">
                        {activeDocs.map((doc) => {
                          const isSelected = selectedDocIds.includes(doc.id);
                          return (
                            <button
                              key={doc.id}
                              onClick={() => handleToggleDoc(doc.id)}
                              className="flex items-center justify-between w-full px-3 py-2 rounded-lg text-left text-sm text-slate-300 hover:text-white hover:bg-white/5 transition-all"
                            >
                              <span className="truncate pr-4">{doc.original_filename}</span>
                              <div className={`w-4.5 h-4.5 rounded border flex items-center justify-center transition-all ${
                                isSelected
                                  ? 'bg-blue-600 border-blue-500 text-white'
                                  : 'border-slate-600'
                              }`}>
                                {isSelected && <Check className="w-3.5 h-3.5 stroke-[3]" />}
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Query mode pill */}
              <div
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium"
                style={{ background: queryModeInfo.bg, border: `1px solid ${queryModeInfo.border}`, color: queryModeInfo.color }}
              >
                <QIcon className="w-3 h-3" />
                {queryModeInfo.label}
              </div>
            </div>
          </div>

          {/* Quick Clear UI Trigger (Mobile Friendly/Fallback) */}
          <button
            onClick={handleNewChat}
            className="md:hidden flex items-center gap-2 text-xs font-semibold text-slate-400 hover:text-teal-400 px-3 py-2 rounded-xl hover:bg-teal-500/10 border border-transparent hover:border-teal-500/20 transition-all self-end sm:self-auto"
          >
            <Plus className="w-3.5 h-3.5" />
            New Chat
          </button>
        </div>

        {/* Main Conversation Container */}
        <div className="flex-1 glass-card flex flex-col overflow-hidden relative border border-slate-800/80 rounded-2xl bg-slate-950/10">
          <div
            className="absolute inset-0 w-full h-full opacity-5 pointer-events-none"
            style={{
              backgroundImage: 'radial-gradient(var(--color-accent-primary) 1px, transparent 1px)',
              backgroundSize: '24px 24px'
            }}
          />

          {/* Messages Log */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar">
            {isLoadingMessages ? (
              <div className="h-full flex flex-col items-center justify-center text-center fade-in">
                <Loader2 className="w-8 h-8 animate-spin text-teal-400 mb-2" />
                <p className="text-sm text-slate-500 font-medium">Loading chat history...</p>
              </div>
            ) : messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center max-w-lg mx-auto fade-in">
                <div
                  className="w-16 h-16 rounded-2xl flex items-center justify-center mb-6 animate-pulse"
                  style={{
                    background: 'linear-gradient(135deg, rgba(59,130,246,0.15), rgba(139,92,246,0.15))',
                    border: '1px solid rgba(59,130,246,0.25)'
                  }}
                >
                  <MessageSquare className="w-8 h-8 text-blue-400" />
                </div>
                <h2 className="text-xl font-bold text-white mb-2">Policy Conversational AI</h2>
                <p className="text-slate-400 text-sm mb-3 leading-relaxed">
                  Ask specific questions about your uploaded health policies. The AI will extract relevant context from your insurance terms and provide clear, plain-language answers.
                </p>

                {/* Mode hint */}
                <div
                  className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs mb-8"
                  style={{ background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.15)', color: '#93c5fd' }}
                >
                  <Globe className="w-3.5 h-3.5" />
                  Currently querying <strong className="text-white ml-0.5">all your documents</strong>. Select specific ones above to narrow the search.
                </div>

                {/* Suggestions */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full">
                  {SUGGESTED_PROMPTS.map((prompt, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleSend(prompt)}
                      className="p-3 text-left text-xs text-slate-300 hover:text-white rounded-xl bg-slate-900/40 border border-slate-800 hover:border-blue-500/30 hover:bg-blue-600/5 transition-all"
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                {messages.map((message, index) => {
                  const isUser = message.role === 'user';
                  return (
                    <div key={index} className={`flex items-start gap-4 ${isUser ? 'justify-end' : 'justify-start'} fade-in`}>
                      {/* Assistant Avatar */}
                      {!isUser && (
                        <div
                          className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 font-semibold"
                          style={{ background: 'linear-gradient(135deg, #14b8a6, #3b82f6)' }}
                        >
                          <Sparkles className="w-4 h-4 text-white" />
                        </div>
                      )}

                      {/* Chat Bubble */}
                      <div
                        className={`max-w-[78%] rounded-2xl px-5 py-3.5 text-sm leading-relaxed border ${
                          isUser
                            ? 'bg-blue-600 border-blue-500/30 text-white rounded-tr-none shadow-md'
                            : 'bg-slate-900/60 border-slate-800/80 text-slate-200 rounded-tl-none backdrop-blur-md shadow-sm'
                        }`}
                      >
                        {isUser ? (
                          <p className="whitespace-pre-wrap">{message.content}</p>
                        ) : (
                          <>
                            <FormattedMessage content={message.content} />
                            {message.sources && message.sources.length > 0 && (
                              <SourceBadges sources={message.sources} />
                            )}
                          </>
                        )}
                      </div>

                      {/* User Avatar */}
                      {isUser && (
                        <div
                          className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 text-sm font-bold text-white shadow-md"
                          style={{ background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)' }}
                        >
                          U
                        </div>
                      )}
                    </div>
                  );
                })}

                {/* Thinking indicator */}
                {isThinking && (
                  <div className="flex items-start gap-4 justify-start fade-in">
                    <div
                      className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 font-semibold"
                      style={{ background: 'linear-gradient(135deg, #14b8a6, #3b82f6)' }}
                    >
                      <Sparkles className="w-4 h-4 text-white animate-pulse" />
                    </div>
                    <div className="bg-slate-900/60 border border-slate-800 text-slate-400 rounded-2xl rounded-tl-none px-5 py-3.5 text-sm flex items-center gap-2 backdrop-blur-md">
                      <Loader2 className="w-4 h-4 animate-spin text-teal-400" />
                      {selectedDocIds.length > 0
                        ? `Searching ${selectedDocIds.length} selected document${selectedDocIds.length > 1 ? 's' : ''}...`
                        : 'Thinking and searching all policies...'}
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Input Bar */}
          <div className="p-4 border-t border-slate-800/80 bg-[#0c1224]/60 backdrop-blur-md flex-shrink-0">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSend(input);
              }}
              className="flex items-center gap-3"
            >
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={
                  selectedDocIds.length > 0
                    ? `Ask about ${selectedDocIds.length} selected document${selectedDocIds.length > 1 ? 's' : ''}...`
                    : 'Ask a question about your policies...'
                }
                disabled={isSending}
                id="chat-input"
                className="flex-1 bg-slate-900/80 border border-slate-800/80 hover:border-slate-700/80 focus:border-teal-500/60 rounded-xl px-4 py-3.5 text-sm text-white placeholder-slate-500 outline-none transition-all"
              />
              <button
                type="submit"
                disabled={isSending || !input.trim()}
                className="btn-primary p-3.5 h-[46px] rounded-xl flex items-center justify-center flex-shrink-0 transition-all disabled:opacity-40 disabled:pointer-events-none"
                style={{ padding: '0.75rem' }}
              >
                <Send className="w-4.5 h-4.5" />
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
