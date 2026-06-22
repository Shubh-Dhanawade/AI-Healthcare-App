'use client';

import { useState, useEffect, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import { documentsApi, aiApi } from '@/lib/apiHelpers';
import { Document } from '@/types';
import { 
  MessageSquare, Send, Sparkles, Check, ChevronDown, Loader2, RefreshCw, X 
} from 'lucide-react';
import toast from 'react-hot-toast';

// ── Response Cleaner ──────────────────────────────────────────────────────────
// Strips model-echoed labels (ASSISTANT:, USER:, context: filename, etc.)
function cleanResponse(raw: string): string {
  return raw
    // Remove leading ASSISTANT: / USER: role labels the model sometimes echoes
    .replace(/^\s*(ASSISTANT|USER|SYSTEM)\s*:\s*/gim, '')
    // Remove raw "context: filename" lines from RAG chunks
    .replace(/^context:\s*.+$/gim, '')
    // Remove "Source: filename" lines
    .replace(/^(source|doc|document):\s*.+$/gim, '')
    // Remove Arabic/Urdu stray characters sometimes prepended
    .replace(/^[\u0600-\u06FF\s]+(?=\n|[A-Z])/g, '')
    // Collapse 3+ blank lines into 2
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

// ── Markdown-Style Renderer ───────────────────────────────────────────────────
// Converts **bold**, bullet lists (•/-/*) and numbered lists to JSX elements
function FormattedMessage({ content }: { content: string }) {
  const cleaned = cleanResponse(content);
  if (!cleaned) return null;

  const lines = cleaned.split('\n');

  const renderInline = (text: string) => {
    // Split on **bold** markers
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={i} className="font-semibold text-white">{part.slice(2, -2)}</strong>;
      }
      // Also handle *italic* or single _italic_
      const italicParts = part.split(/(\*[^*]+\*|_[^_]+_)/g);
      return italicParts.map((p, j) => {
        if ((p.startsWith('*') && p.endsWith('*')) || (p.startsWith('_') && p.endsWith('_'))) {
          return <em key={`${i}-${j}`} className="italic text-slate-300">{p.slice(1, -1)}</em>;
        }
        return <span key={`${i}-${j}`}>{p}</span>;
      });
    });
  };

  const elements: JSX.Element[] = [];
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

    // Bullet point: • - * at line start
    const bulletMatch = trimmed.match(/^([•\-\*])\s+(.+)/);
    if (bulletMatch) {
      flushOrdered();
      bulletBuffer.push(bulletMatch[2]);
      continue;
    }

    // Numbered list: 1. 2. Step 1: etc.
    const orderedMatch = trimmed.match(/^(Step\s*)?(\d+)[.:)]\s+(.+)/i);
    if (orderedMatch) {
      flushBullets();
      orderedBuffer.push({ num: orderedMatch[2], text: orderedMatch[3] });
      continue;
    }

    // Heading: ### or ## lines
    const h3Match = trimmed.match(/^###\s+(.+)/);
    if (h3Match) { flushBullets(); flushOrdered(); elements.push(<p key={i} className="font-semibold text-teal-300 mt-3 mb-1 text-sm">{h3Match[1]}</p>); continue; }
    const h2Match = trimmed.match(/^##\s+(.+)/);
    if (h2Match) { flushBullets(); flushOrdered(); elements.push(<p key={i} className="font-bold text-white mt-3 mb-1">{h2Match[1]}</p>); continue; }

    // Empty line = paragraph break
    if (trimmed === '') {
      flushBullets();
      flushOrdered();
      if (elements.length > 0) elements.push(<div key={`gap-${i}`} className="h-2" />);
      continue;
    }

    // Regular paragraph line
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

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

const SUGGESTED_PROMPTS = [
  "What is my co-pay for room rent?",
  "Are pre-existing conditions covered?",
  "What is the maternity benefit waiting period?",
  "List the major exclusions of this policy."
];

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Fetch user's documents
  const { data: documents = [], isLoading: isLoadingDocs } = useQuery<Document[]>({
    queryKey: ['documents'],
    queryFn: documentsApi.list,
  });

  // Filter completed/summarized documents
  const activeDocs = documents.filter(d => 
    ['completed', 'summarized', 'text_extracted'].includes(d.status)
  );

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
      prev.includes(docId) 
        ? prev.filter(id => id !== docId) 
        : [...prev, docId]
    );
  };

  const handleSend = async (textToSend: string) => {
    if (!textToSend.trim()) return;
    
    // Create new user message
    const userMessage: Message = { role: 'user', content: textToSend };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsSending(true);
    setIsThinking(true);

    let assistantMessageIndex = -1;

    try {
      // Map history for RAG endpoint
      const historyPayload = messages.map(m => ({
        role: m.role,
        content: m.content
      }));

      const token = localStorage.getItem('access_token');
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

      const response = await fetch(`${API_URL}/ai/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          query: textToSend,
          document_ids: selectedDocIds.length > 0 ? selectedDocIds : undefined,
          history: historyPayload
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to initialize stream: ${response.statusText}`);
      }

      // Hide the "Thinking..." loader as soon as we transition to stream writing
      setIsThinking(false);

      // Add empty assistant message to start streaming into
      setMessages(prev => {
        assistantMessageIndex = prev.length;
        return [...prev, { role: 'assistant', content: '' }];
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder('utf-8');
      let accumulatedContent = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          accumulatedContent += chunk;

          // Update the assistant message content
          setMessages(prev => {
            const updated = [...prev];
            if (assistantMessageIndex !== -1 && updated[assistantMessageIndex]) {
              updated[assistantMessageIndex] = {
                role: 'assistant',
                content: accumulatedContent
              };
            }
            return updated;
          });
        }
      }
    } catch (error: any) {
      console.error(error);
      const errorMsg = error.message || "Failed to communicate with AI. Please check if Ollama is running.";
      toast.error(errorMsg);
      setMessages(prev => {
        const updated = [...prev];
        const errorMessage = "❌ Sorry, I encountered an error. Please make sure the local Ollama server is running with the 'gemma3:4b' model, or check if the backend is configured properly.";
        if (assistantMessageIndex !== -1 && updated[assistantMessageIndex]) {
          updated[assistantMessageIndex] = {
            role: 'assistant',
            content: errorMessage
          };
        } else {
          updated.push({
            role: 'assistant',
            content: errorMessage
          });
        }
        return updated;
      });
    } finally {
      setIsSending(false);
      setIsThinking(false);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    toast.success("Chat history cleared");
  };

  const getSelectedDocsLabel = () => {
    if (selectedDocIds.length === 0) return "Query All Policies (Default)";
    if (selectedDocIds.length === 1) {
      const doc = activeDocs.find(d => d.id === selectedDocIds[0]);
      return doc ? doc.original_filename : "1 policy selected";
    }
    return `${selectedDocIds.length} policies selected`;
  };

  return (
    <div className="flex flex-col h-[calc(100vh-100px)] gap-4 fade-in">
      {/* Top Selector Ribbon */}
      <div className="glass-card p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex flex-col gap-1 w-full sm:w-auto">
          <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Select Policies to Query</label>
          
          {/* Custom Dropdown */}
          <div className="relative mt-1" ref={dropdownRef}>
            <button
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              className="flex items-center justify-between w-full sm:w-80 px-4 py-2 bg-slate-900/60 border border-slate-700/50 rounded-xl text-sm font-medium text-white transition-all hover:bg-slate-800/60 focus:border-blue-500/60 outline-none"
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
        </div>

        {/* Action Buttons */}
        {messages.length > 0 && (
          <button
            onClick={handleClearChat}
            className="flex items-center gap-2 text-xs font-semibold text-slate-400 hover:text-red-400 px-3 py-2 rounded-xl hover:bg-red-500/10 border border-transparent hover:border-red-500/20 transition-all self-end sm:self-auto"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Clear Chat
          </button>
        )}
      </div>

      {/* Main Conversation Container */}
      <div className="flex-1 glass-card flex flex-col overflow-hidden relative">
        <div className="absolute inset-0 w-full h-full opacity-5 pointer-events-none" 
          style={{ 
            backgroundImage: 'radial-gradient(var(--color-accent-primary) 1px, transparent 1px)', 
            backgroundSize: '24px 24px' 
          }} 
        />
        
        {/* Messages Log */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-lg mx-auto fade-in">
              <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-6"
                style={{ background: 'linear-gradient(135deg, rgba(59,130,246,0.15), rgba(139,92,246,0.15))', border: '1px solid rgba(59,130,246,0.25)' }}>
                <MessageSquare className="w-8 h-8 text-blue-400" />
              </div>
              <h2 className="text-xl font-bold text-white mb-2">Policy Conversational AI</h2>
              <p className="text-slate-400 text-sm mb-8 leading-relaxed">
                Ask specific questions about your uploaded health policies. The AI will extract relevant context from your insurance terms and provide clear, plain-language answers.
              </p>

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
                      <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 font-semibold"
                        style={{ background: 'linear-gradient(135deg, #14b8a6, #3b82f6)' }}>
                        <Sparkles className="w-4 h-4 text-white" />
                      </div>
                    )}

                    {/* Chat Bubble */}
                    <div 
                      className={`max-w-[78%] rounded-2xl px-5 py-3.5 text-sm leading-relaxed border ${
                        isUser 
                          ? 'bg-blue-600 border-blue-500/30 text-white rounded-tr-none' 
                          : 'bg-slate-900/60 border-slate-800 text-slate-200 rounded-tl-none backdrop-blur-md'
                      }`}
                    >
                      {isUser
                        ? <p className="whitespace-pre-wrap">{message.content}</p>
                        : <FormattedMessage content={message.content} />
                      }
                    </div>

                    {/* User Avatar */}
                    {isUser && (
                      <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 text-sm font-bold text-white"
                        style={{ background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)' }}>
                        U
                      </div>
                    )}
                  </div>
                );
              })}

              {/* Loading AI response */}
              {isThinking && (
                <div className="flex items-start gap-4 justify-start fade-in">
                  <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 font-semibold"
                    style={{ background: 'linear-gradient(135deg, #14b8a6, #3b82f6)' }}>
                    <Sparkles className="w-4 h-4 text-white" />
                  </div>
                  <div className="bg-slate-900/60 border border-slate-800 text-slate-400 rounded-2xl rounded-tl-none px-5 py-3.5 text-sm flex items-center gap-2 backdrop-blur-md">
                    <Loader2 className="w-4 h-4 animate-spin text-teal-400" />
                    Thinking and searching policies...
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Bar */}
        <div className="p-4 border-t border-slate-800 bg-[#0c1224]/60 backdrop-blur-md">
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
              placeholder="Ask a question about your policies..."
              disabled={isSending}
              id="chat-input"
              className="flex-1 bg-slate-900/80 border border-slate-800 hover:border-slate-700 focus:border-blue-500/60 rounded-xl px-4 py-3.5 text-sm text-white placeholder-slate-500 outline-none transition-all"
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
  );
}
