'use client';

import { useState, useEffect, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import { documentsApi, aiApi } from '@/lib/apiHelpers';
import { Document } from '@/types';
import { 
  MessageSquare, Send, Sparkles, Check, ChevronDown, Loader2, RefreshCw, X 
} from 'lucide-react';
import toast from 'react-hot-toast';

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

    try {
      // Map history for RAG endpoint
      const historyPayload = messages.map(m => ({
        role: m.role,
        content: m.content
      }));

      // Call API
      const response = await aiApi.chat(
        textToSend,
        selectedDocIds.length > 0 ? selectedDocIds : undefined,
        historyPayload
      );

      // Add assistant response
      setMessages(prev => [...prev, { role: 'assistant', content: response.response }]);
    } catch (error: any) {
      console.error(error);
      const errorMsg = error.response?.data?.detail || "Failed to communicate with AI. Please check if Ollama is running.";
      toast.error(errorMsg);
      setMessages(prev => [
        ...prev, 
        { 
          role: 'assistant', 
          content: "❌ Sorry, I encountered an error. Please make sure the local Ollama server is running with the 'phi3' model, or check if the backend is configured properly." 
        }
      ]);
    } finally {
      setIsSending(false);
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
                      className={`max-w-[75%] rounded-2xl px-5 py-3.5 text-sm leading-relaxed border ${
                        isUser 
                          ? 'bg-blue-600 border-blue-500/30 text-white rounded-tr-none' 
                          : 'bg-slate-900/60 border-slate-800 text-slate-200 rounded-tl-none backdrop-blur-md'
                      }`}
                    >
                      <div className="whitespace-pre-wrap">{message.content}</div>
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
              {isSending && (
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
