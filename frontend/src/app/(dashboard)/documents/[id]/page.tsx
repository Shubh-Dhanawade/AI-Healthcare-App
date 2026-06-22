'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { documentsApi, aiApi } from '@/lib/apiHelpers';
import { DocumentDetail, RiskAnalysis } from '@/types';
import { useParams, useRouter } from 'next/navigation';
import toast from 'react-hot-toast';
import {
  ArrowLeft, Brain, Shield, FileText, Search, RefreshCw,
  AlertTriangle, CheckCircle, Info, ChevronDown, ChevronUp,
  Send, MessageSquare, Clock, Activity, List
} from 'lucide-react';
import { useState } from 'react';
import DocumentStatusBadge from '@/components/documents/DocumentStatusBadge';
import Link from 'next/link';

function RiskCard({ risk }: { risk: RiskAnalysis }) {
  const [expanded, setExpanded] = useState(false);
  const severityColors: Record<string, string> = {
    high: 'text-red-400',
    medium: 'text-amber-400',
    low: 'text-emerald-400',
  };
  return (
    <div className={`p-4 rounded-xl risk-${risk.severity}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 flex-1">
          <AlertTriangle className={`w-4 h-4 flex-shrink-0 ${severityColors[risk.severity]}`} />
          <p className="text-sm font-medium">{risk.risk_type.replace(/_/g, ' ').toUpperCase()}</p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full severity-${risk.severity}`}>
            {risk.severity.toUpperCase()}
          </span>
          <button onClick={() => setExpanded(!expanded)} className="text-slate-400 hover:text-white">
            {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>
      <p className="text-sm text-slate-300 mt-2 italic">"{risk.clause_text}"</p>
      {expanded && (
        <div className="mt-3 space-y-2 border-t border-white/10 pt-3">
          {risk.explanation && (
            <div>
              <p className="text-xs font-semibold text-slate-400 mb-1">⚠️ Why this is risky:</p>
              <p className="text-sm text-slate-300">{risk.explanation}</p>
            </div>
          )}
          {risk.recommendation && (
            <div>
              <p className="text-xs font-semibold text-slate-400 mb-1">💡 Recommendation:</p>
              <p className="text-sm text-slate-300">{risk.recommendation}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function MetricRing({ score, label, color, reasoning }: { score: number; label: string; color: string; reasoning?: string }) {
  const percentage = Math.round(score * 100);
  return (
    <div className="flex flex-col items-center p-3 rounded-lg bg-slate-900/40 border border-white/5 relative group w-full text-center">
      <div className="relative w-14 h-14 flex items-center justify-center">
        <svg className="w-full h-full transform -rotate-90">
          <circle cx="28" cy="28" r="24" className="text-slate-800" strokeWidth="3" stroke="currentColor" fill="transparent" />
          <circle cx="28" cy="28" r="24" strokeWidth="3" stroke={color} strokeDasharray={2 * Math.PI * 24} strokeDashoffset={2 * Math.PI * 24 * (1 - score)} strokeLinecap="round" fill="transparent" />
        </svg>
        <span className="absolute text-xs font-bold text-white">{percentage}%</span>
      </div>
      <p className="text-[11px] font-semibold mt-2 text-slate-300">{label}</p>
      
      {reasoning && (
        <div className="absolute bottom-full mb-2 hidden group-hover:block w-48 p-2.5 rounded bg-slate-950 border border-slate-800 text-[10px] text-slate-300 z-10 shadow-xl leading-normal text-left">
          {reasoning}
        </div>
      )}
    </div>
  );
}

function ChatResponseCard({ msg }: { msg: any }) {
  const [showSources, setShowSources] = useState(false);
  
  if (msg.isUser) {
    return (
      <div className="flex justify-end gap-3 fade-in mt-3">
        <div className="bg-blue-600/30 border border-blue-500/30 text-slate-200 px-4 py-2.5 rounded-2xl rounded-tr-none max-w-lg shadow-md">
          <p className="text-xs font-bold text-blue-400 mb-0.5">You</p>
          <p className="text-sm">{msg.query}</p>
        </div>
      </div>
    );
  }

  const { evaluation, context, answer } = msg;

  return (
    <div className="space-y-3 bg-slate-800/40 border border-white/5 p-4 rounded-2xl shadow-md fade-in mt-3">
      <div>
        <p className="text-xs font-bold text-emerald-400 mb-1 flex items-center gap-1.5">
          <Brain className="w-3.5 h-3.5" /> AI Assistant
        </p>
        {msg.answer ? (
          <p className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">{answer}</p>
        ) : (
          <div className="flex items-center gap-2 p-1 text-slate-400">
            <span className="w-4 h-4 border-2 border-slate-400/30 border-t-slate-400 rounded-full animate-spin" />
            <p className="text-xs">Thinking...</p>
          </div>
        )}
      </div>



      {context && context.length > 0 && (
        <div className="border-t border-white/5 pt-2">
          <button 
            onClick={() => setShowSources(!showSources)}
            className="flex items-center gap-1 text-[11px] text-blue-400 hover:text-blue-300 font-semibold"
          >
            <List className="w-3 h-3" /> {showSources ? 'Hide' : 'Show'} retrieved document context ({context.length} chunks)
          </button>
          {showSources && (
            <div className="mt-2 space-y-1.5 border-l-2 border-blue-500/30 pl-3">
              {context.map((c: string, idx: number) => (
                <div key={idx} className="p-2 rounded bg-slate-900/30 border border-white/5">
                  <p className="text-[10px] text-slate-400 font-bold mb-0.5">Source Passage #{idx + 1}</p>
                  <p className="text-xs text-slate-300 italic">"...{c}..."</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}


export default function DocumentDetailPage() {
  const params = useParams();
  const docId = params.id as string;
  const router = useRouter();
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<'summary' | 'fields' | 'risks' | 'query'>('summary');
  const [queryInput, setQueryInput] = useState('');
  const [isQuerying, setIsQuerying] = useState(false);
  const [chatHistory, setChatHistory] = useState<any[]>([]);

  const [selectedLanguage, setSelectedLanguage] = useState<string>('English');
  const [translations, setTranslations] = useState<Record<string, {
    summary_text: string;
    coverage_summary?: string;
    exclusions_summary?: string;
    waiting_period_summary?: string;
    premium_summary?: string;
  }>>({});
  const [isTranslating, setIsTranslating] = useState(false);

  const handleLanguageChange = async (lang: string) => {
    setSelectedLanguage(lang);
    if (lang === 'English' || !doc?.summary) return;
    
    // If already translated, use cache
    if (translations[lang]) return;
    
    setIsTranslating(true);
    const toastId = toast.loading(`Translating summary to ${lang}...`);
    try {
      const summary = doc.summary;
      const [tText, tCoverage, tExclusions, tWaiting, tPremium] = await Promise.all([
        summary.summary_text ? aiApi.translate(summary.summary_text, lang) : Promise.resolve({ translated_text: '' }),
        summary.coverage_summary ? aiApi.translate(summary.coverage_summary, lang) : Promise.resolve({ translated_text: '' }),
        summary.exclusions_summary ? aiApi.translate(summary.exclusions_summary, lang) : Promise.resolve({ translated_text: '' }),
        summary.waiting_period_summary ? aiApi.translate(summary.waiting_period_summary, lang) : Promise.resolve({ translated_text: '' }),
        summary.premium_summary ? aiApi.translate(summary.premium_summary, lang) : Promise.resolve({ translated_text: '' }),
      ]);
      
      setTranslations(prev => ({
        ...prev,
        [lang]: {
          summary_text: tText.translated_text,
          coverage_summary: tCoverage.translated_text || undefined,
          exclusions_summary: tExclusions.translated_text || undefined,
          waiting_period_summary: tWaiting.translated_text || undefined,
          premium_summary: tPremium.translated_text || undefined,
        }
      }));
      toast.success(`Translated summary to ${lang}!`, { id: toastId });
    } catch (error) {
      console.error(error);
      toast.error(`Failed to translate summary. Please check if Ollama is running.`, { id: toastId });
      setSelectedLanguage('English');
    } finally {
      setIsTranslating(false);
    }
  };

  const { data: doc, isLoading, refetch } = useQuery<DocumentDetail>({
    queryKey: ['document', docId],
    queryFn: () => documentsApi.getById(docId),
    refetchInterval: doc => {
      const d = doc.state.data;
      // Keep polling while in any non-final state
      return d && ['uploaded', 'processing', 'text_extracted'].includes(d.status) ? 2000 : false;
    },
  });

  const summarizeMutation = useMutation({
    mutationFn: () => aiApi.summarize(docId),
    onSuccess: () => {
      toast.success('Summary generated!');
      queryClient.invalidateQueries({ queryKey: ['document', docId] });
    },
    onError: (err: any) => toast.error(err.response?.data?.detail || 'Summarization failed'),
  });

  const extractFieldsMutation = useMutation({
    mutationFn: () => aiApi.extractFields(docId),
    onSuccess: () => {
      toast.success('Fields extracted!');
      queryClient.invalidateQueries({ queryKey: ['document', docId] });
      setActiveTab('fields');
    },
    onError: (err: any) => toast.error(err.response?.data?.detail || 'Extraction failed'),
  });

  const riskMutation = useMutation({
    mutationFn: () => aiApi.riskAnalysis(docId),
    onSuccess: () => {
      toast.success('Risk analysis complete!');
      queryClient.invalidateQueries({ queryKey: ['document', docId] });
      setActiveTab('risks');
    },
    onError: (err: any) => toast.error(err.response?.data?.detail || 'Risk analysis failed'),
  });

  const extractAllMutation = useMutation({
    mutationFn: async () => {
      const toastId = toast.loading('Running full policy audit (extracting fields and risks)...');
      try {
        await Promise.all([
          aiApi.extractFields(docId),
          aiApi.riskAnalysis(docId)
        ]);
        toast.success('Policy audit complete!', { id: toastId });
      } catch (err: any) {
        toast.error(err.response?.data?.detail || 'Audit failed', { id: toastId });
        throw err;
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['document', docId] });
      setActiveTab('fields');
    }
  });

  const handleSendQuery = async (queryText?: string) => {
    const textToSend = queryText || queryInput;
    if (!textToSend.trim()) return;

    setIsQuerying(true);
    // Add temporary user query to chat history
    const userMsg = { query: textToSend, isUser: true, timestamp: new Date() };
    setChatHistory(prev => [...prev, userMsg]);
    
    if (!queryText) setQueryInput('');

    let assistantMessageIndex = -1;

    try {
      // Map history for RAG endpoint
      const historyPayload = chatHistory.map(m => ({
        role: m.isUser ? 'user' : 'assistant',
        content: m.isUser ? m.query : m.answer
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
          document_ids: [docId],
          history: historyPayload
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to initialize stream: ${response.statusText}`);
      }

      // Add empty assistant message to start streaming into
      setChatHistory(prev => {
        assistantMessageIndex = prev.length;
        return [
          ...prev,
          {
            answer: '',
            context: [],
            evaluation: null,
            isUser: false,
            timestamp: new Date()
          }
        ];
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
          setChatHistory(prev => {
            const updated = [...prev];
            if (assistantMessageIndex !== -1 && updated[assistantMessageIndex]) {
              updated[assistantMessageIndex] = {
                ...updated[assistantMessageIndex],
                answer: accumulatedContent
              };
            }
            return updated;
          });
        }
      }
    } catch (err: any) {
      toast.error(err.message || 'Failed to query model');
      // remove user message if failed
      setChatHistory(prev => prev.filter(m => m.query !== textToSend || !m.isUser));
    } finally {
      setIsQuerying(false);
    }
  };

  const isProcessing = ['uploaded', 'processing', 'text_extracted'].includes(doc?.status || '');
  const canRunAI = doc?.status !== 'uploaded' && doc?.status !== 'processing' && doc?.status !== 'failed';

  if (isLoading) {
    return (
      <div className="space-y-4 fade-in">
        <div className="skeleton h-10 w-48" />
        <div className="skeleton h-32 w-full" />
        <div className="skeleton h-64 w-full" />
      </div>
    );
  }

  if (!doc) return (
    <div className="text-center py-20">
      <p className="text-slate-400">Document not found.</p>
      <Link href="/documents" className="btn-primary mt-4 inline-flex">Back to Documents</Link>
    </div>
  );

  const tabs = [
    { id: 'summary', label: 'AI Summary', icon: <Brain className="w-4 h-4" />, count: doc.summary ? 1 : 0 },
    { id: 'fields', label: 'Extracted Fields', icon: <Search className="w-4 h-4" />, count: doc.extracted_fields.length },
    { id: 'risks', label: 'Risk Analysis', icon: <Shield className="w-4 h-4" />, count: doc.risk_analyses.length },
    { id: 'query', label: 'Chat to AI', icon: <MessageSquare className="w-4 h-4" />, count: chatHistory.length },
  ] as const;

  const riskCounts = {
    high: doc.risk_analyses.filter(r => r.severity === 'high').length,
    medium: doc.risk_analyses.filter(r => r.severity === 'medium').length,
    low: doc.risk_analyses.filter(r => r.severity === 'low').length,
  };

  return (
    <div className="space-y-6 fade-in max-w-5xl">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <button onClick={() => router.back()} className="btn-secondary p-2 mt-0.5">
            <ArrowLeft className="w-4 h-4" />
          </button>
          <div>
            <h1 className="text-xl font-bold text-white truncate max-w-lg">{doc.original_filename}</h1>
            <div className="flex items-center gap-3 mt-1.5 flex-wrap">
              <DocumentStatusBadge status={doc.status} />
              <span className="text-xs text-slate-500">{doc.page_count} page{doc.page_count !== 1 ? 's' : ''}</span>
              {doc.extraction_method && (
                <span className="text-xs text-slate-500">via {doc.extraction_method}</span>
              )}
            </div>
          </div>
        </div>
        <button onClick={() => refetch()} className="btn-secondary p-2">
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* Processing Banner */}
      {isProcessing && (
        <div className="flex items-center gap-3 p-4 rounded-xl"
          style={{ background: 'rgba(245,158,11,0.1)', border: '1px solid rgba(245,158,11,0.3)' }}>
          <div className="w-4 h-4 border-2 border-amber-400/30 border-t-amber-400 rounded-full animate-spin" />
          <div>
            <p className="text-amber-300 text-sm font-medium">
              {doc?.status === 'text_extracted'
                ? 'Building AI index... Generating embeddings and summary in background.'
                : 'Document is being processed. Text extraction in progress...'}
            </p>
            <p className="text-amber-400/60 text-xs mt-0.5">
              {doc?.status === 'text_extracted'
                ? 'You can use the document now — AI features will be ready shortly.'
                : 'This may take a few seconds...'}
            </p>
          </div>
        </div>
      )}

      {/* AI Actions */}
      {canRunAI && (
        <div className="glass-card p-5">
          <h2 className="font-semibold text-sm text-slate-300 mb-4 flex items-center gap-2">
            <Brain className="w-4 h-4 text-blue-400" /> AI Analysis Tools
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <button
              id="summarize-btn"
              onClick={() => { summarizeMutation.mutate(); setActiveTab('summary'); }}
              disabled={summarizeMutation.isPending || !canRunAI}
              className="btn-primary justify-center py-2.5"
            >
              {summarizeMutation.isPending
                ? <><span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Summarizing...</>
                : <><Brain className="w-4 h-4" /> {doc.summary ? 'Re-summarize' : 'AI Summarize'}</>
              }
            </button>
            <button
              id="extract-fields-btn"
              onClick={() => extractFieldsMutation.mutate()}
              disabled={extractFieldsMutation.isPending || !canRunAI}
              className="btn-secondary justify-center py-2.5"
              style={{ color: '#a78bfa', borderColor: 'rgba(139,92,246,0.4)' }}
            >
              {extractFieldsMutation.isPending
                ? <><span className="w-4 h-4 border-2 border-purple-400/30 border-t-purple-400 rounded-full animate-spin" /> Extracting...</>
                : <><Search className="w-4 h-4" /> Extract Fields</>
              }
            </button>
            <button
              id="risk-analysis-btn"
              onClick={() => riskMutation.mutate()}
              disabled={riskMutation.isPending || !canRunAI}
              className="btn-secondary justify-center py-2.5"
              style={{ color: '#f87171', borderColor: 'rgba(239,68,68,0.4)' }}
            >
              {riskMutation.isPending
                ? <><span className="w-4 h-4 border-2 border-red-400/30 border-t-red-400 rounded-full animate-spin" /> Analyzing...</>
                : <><Shield className="w-4 h-4" /> Risk Analysis</>
              }
            </button>
          </div>
        </div>
      )}

      {/* Risk Overview (if analyzed) */}
      {doc.risk_analyses.length > 0 && (
        <div className="grid grid-cols-3 gap-3">
          {[
            { label: 'High Risk', count: riskCounts.high, color: '#ef4444', bg: 'rgba(239,68,68,0.1)' },
            { label: 'Medium Risk', count: riskCounts.medium, color: '#f59e0b', bg: 'rgba(245,158,11,0.1)' },
            { label: 'Low Risk', count: riskCounts.low, color: '#10b981', bg: 'rgba(16,185,129,0.1)' },
          ].map((r) => (
            <div key={r.label} className="p-4 rounded-xl text-center"
              style={{ background: r.bg, border: `1px solid ${r.color}40` }}>
              <p className="text-2xl font-black" style={{ color: r.color }}>{r.count}</p>
              <p className="text-xs text-slate-400">{r.label}</p>
            </div>
          ))}
        </div>
      )}

      {/* Tabs */}
      <div>
        <div className="flex border-b border-slate-700/50 gap-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-5 py-3 text-sm font-medium transition-all border-b-2 -mb-px ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              {tab.icon}
              {tab.label}
              {tab.count > 0 && (
                <span className="text-xs px-1.5 py-0.5 rounded-full bg-blue-500/20 text-blue-300">{tab.count}</span>
              )}
            </button>
          ))}
        </div>

        <div className="pt-6">
          {activeTab === 'summary' && (() => {
            const displayedSummary = selectedLanguage === 'English' 
              ? doc.summary 
              : translations[selectedLanguage] || doc.summary;
            
            if (!doc.summary) {
              return (
                <div className="text-center py-12 glass-card">
                  <Brain className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                  <p className="text-slate-400">No summary yet. Click "AI Summarize" to generate one.</p>
                </div>
              );
            }
            
            return (
              <div className="space-y-4">
                <div className="glass-card p-6 relative overflow-hidden">
                  {isTranslating && (
                    <div className="absolute inset-0 bg-[#0a0f1e]/60 backdrop-blur-sm flex items-center justify-center z-10 transition-all">
                      <div className="flex items-center gap-2">
                        <span className="w-5 h-5 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
                        <span className="text-xs text-slate-400">Translating to {selectedLanguage}...</span>
                      </div>
                    </div>
                  )}
                  <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
                    <h3 className="font-semibold flex items-center gap-2 text-blue-300">
                      <Info className="w-4 h-4" /> Policy Summary
                    </h3>
                    
                    {/* Language Dropdown */}
                    <div className="flex items-center gap-1.5 bg-slate-900/60 border border-slate-700/40 rounded-lg px-2.5 py-1">
                      <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Language:</span>
                      <select
                        value={selectedLanguage}
                        onChange={(e) => handleLanguageChange(e.target.value)}
                        className="bg-transparent border-none text-xs text-white focus:outline-none cursor-pointer"
                      >
                        <option value="English">English</option>
                        <option value="Hindi">Hindi (हिंदी)</option>
                        <option value="Marathi">Marathi (मराठी)</option>
                      </select>
                    </div>
                  </div>
                  <p className="text-slate-300 leading-relaxed whitespace-pre-wrap">{displayedSummary.summary_text}</p>
                </div>
                
                <div className="grid md:grid-cols-2 gap-4 relative">
                  {isTranslating && (
                    <div className="absolute inset-0 bg-[#0a0f1e]/30 backdrop-blur-[2px] z-10 rounded-xl" />
                  )}
                  {[
                    { label: '✅ Coverage', value: displayedSummary.coverage_summary, color: '#10b981' },
                    { label: '❌ Exclusions', value: displayedSummary.exclusions_summary, color: '#ef4444' },
                    { label: '⏰ Waiting Period', value: displayedSummary.waiting_period_summary, color: '#f59e0b' },
                    { label: '💰 Premium', value: displayedSummary.premium_summary, color: '#3b82f6' },
                  ].filter(s => s.value).map((section) => (
                    <div key={section.label} className="glass-card p-5">
                      <h4 className="font-semibold text-sm mb-2" style={{ color: section.color }}>{section.label}</h4>
                      <p className="text-slate-300 text-sm leading-relaxed whitespace-pre-wrap">{section.value}</p>
                    </div>
                  ))}
                </div>
                <p className="text-xs text-slate-500 text-right">Generated by {displayedSummary.model_used || doc.summary.model_used} • {new Date(doc.summary.created_at).toLocaleString()}</p>
              </div>
            );
          })()}

          {/* Fields Tab */}
          {activeTab === 'fields' && (
            <div className="glass-card overflow-hidden">
              {doc.extracted_fields.length === 0 ? (
                <div className="text-center py-12">
                  <FileText className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                  <p className="text-slate-400">No fields extracted yet. Click "Extract Fields".</p>
                </div>
              ) : (
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-slate-700/50">
                      <th className="text-left px-6 py-3 text-xs font-semibold text-slate-400 uppercase">Field</th>
                      <th className="text-left px-6 py-3 text-xs font-semibold text-slate-400 uppercase">Value</th>
                      <th className="text-left px-6 py-3 text-xs font-semibold text-slate-400 uppercase hidden md:table-cell">Category</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-700/30">
                    {doc.extracted_fields.map((field) => (
                      <tr key={field.id} className="hover:bg-white/3">
                        <td className="px-6 py-4 text-sm font-medium text-slate-300">{field.field_name}</td>
                        <td className="px-6 py-4 text-sm text-white">{field.field_value || '—'}</td>
                        <td className="px-6 py-4 hidden md:table-cell">
                          {field.field_category && (
                            <span className="text-xs px-2 py-1 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 capitalize">
                              {field.field_category.replace(/_/g, ' ')}
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}

          {/* Risks Tab */}
          {activeTab === 'risks' && (
            <div className="space-y-3">
              {doc.risk_analyses.length === 0 ? (
                <div className="text-center py-12 glass-card">
                  <Shield className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                  <p className="text-slate-400">No risk analysis yet. Click "Risk Analysis".</p>
                </div>
              ) : (
                <>
                  {doc.risk_analyses
                    .sort((a, b) => ({ high: 0, medium: 1, low: 2 }[a.severity] - { high: 0, medium: 1, low: 2 }[b.severity]))
                    .map((risk) => (
                      <RiskCard key={risk.id} risk={risk} />
                    ))}
                </>
              )}
            </div>
          )}

          {/* RAG Q&A Tab */}
          {activeTab === 'query' && (
            <div className="space-y-4">
              <div className="glass-card p-5">
                <h3 className="font-semibold mb-2 text-blue-300 text-sm flex items-center gap-2">
                  <MessageSquare className="w-4 h-4" /> Chat to AI
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Ask any questions about this health insurance policy. The AI assistant will review the document and provide relevant answers, even if you make typos.
                </p>
              </div>

              {/* Chat history */}
              <div className="space-y-4 max-h-[450px] overflow-y-auto pr-1 border border-white/5 rounded-xl p-3 bg-slate-950/25">
                {chatHistory.length === 0 ? (
                  <div className="text-center py-12">
                    <MessageSquare className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                    <p className="text-sm text-slate-400">No questions asked yet. Choose a suggested query below or type your own!</p>
                    
                    {/* Suggested Questions */}
                    <div className="mt-6 max-w-lg mx-auto flex flex-col gap-2">
                      {[
                        "What is the sum insured under this policy?",
                        "What are the waiting periods for pre-existing diseases?",
                        "What are the room rent limits or copayment terms?",
                        "How do I submit a claim under this policy?"
                      ].map((q, i) => (
                        <button 
                          key={i}
                          onClick={() => { setQueryInput(q); handleSendQuery(q); }}
                          disabled={isQuerying}
                          className="text-xs text-left p-2.5 rounded-lg bg-slate-900/60 border border-white/5 hover:border-blue-500/30 hover:bg-slate-900 text-blue-300 transition-all font-medium flex items-center gap-2"
                        >
                          <span>🔍</span>
                          <span>{q}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {chatHistory.map((msg, i) => (
                      <ChatResponseCard key={i} msg={msg} />
                    ))}
                    {isQuerying && chatHistory[chatHistory.length - 1]?.isUser && (
                      <ChatResponseCard msg={{ isUser: false }} />
                    )}
                  </div>
                )}
              </div>

              {/* Query Input Box */}
              <div className="flex gap-2 border-t border-slate-700/30 pt-3">
                <input 
                  type="text" 
                  value={queryInput}
                  onChange={(e) => setQueryInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && queryInput.trim() && !isQuerying) { handleSendQuery(); } }}
                  placeholder="Ask a question about this document..."
                  disabled={isQuerying}
                  className="form-input flex-1 py-2 text-sm bg-slate-900/60"
                />
                <button 
                  onClick={() => handleSendQuery()}
                  disabled={isQuerying || !queryInput.trim()}
                  className="btn-primary px-4 py-2 flex items-center gap-1.5"
                >
                  {isQuerying ? (
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  ) : (
                    <Send className="w-4 h-4" />
                  )}
                  <span>Send</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
