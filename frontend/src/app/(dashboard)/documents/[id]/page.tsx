'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { documentsApi, aiApi } from '@/lib/apiHelpers';
import { DocumentDetail, RiskAnalysis } from '@/types';
import { useParams, useRouter } from 'next/navigation';
import toast from 'react-hot-toast';
import {
  ArrowLeft, Brain, Shield, FileText, Search, RefreshCw,
  AlertTriangle, CheckCircle, Info, ChevronDown, ChevronUp
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

export default function DocumentDetailPage() {
  const params = useParams();
  const docId = params.id as string;
  const router = useRouter();
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<'summary' | 'fields' | 'risks'>('summary');

  const { data: doc, isLoading, refetch } = useQuery<DocumentDetail>({
    queryKey: ['document', docId],
    queryFn: () => documentsApi.getById(docId),
    refetchInterval: doc => {
      const d = doc.state.data;
      return d && ['uploaded', 'processing'].includes(d.status) ? 3000 : false;
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

  const isProcessing = ['uploaded', 'processing'].includes(doc?.status || '');
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
          <p className="text-amber-300 text-sm">Document is being processed. Text extraction in progress...</p>
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
          {/* Summary Tab */}
          {activeTab === 'summary' && (
            <div className="space-y-4">
              {!doc.summary ? (
                <div className="text-center py-12 glass-card">
                  <Brain className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                  <p className="text-slate-400">No summary yet. Click "AI Summarize" to generate one.</p>
                </div>
              ) : (
                <>
                  <div className="glass-card p-6">
                    <h3 className="font-semibold mb-3 flex items-center gap-2 text-blue-300">
                      <Info className="w-4 h-4" /> Policy Summary
                    </h3>
                    <p className="text-slate-300 leading-relaxed whitespace-pre-wrap">{doc.summary.summary_text}</p>
                  </div>
                  <div className="grid md:grid-cols-2 gap-4">
                    {[
                      { label: '✅ Coverage', value: doc.summary.coverage_summary, color: '#10b981' },
                      { label: '❌ Exclusions', value: doc.summary.exclusions_summary, color: '#ef4444' },
                      { label: '⏰ Waiting Period', value: doc.summary.waiting_period_summary, color: '#f59e0b' },
                      { label: '💰 Premium', value: doc.summary.premium_summary, color: '#3b82f6' },
                    ].filter(s => s.value).map((section) => (
                      <div key={section.label} className="glass-card p-5">
                        <h4 className="font-semibold text-sm mb-2" style={{ color: section.color }}>{section.label}</h4>
                        <p className="text-slate-300 text-sm leading-relaxed whitespace-pre-wrap">{section.value}</p>
                      </div>
                    ))}
                  </div>
                  <p className="text-xs text-slate-500 text-right">Generated by {doc.summary.model_used} • {new Date(doc.summary.created_at).toLocaleString()}</p>
                </>
              )}
            </div>
          )}

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
        </div>
      </div>
    </div>
  );
}
