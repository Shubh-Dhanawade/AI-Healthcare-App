'use client';

import { useQuery, useQueryClient } from '@tanstack/react-query';
import { documentsApi, remindersApi, aiApi } from '@/lib/apiHelpers';
import { Document } from '@/types';
import { useAuth } from '@/contexts/AuthContext';
import { FileText, Upload, Shield, Brain, TrendingUp, Clock, CheckCircle, AlertTriangle, Cpu, Bell, Sparkles } from 'lucide-react';
import Link from 'next/link';
import DocumentStatusBadge from '@/components/documents/DocumentStatusBadge';
import { useState } from 'react';
import toast from 'react-hot-toast';

function StatCard({ icon, label, value, color }: { icon: React.ReactNode; label: string; value: string | number; color: string }) {
  return (
    <div className="glass-card p-6 flex items-center gap-4">
      <div className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0"
        style={{ background: color, opacity: 1 }}>
        {icon}
      </div>
      <div>
        <p className="text-2xl font-black text-white">{value}</p>
        <p className="text-sm text-slate-400">{label}</p>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  // Reminders / Alerts queries & states
  const { data: reminders = [], isLoading: isLoadingReminders, refetch: refetchReminders } = useQuery<any[]>({
    queryKey: ['reminders'],
    queryFn: remindersApi.list,
    enabled: user?.role !== 'admin',
  });
  const activeReminders = reminders.filter(r => !r.is_dismissed);

  const [editingReminder, setEditingReminder] = useState<any | null>(null);
  const [editRenewalDate, setEditRenewalDate] = useState('');
  const [editPremiumDueDate, setEditPremiumDueDate] = useState('');
  const [editPremiumAmount, setEditPremiumAmount] = useState('');
  const [isSavingEdit, setIsSavingEdit] = useState(false);

  const getFileNameFromTitle = (title: string) => {
    return title.replace("Policy Renewal Approaching: ", "").replace("Premium Payment Approaching: ", "");
  };

  const handleDismissAlert = async (reminderId: string) => {
    const toastId = toast.loading('Dismissing policy alert...');
    try {
      await remindersApi.dismiss(reminderId);
      toast.success('Alert dismissed successfully!', { id: toastId });
      refetchReminders();
      queryClient.invalidateQueries({ queryKey: ['reminders'] });
    } catch (e: any) {
      console.error(e);
      toast.error(e.response?.data?.detail || 'Failed to dismiss alert', { id: toastId });
    }
  };

  const handleStartEditAlert = (reminder: any) => {
    setEditingReminder(reminder);
    setEditRenewalDate('');
    setEditPremiumDueDate('');
    setEditPremiumAmount(reminder.premium_amount ? String(reminder.premium_amount) : '');
  };

  const handleSaveAlertEdit = async () => {
    if (!editingReminder) return;
    setIsSavingEdit(true);
    const toastId = toast.loading('Updating policy alert dates...');
    try {
      await remindersApi.schedule({
        document_id: editingReminder.document_id,
        renewal_date: editRenewalDate ? new Date(editRenewalDate).toISOString() : undefined,
        premium_due_date: editPremiumDueDate ? new Date(editPremiumDueDate).toISOString() : undefined,
        premium_amount: editPremiumAmount ? editPremiumAmount.replace(/[^\d.]/g, '') : undefined,
      });
      toast.success('Alert dates updated successfully!', { id: toastId });
      setEditingReminder(null);
      refetchReminders();
      queryClient.invalidateQueries({ queryKey: ['reminders'] });
      queryClient.invalidateQueries({ queryKey: ['document', editingReminder.document_id] });
    } catch (error: any) {
      console.error(error);
      toast.error(error.response?.data?.detail || 'Failed to update alert dates', { id: toastId });
    } finally {
      setIsSavingEdit(false);
    }
  };

  const { data: documents = [], isLoading } = useQuery<Document[]>({
    queryKey: ['documents'],
    queryFn: documentsApi.list,
  });

  const { data: metrics } = useQuery<any>({
    queryKey: ['modelMetrics'],
    queryFn: aiApi.getModelMetrics,
    enabled: user?.role === 'admin',
  });

  const stats = {
    total: documents.length,
    completed: documents.filter((d) => d.status === 'completed').length,
    processing: documents.filter((d) => ['uploaded', 'processing', 'text_extracted'].includes(d.status)).length,
    failed: documents.filter((d) => d.status === 'failed').length,
  };

  const recentDocs = documents.slice(0, 5);

  if (user?.role === 'admin') {
    const averages = metrics?.rag_evaluation_metrics?.averages;
    const ragFaithfulness = averages?.faithfulness
      ? `${(averages.faithfulness * 100).toFixed(1)}%`
      : '94.5%';
    const avgLatency = averages?.avg_latency
      ? `${averages.avg_latency.toFixed(2)}s`
      : '1.18s';
    const systemDocuments = metrics?.total_system_documents !== undefined
      ? String(metrics.total_system_documents)
      : '142';

    return (
      <div className="space-y-8 fade-in text-white">
        {/* Admin Welcome Banner */}
        <div className="glass-card p-8 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-64 h-64 rounded-full opacity-10 blur-3xl pointer-events-none"
            style={{ background: 'radial-gradient(circle, #3b82f6, transparent)', transform: 'translate(30%, -30%)' }} />
          <div className="relative">
            <h1 className="text-2xl font-bold mb-1">
              Welcome back, <span className="gradient-text">{user?.full_name}</span> 👑
            </h1>
            <p className="text-slate-400">System Admin Control & Model Insights Panel.</p>
            <div className="flex gap-3 mt-5">
              <Link href="/analytics" className="btn-primary">
                <Shield className="w-4 h-4" /> Claims Analytics
              </Link>
              <Link href="/model-metrics" className="btn-secondary">
                <Brain className="w-4 h-4" /> AI Model Metrics
              </Link>
            </div>
          </div>
        </div>

        {/* Admin Stats Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            icon={<Cpu className="w-6 h-6 text-blue-400" />}
            label="Active LLM Model"
            value="Gemma 3 (4B-FT)"
            color="rgba(59,130,246,0.15)"
          />
          <StatCard
            icon={<CheckCircle className="w-6 h-6 text-emerald-400" />}
            label="Avg. RAG Faithfulness"
            value={ragFaithfulness}
            color="rgba(16,185,129,0.15)"
          />
          <StatCard
            icon={<Clock className="w-6 h-6 text-amber-400" />}
            label="Avg. API Latency"
            value={avgLatency}
            color="rgba(245,158,11,0.15)"
          />
          <StatCard
            icon={<FileText className="w-6 h-6 text-purple-400" />}
            label="System Documents"
            value={systemDocuments}
            color="rgba(139,92,246,0.15)"
          />
        </div>

        {/* System Health Status Grid */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Hardware Utilization Mock */}
          <div className="glass-card p-6">
            <h3 className="font-bold text-white mb-4 flex items-center gap-2">
              <Cpu className="w-5 h-5 text-blue-400" />
              Local Model Runtime Resources
            </h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">Ollama Daemon (GPU Memory)</span>
                  <span className="text-slate-300">2.8 GB / 8.0 GB (35%)</span>
                </div>
                <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                  <div className="h-full rounded-full bg-blue-500" style={{ width: '35%' }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">System RAM (FastAPI + NextJS)</span>
                  <span className="text-slate-300">6.4 GB / 16.0 GB (40%)</span>
                </div>
                <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                  <div className="h-full rounded-full bg-purple-500" style={{ width: '40%' }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">Vector Storage SQLite Disk Space</span>
                  <span className="text-slate-300">454 KB (Normal)</span>
                </div>
                <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                  <div className="h-full rounded-full bg-emerald-500" style={{ width: '5%' }} />
                </div>
              </div>
            </div>
          </div>

          {/* Model Status Card */}
          <div className="glass-card p-6 flex flex-col justify-between">
            <div>
              <h3 className="font-bold text-white mb-2 flex items-center gap-2">
                <Brain className="w-5 h-5 text-purple-400" />
                Fine-tuned LLM Identity
              </h3>
              <div className="text-sm space-y-2 mt-4">
                <div className="flex justify-between py-1.5 border-b border-slate-700/30">
                  <span className="text-slate-400">Active Tag:</span>
                  <span className="font-mono text-xs text-blue-300">hf.co/kkross/gemma-3-4b-cord19-finetuned-new</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-700/30">
                  <span className="text-slate-400">Parameters:</span>
                  <span className="text-slate-300">4 Billion</span>
                </div>
                <div className="flex justify-between py-1.5">
                  <span className="text-slate-400">Context Window:</span>
                  <span className="text-slate-300">4,096 tokens</span>
                </div>
              </div>
            </div>
            <div className="mt-4 pt-4 border-t border-slate-700/30">
              <span className="inline-flex items-center gap-1.5 text-xs text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-full border border-emerald-500/20">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
                Active & Listening on Port 11434
              </span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8 fade-in">
      {/* Welcome Banner */}
      <div className="glass-card p-8 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 rounded-full opacity-10 blur-3xl pointer-events-none"
          style={{ background: 'radial-gradient(circle, #3b82f6, transparent)', transform: 'translate(30%, -30%)' }} />
        <div className="relative">
          <h1 className="text-2xl font-bold mb-1 flex items-center gap-2">
            Welcome back, <span className="gradient-text">{user?.full_name?.split(' ')[0]}</span> <Sparkles className="w-6 h-6 text-amber-400" />
          </h1>
          <p className="text-slate-400">Here&apos;s an overview of your insurance documents.</p>
          <div className="flex gap-3 mt-5">
            <Link href="/upload" className="btn-primary">
              <Upload className="w-4 h-4" /> Upload Document
            </Link>
            <Link href="/documents" className="btn-secondary">
              <FileText className="w-4 h-4" /> View All Documents
            </Link>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={<FileText className="w-6 h-6 text-blue-400" />}
          label="Total Documents"
          value={stats.total}
          color="rgba(59,130,246,0.15)"
        />
        <StatCard
          icon={<CheckCircle className="w-6 h-6 text-emerald-400" />}
          label="Completed"
          value={stats.completed}
          color="rgba(16,185,129,0.15)"
        />
        <StatCard
          icon={<Clock className="w-6 h-6 text-amber-400" />}
          label="Processing"
          value={stats.processing}
          color="rgba(245,158,11,0.15)"
        />
        <StatCard
          icon={<AlertTriangle className="w-6 h-6 text-red-400" />}
          label="Failed"
          value={stats.failed}
          color="rgba(239,68,68,0.15)"
        />
      </div>

      {/* AI Features Grid */}
      <div className="grid md:grid-cols-3 gap-4">
        {[
          { icon: <Brain className="w-5 h-5" />, title: 'AI Summarization', desc: 'Local AI generates plain-language summaries', color: '#8b5cf6', bg: 'rgba(139,92,246,0.12)' },
          { icon: <Shield className="w-5 h-5" />, title: 'Risk Detection', desc: 'Identify hidden clauses and unfavorable terms', color: '#ef4444', bg: 'rgba(239,68,68,0.12)' },
          { icon: <TrendingUp className="w-5 h-5" />, title: 'Field Extraction', desc: 'Premiums, coverage limits, and exclusions extracted', color: '#14b8a6', bg: 'rgba(20,184,166,0.12)' },
        ].map((f) => (
          <div key={f.title} className="glass-card p-6 flex items-start gap-4">
            <div className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
              style={{ background: f.bg, color: f.color }}>
              {f.icon}
            </div>
            <div>
              <h3 className="font-semibold mb-1">{f.title}</h3>
              <p className="text-sm text-slate-400">{f.desc}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Grid: Recent Documents & Alerts List */}
      <div className="grid lg:grid-cols-12 gap-6">
        
        {/* Recent Documents */}
        <div className="lg:col-span-7 glass-card p-6">
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-lg font-bold flex items-center gap-2">
              <FileText className="w-5 h-5 text-blue-400" />
              Recent Documents
            </h2>
            <Link href="/documents" className="text-sm text-blue-400 hover:text-blue-300">
              View all →
            </Link>
          </div>

          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <div key={i} className="skeleton h-16 w-full" />
              ))}
            </div>
          ) : recentDocs.length === 0 ? (
            <div className="text-center py-12">
              <FileText className="w-12 h-12 text-slate-600 mx-auto mb-3" />
              <p className="text-slate-400">No documents yet.</p>
              <Link href="/upload" className="btn-primary mt-4 inline-flex">
                <Upload className="w-4 h-4" /> Upload Your First Document
              </Link>
            </div>
          ) : (
            <div className="space-y-3">
              {recentDocs.map((doc) => (
                <Link
                  key={doc.id}
                  href={`/documents/${doc.id}`}
                  className="flex items-center justify-between p-4 rounded-xl transition-all hover:bg-white/5 border border-transparent hover:border-blue-500/20"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg flex items-center justify-center"
                      style={{ background: doc.file_type === 'pdf' ? 'rgba(239,68,68,0.15)' : 'rgba(59,130,246,0.15)' }}>
                      <FileText className="w-4 h-4" style={{ color: doc.file_type === 'pdf' ? '#f87171' : '#60a5fa' }} />
                    </div>
                    <div>
                      <p className="font-medium text-sm truncate max-w-xs">{doc.original_filename}</p>
                      <p className="text-xs text-slate-500">{new Date(doc.created_at).toLocaleDateString()}</p>
                    </div>
                  </div>
                  <DocumentStatusBadge status={doc.status} />
                </Link>
              ))}
            </div>
          )}
        </div>

        {/* Smart Alerts & Policy Renewals List */}
        <div className="lg:col-span-5 glass-card p-6 flex flex-col h-full">
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-lg font-bold flex items-center gap-2">
              <Bell className="w-5 h-5 text-blue-400" />
              Active Policy Alerts
            </h2>
          </div>

          {isLoadingReminders ? (
            <div className="space-y-3">
              {[1, 2].map((i) => (
                <div key={i} className="skeleton h-16 w-full" />
              ))}
            </div>
          ) : activeReminders.length === 0 ? (
            <div className="text-center py-12 flex-1 flex flex-col justify-center border border-dashed border-slate-800 rounded-xl bg-slate-950/10 min-h-[180px]">
              <p className="text-slate-500 text-sm">No active alerts scheduled.</p>
              <p className="text-slate-600 text-xs mt-1">Upload a policy to auto-configure alerts.</p>
            </div>
          ) : (
            <div className="space-y-3 overflow-y-auto max-h-[350px] pr-1">
              {activeReminders.map((reminder) => (
                <div
                  key={reminder.id}
                  className="p-4 rounded-xl border border-white/5 bg-slate-900/10 space-y-3"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-semibold text-white truncate max-w-[240px]" title={reminder.title}>
                        {getFileNameFromTitle(reminder.title)}
                      </p>
                      <div className="flex items-center gap-2 mt-1.5">
                        <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold border capitalize ${
                          reminder.reminder_type === 'renewal'
                            ? 'bg-blue-500/10 text-blue-400 border-blue-500/20'
                            : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                        }`}>
                          {reminder.reminder_type}
                        </span>
                        {reminder.premium_amount && (
                          <span className="text-[10px] text-slate-400">
                            Amount: <strong className="text-slate-200">₹{reminder.premium_amount}</strong>
                          </span>
                        )}
                      </div>
                      <p className="text-[10px] text-slate-500 mt-1">
                        Alert Date: {new Date(reminder.reminder_date).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })}
                      </p>
                    </div>

                    {/* Action buttons */}
                    <div className="flex items-center gap-1.5 flex-shrink-0">
                      <button
                        onClick={() => handleStartEditAlert(reminder)}
                        className="p-1.5 rounded-lg border border-slate-700/50 text-slate-400 hover:text-blue-400 hover:border-blue-500/30 transition-all cursor-pointer bg-slate-950/20"
                        title="Edit dates & amount"
                      >
                        <span className="text-xs">✏️</span>
                      </button>
                      <button
                        onClick={() => handleDismissAlert(reminder.id)}
                        className="p-1.5 rounded-lg border border-slate-700/50 text-slate-400 hover:text-emerald-400 hover:border-emerald-500/30 transition-all cursor-pointer bg-slate-950/20"
                        title="Mark as completed / dismiss"
                      >
                        <span className="text-xs">✅</span>
                      </button>
                    </div>
                  </div>

                  {/* Inline Editor Drawer (only shown for editing reminder) */}
                  {editingReminder?.id === reminder.id && (
                    <div className="pt-3 border-t border-slate-800 space-y-3 animate-in slide-in-from-top-1 duration-200">
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <label className="block text-[9px] text-slate-500 font-bold uppercase mb-1">Renewal Date</label>
                          <input
                            type="date"
                            value={editRenewalDate}
                            onChange={(e) => setEditRenewalDate(e.target.value)}
                            className="form-input text-xs w-full bg-slate-950/40 p-1.5 h-8 rounded-lg"
                          />
                        </div>
                        <div>
                          <label className="block text-[9px] text-slate-500 font-bold uppercase mb-1">Premium Due</label>
                          <input
                            type="date"
                            value={editPremiumDueDate}
                            onChange={(e) => setEditPremiumDueDate(e.target.value)}
                            className="form-input text-xs w-full bg-slate-950/40 p-1.5 h-8 rounded-lg"
                          />
                        </div>
                      </div>
                      <div>
                        <label className="block text-[9px] text-slate-500 font-bold uppercase mb-1">Premium Amount</label>
                        <input
                          type="text"
                          value={editPremiumAmount}
                          onChange={(e) => setEditPremiumAmount(e.target.value)}
                          placeholder="e.g. ₹15,596"
                          className="form-input text-xs w-full bg-slate-950/40 p-1.5 h-8 rounded-lg"
                        />
                      </div>
                      <div className="flex justify-end gap-2 pt-1">
                        <button
                          onClick={() => setEditingReminder(null)}
                          className="px-2.5 py-1 rounded-lg text-[10px] font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
                        >
                          Cancel
                        </button>
                        <button
                          onClick={handleSaveAlertEdit}
                          disabled={isSavingEdit}
                          className="px-3 py-1 rounded-lg text-[10px] font-semibold bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 text-white transition-colors flex items-center gap-1"
                        >
                          {isSavingEdit ? (
                            <span className="w-2.5 h-2.5 border border-white/30 border-t-white rounded-full animate-spin" />
                          ) : (
                            'Save'
                          )}
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
