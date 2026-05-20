'use client';

import { useQuery } from '@tanstack/react-query';
import { documentsApi } from '@/lib/apiHelpers';
import { Document } from '@/types';
import { useAuth } from '@/contexts/AuthContext';
import { FileText, Upload, Shield, Brain, TrendingUp, Clock, CheckCircle, AlertTriangle } from 'lucide-react';
import Link from 'next/link';
import DocumentStatusBadge from '@/components/documents/DocumentStatusBadge';

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
  const { data: documents = [], isLoading } = useQuery<Document[]>({
    queryKey: ['documents'],
    queryFn: documentsApi.list,
  });

  const stats = {
    total: documents.length,
    completed: documents.filter((d) => d.status === 'completed').length,
    processing: documents.filter((d) => ['uploaded', 'processing', 'text_extracted'].includes(d.status)).length,
    failed: documents.filter((d) => d.status === 'failed').length,
  };

  const recentDocs = documents.slice(0, 5);

  return (
    <div className="space-y-8 fade-in">
      {/* Welcome Banner */}
      <div className="glass-card p-8 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 rounded-full opacity-10 blur-3xl pointer-events-none"
          style={{ background: 'radial-gradient(circle, #3b82f6, transparent)', transform: 'translate(30%, -30%)' }} />
        <div className="relative">
          <h1 className="text-2xl font-bold mb-1">
            Welcome back, <span className="gradient-text">{user?.full_name?.split(' ')[0]}</span> 👋
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
          { icon: <Brain className="w-5 h-5" />, title: 'AI Summarization', desc: 'Phi-3 Mini generates plain-language summaries', color: '#8b5cf6', bg: 'rgba(139,92,246,0.12)' },
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

      {/* Recent Documents */}
      <div className="glass-card p-6">
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
    </div>
  );
}
