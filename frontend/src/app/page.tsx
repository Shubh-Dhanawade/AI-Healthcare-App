'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { Activity, Shield, FileText, Zap, ArrowRight, Brain, Search, Lock } from 'lucide-react';

const features = [
  {
    icon: <FileText className="w-6 h-6" />,
    title: 'Smart OCR Extraction',
    description: 'Extract text from digital PDFs and scanned documents using PyMuPDF and PaddleOCR.',
    color: 'blue',
  },
  {
    icon: <Brain className="w-6 h-6" />,
    title: 'AI Summarization',
    description: 'Get plain-language summaries of complex insurance policies powered by local AI.',
    color: 'purple',
  },
  {
    icon: <Search className="w-6 h-6" />,
    title: 'Field Extraction',
    description: 'Automatically extract premiums, coverage, exclusions, and deductibles.',
    color: 'teal',
  },
  {
    icon: <Shield className="w-6 h-6" />,
    title: 'Risk Detection',
    description: 'Identify hidden clauses, long waiting periods, and unfavorable conditions.',
    color: 'amber',
  },
];

const colorMap: Record<string, string> = {
  blue: 'rgba(59,130,246,0.15)',
  purple: 'rgba(139,92,246,0.15)',
  teal: 'rgba(20,184,166,0.15)',
  amber: 'rgba(245,158,11,0.15)',
};
const iconColorMap: Record<string, string> = {
  blue: '#3b82f6',
  purple: '#8b5cf6',
  teal: '#14b8a6',
  amber: '#f59e0b',
};

export default function HomePage() {
  const { isAuthenticated } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (isAuthenticated) router.replace('/dashboard');
  }, [isAuthenticated, router]);

  return (
    <main className="hero-bg min-h-screen">
      {/* Nav */}
      <nav className="flex items-center justify-between px-6 py-5 max-w-7xl mx-auto">
        <div className="flex items-center gap-2">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)' }}>
            <Activity className="w-5 h-5 text-white" />
          </div>
          <span className="text-xl font-bold gradient-text">HealthAI</span>
        </div>
        <div className="flex items-center gap-3">
          <button onClick={() => router.push('/login')} className="btn-secondary text-sm">
            Log In
          </button>
          <button onClick={() => router.push('/register')} className="btn-primary text-sm">
            Get Started <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </nav>

      {/* Hero */}
      <section className="max-w-7xl mx-auto px-6 pt-20 pb-24 text-center">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-8 text-sm font-medium"
          style={{ background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.3)', color: '#60a5fa' }}>
          <Zap className="w-4 h-4" /> Powered by local RAG AI
        </div>

        <h1 className="text-5xl md:text-6xl font-black mb-6 leading-tight">
          Understand Your{' '}
          <span className="gradient-text">Insurance Policy</span>
          <br />in Minutes
        </h1>
        <p className="text-lg text-slate-400 max-w-2xl mx-auto mb-10">
          Upload any healthcare insurance document and our AI instantly extracts key fields, generates plain-language summaries, and flags risky clauses — no expertise required.
        </p>

        <div className="flex items-center justify-center gap-4 flex-wrap">
          <button onClick={() => router.push('/register')} className="btn-primary text-base px-8 py-3">
            Start Analyzing Free <ArrowRight className="w-5 h-5" />
          </button>
          <button onClick={() => router.push('/login')} className="btn-secondary text-base px-8 py-3">
            Sign In
          </button>
        </div>

        {/* Stats */}
        <div className="mt-20 grid grid-cols-2 md:grid-cols-4 gap-6">
          {[
            { label: 'Documents Processed', value: '10K+' },
            { label: 'Clauses Detected', value: '250K+' },
            { label: 'AI Accuracy', value: '94%' },
            { label: 'Time Saved', value: '98%' },
          ].map((s) => (
            <div key={s.label} className="glass-card p-6 text-center">
              <p className="text-3xl font-black gradient-text">{s.value}</p>
              <p className="text-sm text-slate-400 mt-1">{s.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="max-w-7xl mx-auto px-6 pb-24">
        <h2 className="text-3xl font-bold text-center mb-12">
          Everything You Need to <span className="gradient-text">Understand Your Policy</span>
        </h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((f) => (
            <div key={f.title} className="glass-card glass-card-hover p-6">
              <div className="w-12 h-12 rounded-xl flex items-center justify-center mb-4"
                style={{ background: colorMap[f.color], color: iconColorMap[f.color] }}>
                {f.icon}
              </div>
              <h3 className="font-bold text-lg mb-2">{f.title}</h3>
              <p className="text-slate-400 text-sm leading-relaxed">{f.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="max-w-7xl mx-auto px-6 pb-24">
        <div className="glass-card p-12 text-center" style={{ background: 'linear-gradient(135deg, rgba(59,130,246,0.1), rgba(139,92,246,0.1))' }}>
          <Lock className="w-10 h-10 mx-auto mb-4" style={{ color: '#60a5fa' }} />
          <h2 className="text-3xl font-bold mb-3">Your Documents Are Secure</h2>
          <p className="text-slate-400 max-w-xl mx-auto mb-8">
            End-to-end JWT authentication, encrypted storage, and local AI processing mean your sensitive insurance data never leaves your control.
          </p>
          <button onClick={() => router.push('/register')} className="btn-primary text-base px-10 py-3">
            Create Free Account <ArrowRight className="w-5 h-5" />
          </button>
        </div>
      </section>
    </main>
  );
}
