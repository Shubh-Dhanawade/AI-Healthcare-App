'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { claimsApi } from '@/lib/apiHelpers';
import { BarChart3, ShieldAlert, Cpu, Layers, HelpCircle, Check, Eye } from 'lucide-react';

interface ClassProportion {
  class: string;
  count: number;
  percentage: number;
}

interface Benchmark {
  model_name: string;
  precision: number;
  recall: number;
  f1_score: number;
  auc_roc: number;
  is_selected: boolean;
}

interface ClaimsStats {
  dataset_info: {
    total_samples: number;
    features: string[];
  };
  smote_proportions: {
    before: ClassProportion[];
    after: ClassProportion[];
  };
  benchmarks: Benchmark[];
  plots: {
    roc_curves: string;
    shap_summary: string;
    lime_explanation: string;
  };
}

export default function AnalyticsPage() {
  const [activeTab, setActiveTab] = useState<'roc' | 'shap' | 'lime'>('roc');

  const { data: stats, isLoading, error } = useQuery<ClaimsStats>({
    queryKey: ['claimsStats'],
    queryFn: claimsApi.getStats,
  });

  const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL?.replace('/api/v1', '') || 'http://localhost:8000';

  if (isLoading) {
    return (
      <div className="space-y-6 fade-in">
        <div className="skeleton h-16 w-1/3" />
        <div className="grid md:grid-cols-2 gap-6">
          <div className="skeleton h-64 w-full" />
          <div className="skeleton h-64 w-full" />
        </div>
        <div className="skeleton h-96 w-full" />
      </div>
    );
  }

  if (error || !stats) {
    return (
      <div className="glass-card p-8 text-center text-red-400">
        <ShieldAlert className="w-12 h-12 mx-auto mb-3" />
        <h3 className="text-lg font-bold mb-1">Failed to load claims statistics</h3>
        <p className="text-sm text-slate-400">Please make sure the backend server is running and data science analysis has been executed.</p>
      </div>
    );
  }

  return (
    <div className="space-y-8 fade-in">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold mb-1 text-white">Claims Underwriting Analytics</h1>
        <p className="text-slate-400 text-sm">Review data imbalance remedies, benchmark classification models, and explore Explainable AI (XAI) attributions.</p>
      </div>

      {/* Grid: SMOTE Proportions */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Before SMOTE */}
        <div className="glass-card p-6">
          <div className="flex items-center gap-2 mb-4">
            <Layers className="w-5 h-5 text-red-400" />
            <h3 className="font-bold text-white">Class Proportions (Before SMOTE)</h3>
          </div>
          <p className="text-xs text-slate-400 mb-5">Original training subset distribution showing high class imbalance (81.2% approvals vs 18.7% denials).</p>
          <div className="space-y-4">
            {stats.smote_proportions.before.map((item) => (
              <div key={item.class}>
                <div className="flex justify-between text-sm mb-1.5">
                  <span className="text-slate-300 font-medium">{item.class}</span>
                  <span className="text-slate-400 font-semibold">{item.count} samples ({item.percentage.toFixed(2)}%)</span>
                </div>
                <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                  <div 
                    className="h-full rounded-full bg-red-500/80" 
                    style={{ width: `${item.percentage}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* After SMOTE */}
        <div className="glass-card p-6">
          <div className="flex items-center gap-2 mb-4">
            <Cpu className="w-5 h-5 text-emerald-400" />
            <h3 className="font-bold text-white">Class Proportions (After SMOTE)</h3>
          </div>
          <p className="text-xs text-slate-400 mb-5">Resampled training subset distribution showing balanced classes after applying SMOTE algorithms (50% approvals vs 50% denials).</p>
          <div className="space-y-4">
            {stats.smote_proportions.after.map((item) => (
              <div key={item.class}>
                <div className="flex justify-between text-sm mb-1.5">
                  <span className="text-slate-300 font-medium">{item.class}</span>
                  <span className="text-slate-400 font-semibold">{item.count} samples ({item.percentage.toFixed(2)}%)</span>
                </div>
                <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                  <div 
                    className="h-full rounded-full bg-emerald-500/80" 
                    style={{ width: `${item.percentage}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Model Benchmarking & Evaluation Table */}
      <div className="glass-card p-6">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="w-5 h-5 text-blue-400" />
          <h3 className="font-bold text-white">Model Evaluation & Benchmarking</h3>
        </div>
        <p className="text-xs text-slate-400 mb-5">Performance scores calculated on the test split. The selected model (Random Forest) was chosen based on its superior AUC-ROC score and stability with explainability frameworks.</p>
        
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-sm">
            <thead>
              <tr className="border-b border-slate-700/60 text-slate-400 font-medium">
                <th className="py-3 px-4">Model Algorithm</th>
                <th className="py-3 px-4">Precision</th>
                <th className="py-3 px-4">Recall (Sensitivity)</th>
                <th className="py-3 px-4">F1-Score</th>
                <th className="py-3 px-4">AUC-ROC Score</th>
                <th className="py-3 px-4">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/40">
              {stats.benchmarks.map((row) => (
                <tr 
                  key={row.model_name}
                  className={`transition-colors ${row.is_selected ? 'bg-blue-500/5 text-white border-l-2 border-blue-500' : 'text-slate-300 hover:bg-white/5'}`}
                >
                  <td className="py-3.5 px-4 font-semibold">{row.model_name}</td>
                  <td className="py-3.5 px-4">{row.precision.toFixed(1)}%</td>
                  <td className="py-3.5 px-4">{row.recall.toFixed(1)}%</td>
                  <td className="py-3.5 px-4">{row.f1_score.toFixed(1)}%</td>
                  <td className="py-3.5 px-4 font-mono">{row.auc_roc.toFixed(3)}</td>
                  <td className="py-3.5 px-4">
                    {row.is_selected ? (
                      <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-500/20 text-blue-400 border border-blue-500/30">
                        <Check className="w-3 h-3" /> Selected Model
                      </span>
                    ) : (
                      <span className="text-slate-500 text-xs">-</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Explainable AI Visualizations */}
      <div className="glass-card p-6">
        <div className="flex items-center gap-2 mb-4">
          <HelpCircle className="w-5 h-5 text-purple-400" />
          <h3 className="font-bold text-white">Explainable AI (XAI) Visualizations</h3>
        </div>
        <p className="text-xs text-slate-400 mb-6">Interactive graphical proofs detailing model classification thresholds, global feature importances (SHAP), and local model justifications (LIME).</p>

        {/* Tab Headers */}
        <div className="flex border-b border-slate-700/60 mb-6 gap-2">
          {[
            { id: 'roc', label: 'ROC Curves', icon: BarChart3 },
            { id: 'shap', label: 'Global (SHAP Summary)', icon: Cpu },
            { id: 'lime', label: 'Local (LIME Explanation)', icon: Eye },
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 px-4 py-2.5 -mb-px font-semibold text-sm border-b-2 transition-all ${
                  isActive 
                    ? 'border-blue-500 text-blue-400' 
                    : 'border-transparent text-slate-400 hover:text-white hover:border-slate-600'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Tab Content */}
        <div className="flex justify-center p-4 rounded-2xl bg-slate-900/30 border border-slate-700/20 overflow-hidden">
          {activeTab === 'roc' && (
            <div className="text-center max-w-2xl w-full">
              <p className="text-sm text-slate-300 mb-4 font-medium">ROC Curve analysis comparing true positive rates vs false positive rates for model benchmarking.</p>
              <img 
                src={`${apiBaseUrl}${stats.plots.roc_curves}`} 
                alt="ROC Curves Comparison Chart"
                className="rounded-xl border border-slate-700/40 bg-white mx-auto shadow-2xl max-h-[500px] object-contain"
              />
            </div>
          )}

          {activeTab === 'shap' && (
            <div className="text-center max-w-2xl w-full">
              <p className="text-sm text-slate-300 mb-4 font-medium">SHAP Global Summary plot detailing mean absolute impact of features (like smoking and BP) across all claims.</p>
              <img 
                src={`${apiBaseUrl}${stats.plots.shap_summary}`} 
                alt="SHAP Global Explanations Summary Plot"
                className="rounded-xl border border-slate-700/40 bg-white mx-auto shadow-2xl max-h-[500px] object-contain"
              />
            </div>
          )}

          {activeTab === 'lime' && (
            <div className="text-center max-w-2xl w-full">
              <p className="text-sm text-slate-300 mb-4 font-medium">LIME Local Explanation showing exactly why a single patient's claim was denied based on specific thresholds.</p>
              <img 
                src={`${apiBaseUrl}${stats.plots.lime_explanation}`} 
                alt="LIME Local Explanation Attribution Plot"
                className="rounded-xl border border-slate-700/40 bg-white mx-auto shadow-2xl max-h-[500px] object-contain"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
