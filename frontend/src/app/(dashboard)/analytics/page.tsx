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
  const [activeTab, setActiveTab] = useState<'roc' | 'shap' | 'lime' | 'predict'>('roc');

  // Input states for interactive claim predictor
  const [age, setAge] = useState<number>(45);
  const [bmi, setBmi] = useState<number>(24.2);
  const [smoker, setSmoker] = useState<number>(0);
  const [preExisting, setPreExisting] = useState<number>(0);
  const [coverageTier, setCoverageTier] = useState<number>(2);
  const [systolic, setSystolic] = useState<number>(120);
  const [diastolic, setDiastolic] = useState<number>(80);

  // Prediction output states
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  const { data: stats, isLoading, error } = useQuery<ClaimsStats>({
    queryKey: ['claimsStats'],
    queryFn: claimsApi.getStats,
  });

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsPredicting(true);
    setPredictionError(null);
    try {
      const res = await claimsApi.predict({
        age,
        bmi,
        smoker,
        pre_existing_conditions: preExisting,
        coverage_tier: coverageTier,
        systolic_bp: systolic,
        diastolic_bp: diastolic
      });
      setPredictionResult(res);
    } catch (err: any) {
      console.error(err);
      setPredictionError(err.response?.data?.detail || "Actuarial prediction failed. Please verify API connection.");
    } finally {
      setIsPredicting(false);
    }
  };

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
        <div className="flex border-b border-slate-700/60 mb-6 gap-2 overflow-x-auto">
          {[
            { id: 'roc', label: 'ROC Curves', icon: BarChart3 },
            { id: 'shap', label: 'Global (SHAP Summary)', icon: Cpu },
            { id: 'lime', label: 'Local (LIME Explanation)', icon: Eye },
            { id: 'predict', label: 'Interactive Claim Predictor', icon: ShieldAlert },
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 px-4 py-2.5 -mb-px font-semibold text-sm border-b-2 transition-all whitespace-nowrap ${
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
        <div className="p-4 rounded-2xl bg-slate-900/30 border border-slate-700/20">
          {activeTab === 'roc' && (
            <div className="text-center max-w-2xl w-full mx-auto">
              <p className="text-sm text-slate-300 mb-4 font-medium">ROC Curve analysis comparing true positive rates vs false positive rates for model benchmarking.</p>
              <img 
                src={`${apiBaseUrl}${stats.plots.roc_curves}`} 
                alt="ROC Curves Comparison Chart"
                className="rounded-xl border border-slate-700/40 bg-white mx-auto shadow-2xl max-h-[500px] object-contain"
              />
            </div>
          )}

          {activeTab === 'shap' && (
            <div className="text-center max-w-2xl w-full mx-auto">
              <p className="text-sm text-slate-300 mb-4 font-medium">SHAP Global Summary plot detailing mean absolute impact of features (like smoking and BP) across all claims.</p>
              <img 
                src={`${apiBaseUrl}${stats.plots.shap_summary}`} 
                alt="SHAP Global Explanations Summary Plot"
                className="rounded-xl border border-slate-700/40 bg-white mx-auto shadow-2xl max-h-[500px] object-contain"
              />
            </div>
          )}

          {activeTab === 'lime' && (
            <div className="text-center max-w-2xl w-full mx-auto">
              <p className="text-sm text-slate-300 mb-4 font-medium">LIME Local Explanation showing exactly why a single patient's claim was denied based on specific thresholds.</p>
              <img 
                src={`${apiBaseUrl}${stats.plots.lime_explanation}`} 
                alt="LIME Local Explanation Attribution Plot"
                className="rounded-xl border border-slate-700/40 bg-white mx-auto shadow-2xl max-h-[500px] object-contain"
              />
            </div>
          )}

          {activeTab === 'predict' && (
            <div className="space-y-8 max-w-5xl mx-auto">
              <div className="grid md:grid-cols-5 gap-8">
                {/* Form column (cols 2) */}
                <div className="md:col-span-2 space-y-4">
                  <h4 className="text-sm font-semibold text-slate-300 border-b border-slate-800 pb-2 flex items-center gap-2">
                    <Layers className="w-4 h-4 text-blue-400" /> Patient Risk Parameters
                  </h4>
                  <form onSubmit={handlePredict} className="space-y-4 text-xs">
                    <div>
                      <label className="block text-slate-400 mb-1 font-medium">Age of Policyholder</label>
                      <input 
                        type="number" 
                        min="0" 
                        max="120"
                        value={age} 
                        onChange={(e) => setAge(parseInt(e.target.value) || 0)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-white text-xs outline-none focus:border-blue-500 transition-colors"
                      />
                    </div>
                    <div>
                      <label className="block text-slate-400 mb-1 font-medium">Body Mass Index (BMI)</label>
                      <input 
                        type="number" 
                        step="0.1" 
                        min="10" 
                        max="60"
                        value={bmi} 
                        onChange={(e) => setBmi(parseFloat(e.target.value) || 0)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-white text-xs outline-none focus:border-blue-500 transition-colors"
                      />
                    </div>
                    <div>
                      <label className="block text-slate-400 mb-1 font-medium">Active Smoker Status</label>
                      <select 
                        value={smoker} 
                        onChange={(e) => setSmoker(parseInt(e.target.value) || 0)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-white text-xs outline-none focus:border-blue-500 transition-colors"
                      >
                        <option value={0}>No (Non-smoker)</option>
                        <option value={1}>Yes (Active Smoker)</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-slate-400 mb-1 font-medium">Pre-existing Health Conditions Count</label>
                      <input 
                        type="number" 
                        min="0" 
                        max="10"
                        value={preExisting} 
                        onChange={(e) => setPreExisting(parseInt(e.target.value) || 0)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-white text-xs outline-none focus:border-blue-500 transition-colors"
                      />
                    </div>
                    <div>
                      <label className="block text-slate-400 mb-1 font-medium">Policy Coverage Tier</label>
                      <select 
                        value={coverageTier} 
                        onChange={(e) => setCoverageTier(parseInt(e.target.value) || 2)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-white text-xs outline-none focus:border-blue-500 transition-colors"
                      >
                        <option value={1}>Basic (Tier 1)</option>
                        <option value={2}>Standard (Tier 2)</option>
                        <option value={3}>Premium (Tier 3)</option>
                      </select>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="block text-slate-400 mb-1 font-medium">Systolic BP (mmHg)</label>
                        <input 
                          type="number" 
                          min="80" 
                          max="200"
                          value={systolic} 
                          onChange={(e) => setSystolic(parseInt(e.target.value) || 120)}
                          className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-white text-xs outline-none focus:border-blue-500 transition-colors"
                        />
                      </div>
                      <div>
                        <label className="block text-slate-400 mb-1 font-medium">Diastolic BP (mmHg)</label>
                        <input 
                          type="number" 
                          min="50" 
                          max="130"
                          value={diastolic} 
                          onChange={(e) => setDiastolic(parseInt(e.target.value) || 80)}
                          className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-white text-xs outline-none focus:border-blue-500 transition-colors"
                        />
                      </div>
                    </div>
                    <button
                      type="submit"
                      disabled={isPredicting}
                      className="w-full py-2.5 bg-blue-600 hover:bg-blue-500 active:bg-blue-700 disabled:opacity-50 text-white rounded-lg font-semibold text-xs tracking-wider uppercase transition-colors shadow-lg shadow-blue-500/20"
                    >
                      {isPredicting ? 'Analyzing Underwriting Risk...' : 'Run Claim Underwrite'}
                    </button>
                  </form>
                  {predictionError && (
                    <div className="p-3 bg-red-950/40 border border-red-500/20 text-red-400 rounded-lg text-xs">
                      {predictionError}
                    </div>
                  )}
                </div>

                {/* Results column (cols 3) */}
                <div className="md:col-span-3 space-y-6">
                  {predictionResult ? (
                    <div className="space-y-6 animate-fade-in">
                      {/* Probability Score & Status */}
                      <div className="grid grid-cols-2 gap-4">
                        <div className="glass-card p-4 flex flex-col justify-center items-center text-center">
                          <span className="text-slate-400 text-xs font-semibold mb-1">Denial Probability</span>
                          <div className={`text-4xl font-extrabold font-mono ${predictionResult.claim_denied ? 'text-red-400' : 'text-emerald-400'}`}>
                            {predictionResult.denial_probability}%
                          </div>
                        </div>
                        <div className={`glass-card p-4 flex flex-col justify-center items-center text-center border ${predictionResult.claim_denied ? 'border-red-500/20 bg-red-500/5' : 'border-emerald-500/20 bg-emerald-500/5'}`}>
                          <span className="text-slate-400 text-xs font-semibold mb-1">Underwriting Verdict</span>
                          <span className={`text-sm font-extrabold uppercase tracking-wider ${predictionResult.claim_denied ? 'text-red-400' : 'text-emerald-400'}`}>
                            {predictionResult.claim_denied ? 'High Risk of Denial' : 'Likely Approved'}
                          </span>
                        </div>
                      </div>

                      {/* Gemma 3 Synthesized explanation */}
                      <div className="glass-card p-5 border border-slate-700/30">
                        <h4 className="text-xs font-bold text-slate-300 mb-2.5 flex items-center gap-1.5">
                          <Cpu className="w-4 h-4 text-purple-400" /> Gemma 3 Underwriting Synthesis
                        </h4>
                        <div className="text-xs text-slate-300 leading-relaxed italic bg-slate-950/40 p-4 rounded-xl border border-slate-800">
                          "{predictionResult.explanation}"
                        </div>
                      </div>

                      {/* Feature Contributions bar chart */}
                      <div className="glass-card p-5">
                        <h4 className="text-xs font-bold text-slate-300 mb-4 flex items-center gap-1.5">
                          <BarChart3 className="w-4 h-4 text-blue-400" /> Local Risk Contributions
                        </h4>
                        <div className="space-y-3">
                          {predictionResult.contributions.map((c: any) => {
                            const isPositive = c.contribution > 0;
                            const pctWidth = Math.min(Math.abs(c.contribution) * 1.5, 100);
                            return (
                              <div key={c.feature} className="text-xs">
                                <div className="flex justify-between text-slate-300 mb-1">
                                  <span className="font-semibold">{c.label} ({c.value})</span>
                                  <span className={`font-mono font-bold ${isPositive ? 'text-red-400' : 'text-emerald-400'}`}>
                                    {isPositive ? `+${c.contribution}%` : `${c.contribution}%`}
                                  </span>
                                </div>
                                <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden flex">
                                  <div className="w-1/2 flex justify-end bg-slate-950">
                                    {!isPositive && (
                                      <div 
                                        className="h-full bg-emerald-500" 
                                        style={{ width: `${pctWidth}%` }}
                                      />
                                    )}
                                  </div>
                                  <div className="w-1/2 bg-slate-950">
                                    {isPositive && (
                                      <div 
                                        className="h-full bg-red-500" 
                                        style={{ width: `${pctWidth}%` }}
                                      />
                                    )}
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="h-full min-h-[300px] flex flex-col items-center justify-center text-center p-8 border border-dashed border-slate-800 rounded-2xl bg-slate-950/20">
                      <Cpu className="w-12 h-12 text-slate-600 mb-3 animate-pulse" />
                      <p className="text-sm font-semibold text-slate-400 mb-1">Interactive Underwriter System Ready</p>
                      <p className="text-xs text-slate-500 max-w-sm">Enter client vitals and policy details on the left, then click 'Run Claim Underwrite' to evaluate actuarial risk and generate custom Gemma 3 summaries.</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
