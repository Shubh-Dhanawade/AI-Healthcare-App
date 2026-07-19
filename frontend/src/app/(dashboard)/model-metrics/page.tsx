'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { aiApi } from '@/lib/apiHelpers';
import { 
  Cpu, ShieldAlert, Eye, LineChart, 
  Settings, Clock, Award, Activity, Database
} from 'lucide-react';

interface LossPoint {
  step: number;
  train_loss: number;
  val_loss: number;
}

interface BenchmarkMetric {
  metric: string;
  before: number;
  after: number;
}

interface RAGAverage {
  faithfulness: number;
  answer_relevance: number;
  context_relevance: number;
  avg_latency: number;
  total_queries: number;
}

interface RAGEvaluation {
  query: string;
  answer: string;
  faithfulness: number;
  answer_relevance: number;
  context_relevance: number;
  latency: number;
  reasoning: string;
}

interface ModelMetricsData {
  fine_tuning_metrics: {
    model_name: string;
    base_model: string;
    dataset_used: string;
    train_samples: number;
    hyperparameters: {
      epochs: number;
      learning_rate: string;
      lora_r: number;
      lora_alpha: number;
      quantization: string;
      max_seq_length: number;
    };
    training_loss_curve: LossPoint[];
    knowledge_benchmarks: BenchmarkMetric[];
  };
  rag_evaluation_metrics: {
    averages: RAGAverage;
    recent_evals: RAGEvaluation[];
  };
}

export default function ModelMetricsPage() {
  const [activeTab, setActiveTab] = useState<'ft' | 'rag'>('ft');
  const [hoveredLossIndex, setHoveredLossIndex] = useState<number | null>(null);

  const { data: metrics, isLoading, error } = useQuery<ModelMetricsData>({
    queryKey: ['modelMetrics'],
    queryFn: aiApi.getModelMetrics,
  });

  if (isLoading) {
    return (
      <div className="space-y-6 fade-in">
        <div className="skeleton h-16 w-1/3" />
        <div className="grid md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="skeleton h-28 w-full" />
          ))}
        </div>
        <div className="skeleton h-96 w-full" />
      </div>
    );
  }

  if (error || !metrics) {
    return (
      <div className="glass-card p-8 text-center text-red-400">
        <ShieldAlert className="w-12 h-12 mx-auto mb-3" />
        <h3 className="text-lg font-bold mb-1">Failed to load AI model metrics</h3>
        <p className="text-sm text-slate-400">Please verify the backend server is running and you are logged in as an admin.</p>
      </div>
    );
  }

  const { fine_tuning_metrics, rag_evaluation_metrics } = metrics;
  const ft = fine_tuning_metrics;
  const rag = rag_evaluation_metrics;

  // SVG dimensions for Loss Curve Chart
  const svgWidth = 600;
  const svgHeight = 250;
  const padding = 40;
  const chartWidth = svgWidth - padding * 2;
  const chartHeight = svgHeight - padding * 2;

  const maxLoss = 3.0;
  const stepsCount = ft.training_loss_curve.length;

  const getCoordinates = (index: number, loss: number) => {
    const x = padding + (index / (stepsCount - 1)) * chartWidth;
    const y = padding + chartHeight - (loss / maxLoss) * chartHeight;
    return { x, y };
  };

  // Build SVG paths for training and validation loss
  let trainPath = '';
  let valPath = '';

  ft.training_loss_curve.forEach((pt, idx) => {
    const trainCoords = getCoordinates(idx, pt.train_loss);
    const valCoords = getCoordinates(idx, pt.val_loss);

    if (idx === 0) {
      trainPath = `M ${trainCoords.x} ${trainCoords.y}`;
      valPath = `M ${valCoords.x} ${valCoords.y}`;
    } else {
      trainPath += ` L ${trainCoords.x} ${trainCoords.y}`;
      valPath += ` L ${valCoords.x} ${valCoords.y}`;
    }
  });

  return (
    <div className="space-y-8 fade-in text-white">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold mb-1 text-white">AI Model Evaluation & RAG Audit</h1>
        <p className="text-slate-400 text-sm">Analyze training loss convergence, domain knowledge gains, and real-time retrieval accuracy (LLM-as-a-Judge).</p>
      </div>

      {/* Tab Switcher */}
      <div className="flex border-b border-slate-700/60 gap-2">
        <button
          onClick={() => setActiveTab('ft')}
          className={`flex items-center gap-2 px-5 py-3 -mb-px font-semibold text-sm border-b-2 transition-all ${
            activeTab === 'ft'
              ? 'border-blue-500 text-blue-400'
              : 'border-transparent text-slate-400 hover:text-white hover:border-slate-600'
          }`}
        >
          <Cpu className="w-4 h-4" />
          Gemma 3 Fine-Tuning Performance
        </button>
        <button
          onClick={() => setActiveTab('rag')}
          className={`flex items-center gap-2 px-5 py-3 -mb-px font-semibold text-sm border-b-2 transition-all ${
            activeTab === 'rag'
              ? 'border-blue-500 text-blue-400'
              : 'border-transparent text-slate-400 hover:text-white hover:border-slate-600'
          }`}
        >
          <Database className="w-4 h-4" />
          RAG Pipeline Audit (LLM-as-a-Judge)
        </button>
      </div>

      {activeTab === 'ft' && (
        <div className="space-y-6">
          {/* Top Hyperparams Summary */}
          <div className="grid md:grid-cols-3 gap-6">
            <div className="glass-card p-6 flex flex-col justify-between">
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wider mb-2 font-bold flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-blue-400" /> Model Identity
                </p>
                <h3 className="text-lg font-bold text-white mb-2">{ft.model_name.split('/').pop()}</h3>
                <p className="text-xs text-slate-400">Base model: <span className="font-mono text-blue-300">{ft.base_model}</span></p>
                <p className="text-xs text-slate-400 mt-1">Dataset: <span className="text-slate-300 font-semibold">{ft.dataset_used}</span> ({ft.train_samples} samples)</p>
              </div>
            </div>

            <div className="glass-card p-6 col-span-2">
              <p className="text-xs text-slate-500 uppercase tracking-wider mb-3 font-bold flex items-center gap-1.5">
                <Settings className="w-3.5 h-3.5 text-blue-400" /> Training Hyperparameters (QLoRA)
              </p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div className="p-3 rounded-lg bg-slate-900/40 border border-slate-800/60">
                  <p className="text-xs text-slate-500">Epochs</p>
                  <p className="font-bold text-white mt-0.5">{ft.hyperparameters.epochs}</p>
                </div>
                <div className="p-3 rounded-lg bg-slate-900/40 border border-slate-800/60">
                  <p className="text-xs text-slate-500">Learning Rate</p>
                  <p className="font-bold text-white mt-0.5 font-mono">{ft.hyperparameters.learning_rate}</p>
                </div>
                <div className="p-3 rounded-lg bg-slate-900/40 border border-slate-800/60">
                  <p className="text-xs text-slate-500">LoRA Rank / Alpha</p>
                  <p className="font-bold text-white mt-0.5">{ft.hyperparameters.lora_r} / {ft.hyperparameters.lora_alpha}</p>
                </div>
                <div className="p-3 rounded-lg bg-slate-900/40 border border-slate-800/60">
                  <p className="text-xs text-slate-500">Quantization</p>
                  <p className="font-bold text-white mt-0.5">{ft.hyperparameters.quantization}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Loss Curve & Benchmark Graph Grid */}
          <div className="grid lg:grid-cols-2 gap-6">
            {/* Interactive Loss Chart */}
            <div className="glass-card p-6">
              <h3 className="font-bold text-white mb-2 flex items-center gap-2">
                <LineChart className="w-5 h-5 text-blue-400" />
                Training & Validation Loss Convergence
              </h3>
              <p className="text-xs text-slate-400 mb-6">Cross-entropy loss tracked over 100 training steps, showing stable optimization without overfitting.</p>
              
              <div className="relative flex justify-center bg-slate-900/20 p-2 rounded-2xl border border-slate-800/40">
                <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full h-auto overflow-visible select-none">
                  {/* Grid Lines */}
                  {[0.5, 1.0, 1.5, 2.0, 2.5, 3.0].map((val) => {
                    const y = padding + chartHeight - (val / maxLoss) * chartHeight;
                    return (
                      <g key={val} className="opacity-10">
                        <line x1={padding} y1={y} x2={padding + chartWidth} y2={y} stroke="#fff" strokeWidth={1} strokeDasharray="4 4" />
                        <text x={padding - 10} y={y + 4} fill="#fff" fontSize={10} textAnchor="end">{val.toFixed(1)}</text>
                      </g>
                    );
                  })}

                  {/* X axis labels */}
                  {ft.training_loss_curve.map((pt, idx) => {
                    if (idx % 2 !== 0 && idx !== stepsCount - 1) return null;
                    const x = padding + (idx / (stepsCount - 1)) * chartWidth;
                    return (
                      <text key={idx} x={x} y={padding + chartHeight + 15} fill="#64748b" fontSize={9} textAnchor="middle" className="opacity-70">
                        Step {pt.step}
                      </text>
                    );
                  })}

                  {/* Curves */}
                  <path d={trainPath} fill="none" stroke="#3b82f6" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round" />
                  <path d={valPath} fill="none" stroke="#f59e0b" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round" strokeDasharray="3 3" />

                  {/* Interactive points & hover line */}
                  {ft.training_loss_curve.map((pt, idx) => {
                    const tCoords = getCoordinates(idx, pt.train_loss);
                    const isHovered = hoveredLossIndex === idx;

                    return (
                      <g key={idx}>
                        {/* Invisible hover area */}
                        <line
                          x1={tCoords.x}
                          y1={padding}
                          x2={tCoords.x}
                          y2={padding + chartHeight}
                          stroke="transparent"
                          strokeWidth={svgWidth / stepsCount}
                          onMouseEnter={() => setHoveredLossIndex(idx)}
                          onMouseLeave={() => setHoveredLossIndex(null)}
                          className="cursor-pointer"
                        />
                        {isHovered && (
                          <g>
                            <line x1={tCoords.x} y1={padding} x2={tCoords.x} y2={padding + chartHeight} stroke="rgba(255,255,255,0.15)" strokeWidth={1.5} />
                            
                            <circle cx={tCoords.x} cy={tCoords.y} r={5} fill="#3b82f6" stroke="#fff" strokeWidth={1.5} />
                            <circle cx={tCoords.x} cy={getCoordinates(idx, pt.val_loss).y} r={5} fill="#f59e0b" stroke="#fff" strokeWidth={1.5} />
                          </g>
                        )}
                      </g>
                    );
                  })}
                </svg>

                {/* Legend overlay */}
                <div className="absolute top-4 right-4 flex items-center gap-4 bg-slate-950/80 px-3 py-1.5 rounded-lg border border-slate-800/60 text-xs">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-0.5 bg-blue-500 inline-block" />
                    <span className="text-slate-300">Train Loss</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-0.5 bg-amber-500 border-dashed border-t inline-block" />
                    <span className="text-slate-300">Val Loss</span>
                  </div>
                </div>

                {/* Hover Tooltip Details */}
                {hoveredLossIndex !== null && (
                  <div className="absolute bottom-4 left-4 bg-slate-950/90 px-3 py-2 rounded-lg border border-slate-800/80 text-xs space-y-0.5">
                    <p className="font-semibold text-slate-200">Step {ft.training_loss_curve[hoveredLossIndex].step}</p>
                    <p className="text-blue-400">Train Loss: <span className="font-bold">{ft.training_loss_curve[hoveredLossIndex].train_loss.toFixed(3)}</span></p>
                    <p className="text-amber-400">Val Loss: <span className="font-bold">{ft.training_loss_curve[hoveredLossIndex].val_loss.toFixed(3)}</span></p>
                  </div>
                )}
              </div>
            </div>

            {/* Knowledge Benchmarks */}
            <div className="glass-card p-6 flex flex-col justify-between">
              <div>
                <h3 className="font-bold text-white mb-2 flex items-center gap-2">
                  <Award className="w-5 h-5 text-purple-400" />
                  Medical knowledge Fine-Tuning Gains
                </h3>
                <p className="text-xs text-slate-400 mb-6">Language evaluation scores comparing the base model vs the fine-tuned CORD-19 model on scientific Q&A splits.</p>
                
                <div className="space-y-4">
                  {ft.knowledge_benchmarks.map((row) => {
                    const diff = row.after - row.before;
                    return (
                      <div key={row.metric} className="p-4 rounded-xl bg-slate-900/30 border border-slate-800/40">
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-sm font-semibold text-slate-200">{row.metric} Score</span>
                          <span className="text-xs font-semibold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded">
                            +{diff.toFixed(1)}% improvement
                          </span>
                        </div>
                        <div className="space-y-1.5 text-xs text-slate-400">
                          <div className="flex items-center gap-2">
                            <span className="w-16">Finetuned:</span>
                            <div className="flex-1 bg-slate-800 h-2.5 rounded-full overflow-hidden">
                              <div className="h-full bg-purple-500 rounded-full" style={{ width: `${row.after}%` }} />
                            </div>
                            <span className="w-10 text-right font-bold text-white">{row.after}%</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="w-16">Base Model:</span>
                            <div className="flex-1 bg-slate-800/50 h-2 rounded-full overflow-hidden">
                              <div className="h-full bg-slate-600 rounded-full" style={{ width: `${row.before}%` }} />
                            </div>
                            <span className="w-10 text-right text-slate-500">{row.before}%</span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'rag' && (
        <div className="space-y-6">
          {/* RAG Stat Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="glass-card p-6 flex flex-col justify-between">
              <div>
                <p className="text-xs text-slate-500 uppercase font-semibold mb-1">Avg. Faithfulness</p>
                <h3 className="text-3xl font-black text-emerald-400">{(rag.averages.faithfulness * 100).toFixed(1)}%</h3>
              </div>
              <p className="text-[10px] text-slate-500 mt-2">Measures how grounded the LLM's claims are in the source text.</p>
            </div>
            <div className="glass-card p-6 flex flex-col justify-between">
              <div>
                <p className="text-xs text-slate-500 uppercase font-semibold mb-1">Avg. Answer Relevance</p>
                <h3 className="text-3xl font-black text-blue-400">{(rag.averages.answer_relevance * 100).toFixed(1)}%</h3>
              </div>
              <p className="text-[10px] text-slate-500 mt-2">Measures if the LLM actually answers the user's question directly.</p>
            </div>
            <div className="glass-card p-6 flex flex-col justify-between">
              <div>
                <p className="text-xs text-slate-500 uppercase font-semibold mb-1">Avg. Context Relevance</p>
                <h3 className="text-3xl font-black text-purple-400">{(rag.averages.context_relevance * 100).toFixed(1)}%</h3>
              </div>
              <p className="text-[10px] text-slate-500 mt-2">Measures similarity distance of retrieved chunk snippets.</p>
            </div>
            <div className="glass-card p-6 flex flex-col justify-between">
              <div>
                <p className="text-xs text-slate-500 uppercase font-semibold mb-1">Avg. Query Latency</p>
                <h3 className="text-3xl font-black text-amber-500">{rag.averages.avg_latency.toFixed(2)}s</h3>
              </div>
              <p className="text-[10px] text-slate-500 mt-2">Average response generation time including vector search.</p>
            </div>
          </div>

          {/* RAG Log Table */}
          <div className="glass-card p-6">
            <h3 className="font-bold text-white mb-2 flex items-center gap-2">
              <Database className="w-5 h-5 text-blue-400" />
              Live RAG Audit Log
            </h3>
            <p className="text-xs text-slate-400 mb-6">Recent user prompts run through the LLM-as-a-Judge RAG evaluation process, auditing output grounding.</p>

            <div className="space-y-4">
              {rag.recent_evals.map((row, idx) => (
                <div key={idx} className="p-5 rounded-2xl bg-slate-900/30 border border-slate-800/40 hover:border-slate-700/60 transition-all space-y-3">
                  <div className="flex justify-between items-start gap-4 flex-wrap">
                    <div className="space-y-1">
                      <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider">User Query</p>
                      <p className="text-sm font-semibold text-white">"{row.query}"</p>
                    </div>
                    
                    {/* Badge metrics pills */}
                    <div className="flex gap-2 text-xs">
                      <div className="px-2.5 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-center gap-1">
                        <span>Faithfulness:</span>
                        <span className="font-bold">{row.faithfulness.toFixed(2)}</span>
                      </div>
                      <div className="px-2.5 py-1 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400 flex items-center gap-1">
                        <span>Relevance:</span>
                        <span className="font-bold">{row.answer_relevance.toFixed(2)}</span>
                      </div>
                      <div className="px-2.5 py-1 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-400 flex items-center gap-1">
                        <Clock className="w-3.5 h-3.5" />
                        <span className="font-bold">{row.latency}s</span>
                      </div>
                    </div>
                  </div>

                  <div className="p-3.5 rounded-xl bg-slate-950/50 border border-slate-900 text-xs text-slate-300">
                    <p className="text-slate-500 uppercase tracking-wider font-semibold text-[10px] mb-1">Generated Answer</p>
                    <p className="leading-relaxed">{row.answer}</p>
                  </div>

                  <div className="flex items-start gap-2 text-xs text-slate-400">
                    <Eye className="w-4 h-4 text-purple-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <span className="text-purple-400 font-semibold">Judge Audit Reasoning:</span>{' '}
                      <span className="text-slate-400 italic">"{row.reasoning}"</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
