'use client';

import { useSearchParams, useRouter } from 'next/navigation';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { documentsApi, aiApi } from '@/lib/apiHelpers';
import { CompareResponse, Document, DocumentDetail, DocumentStatus } from '@/types';
import Link from 'next/link';
import { useState, useEffect, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  ArrowLeft, Scale, Brain, ShieldAlert, DollarSign,
  Heart, ShieldCheck, AlertTriangle, AlertCircle, Sparkles, Check,
  Upload, FileText, Image, X, CheckCircle, Search, CloudUpload, HelpCircle,
  Loader2, RotateCcw, Plus, Trash2, Shield
} from 'lucide-react';
import toast from 'react-hot-toast';

const MAX_SIZE_MB = 50;
const ACCEPTED_TYPES = {
  'application/pdf': ['.pdf'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png': ['.png'],
  'image/tiff': ['.tiff'],
  'image/webp': ['.webp'],
};

interface ProcessingFile {
  id: string; // temp unique id
  name: string;
  size: number;
  status: 'uploading' | 'extracting' | 'summarizing' | 'extracting_fields' | 'analyzing_risks' | 'ready' | 'failed';
  progress: number;
  error?: string;
  documentId?: string;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function ComparePage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const queryClient = useQueryClient();

  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [isComparing, setIsComparing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [processingFiles, setProcessingFiles] = useState<Record<string, ProcessingFile>>({});

  // Fetch all user documents
  const { data: documents = [], isLoading: isDocsLoading } = useQuery<Document[]>({
    queryKey: ['documents'],
    queryFn: documentsApi.list,
    refetchInterval: (query) => {
      // If we have files in processing/extracting state, poll to update user library statuses
      const hasActiveDocs = query.state.data?.some(doc => ['uploaded', 'processing', 'text_extracted'].includes(doc.status));
      return hasActiveDocs ? 3000 : false;
    }
  });

  // Handle URL deep links like /compare?ids=doc1,doc2
  useEffect(() => {
    const ids = searchParams.get('ids');
    if (ids) {
      const parsed = ids.split(',').filter(Boolean);
      if (parsed.length >= 2) {
        setSelectedIds(parsed);
        setIsComparing(true);
      }
    } else {
      setIsComparing(false);
    }
  }, [searchParams]);

  // Main comparison react-query
  const { data: compareData, isLoading: isCompareLoading, error: compareError } = useQuery<CompareResponse>({
    queryKey: ['compare', selectedIds.join(',')],
    queryFn: () => documentsApi.compare(selectedIds),
    enabled: isComparing && selectedIds.length >= 2,
    retry: false,
  });

  // Start the background processing pipeline
  const startProcessingPipeline = async (tempId: string, file: File) => {
    try {
      // 1. Upload file
      const doc = await documentsApi.upload(file, (progress) => {
        setProcessingFiles(prev => ({
          ...prev,
          [tempId]: prev[tempId] ? { ...prev[tempId], progress } : prev[tempId]
        }));
      });

      const docId = doc.id;
      setProcessingFiles(prev => ({
        ...prev,
        [tempId]: prev[tempId] ? { ...prev[tempId], status: 'extracting', documentId: docId } : prev[tempId]
      }));

      // 2. Poll for text extraction
      let attempts = 0;
      const maxAttempts = 60; // 2 minutes max
      let currentDoc = doc;

      while (attempts < maxAttempts) {
        currentDoc = await documentsApi.getById(docId);
        if (currentDoc.status === 'failed') {
          throw new Error('Text extraction failed on server.');
        }
        if (['text_extracted', 'summarized', 'completed'].includes(currentDoc.status)) {
          break;
        }
        await new Promise(r => setTimeout(r, 2000));
        attempts++;
      }

      if (attempts >= maxAttempts) {
        throw new Error('Text extraction timed out.');
      }

      // 3. AI Pipeline: Summarize
      if (currentDoc.status === 'text_extracted') {
        setProcessingFiles(prev => ({
          ...prev,
          [tempId]: prev[tempId] ? { ...prev[tempId], status: 'summarizing' } : prev[tempId]
        }));
        await aiApi.summarize(docId);
        currentDoc = await documentsApi.getById(docId);
      }

      // 4. AI Pipeline: Extract Fields
      if (currentDoc.status === 'summarized') {
        setProcessingFiles(prev => ({
          ...prev,
          [tempId]: prev[tempId] ? { ...prev[tempId], status: 'extracting_fields' } : prev[tempId]
        }));
        await aiApi.extractFields(docId);
        currentDoc = await documentsApi.getById(docId);
      }

      // 5. AI Pipeline: Risk Analysis
      if (currentDoc.risk_analyses && currentDoc.risk_analyses.length === 0) {
        setProcessingFiles(prev => ({
          ...prev,
          [tempId]: prev[tempId] ? { ...prev[tempId], status: 'analyzing_risks' } : prev[tempId]
        }));
        await aiApi.riskAnalysis(docId);
      }

      // 6. Complete
      setProcessingFiles(prev => {
        const updated = { ...prev };
        delete updated[tempId]; // Clean up from active upload list once ready
        return updated;
      });

      toast.success(`${file.name} is processed and ready!`);

      // Auto-select
      setSelectedIds(prev => {
        if (prev.includes(docId)) return prev;
        if (prev.length >= 3) {
          toast.error('Could not auto-select. Already selected maximum of 3 policies.');
          return prev;
        }
        return [...prev, docId];
      });

      // Refresh documents query
      queryClient.invalidateQueries({ queryKey: ['documents'] });

    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || 'Processing failed';
      setProcessingFiles(prev => ({
        ...prev,
        [tempId]: prev[tempId] ? { ...prev[tempId], status: 'failed', error: errorMsg } : prev[tempId]
      }));
      toast.error(`Failed to process ${file.name}: ${errorMsg}`);
    }
  };

  // Dropzone setup
  const onDrop = useCallback((accepted: File[], rejected: any[]) => {
    if (rejected.length > 0) {
      const reason = rejected[0]?.errors?.[0]?.message || 'Invalid file';
      toast.error(`File rejected: ${reason}`);
      return;
    }

    const currentSelectedCount = selectedIds.length;
    const currentProcessingCount = Object.keys(processingFiles).length;
    if (currentSelectedCount + currentProcessingCount + accepted.length > 3) {
      toast.error('You can compare a maximum of 3 policies.');
      return;
    }

    accepted.forEach(file => {
      const tempId = Math.random().toString(36).substring(7);
      const newFile: ProcessingFile = {
        id: tempId,
        name: file.name,
        size: file.size,
        status: 'uploading',
        progress: 0,
      };

      setProcessingFiles(prev => ({ ...prev, [tempId]: newFile }));
      startProcessingPipeline(tempId, file);
    });
  }, [selectedIds, processingFiles]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxSize: MAX_SIZE_MB * 1024 * 1024,
    multiple: true,
  });

  const clearProcessingFile = (tempId: string) => {
    setProcessingFiles(prev => {
      const updated = { ...prev };
      delete updated[tempId];
      return updated;
    });
  };

  const handleCompareClick = () => {
    if (selectedIds.length < 2) {
      toast.error('Please select at least 2 policies to compare.');
      return;
    }
    router.push(`/compare?ids=${selectedIds.join(',')}`);
    setIsComparing(true);
  };

  const handleBackToSelection = () => {
    router.push('/compare');
    setIsComparing(false);
  };

  const toggleSelectDocument = (docId: string, status: string) => {
    if (status !== 'completed' && status !== 'summarized') {
      toast.error('This policy is still being analyzed. Please wait.');
      return;
    }
    setSelectedIds(prev => {
      if (prev.includes(docId)) {
        return prev.filter(id => id !== docId);
      }
      if (prev.length >= 3) {
        toast.error('You can compare a maximum of 3 policies.');
        return prev;
      }
      return [...prev, docId];
    });
  };

  const filteredDocs = documents.filter(d =>
    d.original_filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getStepIcon = (status: string) => {
    switch (status) {
      case 'uploading':
        return <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />;
      case 'extracting':
        return <Loader2 className="w-4 h-4 text-amber-400 animate-spin" />;
      case 'summarizing':
        return <Brain className="w-4 h-4 text-violet-400 animate-pulse" />;
      case 'extracting_fields':
        return <Search className="w-4 h-4 text-teal-400 animate-pulse" />;
      case 'analyzing_risks':
        return <ShieldAlert className="w-4 h-4 text-red-400 animate-pulse" />;
      case 'ready':
        return <CheckCircle className="w-4 h-4 text-emerald-400" />;
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <Loader2 className="w-4 h-4 text-slate-500 animate-spin" />;
    }
  };

  const getStepMessage = (status: string, progress: number) => {
    switch (status) {
      case 'uploading':
        return `Uploading document... (${progress}%)`;
      case 'extracting':
        return 'Extracting policy text...';
      case 'summarizing':
        return 'Generating AI summary...';
      case 'extracting_fields':
        return 'Extracting fields (premium, copay, etc)...';
      case 'analyzing_risks':
        return 'Running risk analysis...';
      case 'ready':
        return 'Ready!';
      case 'failed':
        return 'Processing failed';
      default:
        return 'Processing...';
    }
  };

  const getFieldValue = (doc: DocumentDetail, name: string) => {
    const field = doc.extracted_fields.find(
      (f) => f.field_name.toLowerCase() === name.toLowerCase() ||
        f.field_name.toLowerCase().replace(/_/g, ' ') === name.toLowerCase()
    );
    return field?.field_value || '—';
  };

  const getGoodPoints = (doc: DocumentDetail, synthesis: any) => {
    const points: string[] = [];

    // 1. Check feature winners from AI synthesis
    if (synthesis?.feature_winners) {
      synthesis.feature_winners.forEach((winner: any) => {
        const docName = doc.original_filename.toLowerCase();
        const policyName = getFieldValue(doc, 'policy name').toLowerCase();
        const insurerName = getFieldValue(doc, 'insurer name').toLowerCase();
        const winnerName = winner.winner.toLowerCase();

        if (
          winnerName !== 'tie' &&
          (docName.includes(winnerName) ||
            winnerName.includes(docName.replace(/\.[^/.]+$/, "")) ||
            policyName.includes(winnerName) ||
            winnerName.includes(policyName) ||
            insurerName.includes(winnerName))
        ) {
          points.push(`${winner.feature}: ${winner.reason}`);
        }
      });
    }

    // 2. Fallback rule-based strengths
    const coPay = getFieldValue(doc, 'co payment').toLowerCase();
    if (coPay.includes('no') || coPay.includes('0%') || coPay === '—' || coPay === 'none' || coPay === 'null') {
      if (points.length < 3) points.push("Co-Payment: No co-payment required");
    }

    const deductible = getFieldValue(doc, 'deductible').toLowerCase();
    if (deductible.includes('no') || deductible.includes('0') || deductible === '—' || deductible === 'none' || deductible === 'null') {
      if (points.length < 3) points.push("Deductible: No deductible before claims");
    }

    const roomRent = getFieldValue(doc, 'room rent limit').toLowerCase();
    if (roomRent.includes('no limit') || roomRent.includes('up to sum') || roomRent.includes('single') || roomRent.includes('no cap')) {
      if (points.length < 3) points.push("Room Rent: No cap or sub-limits on rooms");
    }

    const network = getFieldValue(doc, 'network hospitals').toLowerCase();
    if (network.includes('5,000') || network.includes('10,000') || network.includes('large') || network.includes('5000')) {
      if (points.length < 3) points.push("Network: Extensive cashless network");
    }

    if (points.length === 0) {
      points.push("Coverage: Comprehensive hospitalization cover");
      points.push("Claims: Cashless settlement support");
    } else if (points.length === 1) {
      points.push("Claims: Standard cashless claim process");
    }

    return points.slice(0, 3);
  };

  const getRiskColor = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'high':
        return 'text-red-400 bg-red-400/10 border-red-500/20';
      case 'medium':
        return 'text-amber-400 bg-amber-400/10 border-amber-500/20';
      case 'low':
      default:
        return 'text-emerald-400 bg-emerald-400/10 border-emerald-500/20';
    }
  };

  // ----------------------------------------------------
  // RENDER VIEW: SELECTION SCREEN
  // ----------------------------------------------------
  if (!isComparing) {
    const processingFilesList = Object.values(processingFiles);

    return (
      <div className="space-y-8 pb-12 fade-in">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold gradient-text flex items-center gap-2">
            <Scale className="w-6 h-6 text-blue-400" /> Compare Policies
          </h1>
          <p className="text-slate-400 text-sm">Upload new insurance documents or select from your library to generate an AI comparative dashboard.</p>
        </div>

        {/* Selected Counter & Button Bar */}
        <div className="glass-card p-5 border border-blue-500/20 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-blue-500/15 text-blue-400 flex items-center justify-center border border-blue-500/30">
              <Scale className="w-5 h-5" />
            </div>
            <div>
              <p className="font-semibold text-white text-sm">Selection Status</p>
              <p className="text-slate-400 text-xs">
                Selected <span className="text-blue-400 font-bold">{selectedIds.length}</span> of <span className="text-slate-200">3</span> policies.
                {selectedIds.length < 2 && ' Select at least 2 to compare.'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3 w-full sm:w-auto">
            {selectedIds.length > 0 && (
              <button
                onClick={() => setSelectedIds([])}
                className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors w-full sm:w-auto text-center"
              >
                Clear Selected
              </button>
            )}
            <button
              onClick={handleCompareClick}
              disabled={selectedIds.length < 2}
              className={`w-full sm:w-auto px-6 py-2.5 rounded-xl font-semibold text-sm transition-all flex items-center justify-center gap-2 shadow-lg ${selectedIds.length >= 2
                ? 'bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 text-white shadow-blue-500/25 hover:scale-[1.02] active:scale-[0.98]'
                : 'bg-slate-800 text-slate-500 border border-slate-700/50 cursor-not-allowed'
                }`}
            >
              <Scale className="w-4 h-4" /> Compare Selected Policies
            </button>
          </div>
        </div>

        {/* Dashboard Grid */}
        <div className="grid lg:grid-cols-12 gap-8">
          {/* Library Right Column */}
          <div className="col-span-12">
            <div className="glass-card p-6 space-y-4 flex flex-col h-full min-h-[400px]">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-slate-800/80 pb-4">
                <div>
                  <h3 className="font-bold text-white text-base">Select from Library</h3>
                  <p className="text-xs text-slate-500 mt-0.5">Choose policies from your uploaded collection</p>
                </div>
                <div className="relative w-full sm:w-60">
                  <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500" />
                  <input
                    type="text"
                    className="form-input pl-8 py-1.5 text-xs rounded-lg"
                    placeholder="Search documents..."
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                  />
                </div>
              </div>

              {/* Document List */}
              <div className="flex-1 overflow-y-auto max-h-[450px] space-y-2 pr-1">
                {isDocsLoading ? (
                  <div className="space-y-2 py-4">
                    {[1, 2, 3].map(i => <div key={i} className="skeleton h-14 w-full rounded-xl" />)}
                  </div>
                ) : filteredDocs.length === 0 ? (
                  <div className="text-center py-16">
                    <FileText className="w-12 h-12 text-slate-700 mx-auto mb-2" />
                    <p className="text-slate-400 text-xs">No documents found</p>
                    <p className="text-slate-600 text-[10px] mt-1">Upload files on the left to start comparison.</p>
                  </div>
                ) : (
                  filteredDocs.map((doc) => {
                    const isSelectable = doc.status === 'completed' || doc.status === 'summarized';
                    const isSelected = selectedIds.includes(doc.id);

                    return (
                      <div
                        key={doc.id}
                        onClick={() => toggleSelectDocument(doc.id, doc.status)}
                        className={`flex items-center justify-between p-3 rounded-xl border transition-all cursor-pointer select-none ${isSelected
                          ? 'bg-blue-500/5 border-blue-500/30'
                          : 'bg-slate-900/10 border-slate-800/80 hover:bg-white/1'
                          } ${!isSelectable ? 'opacity-55' : ''}`}
                      >
                        <div className="flex items-center gap-3 min-w-0 flex-1">
                          <input
                            type="checkbox"
                            className="w-4 h-4 rounded border-slate-700 bg-[#0c1322] text-blue-500 focus:ring-blue-500/20 disabled:opacity-30"
                            checked={isSelected}
                            disabled={!isSelectable}
                            onChange={() => { }} // Handled by outer container click
                          />
                          <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                            style={{ background: doc.file_type === 'pdf' ? 'rgba(239,68,68,0.1)' : 'rgba(59,130,246,0.1)' }}>
                            <FileText className="w-3.5 h-3.5" style={{ color: doc.file_type === 'pdf' ? '#f87171' : '#60a5fa' }} />
                          </div>
                          <div className="min-w-0 flex-1">
                            <h4 className="text-xs font-semibold text-slate-200 truncate pr-2" title={doc.original_filename}>
                              {doc.original_filename}
                            </h4>
                            <p className="text-[10px] text-slate-500 flex items-center gap-1.5 capitalize mt-0.5">
                              <span>{doc.file_type}</span>
                              <span>•</span>
                              <span>{formatBytes(doc.file_size_bytes)}</span>
                              {doc.page_count > 0 && (
                                <>
                                  <span>•</span>
                                  <span>{doc.page_count} Pages</span>
                                </>
                              )}
                            </p>
                          </div>
                        </div>

                        {/* Status / Check indicators */}
                        <div>
                          {!isSelectable ? (
                            <span className="text-[9px] px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/20 text-amber-400 flex items-center gap-1">
                              <Loader2 className="w-2.5 h-2.5 animate-spin" /> Processing
                            </span>
                          ) : isSelected ? (
                            <span className="w-5 h-5 rounded-full bg-blue-500/20 border border-blue-500/40 flex items-center justify-center text-blue-400">
                              <Check className="w-3.5 h-3.5" />
                            </span>
                          ) : null}
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>

        </div>
      </div>
    );
  }

  // ----------------------------------------------------
  // RENDER VIEW: LOADING STATE DURING COMPARISON
  // ----------------------------------------------------
  if (isCompareLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-[70vh] gap-4">
        <div className="relative w-16 h-16">
          <div className="absolute inset-0 rounded-full border-4 border-blue-500/20 border-t-blue-500 animate-spin" />
          <div className="absolute inset-2 rounded-full border-4 border-violet-500/20 border-t-violet-500 animate-spin [animation-duration:1.5s]" />
        </div>
        <div className="text-center mt-2">
          <p className="text-white font-semibold">Running AI Comparative Synthesis...</p>
          <p className="text-slate-500 text-sm mt-1">Analyzing coverage parameters, limits, exclusions & risks side-by-side</p>
        </div>
      </div>
    );
  }

  // ----------------------------------------------------
  // RENDER VIEW: FAILURE / ERROR STATE
  // ----------------------------------------------------
  if (compareError || !compareData) {
    return (
      <div className="flex flex-col items-center justify-center h-[70vh] text-center p-6">
        <AlertCircle className="w-16 h-16 text-red-500 mb-4" />
        <h2 className="text-xl font-bold text-white mb-2">Comparison Failed</h2>
        <p className="text-slate-400 max-w-md mb-6">
          {compareError instanceof Error ? compareError.message : 'An error occurred while generating the side-by-side comparison.'}
        </p>
        <button onClick={handleBackToSelection} className="btn-primary">
          <RotateCcw className="w-4 h-4" /> Modify Comparison Selection
        </button>
      </div>
    );
  }

  // ----------------------------------------------------
  // RENDER VIEW: SIDE-BY-SIDE COMPARISON DASHBOARD
  // ----------------------------------------------------
  const { documents: comparedDocs, comparison_synthesis } = compareData;
  const colCount = comparedDocs.length;

  const parameters = [
    { name: 'Insurer', key: 'insurer name', category: 'General' },
    { name: 'Plan Name', key: 'policy name', category: 'General' },
    { name: 'Policy Number', key: 'policy number', category: 'General' },
    { name: 'Policy Term', key: 'policy term', category: 'General' },
    { name: 'Coverage Type', key: 'coverage type', category: 'General' },
    { name: 'Sum Insured', key: 'sum insured', category: 'Coverage & Limits' },
    { name: 'Room Rent Limit', key: 'room rent limit', category: 'Coverage & Limits' },
    { name: 'Maternity Benefit', key: 'maternity coverage', category: 'Coverage & Limits' },
    { name: 'Pre-Existing Disease', key: 'pre existing coverage', category: 'Coverage & Limits' },
    { name: 'Network Hospital Info', key: 'network hospitals', category: 'Coverage & Limits' },
    { name: 'Premium Amount', key: 'premium amount', category: 'Costs & Fees' },
    { name: 'Deductible', key: 'deductible', category: 'Costs & Fees' },
    { name: 'Co-Payment', key: 'co payment', category: 'Costs & Fees' },
    { name: 'Waiting Period', key: 'waiting period', category: 'Restrictions & Claims' },
    { name: 'Claim Process', key: 'claim process', category: 'Restrictions & Claims' },
  ];

  const categories = Array.from(new Set(parameters.map(p => p.category)));

  // Find the best policy ID based on wins or recommendation text
  const getBestPolicyId = () => {
    if (!comparison_synthesis) return null;

    // Count feature wins
    const winCounts = comparedDocs.map((doc) => {
      let wins = 0;
      const docName = doc.original_filename.toLowerCase();
      const policyName = getFieldValue(doc, 'policy name').toLowerCase();
      const insurerName = getFieldValue(doc, 'insurer name').toLowerCase();

      comparison_synthesis.feature_winners?.forEach((winner) => {
        const winnerName = winner.winner.toLowerCase();
        if (
          winnerName !== 'tie' &&
          (docName.includes(winnerName) ||
            winnerName.includes(docName.replace(/\.[^/.]+$/, "")) ||
            policyName.includes(winnerName) ||
            winnerName.includes(policyName) ||
            insurerName.includes(winnerName))
        ) {
          wins++;
        }
      });
      return { id: doc.id, wins };
    });

    const maxWins = Math.max(...winCounts.map(w => w.wins));
    if (maxWins > 0) {
      const winners = winCounts.filter(w => w.wins === maxWins);
      if (winners.length === 1) {
        return winners[0].id;
      }
    }

    // Fallback: Check if verdict text mentions the policy name or filename
    const verdictLower = comparison_synthesis.verdict?.toLowerCase() || '';
    for (const doc of comparedDocs) {
      const docName = doc.original_filename.toLowerCase();
      const policyName = getFieldValue(doc, 'policy name').toLowerCase();
      const insurerName = getFieldValue(doc, 'insurer name').toLowerCase();
      if (
        (policyName && policyName !== '—' && verdictLower.includes(policyName)) ||
        verdictLower.includes(docName.replace(/\.[^/.]+$/, "")) ||
        (insurerName && insurerName !== '—' && verdictLower.includes(insurerName))
      ) {
        return doc.id;
      }
    }

    return null;
  };

  const bestPolicyId = getBestPolicyId();

  return (
    <div className="space-y-8 pb-12 fade-in">
      {/* Header with Navigation */}
      <div className="flex items-center gap-4">
        <button
          onClick={handleBackToSelection}
          className="p-2.5 rounded-xl border border-slate-700 bg-slate-800/40 text-slate-400 hover:text-white hover:bg-slate-800/80 transition flex items-center gap-1.5 font-medium text-xs"
        >
          <ArrowLeft className="w-4 h-4" /> Change Policies
        </button>
        <div>
          <h1 className="text-2xl font-bold gradient-text flex items-center gap-2">
            <Scale className="w-6 h-6 text-blue-400" /> Comparison Analysis
          </h1>
          <p className="text-slate-400 text-sm">Comparing {colCount} health insurance policies side-by-side</p>
        </div>
      </div>

      {/* AI Synthesis Summary Card — Per-Policy Side-by-Side */}
      <div className="glass-card relative overflow-hidden border border-blue-500/20 shadow-[0_0_30px_rgba(59,130,246,0.05)]">
        <div className="absolute top-0 right-0 w-96 h-96 rounded-full opacity-10 blur-3xl pointer-events-none"
          style={{ background: 'radial-gradient(circle, #8b5cf6, transparent)', transform: 'translate(40%, -40%)' }} />
        <div className="absolute bottom-0 left-0 w-64 h-64 rounded-full opacity-5 blur-3xl pointer-events-none"
          style={{ background: 'radial-gradient(circle, #06b6d4, transparent)', transform: 'translate(-30%, 30%)' }} />

        {/* Card Header */}
        <div className="flex items-center gap-2.5 px-6 pt-6 pb-4 border-b border-slate-700/40">
          <div className="w-8 h-8 rounded-lg bg-violet-500/20 flex items-center justify-center text-violet-400 border border-violet-500/30">
            <Sparkles className="w-4 h-4 animate-pulse" />
          </div>
          <div>
            <h2 className="font-bold text-white text-base">AI Comparison Synthesis</h2>
            <p className="text-xs text-slate-500">Side-by-side AI generated comparative analysis</p>
          </div>
        </div>

        {/* Policy Columns */}
        <div className={`grid ${colCount === 3 ? 'md:grid-cols-3' : 'md:grid-cols-2'} divide-x divide-slate-700/40`}>
          {comparedDocs.map((doc, idx) => {
            const policyNameVal = getFieldValue(doc, 'policy name');
            const insurerNameVal = getFieldValue(doc, 'insurer name');
            const label = (policyNameVal && policyNameVal !== '—') ? policyNameVal : doc.original_filename.replace(/\.[^/.]+$/, '');
            const insurer = (insurerNameVal && insurerNameVal !== '—') ? insurerNameVal : 'Policy';
            const isBest = doc.id === bestPolicyId;
            const accentColors = [
              { bg: 'bg-blue-500/10', border: 'border-blue-500/20', text: 'text-blue-400', dot: 'bg-blue-400' },
              { bg: 'bg-violet-500/10', border: 'border-violet-500/20', text: 'text-violet-400', dot: 'bg-violet-400' },
              { bg: 'bg-teal-500/10', border: 'border-teal-500/20', text: 'text-teal-400', dot: 'bg-teal-400' },
            ];
            const accent = accentColors[idx % accentColors.length];

            // Parse key differences per-policy from the synthesis text
            const synthLines = comparison_synthesis.synthesis
              .split('\n')
              .filter((l: string) => l.trim());

            // Filter lines that reference this policy by name / insurer / index
            const labelLow = label.toLowerCase();
            const insurerLow = insurer.toLowerCase().split(' ')[0]; // first word of insurer
            const policyKeywords = [labelLow, insurerLow].filter(Boolean);

            const relevantDiff = synthLines.filter((l: string) => {
              const ll = l.toLowerCase();
              return policyKeywords.some(kw => kw.length > 3 && ll.includes(kw));
            });
            const diffLines = relevantDiff.length > 0 ? relevantDiff : synthLines.slice(idx * 2, idx * 2 + 3);

            // Parse best_for per-policy
            const bestForLines = comparison_synthesis.best_for
              .split('\n')
              .filter((l: string) => l.trim());
            const relevantBest = bestForLines.filter((l: string) => {
              const ll = l.toLowerCase();
              return policyKeywords.some(kw => kw.length > 3 && ll.includes(kw));
            });
            const bestLines = relevantBest.length > 0 ? relevantBest : [bestForLines[idx] || bestForLines[0] || ''];

            return (
              <div key={doc.id} className="p-5 md:p-6 space-y-5 relative">
                {/* Policy Label */}
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className={`w-2 h-2 rounded-full flex-shrink-0 ${accent.dot}`} />
                    <div className="min-w-0">
                      <p className={`text-[10px] font-bold uppercase tracking-widest ${accent.text}`}>{insurer}</p>
                      <p className="text-sm font-bold text-white truncate" title={label}>{label}</p>
                    </div>
                  </div>
                  {isBest && (
                    <span className="text-[9px] font-extrabold px-2 py-0.5 rounded-full bg-amber-500/15 border border-amber-500/30 text-amber-400 uppercase tracking-wide whitespace-nowrap flex-shrink-0">
                      ★ Best Value
                    </span>
                  )}
                </div>

                {/* Key Differences */}
                <div className="space-y-2">
                  <p className="text-[10px] font-bold text-violet-400 uppercase tracking-widest flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-violet-400 inline-block" />
                    Key Differences
                  </p>
                  <ul className="space-y-1.5">
                    {diffLines.slice(0, 4).map((line: string, li: number) => {
                      const cleanLine = line.replace(/^[-•*]\s*/, '').replace(/\*\*/g, '').trim();
                      return cleanLine ? (
                        <li key={li} className="flex items-start gap-2 text-xs text-slate-300 leading-relaxed">
                          <span className="mt-1.5 w-1 h-1 rounded-full bg-violet-500/60 flex-shrink-0" />
                          <span>{cleanLine}</span>
                        </li>
                      ) : null;
                    })}
                  </ul>
                </div>

                {/* Divider */}
                <div className="border-t border-slate-700/30" />

                {/* Best Suited For */}
                <div className="space-y-2">
                  <p className="text-[10px] font-bold text-teal-400 uppercase tracking-widest flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-teal-400 inline-block" />
                    Best Suited For
                  </p>
                  <ul className="space-y-1.5">
                    {bestLines.slice(0, 3).map((line: string, li: number) => {
                      const cleanLine = line.replace(/^[-•*]\s*/, '').replace(/\*\*/g, '').trim();
                      return cleanLine ? (
                        <li key={li} className="flex items-start gap-2 text-xs text-slate-300 leading-relaxed">
                          <Check className="w-3.5 h-3.5 text-teal-400 flex-shrink-0 mt-0.5" />
                          <span>{cleanLine}</span>
                        </li>
                      ) : null;
                    })}
                  </ul>
                </div>
              </div>
            );
          })}
        </div>

        {/* Shared Advisor Verdict — Full Width Banner */}
        <div className="mx-4 mb-5 rounded-xl bg-gradient-to-r from-blue-500/8 via-violet-500/8 to-blue-500/8 border border-blue-500/15 p-4 flex items-start gap-3">
          <div className="w-8 h-8 rounded-lg bg-blue-500/15 flex items-center justify-center flex-shrink-0 border border-blue-500/25 mt-0.5">
            <Brain className="w-4 h-4 text-blue-400" />
          </div>
          <div>
            <p className="text-[10px] font-bold text-blue-400 uppercase tracking-widest mb-1.5">Advisor Verdict</p>
            <p className="text-white text-sm font-medium leading-relaxed">{comparison_synthesis.verdict}</p>
          </div>
        </div>
      </div>

      {/* Feature Winner Highlights Card Grid */}
      {comparison_synthesis.feature_winners && comparison_synthesis.feature_winners.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-amber-500/10 text-amber-400 flex items-center justify-center border border-amber-500/20">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white">AI Feature Winner Analysis</h2>
              <p className="text-xs text-slate-500">Highlighting the strongest benefits of each policy option</p>
            </div>
          </div>
          <div className="grid sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {comparison_synthesis.feature_winners.map((winner, idx) => (
              <div
                key={idx}
                className="glass-card p-4 border border-slate-800/80 bg-slate-900/10 hover:border-slate-700/60 transition flex flex-col justify-between"
              >
                <div>
                  <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block mb-1">
                    {winner.feature}
                  </span>
                  <h4 className="text-amber-400 font-bold text-sm mb-1.5 flex items-center gap-1">
                    <Check className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                    <span className="truncate" title={winner.winner}>{winner.winner}</span>
                  </h4>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed border-t border-slate-800/40 pt-2 mt-2">
                  {winner.reason}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Side-by-Side Card Comparison Dashboard */}
      <div className={`grid gap-6 ${colCount === 3 ? 'md:grid-cols-3' : 'md:grid-cols-2'}`}>
        {comparedDocs.map((doc, idx) => {
          // Calculate overall risk
          const highCount = doc.risk_analyses?.filter(r => r.severity === 'high').length || 0;
          const medCount = doc.risk_analyses?.filter(r => r.severity === 'medium').length || 0;
          let riskText = 'Low Risk';
          let riskSeverity = 'low';
          if (highCount > 0) {
            riskText = 'High Risk';
            riskSeverity = 'high';
          } else if (medCount > 0) {
            riskText = 'Medium Risk';
            riskSeverity = 'medium';
          }

          const premium = getFieldValue(doc, 'premium amount');
          const sumInsured = getFieldValue(doc, 'sum insured');
          const coPay = getFieldValue(doc, 'co payment');
          const deductible = getFieldValue(doc, 'deductible');

          const isBestValue = doc.id === bestPolicyId;

          const insurerNameVal = getFieldValue(doc, 'insurer name');
          const insurerName = (insurerNameVal && insurerNameVal !== '—') ? insurerNameVal : 'General Insurer';

          const policyNameVal = getFieldValue(doc, 'policy name');
          const policyName = (policyNameVal && policyNameVal !== '—') ? policyNameVal : doc.original_filename;

          return (
            <div
              key={doc.id}
              className={`glass-card relative overflow-hidden p-6 border transition-all duration-300 flex flex-col justify-between ${isBestValue
                ? 'border-amber-500/40 bg-gradient-to-b from-[#0d1322]/80 via-[#0d1322]/40 to-transparent shadow-[0_0_40px_rgba(245,158,11,0.08)]'
                : 'border-slate-800/80 bg-[#0d1322]/40'
                } hover:border-blue-500/30`}
            >
              {/* Top accent gradient / Best value badge */}
              {isBestValue ? (
                <div className="absolute top-0 right-0 bg-gradient-to-l from-amber-500 to-orange-500 text-slate-950 text-[10px] font-extrabold px-3 py-1 rounded-bl-xl uppercase tracking-wider flex items-center gap-1 shadow-lg z-10 animate-pulse">
                  <Sparkles className="w-3 h-3 fill-slate-950" /> Best Value Option
                </div>
              ) : (
                <div className="absolute top-0 left-0 right-0 h-[3px] bg-gradient-to-r from-blue-500 to-violet-600" />
              )}

              <div className="space-y-6">
                {/* Header */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-[10px] font-bold text-slate-500 tracking-wider uppercase">POLICY #{idx + 1}</span>
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded border capitalize ${getRiskColor(riskSeverity)}`}>
                      {riskText}
                    </span>
                  </div>
                  <div>
                    <span className="text-xs text-blue-400 font-bold tracking-wide uppercase block">
                      {insurerName}
                    </span>
                    <h3 className="text-lg font-bold text-white leading-snug truncate" title={policyName}>
                      {policyName}
                    </h3>
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5 pt-1">
                    <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-medium">
                      {doc.page_count} Pages
                    </span>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 capitalize font-medium">
                      {doc.file_type}
                    </span>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-medium">
                      Term: {getFieldValue(doc, 'policy term')}
                    </span>
                  </div>
                </div>

                {/* Key Strengths / Good Points */}
                <div className="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/10 space-y-2">
                  <div className="flex items-center gap-1.5 text-emerald-400 font-bold text-xs">
                    <Sparkles className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
                    Key Strengths
                  </div>
                  <ul className="space-y-2">
                    {getGoodPoints(doc, comparison_synthesis).map((point, pIdx) => (
                      <li key={pIdx} className="text-xs text-slate-300 flex items-start gap-2 leading-relaxed">
                        <Check className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                        <span>{point}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Pricing / Sum Insured Callout */}
                <div className="bg-slate-950/20 border border-slate-800/60 rounded-2xl p-4 flex items-center justify-between gap-4">
                  <div>
                    <p className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Premium Amount</p>
                    <p className="text-base font-extrabold text-white mt-0.5 truncate max-w-[130px] md:max-w-[150px]" title={premium}>{premium}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Sum Insured</p>
                    <p className="text-base font-extrabold text-blue-400 mt-0.5 truncate max-w-[130px] md:max-w-[150px]" title={sumInsured}>{sumInsured}</p>
                  </div>
                </div>

                {/* Policy Parameters Grid */}
                <div className="space-y-3.5 pt-2">
                  <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest border-b border-slate-800/40 pb-1.5">
                    Plan Details
                  </h4>

                  {/* Co-Pay */}
                  <div className="flex items-center justify-between text-xs py-1 border-b border-slate-800/20">
                    <span className="text-slate-500 flex items-center gap-2">
                      <DollarSign className="w-3.5 h-3.5 text-slate-500" /> Co-Payment
                    </span>
                    <span className="font-semibold text-slate-200">{coPay}</span>
                  </div>

                  {/* Deductible */}
                  <div className="flex items-center justify-between text-xs py-1 border-b border-slate-800/20">
                    <span className="text-slate-500 flex items-center gap-2">
                      <Scale className="w-3.5 h-3.5 text-slate-500" /> Deductible
                    </span>
                    <span className="font-semibold text-slate-200">{deductible}</span>
                  </div>

                  {/* Coverage Type */}
                  <div className="flex items-center justify-between text-xs py-1 border-b border-slate-800/20">
                    <span className="text-slate-500 flex items-center gap-2">
                      <Heart className="w-3.5 h-3.5 text-slate-500" /> Coverage Type
                    </span>
                    <span className="font-semibold text-slate-200">{getFieldValue(doc, 'coverage type')}</span>
                  </div>

                  {/* Room Rent */}
                  <div className="flex items-center justify-between text-xs py-1 border-b border-slate-800/20">
                    <span className="text-slate-500 flex items-center gap-2">
                      <Shield className="w-3.5 h-3.5 text-slate-500" /> Room Rent Limit
                    </span>
                    <span className="font-semibold text-slate-200 text-right truncate max-w-[150px]" title={getFieldValue(doc, 'room rent limit')}>
                      {getFieldValue(doc, 'room rent limit')}
                    </span>
                  </div>

                  {/* Network Hospitals */}
                  <div className="flex items-center justify-between text-xs py-1 border-b border-slate-800/20">
                    <span className="text-slate-500 flex items-center gap-2">
                      <ShieldCheck className="w-3.5 h-3.5 text-slate-500" /> Network Info
                    </span>
                    <span className="font-semibold text-slate-200 text-right truncate max-w-[150px]" title={getFieldValue(doc, 'network hospitals')}>
                      {getFieldValue(doc, 'network hospitals')}
                    </span>
                  </div>

                  {/* Maternity Coverage */}
                  <div className="flex items-center justify-between text-xs py-1 border-b border-slate-800/20">
                    <span className="text-slate-500 flex items-center gap-2">
                      <Sparkles className="w-3.5 h-3.5 text-slate-500" /> Maternity Cover
                    </span>
                    <span className="font-semibold text-slate-200 text-right truncate max-w-[150px]" title={getFieldValue(doc, 'maternity coverage')}>
                      {getFieldValue(doc, 'maternity coverage')}
                    </span>
                  </div>

                  {/* Pre Existing Waiting */}
                  <div className="flex items-center justify-between text-xs py-1 border-b border-slate-800/20">
                    <span className="text-slate-500 flex items-center gap-2">
                      <RotateCcw className="w-3.5 h-3.5 text-slate-500" /> Pre-Existing Wait
                    </span>
                    <span className="font-semibold text-slate-200 text-right truncate max-w-[150px]" title={getFieldValue(doc, 'pre existing coverage')}>
                      {getFieldValue(doc, 'pre existing coverage')}
                    </span>
                  </div>

                  {/* Waiting Period */}
                  <div className="flex items-center justify-between text-xs py-1 border-b border-slate-800/20">
                    <span className="text-slate-500 flex items-center gap-2">
                      <Loader2 className="w-3.5 h-3.5 text-slate-500" /> Initial Wait Period
                    </span>
                    <span className="font-semibold text-slate-200 text-right truncate max-w-[150px]" title={getFieldValue(doc, 'waiting period')}>
                      {getFieldValue(doc, 'waiting period')}
                    </span>
                  </div>

                  {/* Claim Process */}
                  <div className="flex items-center justify-between text-xs py-1">
                    <span className="text-slate-500 flex items-center gap-2">
                      <FileText className="w-3.5 h-3.5 text-slate-500" /> Claim Process
                    </span>
                    <span className="font-semibold text-slate-200 text-right truncate max-w-[150px]" title={getFieldValue(doc, 'claim process')}>
                      {getFieldValue(doc, 'claim process')}
                    </span>
                  </div>
                </div>
              </div>

              {/* Risk Analyses (Bottom Card Section) */}
              <div className="border-t border-slate-800/80 pt-5 mt-6 space-y-3">
                <h4 className="text-xs font-bold text-red-400/90 uppercase tracking-widest flex items-center gap-1.5">
                  <ShieldAlert className="w-4 h-4 text-red-400" /> Critical Risk Clauses
                </h4>

                {doc.risk_analyses.length === 0 ? (
                  <div className="flex items-center gap-1.5 text-emerald-400 text-xs py-2.5 bg-emerald-400/5 border border-emerald-500/10 rounded-xl px-3 font-medium">
                    <ShieldCheck className="w-4 h-4 text-emerald-400" /> No critical risk clauses found.
                  </div>
                ) : (
                  <div className="space-y-3">
                    {doc.risk_analyses.slice(0, 2).map((risk, index) => (
                      <div key={risk.id} className="bg-[#0c1322]/60 border border-slate-800/80 rounded-xl p-3.5 space-y-1.5">
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-[9px] text-slate-500 font-bold uppercase">Clause #{index + 1}</span>
                          <span className={`text-[9px] font-bold px-1.5 rounded uppercase ${risk.severity === 'high' ? 'text-red-400 bg-red-400/10' : 'text-amber-400 bg-amber-400/10'
                            }`}>
                            {risk.severity} Severity
                          </span>
                        </div>
                        <p className="text-xs font-semibold text-white italic truncate" title={risk.clause_text}>
                          &ldquo;{risk.clause_text}&rdquo;
                        </p>
                        <p className="text-[11px] text-slate-400 leading-normal">
                          {risk.explanation}
                        </p>
                        {risk.recommendation && (
                          <p className="text-[10px] text-blue-400 font-semibold pt-1.5 border-t border-slate-800/40 mt-1">
                            Rec: {risk.recommendation}
                          </p>
                        )}
                      </div>
                    ))}
                    {doc.risk_analyses.length > 2 && (
                      <div className="text-right">
                        <Link
                          href={`/documents/${doc.id}?tab=risks`}
                          className="text-[11px] text-blue-400 hover:text-blue-300 font-semibold inline-flex items-center gap-1"
                        >
                          +{doc.risk_analyses.length - 2} more risks →
                        </Link>
                      </div>
                    )}
                  </div>
                )}
              </div>

            </div>
          );
        })}
      </div>
    </div>
  );
}
