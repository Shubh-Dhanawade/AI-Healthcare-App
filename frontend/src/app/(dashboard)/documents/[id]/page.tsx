'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { documentsApi, aiApi, exportApi } from '@/lib/apiHelpers';
import { DocumentDetail, RiskAnalysis } from '@/types';
import { useParams, useRouter } from 'next/navigation';
import toast from 'react-hot-toast';
import {
  ArrowLeft, Brain, Shield, FileText, Search, RefreshCw,
  AlertTriangle, Info, ChevronDown, ChevronUp,
  Send, MessageSquare, List, Loader2, Download, Mail
} from 'lucide-react';
import { useState, useEffect, useRef } from 'react';
import DocumentStatusBadge from '@/components/documents/DocumentStatusBadge';
import Link from 'next/link';

const loadScript = (src: string) => {
  return new Promise((resolve, reject) => {
    if (typeof window === 'undefined') {
      resolve(false);
      return;
    }
    if (document.querySelector(`script[src="${src}"]`)) {
      resolve(true);
      return;
    }
    const script = document.createElement('script');
    script.src = src;
    script.onload = () => resolve(true);
    script.onerror = () => reject(new Error(`Failed to load script ${src}`));
    document.head.appendChild(script);
  });
};



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

function cleanResponse(raw: string): string {
  if (!raw) return '';
  return raw
    .replace(/\[SOURCES:[^\]]*\]/g, '')           // strip the sources tag
    .replace(/\s*\bIn\s+[a-zA-Z0-9_\-\.]+\.(?:pdf|docx)\s*-\s*Page\s*\d+(?:\s*context)?\.?/gi, '') // strip any inline citations
    .replace(/\n*\b(?:Reference|Source)s?:[\s\S]*$/gi, '') // strip any trailing inline references/sources
    .trim();
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
          <p className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">{cleanResponse(answer)}</p>
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

const generateSimplePrintHTML = (doc: DocumentDetail, selectedLanguage: string) => {
  // Build fields list
  let fieldsRows = '';
  if (doc.extracted_fields && doc.extracted_fields.length > 0) {
    doc.extracted_fields.forEach(f => {
      fieldsRows += `
        <div style="padding: 10px 0; border-bottom: 1px solid #e2e8f0; display: flex; justify-content: space-between; font-size: 13px;">
          <span style="font-weight: bold; color: #000000; width: 40%;">${f.field_name}:</span>
          <span style="color: #000000; width: 55%; text-align: left;">${f.field_value || '—'}</span>
        </div>
      `;
    });
  } else {
    fieldsRows = '<div style="padding: 10px 0; text-align: center; color: #000000;">No extracted fields available.</div>';
  }

  // Build risk cards
  let risksContent = '';
  if (doc.risk_analyses && doc.risk_analyses.length > 0) {
    doc.risk_analyses.forEach(r => {
      const severityColor = r.severity === 'high' ? '#dc2626' : (r.severity === 'medium' ? '#d97706' : '#059669');
      risksContent += `
        <div style="border: 1px solid #e2e8f0; border-left: 5px solid ${severityColor}; padding: 15px; border-radius: 6px; margin-bottom: 15px; background-color: #ffffff;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: bold; font-size: 13px; color: #000000;">${r.risk_type.replace(/_/g, ' ').toUpperCase()}</span>
            <span style="font-size: 11px; font-weight: bold; color: ${severityColor}; border: 1px solid ${severityColor}; padding: 2px 8px; border-radius: 4px;">${r.severity.toUpperCase()}</span>
          </div>
          <p style="margin: 0 0 6px 0; font-size: 12.5px; color: #000000; font-style: italic;">"${r.clause_text}"</p>
          ${r.explanation ? `<p style="margin: 0 0 4px 0; font-size: 12px; color: #000000;"><strong>Analysis:</strong> ${r.explanation}</p>` : ''}
          ${r.recommendation ? `<p style="margin: 0; font-size: 12px; color: #dc2626; font-weight: bold;"><strong>Recommendation:</strong> ${r.recommendation}</p>` : ''}
        </div>
      `;
    });
  } else {
    risksContent = '<p style="color: #059669; font-weight: bold; font-size: 13px;">No critical risks identified.</p>';
  }

  return `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
          color: #000000;
          line-height: 1.6;
          margin: 0;
          padding: 30px;
          background: #ffffff;
        }
        .container {
          max-width: 800px;
          margin: 0 auto;
          background-color: white;
        }
        .header {
          border-bottom: 3px solid #000000;
          padding-bottom: 12px;
          margin-bottom: 30px;
        }
        .header h1 {
          margin: 0;
          font-size: 26px;
          font-weight: bold;
          color: #000000;
        }
        .meta-line {
          font-size: 12px;
          color: #000000;
          margin-top: 5px;
          opacity: 0.8;
        }
        .section {
          margin-bottom: 35px;
        }
        .section-title {
          font-size: 15px;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          color: #000000;
          background: #f1f5f9;
          padding: 8px 12px;
          border-left: 5px solid #000000;
          font-weight: bold;
          margin-bottom: 15px;
        }
        .summary-text {
          font-size: 13.5px;
          color: #000000;
          text-align: justify;
          margin-bottom: 20px;
        }
        .info-card {
          border: 1px solid #e2e8f0;
          padding: 15px;
          border-radius: 6px;
          margin-bottom: 15px;
          background: #ffffff;
        }
        .info-card-title {
          font-weight: bold;
          font-size: 13px;
          color: #000000;
          margin-bottom: 5px;
        }
        .info-card-content {
          font-size: 12.5px;
          color: #000000;
          margin: 0;
          white-space: pre-wrap;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>Healthcare Policy Analysis Report</h1>
          <div class="meta-line">
            Document: ${doc.original_filename} &bull; Processed: ${new Date(doc.created_at).toLocaleDateString()} &bull; Safety Score: ${doc.safety_score}/100
          </div>
        </div>

        <div class="section">
          <div class="section-title">AI Executive Summary (${selectedLanguage})</div>
          <div class="summary-text">${doc.summary?.summary_text || 'No summary available.'}</div>
          
          ${doc.summary?.coverage_summary ? `
            <div class="info-card">
              <div class="info-card-title">✅ Covered Items</div>
              <pre class="info-card-content">${doc.summary.coverage_summary}</pre>
            </div>
          ` : ''}
          
          ${doc.summary?.exclusions_summary ? `
            <div class="info-card">
              <div class="info-card-title">❌ Excluded Items</div>
              <pre class="info-card-content">${doc.summary.exclusions_summary}</pre>
            </div>
          ` : ''}
          
          ${doc.summary?.waiting_period_summary ? `
            <div class="info-card">
              <div class="info-card-title">⏰ Waiting Periods</div>
              <pre class="info-card-content">${doc.summary.waiting_period_summary}</pre>
            </div>
          ` : ''}
          
          ${doc.summary?.premium_summary ? `
            <div class="info-card">
              <div class="info-card-title">💰 Premium Details</div>
              <pre class="info-card-content">${doc.summary.premium_summary}</pre>
            </div>
          ` : ''}
        </div>

        <div class="section">
          <div class="section-title">Extracted Policy Parameters</div>
          <div style="border: 1px solid #e2e8f0; padding: 15px; border-radius: 6px; background-color: #ffffff;">
            ${fieldsRows}
          </div>
        </div>

        <div class="section">
          <div class="section-title">Critical Risk Audit</div>
          <div>
            ${risksContent}
          </div>
        </div>
      </div>
    </body>
    </html>
  `;
};

export default function DocumentDetailPage() {
  const params = useParams();
  const docId = params.id as string;
  const router = useRouter();
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<'summary' | 'fields' | 'risks' | 'query'>('summary');
  const [queryInput, setQueryInput] = useState('');
  const [isQuerying, setIsQuerying] = useState(false);
  const [chatHistory, setChatHistory] = useState<any[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const sessionIdRef = useRef<string | null>(null);

  // Export / Sharing states
  const [showEmailForm, setShowEmailForm] = useState(false);
  const [emailInput, setEmailInput] = useState('');
  const [sendingEmail, setSendingEmail] = useState(false);
  const [showExportDropdown, setShowExportDropdown] = useState(false);

  const handleDownload = async () => {
    if (!doc) return;
    const toastId = toast.loading('Generating and downloading PDF report...');

    try {
      // 1. Load html2pdf from CDN
      await loadScript('https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js');

      // 2. Generate the simple print HTML report template
      const printHTML = generateSimplePrintHTML(doc, selectedLanguage);

      // 3. Create a temporary hidden iframe container
      const iframe = document.createElement('iframe');
      iframe.style.position = 'fixed';
      iframe.style.left = '-9999px';
      iframe.style.top = '-9999px';
      iframe.style.width = '800px';
      iframe.style.height = '1100px';
      document.body.appendChild(iframe);

      const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;
      if (!iframeDoc) {
        throw new Error('Failed to access iframe context');
      }

      iframeDoc.write(printHTML);
      iframeDoc.close();

      // Wait for the stylesheet rules inside the iframe to mount and paint
      setTimeout(async () => {
        try {
          const opt = {
            margin: 10,
            filename: `HealthAI_Report_${doc.original_filename.replace(/\.[^/.]+$/, "") || docId}.pdf`,
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: {
              scale: 2,
              useCORS: true,
              backgroundColor: '#ffffff' // Clear white background
            },
            jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
          };

          // @ts-ignore
          await html2pdf().from(iframeDoc.body).set(opt).save();

          toast.success('PDF report downloaded successfully!', { id: toastId });
        } catch (pdfError) {
          console.error('PDF generation inside iframe failed, falling back to window print:', pdfError);
          const printWindow = window.open('', '_blank');
          if (printWindow) {
            printWindow.document.write(printHTML);
            printWindow.document.close();
            printWindow.focus();
            setTimeout(() => {
              printWindow.print();
            }, 800);
            toast.success('Print to PDF dialog opened successfully!', { id: toastId });
          } else {
            toast.error('Pop-up blocked. Please allow pop-ups to print the PDF report.', { id: toastId });
          }
        } finally {
          // Cleanup iframe
          if (document.body.contains(iframe)) {
            document.body.removeChild(iframe);
          }
        }
      }, 500);

    } catch (error) {
      console.error('Download error:', error);
      toast.error('Failed to generate PDF report', { id: toastId });
    }
  };

  const handleWhatsAppShare = () => {
    if (!doc) return;

    let text = `🏥 *HealthAI Policy Audit Report*\n`;
    text += `*Policy Name:* ${doc.original_filename}\n\n`;

    if (doc.summary) {
      text += `*Summary in Brief:*\n${doc.summary.summary_text}\n\n`;
    }

    if (doc.extracted_fields && doc.extracted_fields.length > 0) {
      text += `*Key Policy Details:*\n`;
      const keyFields = ['policy_name', 'insurer_name', 'sum_insured', 'premium_amount', 'deductible', 'co_payment'];
      const fieldsToPrint = doc.extracted_fields.filter(f =>
        keyFields.includes(f.field_name.toLowerCase().replace(/\s/g, '_')) ||
        keyFields.includes(f.field_name.toLowerCase())
      );

      const printedFields = fieldsToPrint.length > 0 ? fieldsToPrint : doc.extracted_fields;
      printedFields.slice(0, 6).forEach(f => {
        text += `• ${f.field_name}: ${f.field_value || '—'}\n`;
      });
      text += `\n`;
    }

    if (doc.risk_analyses && doc.risk_analyses.length > 0) {
      const highRisks = doc.risk_analyses.filter(r => r.severity === 'high');
      if (highRisks.length > 0) {
        text += `*⚠️ Critical Risks Detected:*\n`;
        highRisks.slice(0, 3).forEach(r => {
          text += `• ${r.risk_type.replace(/_/g, ' ').toUpperCase()} (${r.severity.toUpperCase()}): ${r.clause_text}\n`;
        });
        text += `\n`;
      }
    }

    text += `Generated by HealthAI.`;
    const whatsappUrl = `https://api.whatsapp.com/send?text=${encodeURIComponent(text)}`;
    window.open(whatsappUrl, '_blank');
  };

  const handleSendEmail = async () => {
    if (!emailInput.trim()) return;
    setSendingEmail(true);
    const toastId = toast.loading(`Sending email to ${emailInput}...`);
    try {
      const res = await exportApi.emailReport(docId, emailInput);
      toast.success(res.message || 'Email sent successfully!', { id: toastId });
      setShowEmailForm(false);
      setEmailInput('');
    } catch (error: any) {
      console.error('Email error:', error);
      toast.error(error.response?.data?.detail || 'Failed to send email', { id: toastId });
    } finally {
      setSendingEmail(false);
    }
  };

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

  // Poll while document is processing OR while AI analysis results are still being generated
  const { data: doc, isLoading, refetch } = useQuery<DocumentDetail>({
    queryKey: ['document', docId],
    queryFn: () => documentsApi.getById(docId),
    refetchInterval: (query) => {
      const d = query.state.data;
      if (!d) return false;
      if (['uploaded', 'processing', 'text_extracted'].includes(d.status)) return 2000;
      // Keep polling until ALL auto-generated results are present
      const anyMissing = !d.summary || d.extracted_fields.length === 0 || d.risk_analyses.length === 0;
      if (anyMissing && ['completed', 'summarized'].includes(d.status)) return 3000;
      return false;
    },
  });

  const summarizeMutation = useMutation({
    mutationFn: () => documentsApi.runSummary(docId),
    onSuccess: () => {
      toast.success('Summary regeneration started! Results will update automatically.');
      queryClient.invalidateQueries({ queryKey: ['document', docId] });
    },
    onError: (err: any) => toast.error(err.response?.data?.detail || 'Summarization failed'),
  });

  const runFieldsMutation = useMutation({
    mutationFn: () => documentsApi.runFields(docId),
    onSuccess: () => {
      toast.success('Fields extraction started in background!');
      queryClient.invalidateQueries({ queryKey: ['document', docId] });
    },
    onError: (err: any) => toast.error(err.response?.data?.detail || 'Failed to start fields extraction'),
  });

  const runRisksMutation = useMutation({
    mutationFn: () => documentsApi.runRisks(docId),
    onSuccess: () => {
      toast.success('Risk analysis started in background!');
      queryClient.invalidateQueries({ queryKey: ['document', docId] });
    },
    onError: (err: any) => toast.error(err.response?.data?.detail || 'Failed to start risk analysis'),
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
          document_id: docId,
          session_id: sessionIdRef.current,
          history: historyPayload
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to initialize stream: ${response.statusText}`);
      }

      // Capture the session_id from the response header (for subsequent messages)
      const returnedSessionId = response.headers.get('X-Chat-Session-Id');
      if (returnedSessionId && !sessionIdRef.current) {
        sessionIdRef.current = returnedSessionId;
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



  // Load persistent chat history from the database when Chat tab is opened
  useEffect(() => {
    if (activeTab !== 'query' || !docId) return;
    if (chatHistory.length > 0 || sessionIdRef.current) return; // already loaded
    setChatLoading(true);
    aiApi.getDocumentChatHistory(docId)
      .then(({ session_id, messages }) => {
        sessionIdRef.current = session_id;
        if (messages.length > 0) {
          const hydrated = messages.map((m) =>
            m.role === 'user'
              ? { query: m.content, isUser: true, timestamp: new Date(m.created_at) }
              : { answer: m.content, context: m.sources || [], evaluation: null, isUser: false, timestamp: new Date(m.created_at) }
          );
          setChatHistory(hydrated);
        }
      })
      .catch((err) => {
        console.warn('Could not load chat history:', err);
      })
      .finally(() => setChatLoading(false));
  }, [activeTab, docId]);





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

  const displayedSummary = doc.summary
    ? (selectedLanguage === 'English'
      ? doc.summary
      : translations[selectedLanguage] || doc.summary) as any
    : null;

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

      {/* AI Actions — Auto-triggered, manual buttons hidden during auto-processing */}
      {canRunAI && (
        <div className="glass-card p-5">
          <h2 className="font-semibold text-sm text-slate-300 mb-3 flex items-center gap-2">
            <Brain className="w-4 h-4 text-blue-400" /> AI Analysis Progress
          </h2>

          {/* AUTO-PROCESSING indicator during text extraction phase */}
          {doc.status === 'text_extracted' && (
            <div className="mb-4 flex items-center gap-3 px-4 py-3 rounded-xl"
              style={{ background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.3)' }}>
              <div className="w-4 h-4 border-2 border-blue-400/30 border-t-blue-400 rounded-full animate-spin flex-shrink-0" />
              <div className="flex-1">
                <p className="text-blue-300 font-medium text-xs">Auto-launching AI analyses...</p>
                <p className="text-slate-500 text-[10px] mt-0.5">Summary, field extraction, and risk analysis will start automatically.</p>
              </div>
            </div>
          )}

          {/* Summary auto-processing banner */}
          {(summarizeMutation.isPending || (!doc.summary && ['completed', 'summarized'].includes(doc.status))) && (
            <div className="mb-3 flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm"
              style={{ background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.25)' }}>
              <Loader2 className="w-4 h-4 text-blue-400 animate-spin flex-shrink-0" />
              <div>
                <p className="text-blue-300 font-medium text-xs">Generating summary in background…</p>
                <p className="text-slate-500 text-[10px] mt-0.5">This will be brief and to the point.</p>
              </div>
            </div>
          )}

          {/* Fields background banner — shown while queuing (isPending) */}
          {runFieldsMutation.isPending && (
            <div
              className="mb-3 flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm"
              style={{ background: 'rgba(139,92,246,0.1)', border: '1px solid rgba(139,92,246,0.3)' }}
            >
              <Loader2 className="w-4 h-4 text-violet-400 animate-spin flex-shrink-0" />
              <div>
                <p className="text-violet-300 font-medium text-xs">Extracting fields in background…</p>
                <p className="text-slate-500 text-[10px] mt-0.5">Key policy details will be extracted automatically.</p>
              </div>
            </div>
          )}

          {/* Fields polling banner — after 202, while results are missing */}
          {!runFieldsMutation.isPending && ['completed', 'summarized'].includes(doc.status) && doc.extracted_fields.length === 0 && (
            <div
              className="mb-3 flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm"
              style={{ background: 'rgba(139,92,246,0.07)', border: '1px solid rgba(139,92,246,0.2)' }}
            >
              <Loader2 className="w-4 h-4 text-violet-300 animate-spin flex-shrink-0" />
              <div>
                <p className="text-violet-300 font-medium text-xs">Extracting fields in background…</p>
                <p className="text-slate-500 text-[10px] mt-0.5">Results will populate here automatically.</p>
              </div>
            </div>
          )}

          {/* Risks background banner — shown while queuing (isPending) */}
          {runRisksMutation.isPending && (
            <div
              className="mb-3 flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm"
              style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)' }}
            >
              <Loader2 className="w-4 h-4 text-red-400 animate-spin flex-shrink-0" />
              <div>
                <p className="text-red-300 font-medium text-xs">Running risk analysis in background…</p>
                <p className="text-slate-500 text-[10px] mt-0.5">Identifying potential risks and exclusions.</p>
              </div>
            </div>
          )}

          {/* Risks polling banner — after 202, while results are missing */}
          {!runRisksMutation.isPending && ['completed', 'summarized'].includes(doc.status) && doc.risk_analyses.length === 0 && (
            <div
              className="mb-3 flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm"
              style={{ background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.18)' }}
            >
              <Loader2 className="w-4 h-4 text-red-300 animate-spin flex-shrink-0" />
              <div>
                <p className="text-red-300 font-medium text-xs">Risk analysis running in background…</p>
                <p className="text-slate-500 text-[10px] mt-0.5">Results will populate here automatically.</p>
              </div>
            </div>
          )}

          {/* Manual re-trigger buttons (hidden during initial auto-processing) */}
          {['completed', 'summarized'].includes(doc.status) && (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-3 border-t border-white/5">
              <p className="col-span-full text-xs text-slate-500 mb-2">Re-run analyses manually if needed:</p>

              {/* Re-summarize */}
              <button
                id="summarize-btn"
                onClick={() => { summarizeMutation.mutate(); setActiveTab('summary'); }}
                disabled={summarizeMutation.isPending || !canRunAI}
                className="btn-secondary justify-center py-2.5"
              >
                {summarizeMutation.isPending
                  ? <><span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Summarizing...</>
                  : <><Brain className="w-4 h-4" /> Re-summarize</>
                }
              </button>

              {/* Extract Fields — separate background button */}
              <button
                id="extract-fields-btn"
                onClick={() => runFieldsMutation.mutate()}
                disabled={runFieldsMutation.isPending || !canRunAI}
                className="btn-secondary justify-center py-2.5"
                style={{ color: '#a78bfa', borderColor: 'rgba(139,92,246,0.4)' }}
              >
                {runFieldsMutation.isPending
                  ? <><Loader2 className="w-4 h-4 animate-spin" /> Starting...</>
                  : <><Search className="w-4 h-4" /> Re-extract Fields</>
                }
              </button>

              {/* Risk Analysis — separate background button */}
              <button
                id="risk-analysis-btn"
                onClick={() => { runRisksMutation.mutate(); }}
                disabled={runRisksMutation.isPending || !canRunAI}
                className="btn-secondary justify-center py-2.5"
                style={{ color: '#f87171', borderColor: 'rgba(239,68,68,0.4)' }}
              >
                {runRisksMutation.isPending
                  ? <><Loader2 className="w-4 h-4 animate-spin" /> Starting...</>
                  : <><Shield className="w-4 h-4" /> Re-run Risk Analysis</>
                }
              </button>
            </div>
          )}
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
              className={`flex items-center gap-2 px-5 py-3 text-sm font-medium transition-all border-b-2 -mb-px ${activeTab === tab.id
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
            if (!displayedSummary) {
              return (
                <div className="text-center py-12 glass-card">
                  <Brain className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                  <p className="text-slate-400">No summary yet. Click "AI Summarize" to generate one.</p>
                </div>
              );
            }

            return (
              <div className="space-y-4" id="summary-report-pdf">
                {/* Summary in Brief & Export/Share Panel */}
                <div className="glass-card p-6 border border-white/5 space-y-4">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div>
                      <h2 className="text-lg font-bold text-white flex items-center gap-2">
                        <Brain className="w-5 h-5 text-blue-400 animate-pulse" /> Summary
                      </h2>
                      <p className="text-slate-400 text-xs mt-0.5">
                        Executive summary of the policy (comprehensive review)
                      </p>
                    </div>
                    {/* Export, Share and Language Actions */}
                    <div className="flex items-center gap-2 relative" data-html2canvas-ignore="true">
                      {/* Language Selector */}
                      <div className="flex items-center gap-1.5 bg-slate-950/60 border border-slate-700/40 rounded-xl px-2.5 py-1.5 mr-2">
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

                      {/* Dropdown Container */}
                      <div className="relative">
                        <button
                          onClick={() => setShowExportDropdown(!showExportDropdown)}
                          className="btn-secondary text-xs py-2 px-3.5 flex items-center gap-1.5 hover:scale-[1.02] active:scale-[0.98] transition-all cursor-pointer border border-blue-500/30 text-blue-400"
                          title="Export options"
                        >
                          <Download className="w-3.5 h-3.5" />
                          <span>Export</span>
                          <ChevronDown className={`w-3 h-3 transition-transform duration-200 ${showExportDropdown ? 'rotate-180' : ''}`} />
                        </button>

                        {/* Export Dropdown Menu */}
                        {showExportDropdown && (
                          <div className="absolute right-0 mt-2 w-48 rounded-xl bg-[#0b0f19] border border-white/10 shadow-2xl p-1.5 z-50 animate-in fade-in slide-in-from-top-2 duration-150">
                            {/* Download PDF Option */}
                            <button
                              onClick={() => {
                                handleDownload();
                                setShowExportDropdown(false);
                              }}
                              className="w-full flex items-center gap-2.5 px-3 py-2 text-xs text-left rounded-lg text-slate-200 hover:bg-white/5 hover:text-white transition-all cursor-pointer"
                            >
                              <FileText className="w-3.5 h-3.5 text-blue-400" />
                              <span>Download PDF</span>
                            </button>

                            {/* WhatsApp Option */}
                            <button
                              onClick={() => {
                                handleWhatsAppShare();
                                setShowExportDropdown(false);
                              }}
                              className="w-full flex items-center gap-2.5 px-3 py-2 text-xs text-left rounded-lg text-slate-200 hover:bg-emerald-500/10 hover:text-emerald-400 transition-all cursor-pointer"
                            >
                              <svg className="w-3.5 h-3.5 fill-current text-emerald-400" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path d="M.057 24l1.687-6.163c-1.041-1.804-1.588-3.849-1.587-5.946C.06 5.348 5.397.01 12.008.01c3.202.001 6.212 1.246 8.477 3.514 2.266 2.268 3.507 5.28 3.505 8.484-.004 6.657-5.34 11.997-11.953 11.997-2.005-.001-3.973-.502-5.717-1.458zm6.575-3.466l.393.233c1.524.905 3.284 1.382 5.083 1.382 5.861 0 10.629-4.767 10.631-10.631.001-2.84-1.093-5.509-3.079-7.502-1.986-1.992-4.63-3.089-7.555-3.09-5.886 0-10.655 4.767-10.658 10.632-.001 1.956.513 3.864 1.489 5.586l.248.441-1.03 3.763zm14.195-7.616c-.3-.15-1.77-.874-2.048-.975-.278-.102-.48-.153-.681.15-.202.302-.78.975-.957 1.177-.177.203-.355.228-.655.078-3.002-1.5-4.225-2.655-5.632-5.07-.375-.644.375-.598 1.074-1.997.12-.24.06-.454-.03-.604-.09-.15-.681-1.643-.933-2.247-.245-.588-.493-.509-.681-.519-.177-.009-.38-.01-.582-.01-.202 0-.531.076-.81.381-.278.305-1.062 1.037-1.062 2.53 0 1.493 1.088 2.936 1.239 3.138.152.203 2.14 3.267 5.185 4.578 2.457 1.058 3.093 1.012 3.655.96.67-.063 1.77-.723 2.022-1.396.253-.673.253-1.25.177-1.397-.076-.146-.278-.223-.578-.374z" />
                              </svg>
                              <span>Share via WhatsApp</span>
                            </button>

                            {/* Email Option */}
                            <button
                              onClick={() => {
                                setShowEmailForm(!showEmailForm);
                                setShowExportDropdown(false);
                              }}
                              className="w-full flex items-center gap-2.5 px-3 py-2 text-xs text-left rounded-lg text-slate-200 hover:bg-blue-600/20 hover:text-blue-400 transition-all cursor-pointer"
                            >
                              <Mail className="w-3.5 h-3.5 text-blue-400" />
                              <span>Send via Email</span>
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-950/40 border border-white/5 rounded-xl p-4 relative overflow-hidden">
                    {isTranslating && (
                      <div className="absolute inset-0 bg-[#0a0f1e]/60 backdrop-blur-sm flex items-center justify-center z-10 transition-all">
                        <div className="flex items-center gap-2">
                          <span className="w-5 h-5 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
                          <span className="text-xs text-slate-400">Translating summary to {selectedLanguage}...</span>
                        </div>
                      </div>
                    )}
                    <p className="text-slate-200 text-sm leading-relaxed whitespace-pre-wrap">
                      {displayedSummary?.summary_text}
                    </p>
                  </div>

                  {/* Detailed Policy Breakdowns */}
                  <div className="mt-6 border-t border-white/5 pt-6 space-y-4">
                    <h3 className="text-xs font-bold text-slate-400 tracking-wider uppercase">Policy Details & Exclusions</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {[
                        { label: '✅ Coverage & Benefits', value: displayedSummary.coverage_summary, color: '#10b981', border: 'border-emerald-500/10', bg: 'bg-emerald-500/5' },
                        { label: '❌ Exclusions & Limits', value: displayedSummary.exclusions_summary, color: '#ef4444', border: 'border-red-500/10', bg: 'bg-red-500/5' },
                        { label: '⏰ Waiting Periods', value: displayedSummary.waiting_period_summary, color: '#f59e0b', border: 'border-amber-500/10', bg: 'bg-amber-500/5' },
                        { label: '💰 Premium & Charges', value: displayedSummary.premium_summary, color: '#3b82f6', border: 'border-blue-500/10', bg: 'bg-blue-500/5' },
                      ].filter(s => s.value).map((section) => (
                        <div key={section.label} className={`p-4 rounded-xl border ${section.border} ${section.bg}`}>
                          <h4 className="font-bold text-xs uppercase tracking-wider mb-2.5" style={{ color: section.color }}>{section.label}</h4>
                          <div className="text-slate-300 text-xs leading-relaxed space-y-2 whitespace-pre-wrap">
                            {section.value}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Email Inline Form */}
                  {showEmailForm && (
                    <div className="border border-white/5 bg-slate-950/30 rounded-xl p-4 flex flex-col sm:flex-row items-center gap-3 max-w-md">
                      <input
                        type="email"
                        placeholder="Enter recipient email..."
                        value={emailInput}
                        onChange={(e) => setEmailInput(e.target.value)}
                        className="form-input text-xs py-2 px-3 flex-1 bg-slate-950/50"
                      />
                      <div className="flex gap-2 w-full sm:w-auto">
                        <button
                          onClick={handleSendEmail}
                          disabled={sendingEmail || !emailInput.trim()}
                          className="btn-primary text-xs py-2 px-4 flex-1 sm:flex-initial justify-center cursor-pointer"
                        >
                          {sendingEmail ? (
                            <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                          ) : (
                            'Send'
                          )}
                        </button>
                        <button
                          onClick={() => { setShowEmailForm(false); setEmailInput(''); }}
                          className="btn-secondary text-xs py-2 px-3 flex-1 sm:flex-initial justify-center cursor-pointer border-transparent"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}
                </div>
                <p className="text-xs text-slate-500 text-right">Generated by {displayedSummary.model_used || doc.summary?.model_used || 'AI model'} • {doc.summary?.created_at ? new Date(doc.summary.created_at).toLocaleString() : ''}</p>
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
                {chatLoading ? (
                  <div className="flex items-center justify-center py-12 gap-3">
                    <span className="w-5 h-5 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
                    <p className="text-sm text-slate-400">Loading chat history...</p>
                  </div>
                ) : chatHistory.length === 0 ? (
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
