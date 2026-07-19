'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { documentsApi, aiApi, exportApi, remindersApi } from '@/lib/apiHelpers';
import { DocumentDetail, RiskAnalysis } from '@/types';
import { useParams, useRouter } from 'next/navigation';
import toast from 'react-hot-toast';
import {
  ArrowLeft, Brain, Shield, FileText, Search, RefreshCw,
  AlertTriangle, Info, ChevronDown, ChevronUp,
  Send, MessageSquare, List, Loader2, Download, Mail,
  Volume2, VolumeX
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
  const [activeTab, setActiveTab] = useState<'summary' | 'fields' | 'risks' | 'checklist' | 'query'>('summary');
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

  // Claims Checklist states
  const [treatmentTypeInput, setTreatmentTypeInput] = useState('');
  const [checklistData, setChecklistData] = useState<any>(null);
  const [isGeneratingChecklist, setIsGeneratingChecklist] = useState(false);

  // Reminders / Alerts states
  const [renewalDate, setRenewalDate] = useState('');
  const [premiumDueDate, setPremiumDueDate] = useState('');
  const [premiumAmount, setPremiumAmount] = useState('');
  const [isSavingReminders, setIsSavingReminders] = useState(false);

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
            filename: `HealthPolicyLens_Report_${doc.original_filename.replace(/\.[^/.]+$/, "") || docId}.pdf`,
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

    let text = `🏥 *HealthPolicyLens Policy Audit Report*\n`;
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

    text += `Generated by HealthPolicyLens.`;
    const whatsappUrl = `https://api.whatsapp.com/send?text=${encodeURIComponent(text)}`;
    window.open(whatsappUrl, '_blank');
  };

  const handleSendEmail = async () => {
    if (!emailInput.trim()) return;
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(emailInput.trim())) {
      toast.error('Please enter a valid email address');
      return;
    }
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
    extracted_fields?: any[];
    risk_analyses?: any[];
    checklistData?: any;
  }>>({});
  const [isTranslating, setIsTranslating] = useState(false);
  const [speakingTab, setSpeakingTab] = useState<'summary' | 'fields' | 'risks' | 'checklist' | null>(null);

  const safeTranslate = async (text: string | undefined, lang: string) => {
    if (!text || !text.trim()) return { translated_text: '' };
    try {
      return await aiApi.translate(text, lang);
    } catch (err) {
      console.error('Translation error for text:', text, err);
      return { translated_text: text };
    }
  };

  const handleLanguageChange = async (lang: string) => {
    setSelectedLanguage(lang);
    if (lang === 'English' || !doc) return;
    if (translations[lang]) return;
    setIsTranslating(true);
    const toastId = toast.loading(`Translating page content to ${lang}...`);
    try {
      const summary = doc.summary;
      const summaryPromises = summary ? [
        safeTranslate(summary.summary_text, lang),
        safeTranslate(summary.coverage_summary, lang),
        safeTranslate(summary.exclusions_summary, lang),
        safeTranslate(summary.waiting_period_summary, lang),
        safeTranslate(summary.premium_summary, lang),
      ] : Array(5).fill(Promise.resolve({ translated_text: '' }));

      const fieldsPromises = (doc.extracted_fields || []).flatMap(f => [
        safeTranslate(f.field_name, lang),
        safeTranslate(f.field_value, lang),
      ]);

      const risksPromises = (doc.risk_analyses || []).flatMap(r => [
        safeTranslate(r.clause_text, lang),
        safeTranslate(r.explanation, lang),
        safeTranslate(r.recommendation, lang),
      ]);

      const checklistPromises = checklistData ? [
        safeTranslate(checklistData.estimated_approval_days, lang),
        ...checklistData.checklist.flatMap((item: any) => [
          safeTranslate(item.document_name, lang),
          safeTranslate(item.importance, lang),
          safeTranslate(item.description, lang)
        ]),
        ...checklistData.claim_steps.map((step: string) => safeTranslate(step, lang))
      ] : [];

      const results = await Promise.all([
        ...summaryPromises,
        ...fieldsPromises,
        ...risksPromises,
        ...checklistPromises
      ]);

      let ptr = 0;
      const tText = results[ptr++];
      const tCoverage = results[ptr++];
      const tExclusions = results[ptr++];
      const tWaiting = results[ptr++];
      const tPremium = results[ptr++];

      const translatedFields = (doc.extracted_fields || []).map(f => {
        const name = f.field_name ? results[ptr++].translated_text : f.field_name;
        const val = f.field_value ? results[ptr++].translated_text : f.field_value;
        return {
          ...f,
          field_name: name || f.field_name,
          field_value: val || f.field_value
        };
      });

      const translatedRisks = (doc.risk_analyses || []).map(r => {
        const clause = r.clause_text ? results[ptr++].translated_text : r.clause_text;
        const expl = r.explanation ? results[ptr++].translated_text : r.explanation;
        const rec = r.recommendation ? results[ptr++].translated_text : r.recommendation;
        return {
          ...r,
          clause_text: clause || r.clause_text,
          explanation: expl || r.explanation,
          recommendation: rec || r.recommendation
        };
      });

      let translatedChecklist = null;
      if (checklistData) {
        const estTimeline = results[ptr++].translated_text;
        const items = [];
        for (let i = 0; i < checklistData.checklist.length; i++) {
          const docName = results[ptr++].translated_text;
          const imp = results[ptr++].translated_text;
          const desc = results[ptr++].translated_text;
          items.push({ document_name: docName, importance: imp, description: desc });
        }
        const steps = [];
        for (let i = 0; i < checklistData.claim_steps.length; i++) {
          steps.push(results[ptr++].translated_text);
        }
        translatedChecklist = {
          checklist: items,
          claim_steps: steps,
          estimated_approval_days: estTimeline,
        };
      }

      setTranslations(prev => ({
        ...prev,
        [lang]: {
          summary_text: tText.translated_text,
          coverage_summary: tCoverage.translated_text || undefined,
          exclusions_summary: tExclusions.translated_text || undefined,
          waiting_period_summary: tWaiting.translated_text || undefined,
          premium_summary: tPremium.translated_text || undefined,
          extracted_fields: translatedFields,
          risk_analyses: translatedRisks,
          checklistData: translatedChecklist
        }
      }));
      toast.success(`Translated page content to ${lang}!`, { id: toastId });
    } catch (error) {
      console.error(error);
      toast.error(`Failed to translate. Please check if Ollama is running.`, { id: toastId });
      setSelectedLanguage('English');
    } finally {
      setIsTranslating(false);
    }
  };

  const speakText = (text: string, lang: string, onStop?: () => void) => {
    if (typeof window === 'undefined' || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    if (!text) return;
    const utterance = new SpeechSynthesisUtterance(text);
    let langCode = 'en-US';
    if (lang === 'Hindi') langCode = 'hi-IN';
    else if (lang === 'Marathi') langCode = 'mr-IN';
    utterance.lang = langCode;
    const voices = window.speechSynthesis.getVoices();
    const matchingVoice = voices.find(v => v.lang.startsWith(langCode) || v.lang.includes(langCode.replace('-', '_')));
    if (matchingVoice) utterance.voice = matchingVoice;
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    if (onStop) { utterance.onend = onStop; utterance.onerror = onStop; }
    window.speechSynthesis.speak(utterance);
  };

  const getSummarySpeakText = (displayedSummary: any, lang: string) => {
    if (!displayedSummary) return '';
    let text = '';
    if (lang === 'Hindi') {
      text += `दस्तावेज़ का सारांश: ${displayedSummary.summary_text || ''}\n\n`;
      if (displayedSummary.coverage_summary) text += `कवरेज और लाभ: ${displayedSummary.coverage_summary}\n\n`;
      if (displayedSummary.exclusions_summary) text += `बहिष्करण और सीमाएं: ${displayedSummary.exclusions_summary}\n\n`;
      if (displayedSummary.waiting_period_summary) text += `प्रतीक्षा अवधि: ${displayedSummary.waiting_period_summary}\n\n`;
      if (displayedSummary.premium_summary) text += `प्रीमियम और शुल्क: ${displayedSummary.premium_summary}\n\n`;
    } else if (lang === 'Marathi') {
      text += `दस्तएवजाचा सारांश: ${displayedSummary.summary_text || ''}\n\n`;
      if (displayedSummary.coverage_summary) text += `कव्हरेज आणि फायदे: ${displayedSummary.coverage_summary}\n\n`;
      if (displayedSummary.exclusions_summary) text += `वगळलेले मुद्दे आणि मर्यादा: ${displayedSummary.exclusions_summary}\n\n`;
      if (displayedSummary.waiting_period_summary) text += `प्रतीक्षा कालावधी: ${displayedSummary.waiting_period_summary}\n\n`;
      if (displayedSummary.premium_summary) text += `प्रीमियम आणि शुल्क: ${displayedSummary.premium_summary}\n\n`;
    } else {
      text += `Document Summary: ${displayedSummary.summary_text || ''}\n\n`;
      if (displayedSummary.coverage_summary) text += `Coverage and Benefits: ${displayedSummary.coverage_summary}\n\n`;
      if (displayedSummary.exclusions_summary) text += `Exclusions and Limits: ${displayedSummary.exclusions_summary}\n\n`;
      if (displayedSummary.waiting_period_summary) text += `Waiting Periods: ${displayedSummary.waiting_period_summary}\n\n`;
      if (displayedSummary.premium_summary) text += `Premium and Charges: ${displayedSummary.premium_summary}\n\n`;
    }
    return text;
  };

  const getFieldsSpeakText = (fields: any[], lang: string) => {
    if (!fields || fields.length === 0) return '';
    let text = '';
    if (lang === 'Hindi') {
      text += `निकालने गए फ़ील्ड विवरण निम्नानुसार हैं:\n`;
      fields.forEach(f => { text += `${f.field_name}: ${f.field_value || 'लागू नहीं'}.\n`; });
    } else if (lang === 'Marathi') {
      text += `काढून घेतलेले फील्ड तपशील खालीलप्रमाणे आहेत:\n`;
      fields.forEach(f => { text += `${f.field_name}: ${f.field_value || 'लागू नाही'}.\n`; });
    } else {
      text += `Here are the extracted fields:\n`;
      fields.forEach(f => { text += `${f.field_name}: ${f.field_value || 'Not available'}.\n`; });
    }
    return text;
  };

  const getRisksSpeakText = (risks: any[], lang: string) => {
    if (!risks || risks.length === 0) return '';
    let text = '';
    if (lang === 'Hindi') {
      text += `जोखिम विश्लेषण विवरण निम्नानुसार हैं:\n`;
      risks.forEach((r, idx) => {
        text += `जोखिम ${idx + 1}: प्रकार ${r.risk_type.replace(/_/g, ' ')}, गंभीरता ${r.severity}.\n`;
        if (r.clause_text) text += `दस्तावेज़ खंड: ${r.clause_text}.\n`;
        if (r.explanation) text += `स्पष्टीकरण: ${r.explanation}.\n`;
        if (r.recommendation) text += `सिफारिश: ${r.recommendation}.\n`;
      });
    } else if (lang === 'Marathi') {
      text += `धोका विश्लेषण तपशील खालीलप्रमाणे आहेत:\n`;
      risks.forEach((r, idx) => {
        text += `धोका ${idx + 1}: प्रकार ${r.risk_type.replace(/_/g, ' ')}, तीव्रता ${r.severity}.\n`;
        if (r.clause_text) text += `दस्तएवज खंड: ${r.clause_text}.\n`;
        if (r.explanation) text += `स्पष्टीकरण: ${r.explanation}.\n`;
        if (r.recommendation) text += `शिफारस: ${r.recommendation}.\n`;
      });
    } else {
      text += `Here is the risk analysis:\n`;
      risks.forEach((r, idx) => {
        text += `Risk ${idx + 1}: Type ${r.risk_type.replace(/_/g, ' ')}, Severity ${r.severity}.\n`;
        if (r.clause_text) text += `Clause text: ${r.clause_text}.\n`;
        if (r.explanation) text += `Explanation: ${r.explanation}.\n`;
        if (r.recommendation) text += `Recommendation: ${r.recommendation}.\n`;
      });
    }
    return text;
  };

  const getChecklistSpeakText = (data: any, lang: string) => {
    if (!data) return '';
    let text = '';
    if (lang === 'Hindi') {
      text += `चिकित्सा उपचार के लिए दावे की चेकलिस्ट निम्नानुसार है:\n`;
      text += `अनुमानित दावा स्वीकृति समयरेखा: ${data.estimated_approval_days}.\n\n`;
      text += `आवश्यक दस्तावेज:\n`;
      (data.checklist || []).forEach((item: any) => {
        text += `दस्तावेज: ${item.document_name}. महत्व: ${item.importance}. निर्देश: ${item.description}.\n`;
      });
      text += `\nचरण-दर-चरण दावा प्रक्रिया दिशानिर्देश:\n`;
      (data.claim_steps || []).forEach((step: string, idx: number) => {
        text += `चरण ${idx + 1}: ${step}.\n`;
      });
    } else if (lang === 'Marathi') {
      text += `वैद्यकीय उपचारांसाठी दाव्याची चेकलिस्ट खालीलप्रमाणे आहे:\n`;
      text += `अंदाजित दावा मंजुरीची मुदत: ${data.estimated_approval_days}.\n\n`;
      text += `आवश्यक कागदपत्रे:\n`;
      (data.checklist || []).forEach((item: any) => {
        text += `कागदपत्र: ${item.document_name}. महत्त्व: ${item.importance}. सूचना: ${item.description}.\n`;
      });
      text += `\nचरण-दर-चरण दावा प्रक्रिया मार्गदर्शक तत्त्वे:\n`;
      (data.claim_steps || []).forEach((step: string, idx: number) => {
        text += `चरण ${idx + 1}: ${step}.\n`;
      });
    } else {
      text += `Here is your claim checklist for treatment.\n`;
      text += `Estimated claim approval timeline is ${data.estimated_approval_days}.\n\n`;
      text += `Required claims documents:\n`;
      (data.checklist || []).forEach((item: any) => {
        text += `Document: ${item.document_name}. Importance: ${item.importance}. Instructions: ${item.description}.\n`;
      });
      text += `\nStep-by-step claim guidelines:\n`;
      (data.claim_steps || []).forEach((step: string, idx: number) => {
        text += `Step ${idx + 1}: ${step}.\n`;
      });
    }
    return text;
  };

  const handlePlaySpeech = (tab: 'summary' | 'fields' | 'risks' | 'checklist') => {
    if (!doc) return;
    if (speakingTab === tab) {
      if (typeof window !== 'undefined' && window.speechSynthesis) window.speechSynthesis.cancel();
      setSpeakingTab(null);
      return;
    }
    let textToSpeak = '';
    if (tab === 'summary') {
      textToSpeak = getSummarySpeakText(displayedSummary, selectedLanguage);
    } else if (tab === 'fields') {
      const currentFields = selectedLanguage === 'English'
        ? doc.extracted_fields
        : (translations[selectedLanguage]?.extracted_fields || doc.extracted_fields);
      textToSpeak = getFieldsSpeakText(currentFields, selectedLanguage);
    } else if (tab === 'risks') {
      const currentRisks = selectedLanguage === 'English'
        ? doc.risk_analyses
        : (translations[selectedLanguage]?.risk_analyses || doc.risk_analyses);
      textToSpeak = getRisksSpeakText(currentRisks, selectedLanguage);
    } else if (tab === 'checklist') {
      const currentChecklist = selectedLanguage === 'English'
        ? checklistData
        : (translations[selectedLanguage]?.checklistData || checklistData);
      textToSpeak = getChecklistSpeakText(currentChecklist, selectedLanguage);
    }
    if (!textToSpeak) return;
    setSpeakingTab(tab);
    speakText(textToSpeak, selectedLanguage, () => { setSpeakingTab(null); });
  };

  useEffect(() => {
    return () => {
      if (typeof window !== 'undefined' && window.speechSynthesis) window.speechSynthesis.cancel();
    };
  }, []);



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


  // Client side date parsing helpers
  const parseClientDate = (str: string): string | null => {
    if (!str) return null;
    const m1 = str.match(/(\d{1,2})[\-\/](\d{1,2})[\-\/](\d{4})/);
    if (m1) {
      let d = parseInt(m1[1]);
      let m = parseInt(m1[2]);
      let y = parseInt(m1[3]);
      if (d > 1900) { y = d; d = parseInt(m1[3]); }
      try {
        return `${y}-${String(m).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
      } catch(e) {}
    }
    const m2 = str.match(/(\d{4})[\-\/](\d{1,2})[\-\/](\d{1,2})/);
    if (m2) {
      let y = parseInt(m2[1]);
      let m = parseInt(m2[2]);
      let d = parseInt(m2[3]);
      try {
        return `${y}-${String(m).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
      } catch(e) {}
    }
    const months: Record<string, number> = {
      jan: 1, feb: 2, mar: 3, apr: 4, may: 5, jun: 6,
      jul: 7, aug: 8, sep: 9, oct: 10, nov: 11, dec: 12
    };
    const m3 = str.match(/(\d{1,2})\s*[\-\s\/]?\s*([A-Za-z]{3})[a-z]*\s*[\-\s\/]?\s*(\d{4})/i);
    if (m3) {
      const d = parseInt(m3[1]);
      const m = months[m3[2].toLowerCase()];
      const y = parseInt(m3[3]);
      if (m !== undefined) {
        try {
          return `${y}-${String(m).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
        } catch(e) {}
      }
    }
    return null;
  };

  const parseClientDateFromPeriod = (str: string): string | null => {
    if (!str) return null;
    const parts = str.toLowerCase().split(/\bto\b/);
    if (parts.length > 1) {
      return parseClientDate(parts[parts.length - 1].trim());
    }
    return parseClientDate(str);
  };

  // Reminders states auto loader
  useEffect(() => {
    if (doc) {
      let rDate = '';
      let pDate = '';
      let pAmount = '';

      if (doc.renewal_date) {
        rDate = new Date(doc.renewal_date).toISOString().split('T')[0];
      } else {
        const renewalField = doc.extracted_fields?.find(f => {
          const name = f.field_name.toLowerCase();
          return name.includes('renewal') || name.includes('expiry') || name.includes('valid to') || name.includes('end date') || name.includes('to date');
        });
        if (renewalField?.field_value) {
          rDate = parseClientDate(renewalField.field_value) || '';
        } else {
          const periodField = doc.extracted_fields?.find(f => f.field_name.toLowerCase().includes('period') || f.field_name.toLowerCase().includes('term'));
          if (periodField?.field_value) {
            rDate = parseClientDateFromPeriod(periodField.field_value) || '';
          }
        }
      }

      if (doc.premium_due_date) {
        pDate = new Date(doc.premium_due_date).toISOString().split('T')[0];
      } else {
        const dueField = doc.extracted_fields?.find(f => f.field_name.toLowerCase().includes('due date') || f.field_name.toLowerCase().includes('payment due'));
        if (dueField?.field_value) {
          pDate = parseClientDate(dueField.field_value) || '';
        } else if (rDate) {
          pDate = rDate;
        }
      }

      const premField = doc.extracted_fields?.find(f => {
        const name = f.field_name.toLowerCase();
        return name.includes('premium') && !name.includes('due') && !name.includes('frequency');
      });
      if (premField?.field_value) {
        pAmount = premField.field_value;
      }

      if (rDate) setRenewalDate(rDate);
      if (pDate) setPremiumDueDate(pDate);
      if (pAmount) setPremiumAmount(pAmount);
    }
  }, [doc]);

  const handleSaveReminders = async () => {
    setIsSavingReminders(true);
    const toastId = toast.loading('Scheduling policy alerts...');
    try {
      await remindersApi.schedule({
        document_id: docId,
        renewal_date: renewalDate ? new Date(renewalDate).toISOString() : undefined,
        premium_due_date: premiumDueDate ? new Date(premiumDueDate).toISOString() : undefined,
        premium_amount: premiumAmount ? parseFloat(premiumAmount.replace(/[^\d.]/g, '')) : undefined,
      });
      toast.success('Smart renewal and premium alerts scheduled successfully!', { id: toastId });
      queryClient.invalidateQueries({ queryKey: ['reminders'] });
      queryClient.invalidateQueries({ queryKey: ['document', docId] });
    } catch (error: any) {
      console.error(error);
      toast.error(error.response?.data?.detail || 'Failed to schedule alerts', { id: toastId });
    } finally {
      setIsSavingReminders(false);
    }
  };

  const handleGenerateChecklist = async () => {
    if (!treatmentTypeInput) return;
    setIsGeneratingChecklist(true);
    const toastId = toast.loading(`Generating claims checklist for ${treatmentTypeInput}...`);
    try {
      const data = await aiApi.claimsChecklist(docId, treatmentTypeInput);
      setChecklistData(data);
      toast.success('Checklist generated successfully!', { id: toastId });
    } catch (error: any) {
      console.error(error);
      toast.error(error.response?.data?.detail || 'Failed to generate claims checklist. Check if Ollama is running.', { id: toastId });
    } finally {
      setIsGeneratingChecklist(false);
    }
  };


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

  const displayedChecklist = selectedLanguage === 'English'
    ? checklistData
    : (translations[selectedLanguage]?.checklistData || checklistData);

  const tabs = [
    { id: 'summary', label: 'AI Summary', icon: <Brain className="w-4 h-4" />, count: doc.summary ? 1 : 0 },
    { id: 'fields', label: 'Extracted Fields', icon: <Search className="w-4 h-4" />, count: doc.extracted_fields.length },
    { id: 'risks', label: 'Risk Analysis', icon: <Shield className="w-4 h-4" />, count: doc.risk_analyses.length },
    { id: 'checklist', label: 'Claims Checklist', icon: <List className="w-4 h-4" />, count: checklistData ? 1 : 0 },
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

      {/* Policy Reminders & Alerts Scheduler */}
      {canRunAI && (
        <div className="glass-card p-5 border border-white/5 space-y-4">
          <h2 className="font-semibold text-sm text-slate-300 flex items-center gap-2">
            <span className="text-blue-400">⏰</span> Smart Renewal & Premium Alerts
          </h2>
          <p className="text-xs text-slate-400">
            Set dates for your policy renewal and premium payments. The system will automatically calculate and trigger renewal alerts 7 days prior, and premium reminders 5 days prior.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 items-end">
            <div>
              <label className="block text-[10px] text-slate-500 font-bold uppercase tracking-wider mb-1.5">Renewal Date</label>
              <input
                type="date"
                value={renewalDate}
                onChange={(e) => setRenewalDate(e.target.value)}
                className="form-input text-xs w-full bg-slate-950/40"
              />
            </div>
            <div>
              <label className="block text-[10px] text-slate-500 font-bold uppercase tracking-wider mb-1.5">Premium Due Date</label>
              <input
                type="date"
                value={premiumDueDate}
                onChange={(e) => setPremiumDueDate(e.target.value)}
                className="form-input text-xs w-full bg-slate-950/40"
              />
            </div>
            <div>
              <label className="block text-[10px] text-slate-500 font-bold uppercase tracking-wider mb-1.5">Premium Amount</label>
              <input
                type="text"
                placeholder="e.g. ₹15,596"
                value={premiumAmount}
                onChange={(e) => setPremiumAmount(e.target.value)}
                className="form-input text-xs w-full bg-slate-950/40"
              />
            </div>
            <div className="sm:col-span-3 flex justify-end">
              <button
                onClick={handleSaveReminders}
                disabled={isSavingReminders}
                className="btn-primary text-xs py-2 px-4 flex items-center gap-1.5"
              >
                {isSavingReminders ? (
                  <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                ) : (
                  'Save Alert Dates'
                )}
              </button>
            </div>
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
                      {/* Text to Speech Button */}
                      <button
                        onClick={() => handlePlaySpeech('summary')}
                        className={`btn-secondary text-xs py-2 px-3 flex items-center gap-1.5 hover:scale-[1.02] active:scale-[0.98] transition-all cursor-pointer border ${
                          speakingTab === 'summary' 
                            ? 'border-red-500/30 text-red-400 bg-red-500/5' 
                            : 'border-blue-500/30 text-blue-400'
                        }`}
                        title={speakingTab === 'summary' ? 'Stop read aloud' : 'Read aloud summary'}
                      >
                        {speakingTab === 'summary' ? (
                          <>
                            <VolumeX className="w-3.5 h-3.5 animate-pulse" />
                            <span>Stop Speech</span>
                          </>
                        ) : (
                          <>
                            <Volume2 className="w-3.5 h-3.5" />
                            <span>Listen</span>
                          </>
                        )}
                      </button>

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

                  <div className="bg-slate-950/40 border border-white/5 rounded-xl p-5 relative overflow-hidden">
                    {isTranslating && (
                      <div className="absolute inset-0 bg-[#0a0f1e]/60 backdrop-blur-sm flex items-center justify-center z-10 transition-all">
                        <div className="flex items-center gap-2">
                          <span className="w-5 h-5 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
                          <span className="text-xs text-slate-400">Translating summary to {selectedLanguage}...</span>
                        </div>
                      </div>
                    )}
                    <div className="space-y-3 text-slate-200 text-sm leading-relaxed">
                      {(displayedSummary?.summary_text || '').split(/\n{2,}|\n(?=\S)/).filter(Boolean).map((para: string, i: number) => (
                        <p key={i} className="leading-7 text-slate-200">{para.trim()}</p>
                      ))}
                    </div>
                  </div>

                  {/* Detailed Policy Breakdowns */}
                  <div className="mt-6 border-t border-white/5 pt-6 space-y-4">
                    <h3 className="text-xs font-bold text-slate-400 tracking-wider uppercase">Policy Details & Exclusions</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {[
                        { label: '✅ Coverage & Benefits', value: displayedSummary.coverage_summary, color: '#10b981', border: 'border-emerald-500/20', bg: 'bg-emerald-500/5', dotColor: 'bg-emerald-400' },
                        { label: '❌ Exclusions & Limits', value: displayedSummary.exclusions_summary, color: '#ef4444', border: 'border-red-500/20', bg: 'bg-red-500/5', dotColor: 'bg-red-400' },
                        { label: '⏰ Waiting Periods', value: displayedSummary.waiting_period_summary, color: '#f59e0b', border: 'border-amber-500/20', bg: 'bg-amber-500/5', dotColor: 'bg-amber-400' },
                        { label: '💰 Premium & Charges', value: displayedSummary.premium_summary, color: '#3b82f6', border: 'border-blue-500/20', bg: 'bg-blue-500/5', dotColor: 'bg-blue-400' },
                      ].filter(s => s.value).map((section) => {
                        const bullets = (section.value || '')
                          .split(/\n/)
                          .map((line: string) => line.replace(/^[•\-*\s\u2022\uf0b7]+/, '').trim())
                          .filter((line: string) => line.length > 0);
                        return (
                          <div key={section.label} className={`p-4 rounded-xl border ${section.border} ${section.bg}`}>
                            <h4 className="font-bold text-xs uppercase tracking-wider mb-3" style={{ color: section.color }}>{section.label}</h4>
                            <ul className="space-y-2">
                              {bullets.map((bullet: string, idx: number) => (
                                <li key={idx} className="flex items-start gap-2 text-xs text-slate-200 leading-relaxed">
                                  <span className={`mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0 ${section.dotColor}`} />
                                  <span>{bullet}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        );
                      })}
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
            <div className="space-y-4 animate-in fade-in duration-200">
              {doc.extracted_fields.length > 0 && (
                <div className="flex justify-between items-center bg-slate-900/40 border border-white/5 rounded-xl px-5 py-3 gap-4">
                  <p className="text-xs text-slate-400 font-medium">
                    {selectedLanguage === 'English'
                      ? 'Extracted key fields from your policy document.'
                      : `पॉलिसी दस्तावेज़ से निकाले गए प्रमुख फ़ील्ड (${selectedLanguage})`}
                  </p>
                  <button
                    onClick={() => handlePlaySpeech('fields')}
                    className={`btn-secondary text-xs py-2 px-3.5 flex items-center gap-1.5 hover:scale-[1.02] active:scale-[0.98] transition-all cursor-pointer border flex-shrink-0 ${
                      speakingTab === 'fields'
                        ? 'border-red-500/30 text-red-400 bg-red-500/5'
                        : 'border-blue-500/30 text-blue-400'
                    }`}
                    title={speakingTab === 'fields' ? 'Stop read aloud' : 'Read aloud fields'}
                  >
                    {speakingTab === 'fields' ? (
                      <>
                        <VolumeX className="w-3.5 h-3.5 animate-pulse" />
                        <span>Stop Speech</span>
                      </>
                    ) : (
                      <>
                        <Volume2 className="w-3.5 h-3.5" />
                        <span>Listen to Fields</span>
                      </>
                    )}
                  </button>
                </div>
              )}

              <div className="glass-card overflow-hidden relative">
                {isTranslating && (
                  <div className="absolute inset-0 bg-[#0a0f1e]/60 backdrop-blur-sm flex items-center justify-center z-10 transition-all">
                    <div className="flex items-center gap-2">
                      <span className="w-5 h-5 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
                      <span className="text-xs text-slate-400">Translating fields to {selectedLanguage}...</span>
                    </div>
                  </div>
                )}
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
                      {(selectedLanguage === 'English' ? doc.extracted_fields : (translations[selectedLanguage]?.extracted_fields || doc.extracted_fields)).map((field) => (
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
            </div>
          )}

          {/* Risks Tab */}
          {activeTab === 'risks' && (
            <div className="space-y-3 animate-in fade-in duration-200 relative">
              {isTranslating && (
                <div className="absolute inset-0 bg-[#0a0f1e]/60 backdrop-blur-sm flex items-center justify-center z-10 transition-all rounded-xl">
                  <div className="flex items-center gap-2">
                    <span className="w-5 h-5 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
                    <span className="text-xs text-slate-400">Translating risks to {selectedLanguage}...</span>
                  </div>
                </div>
              )}
              {doc.risk_analyses.length === 0 ? (
                <div className="text-center py-12 glass-card">
                  <Shield className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                  <p className="text-slate-400">No risk analysis yet. Click "Risk Analysis".</p>
                </div>
              ) : (
                <>
                  <div className="flex justify-between items-center bg-slate-900/40 border border-white/5 rounded-xl px-5 py-3 gap-4">
                    <p className="text-xs text-slate-400 font-medium">
                      {selectedLanguage === 'English'
                        ? 'Analyzed risk terms and potential high-exposure clauses.'
                        : `जोखिम शर्तों और संभावित उच्च-जोखिम वाले खंडों का विश्लेषण (${selectedLanguage})`}
                    </p>
                    <button
                      onClick={() => handlePlaySpeech('risks')}
                      className={`btn-secondary text-xs py-2 px-3.5 flex items-center gap-1.5 hover:scale-[1.02] active:scale-[0.98] transition-all cursor-pointer border flex-shrink-0 ${
                        speakingTab === 'risks'
                          ? 'border-red-500/30 text-red-400 bg-red-500/5'
                          : 'border-blue-500/30 text-blue-400'
                      }`}
                      title={speakingTab === 'risks' ? 'Stop read aloud' : 'Read aloud risks'}
                    >
                      {speakingTab === 'risks' ? (
                        <>
                          <VolumeX className="w-3.5 h-3.5 animate-pulse" />
                          <span>Stop Speech</span>
                        </>
                      ) : (
                        <>
                          <Volume2 className="w-3.5 h-3.5" />
                          <span>Listen to Risks</span>
                        </>
                      )}
                    </button>
                  </div>

                  {(selectedLanguage === 'English' ? doc.risk_analyses : (translations[selectedLanguage]?.risk_analyses || doc.risk_analyses))
                    .sort((a, b) => {
                      const severityWeight = { high: 0, medium: 1, low: 2 };
                      const aVal = severityWeight[a.severity as keyof typeof severityWeight] ?? 3;
                      const bVal = severityWeight[b.severity as keyof typeof severityWeight] ?? 3;
                      return aVal - bVal;
                    })
                    .map((risk) => (
                      <RiskCard key={risk.id} risk={risk} />
                    ))}
                </>
              )}
            </div>
          )}

          {/* Claims Checklist Tab */}
          {activeTab === 'checklist' && (
            <div className="glass-card p-6 border border-white/5 space-y-6 animate-in fade-in duration-200 relative">
              {isTranslating && (
                <div className="absolute inset-0 bg-[#0a0f1e]/60 backdrop-blur-sm flex items-center justify-center z-10 transition-all rounded-xl">
                  <div className="flex items-center gap-2">
                    <span className="w-5 h-5 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
                    <span className="text-xs text-slate-400">Translating claims checklist to {selectedLanguage}...</span>
                  </div>
                </div>
              )}
              
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-slate-700/40 pb-4">
                <div>
                  <h3 className="font-bold text-sm text-white flex items-center gap-2">
                    <List className="w-4 h-4 text-blue-400" /> Dynamic Claims Documentation Checklist
                  </h3>
                  <p className="text-xs text-slate-400 mt-1">
                    Generate the exact list of documents and steps required to submit a claim for a specific treatment.
                  </p>
                </div>
                
                {/* Speech Button for Checklist (Narrator) */}
                {checklistData && (
                  <button
                    onClick={() => handlePlaySpeech('checklist')}
                    className={`btn-secondary text-xs py-2 px-3.5 flex items-center gap-1.5 hover:scale-[1.02] active:scale-[0.98] transition-all cursor-pointer border flex-shrink-0 ${
                      speakingTab === 'checklist'
                        ? 'border-red-500/30 text-red-400 bg-red-500/5'
                        : 'border-blue-500/30 text-blue-400'
                    }`}
                    title={speakingTab === 'checklist' ? 'Stop read aloud' : 'Read aloud claims checklist'}
                  >
                    {speakingTab === 'checklist' ? (
                      <>
                        <VolumeX className="w-3.5 h-3.5 animate-pulse" />
                        <span>Stop Speech</span>
                      </>
                    ) : (
                      <>
                        <Volume2 className="w-3.5 h-3.5" />
                        <span>Listen to Checklist</span>
                      </>
                    )}
                  </button>
                )}
              </div>

              {/* Input Form */}
              <div className="flex flex-col sm:flex-row items-end gap-3 max-w-lg">
                <div className="flex-1 w-full">
                  <label className="block text-[10px] text-slate-500 font-bold uppercase tracking-wider mb-1.5">Treatment or Surgery Type</label>
                  <select
                    value={treatmentTypeInput}
                    onChange={(e) => setTreatmentTypeInput(e.target.value)}
                    className="form-input text-xs w-full bg-slate-950/40"
                  >
                    <option value="" disabled>Select a treatment...</option>
                    <option value="Cataract Surgery">Cataract Surgery (मोतीबिंदू शस्त्रक्रिया)</option>
                    <option value="Heart Bypass / CABG">Heart Bypass / CABG (बायपास शस्त्रक्रिया)</option>
                    <option value="Knee Replacement">Knee Replacement (गुडघा प्रत्यारोपण)</option>
                    <option value="Accidental Fracture Cover">Accidental Fracture Cover (अपघाती फ्रॅक्चर)</option>
                    <option value="Kidney Dialysis">Kidney Dialysis (डायलिसिस)</option>
                    <option value="Maternity Delivery">Maternity Delivery (प्रसूती)</option>
                  </select>
                </div>
                <button
                  onClick={handleGenerateChecklist}
                  disabled={isGeneratingChecklist || !treatmentTypeInput}
                  className="btn-primary text-xs py-2.5 px-4 w-full sm:w-auto justify-center flex items-center gap-1.5"
                >
                  {isGeneratingChecklist ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      <span>Generating...</span>
                    </>
                  ) : (
                    'Generate Checklist'
                  )}
                </button>
              </div>

              {/* Checklist Output */}
              {displayedChecklist ? (
                <div className="space-y-6 pt-4 border-t border-slate-700/30 animate-in fade-in duration-200">
                  {/* Estimated timeline banner */}
                  <div className="p-4 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-between">
                    <p className="text-xs font-semibold text-blue-300">Estimated Claim Approval Timeline:</p>
                    <span className="text-xs px-2.5 py-1 rounded-full bg-blue-500/20 text-blue-200 border border-blue-500/30 font-bold">
                      {displayedChecklist.estimated_approval_days}
                    </span>
                  </div>

                  {/* Documents table */}
                  <div className="space-y-3">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">Required Claims Documents</h4>
                    <div className="border border-white/5 rounded-xl overflow-hidden">
                      <table className="w-full">
                        <thead>
                          <tr className="border-b border-slate-700/50 bg-slate-900/20">
                            <th className="text-left px-5 py-3 text-xs font-semibold text-slate-400 uppercase w-1/3">Document Required</th>
                            <th className="text-left px-5 py-3 text-xs font-semibold text-slate-400 uppercase w-1/6">Importance</th>
                            <th className="text-left px-5 py-3 text-xs font-semibold text-slate-400 uppercase w-1/2">Instructions</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-700/30">
                          {displayedChecklist.checklist.map((item: any, idx: number) => (
                            <tr key={idx} className="hover:bg-white/3">
                              <td className="px-5 py-3.5 text-sm font-medium text-slate-300">{item.document_name}</td>
                              <td className="px-5 py-3.5 text-xs">
                                <span className={`px-2 py-0.5 rounded-full font-bold border capitalize ${
                                  item.importance.toLowerCase().includes('mandatory') || item.importance.includes('अनिवार्य')
                                    ? 'bg-red-500/10 text-red-400 border-red-500/20'
                                    : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                                }`}>
                                  {item.importance}
                                </span>
                              </td>
                              <td className="px-5 py-3.5 text-xs text-slate-400 leading-relaxed">{item.description}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Claim process steps */}
                  <div className="space-y-3">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">Step-by-Step Claim Guidelines</h4>
                    <ol className="space-y-3">
                      {displayedChecklist.claim_steps.map((step: string, idx: number) => (
                        <li key={idx} className="flex gap-4 p-4 rounded-xl border border-white/5 bg-slate-950/20">
                          <span className="w-6 h-6 rounded-full bg-blue-500/10 border border-blue-500/30 flex items-center justify-center text-xs font-bold text-blue-300 flex-shrink-0">
                            {idx + 1}
                          </span>
                          <p className="text-xs text-slate-300 leading-relaxed pt-0.5">{step}</p>
                        </li>
                      ))}
                    </ol>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 bg-slate-950/10 border border-dashed border-slate-700/30 rounded-xl">
                  <List className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                  <p className="text-slate-400 text-sm">Select a treatment and generate a claims checklist.</p>
                </div>
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
