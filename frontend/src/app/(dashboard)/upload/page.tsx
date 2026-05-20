'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { documentsApi } from '@/lib/apiHelpers';
import toast from 'react-hot-toast';
import { useRouter } from 'next/navigation';
import { Upload, FileText, Image, X, CheckCircle, AlertCircle, CloudUpload } from 'lucide-react';

const MAX_SIZE_MB = 50;
const ACCEPTED_TYPES = {
  'application/pdf': ['.pdf'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png': ['.png'],
  'image/tiff': ['.tiff'],
  'image/webp': ['.webp'],
};

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState('');
  const router = useRouter();

  const onDrop = useCallback((accepted: File[], rejected: any[]) => {
    setError('');
    setDone(false);
    setProgress(0);
    if (rejected.length > 0) {
      const reason = rejected[0]?.errors?.[0]?.message || 'Invalid file';
      setError(reason);
      return;
    }
    if (accepted[0]) setFile(accepted[0]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxSize: MAX_SIZE_MB * 1024 * 1024,
    maxFiles: 1,
    multiple: false,
  });

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError('');
    setProgress(0);
    try {
      const doc = await documentsApi.upload(file, setProgress);
      setDone(true);
      toast.success('Document uploaded successfully!');
      setTimeout(() => router.push(`/documents/${doc.id}`), 1500);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Upload failed. Please try again.');
      toast.error('Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const isPdf = file?.type === 'application/pdf';

  return (
    <div className="max-w-2xl mx-auto space-y-6 fade-in">
      <div>
        <h1 className="text-2xl font-bold gradient-text mb-1">Upload Document</h1>
        <p className="text-slate-400">Upload your healthcare insurance document for AI analysis.</p>
      </div>

      {/* Upload Zone */}
      <div
        {...getRootProps()}
        className={`upload-zone p-12 text-center outline-none ${isDragActive ? 'upload-zone-active' : ''}`}
      >
        <input {...getInputProps()} id="file-dropzone" />
        <div className="flex flex-col items-center gap-4">
          <div className="w-20 h-20 rounded-2xl flex items-center justify-center"
            style={{ background: isDragActive ? 'rgba(59,130,246,0.2)' : 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.3)' }}>
            <CloudUpload className="w-9 h-9" style={{ color: '#3b82f6' }} />
          </div>
          {isDragActive ? (
            <p className="text-blue-400 font-semibold text-lg">Drop it here!</p>
          ) : (
            <>
              <div>
                <p className="text-white font-semibold text-lg">Drag & drop your document</p>
                <p className="text-slate-400 text-sm mt-1">or <span className="text-blue-400">browse files</span></p>
              </div>
              <div className="flex items-center gap-3 text-xs text-slate-500">
                <span className="flex items-center gap-1"><FileText className="w-3.5 h-3.5 text-red-400" /> PDF</span>
                <span>•</span>
                <span className="flex items-center gap-1"><Image className="w-3.5 h-3.5 text-blue-400" /> JPG, PNG, TIFF, WEBP</span>
                <span>•</span>
                <span>Max {MAX_SIZE_MB}MB</span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-3 p-4 rounded-xl"
          style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)' }}>
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
          <p className="text-red-300 text-sm">{error}</p>
        </div>
      )}

      {/* Selected File Preview */}
      {file && !done && (
        <div className="glass-card p-5">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0"
              style={{ background: isPdf ? 'rgba(239,68,68,0.15)' : 'rgba(59,130,246,0.15)' }}>
              {isPdf
                ? <FileText className="w-6 h-6 text-red-400" />
                : <Image className="w-6 h-6 text-blue-400" />
              }
            </div>
            <div className="flex-1 min-w-0">
              <p className="font-medium truncate">{file.name}</p>
              <p className="text-sm text-slate-400">{formatBytes(file.size)} • {file.type}</p>
            </div>
            {!uploading && (
              <button onClick={() => { setFile(null); setProgress(0); }} className="text-slate-500 hover:text-slate-300">
                <X className="w-5 h-5" />
              </button>
            )}
          </div>

          {/* Progress bar */}
          {uploading && (
            <div className="mt-4">
              <div className="flex justify-between text-xs text-slate-400 mb-1.5">
                <span>Uploading...</span>
                <span>{progress}%</span>
              </div>
              <div className="progress-bar">
                <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Success */}
      {done && (
        <div className="flex items-center gap-3 p-4 rounded-xl"
          style={{ background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)' }}>
          <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0" />
          <p className="text-emerald-300 text-sm">Upload successful! Redirecting to document view...</p>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-3">
        <button
          id="upload-btn"
          onClick={handleUpload}
          disabled={!file || uploading || done}
          className="btn-primary flex-1 justify-center py-3"
        >
          {uploading ? (
            <span className="flex items-center gap-2">
              <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Uploading {progress}%
            </span>
          ) : done ? (
            <><CheckCircle className="w-4 h-4" /> Uploaded!</>
          ) : (
            <><Upload className="w-4 h-4" /> Upload Document</>
          )}
        </button>
        {file && !uploading && !done && (
          <button onClick={() => { setFile(null); setProgress(0); setError(''); }} className="btn-secondary">
            Cancel
          </button>
        )}
      </div>

      {/* Instructions */}
      <div className="glass-card p-6 space-y-3">
        <h3 className="font-semibold text-sm text-slate-300">After uploading, you can:</h3>
        <ul className="space-y-2">
          {[
            { icon: '🔍', text: 'Text is automatically extracted from your document' },
            { icon: '🤖', text: 'Run AI summarization to get plain-language explanations' },
            { icon: '📋', text: 'Extract key fields like premiums, coverage, and deductibles' },
            { icon: '⚠️', text: 'Detect risky clauses and hidden conditions' },
          ].map((item, i) => (
            <li key={i} className="flex items-start gap-2 text-sm text-slate-400">
              <span>{item.icon}</span>
              <span>{item.text}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
