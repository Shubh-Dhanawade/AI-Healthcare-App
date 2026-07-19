'use client';

import { useState, useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { documentsApi } from '@/lib/apiHelpers';
import toast from 'react-hot-toast';
import { useRouter } from 'next/navigation';
import {
  Upload, FileText, Image, X, CheckCircle, AlertCircle,
  CloudUpload, GripVertical,
} from 'lucide-react';

const MAX_SIZE_MB = 50;
const ACCEPTED_TYPES = {
  'application/pdf': ['.pdf'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png': ['.png'],
  'image/tiff': ['.tiff'],
  'image/webp': ['.webp'],
};

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

interface ImageItem { id: string; file: File; previewUrl: string; }

export default function UploadPage() {
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [images, setImages] = useState<ImageItem[]>([]);
  const [progress, setProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState('');
  const [draggingId, setDraggingId] = useState<string | null>(null);
  const dragOverId = useRef<string | null>(null);
  const router = useRouter();

  // Derive mode from what's been dropped
  const mode: 'idle' | 'pdf' | 'images' = pdfFile ? 'pdf' : images.length > 0 ? 'images' : 'idle';

  const onDrop = useCallback((accepted: File[], rejected: any[]) => {
    setError(''); setDone(false); setProgress(0);
    if (rejected.length > 0) {
      setError(rejected[0]?.errors?.[0]?.message || 'Some files were rejected.');
    }
    if (accepted.length === 0) return;

    // Separate PDFs and images
    const pdfs = accepted.filter((f) => f.type === 'application/pdf');
    const imgs = accepted.filter((f) => f.type !== 'application/pdf');

    if (pdfs.length > 0) {
      // PDF takes priority; only one allowed
      setPdfFile(pdfs[0]);
      setImages([]);
    } else if (imgs.length > 0) {
      // Image(s) dropped — add to existing image list
      setPdfFile(null);
      const newItems: ImageItem[] = imgs.map((f) => ({
        id: Math.random().toString(36).slice(2),
        file: f,
        previewUrl: URL.createObjectURL(f),
      }));
      setImages((prev) => [...prev, ...newItems]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxSize: MAX_SIZE_MB * 1024 * 1024,
    multiple: true,
  });

  const removeImage = (id: string) => {
    setImages((prev) => {
      const item = prev.find((i) => i.id === id);
      if (item) URL.revokeObjectURL(item.previewUrl);
      return prev.filter((i) => i.id !== id);
    });
  };

  const reset = () => {
    images.forEach((i) => URL.revokeObjectURL(i.previewUrl));
    setPdfFile(null); setImages([]); setProgress(0); setError(''); setDone(false);
  };

  // Drag-to-reorder
  const handleDragStart = (id: string) => setDraggingId(id);
  const handleDragEnter = (id: string) => { dragOverId.current = id; };
  const handleDragEnd = () => {
    if (draggingId && dragOverId.current && draggingId !== dragOverId.current) {
      setImages((prev) => {
        const from = prev.findIndex((i) => i.id === draggingId);
        const to = prev.findIndex((i) => i.id === dragOverId.current);
        if (from < 0 || to < 0) return prev;
        const next = [...prev];
        const [moved] = next.splice(from, 1);
        next.splice(to, 0, moved);
        return next;
      });
    }
    setDraggingId(null); dragOverId.current = null;
  };

  const handleUpload = async () => {
    if (mode === 'idle') return;
    setUploading(true); setError(''); setProgress(0);
    try {
      let doc: any;
      if (mode === 'pdf') {
        doc = await documentsApi.upload(pdfFile!, setProgress);
      } else {
        doc = await documentsApi.uploadImages(images.map((i) => i.file), setProgress);
        images.forEach((i) => URL.revokeObjectURL(i.previewUrl));
      }
      setDone(true);
      toast.success('Uploaded successfully!');
      setTimeout(() => router.push(`/documents/${doc.id}`), 1400);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Upload failed. Please try again.');
      toast.error('Upload failed');
    } finally { setUploading(false); }
  };

  const canUpload = mode !== 'idle' && !uploading && !done;
  const totalSize = mode === 'pdf' ? (pdfFile?.size ?? 0) : images.reduce((a, i) => a + i.file.size, 0);

  return (
    <div className="max-w-2xl mx-auto space-y-5 fade-in">
      <div>
        <h1 className="text-2xl font-bold gradient-text mb-1">Upload Document</h1>
        <p className="text-slate-400 text-sm">PDF or multiple images of a single insurance report.</p>
      </div>

      {/* ── Drop zone ── */}
      <div
        {...getRootProps()}
        className={`upload-zone p-10 text-center outline-none ${isDragActive ? 'upload-zone-active' : ''}`}
      >
        <input {...getInputProps()} id="file-dropzone" />
        <div className="flex flex-col items-center gap-4">
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center transition-all"
            style={{
              background: isDragActive ? 'rgba(59,130,246,0.2)' : 'rgba(59,130,246,0.08)',
              border: '1px solid rgba(59,130,246,0.3)',
            }}
          >
            <CloudUpload className="w-8 h-8" style={{ color: '#3b82f6' }} />
          </div>
          {isDragActive ? (
            <p className="text-blue-400 font-semibold">Drop here</p>
          ) : (
            <>
              <div>
                <p className="text-white font-semibold">Drag &amp; drop your document</p>
                <p className="text-slate-400 text-sm mt-1">or <span className="text-blue-400">browse files</span></p>
              </div>
              <p className="text-xs text-slate-500">PDF • JPG • PNG • TIFF • WEBP &nbsp;·&nbsp; Max {MAX_SIZE_MB}MB each</p>
            </>
          )}
        </div>
      </div>

      {/* ── Error ── */}
      {error && (
        <div className="flex items-center gap-3 p-4 rounded-xl"
          style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)' }}>
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
          <p className="text-red-300 text-sm">{error}</p>
        </div>
      )}

      {/* ── PDF preview ── */}
      {mode === 'pdf' && !done && (
        <div className="glass-card p-5">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0"
              style={{ background: 'rgba(239,68,68,0.15)' }}>
              <FileText className="w-6 h-6 text-red-400" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="font-medium truncate">{pdfFile!.name}</p>
              <p className="text-sm text-slate-400">{formatBytes(pdfFile!.size)}</p>
            </div>
            {!uploading && (
              <button onClick={reset} className="text-slate-500 hover:text-slate-300">
                <X className="w-5 h-5" />
              </button>
            )}
          </div>
          {uploading && <ProgressBar progress={progress} />}
        </div>
      )}

      {/* ── Image thumbnails ── */}
      {mode === 'images' && !done && (
        <div className="glass-card p-4 space-y-4">
          {/* Header row */}
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-slate-300 flex items-center gap-2">
              <Image className="w-4 h-4 text-blue-400" />
              {images.length} image{images.length !== 1 ? 's' : ''}
              <span className="text-slate-500 font-normal">· {formatBytes(totalSize)}</span>
            </p>
            {!uploading && (
              <button onClick={reset} className="text-xs text-slate-500 hover:text-red-400 transition-colors">
                Clear all
              </button>
            )}
          </div>

          {/* Thumbnail grid */}
          <div className="grid grid-cols-4 sm:grid-cols-5 gap-2.5">
            {images.map((item, idx) => (
              <div
                key={item.id}
                draggable={!uploading}
                onDragStart={() => handleDragStart(item.id)}
                onDragEnter={() => handleDragEnter(item.id)}
                onDragEnd={handleDragEnd}
                onDragOver={(e) => e.preventDefault()}
                className={`relative group rounded-xl overflow-hidden border transition-all select-none ${
                  uploading ? 'cursor-default' : 'cursor-grab active:cursor-grabbing'
                } ${
                  draggingId === item.id
                    ? 'opacity-40 scale-95 border-blue-500/60'
                    : 'border-white/10 hover:border-blue-400/50'
                }`}
                style={{ aspectRatio: '3/4', background: '#0b0f1a' }}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={item.previewUrl} alt={`Page ${idx + 1}`} className="w-full h-full object-cover" />

                {/* Page badge */}
                <div className="absolute top-1 left-1 bg-black/70 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-md">
                  {idx + 1}
                </div>

                {/* Drag handle */}
                {!uploading && (
                  <div className="absolute top-1 right-5 opacity-0 group-hover:opacity-100 transition-opacity">
                    <GripVertical className="w-3 h-3 text-white/60" />
                  </div>
                )}

                {/* Remove */}
                {!uploading && (
                  <button
                    onClick={() => removeImage(item.id)}
                    className="absolute top-0.5 right-0.5 w-5 h-5 rounded-full bg-black/60 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-500/80"
                  >
                    <X className="w-3 h-3 text-white" />
                  </button>
                )}

                {/* Name on hover */}
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent px-1 py-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                  <p className="text-[9px] text-white/70 truncate">{item.file.name}</p>
                </div>
              </div>
            ))}

            {/* Add more drop target */}
            {!uploading && (
              <div
                {...getRootProps()}
                className="rounded-xl border border-dashed border-white/20 flex items-center justify-center cursor-pointer hover:border-blue-400/50 transition-colors"
                style={{ aspectRatio: '3/4' }}
              >
                <input {...getInputProps()} />
                <span className="text-2xl text-slate-600 hover:text-slate-400 transition-colors">+</span>
              </div>
            )}
          </div>

          {uploading && <ProgressBar progress={progress} />}
        </div>
      )}

      {/* ── Success ── */}
      {done && (
        <div className="flex items-center gap-3 p-4 rounded-xl"
          style={{ background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)' }}>
          <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0" />
          <p className="text-emerald-300 text-sm">Upload successful! Redirecting...</p>
        </div>
      )}

      {/* ── Actions ── */}
      <div className="flex gap-3">
        <button
          id="upload-btn"
          onClick={handleUpload}
          disabled={!canUpload}
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
            <><Upload className="w-4 h-4" /> Upload{mode === 'images' && images.length > 1 ? ` (${images.length} pages)` : ''}</>
          )}
        </button>
        {mode !== 'idle' && !uploading && !done && (
          <button onClick={reset} className="btn-secondary">Cancel</button>
        )}
      </div>
    </div>
  );
}

function ProgressBar({ progress }: { progress: number }) {
  return (
    <div className="mt-3">
      <div className="flex justify-between text-xs text-slate-400 mb-1.5">
        <span>Uploading...</span><span>{progress}%</span>
      </div>
      <div className="progress-bar"><div className="progress-bar-fill" style={{ width: `${progress}%` }} /></div>
    </div>
  );
}
