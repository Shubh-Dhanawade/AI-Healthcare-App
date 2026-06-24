'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { documentsApi } from '@/lib/apiHelpers';
import { Document } from '@/types';
import Link from 'next/link';
import { FileText, Upload, Trash2, Eye, Search, Scale } from 'lucide-react';
import { useState } from 'react';
import DocumentStatusBadge from '@/components/documents/DocumentStatusBadge';
import toast from 'react-hot-toast';

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DocumentsListPage() {
  const [search, setSearch] = useState('');
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const queryClient = useQueryClient();

  const { data: documents = [], isLoading } = useQuery<Document[]>({
    queryKey: ['documents'],
    queryFn: documentsApi.list,
    refetchInterval: 5000, // Poll for status updates
  });

  const deleteMutation = useMutation({
    mutationFn: documentsApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      toast.success('Document deleted');
    },
    onError: () => toast.error('Failed to delete document'),
  });

  const filtered = documents.filter((d) =>
    d.original_filename.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6 fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold gradient-text">Documents</h1>
          <p className="text-slate-400 text-sm">{documents.length} document{documents.length !== 1 ? 's' : ''}</p>
        </div>
        <Link href="/upload" className="btn-primary">
          <Upload className="w-4 h-4" /> Upload New
        </Link>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
        <input
          type="text"
          className="form-input pl-10 "
          placeholder="Search documents..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      {/* Inline Compare Bar — shown when documents are selected */}
      {selectedIds.length > 0 && (
        <div className="glass-card p-5 border border-blue-500/20 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-blue-500/15 text-blue-400 flex items-center justify-center border border-blue-500/30">
              <Scale className="w-5 h-5" />
            </div>
            <div>
              <p className="font-semibold text-white text-sm">Policy Comparison</p>
              <p className="text-slate-400 text-xs">
                Selected <span className="text-blue-400 font-bold">{selectedIds.length}</span> of{' '}
                <span className="text-slate-200">3</span> policies.
                {selectedIds.length < 2 && ' Select at least 2 to compare.'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3 w-full sm:w-auto">
            <button
              onClick={() => setSelectedIds([])}
              className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors w-full sm:w-auto text-center"
            >
              Clear
            </button>
            <Link
              href={`/compare?ids=${selectedIds.join(',')}`}
              className={`w-full sm:w-auto px-6 py-2.5 rounded-xl font-semibold text-sm transition-all flex items-center justify-center gap-2 shadow-lg ${selectedIds.length >= 2
                ? 'bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 text-white shadow-blue-500/25 hover:scale-[1.02] active:scale-[0.98]'
                : 'bg-slate-800 text-slate-500 border border-slate-700/50 cursor-not-allowed pointer-events-none'
                }`}
            >
              <Scale className="w-4 h-4" /> Compare Selected
            </Link>
          </div>
        </div>
      )}

      {/* Documents Table */}
      <div className="glass-card overflow-hidden">
        {isLoading ? (
          <div className="p-6 space-y-3">
            {[1, 2, 3, 4].map((i) => <div key={i} className="skeleton h-16 w-full" />)}
          </div>
        ) : filtered.length === 0 ? (
          <div className="text-center py-16">
            <FileText className="w-14 h-14 text-slate-600 mx-auto mb-3" />
            <p className="text-slate-400 font-medium">No documents found</p>
            {search ? (
              <p className="text-slate-500 text-sm mt-1">Try a different search term</p>
            ) : (
              <Link href="/upload" className="btn-primary mt-5 inline-flex">
                <Upload className="w-4 h-4" /> Upload First Document
              </Link>
            )}
          </div>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-700/50">
                <th className="w-12 px-6 py-4"></th>
                <th className="text-left px-2 py-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">Document</th>
                <th className="text-left px-6 py-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">Status</th>
                <th className="text-left px-6 py-4 text-xs font-semibold text-slate-400 uppercase tracking-wider hidden md:table-cell">Size</th>
                <th className="text-left px-6 py-4 text-xs font-semibold text-slate-400 uppercase tracking-wider hidden lg:table-cell">Uploaded</th>
                <th className="text-right px-6 py-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/30">
              {filtered.map((doc) => {
                const isSelectable = doc.status === 'completed' || doc.status === 'summarized';
                const isSelected = selectedIds.includes(doc.id);
                return (
                  <tr key={doc.id} className={`hover:bg-white/3 transition-colors ${isSelected ? 'bg-blue-500/5 hover:bg-blue-500/10' : ''}`}>
                    <td className="px-6 py-4 text-center">
                      <input
                        type="checkbox"
                        className="w-4 h-4 rounded border-slate-700 bg-[#0c1322] text-blue-500 focus:ring-blue-500/20 focus:ring-offset-0 transition cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
                        disabled={!isSelectable}
                        checked={isSelected}
                        onChange={(e) => {
                          if (e.target.checked) {
                            if (selectedIds.length >= 3) {
                              toast.error('You can compare a maximum of 3 policies.');
                              return;
                            }
                            setSelectedIds([...selectedIds, doc.id]);
                          } else {
                            setSelectedIds(selectedIds.filter((id) => id !== doc.id));
                          }
                        }}
                      />
                    </td>
                    <td className="px-2 py-4">
                      <div className="flex items-center gap-3">
                        <div className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
                          style={{ background: doc.file_type === 'pdf' ? 'rgba(239,68,68,0.15)' : 'rgba(59,130,246,0.15)' }}>
                          <FileText className="w-4 h-4" style={{ color: doc.file_type === 'pdf' ? '#f87171' : '#60a5fa' }} />
                        </div>
                        <div>
                          <p className="font-medium text-sm truncate max-w-[200px] md:max-w-xs">{doc.original_filename}</p>
                          <p className="text-xs text-slate-500 capitalize">{doc.file_type} • {doc.page_count} page{doc.page_count !== 1 ? 's' : ''}</p>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <DocumentStatusBadge status={doc.status} />
                    </td>
                    <td className="px-6 py-4 text-sm text-slate-400 hidden md:table-cell">
                      {formatBytes(doc.file_size_bytes)}
                    </td>
                    <td className="px-6 py-4 text-sm text-slate-400 hidden lg:table-cell">
                      {new Date(doc.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center justify-end gap-2">
                        <Link href={`/documents/${doc.id}`} className="btn-secondary py-1.5 px-3 text-xs">
                          <Eye className="w-3.5 h-3.5" /> View
                        </Link>
                        <button
                          onClick={() => {
                            if (confirm('Delete this document and all its analysis data?')) {
                              deleteMutation.mutate(doc.id);
                              setSelectedIds(selectedIds.filter((id) => id !== doc.id));
                            }
                          }}
                          className="btn-danger py-1.5 px-3 text-xs"
                          disabled={deleteMutation.isPending}
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

    </div>
  );
}

