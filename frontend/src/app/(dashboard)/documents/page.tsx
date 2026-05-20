'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { documentsApi } from '@/lib/apiHelpers';
import { Document } from '@/types';
import Link from 'next/link';
import { FileText, Upload, Trash2, Eye, Search } from 'lucide-react';
import { useState } from 'react';
import DocumentStatusBadge from '@/components/documents/DocumentStatusBadge';
import toast from 'react-hot-toast';

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DocumentsListPage() {
  const [search, setSearch] = useState('');
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
          className="form-input pl-10"
          placeholder="Search documents..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

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
                <th className="text-left px-6 py-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">Document</th>
                <th className="text-left px-6 py-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">Status</th>
                <th className="text-left px-6 py-4 text-xs font-semibold text-slate-400 uppercase tracking-wider hidden md:table-cell">Size</th>
                <th className="text-left px-6 py-4 text-xs font-semibold text-slate-400 uppercase tracking-wider hidden lg:table-cell">Uploaded</th>
                <th className="text-right px-6 py-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/30">
              {filtered.map((doc) => (
                <tr key={doc.id} className="hover:bg-white/3 transition-colors">
                  <td className="px-6 py-4">
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
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
