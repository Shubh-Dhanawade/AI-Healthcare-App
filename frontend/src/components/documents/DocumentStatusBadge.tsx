'use client';

import { DocumentStatus } from '@/types';

const statusConfig: Record<DocumentStatus, { label: string; class: string; dot?: boolean }> = {
  uploaded: { label: 'Uploaded', class: 'status-uploaded' },
  processing: { label: 'Processing', class: 'status-processing', dot: true },
  text_extracted: { label: 'Text Extracted', class: 'status-text_extracted' },
  summarized: { label: 'Summarized', class: 'status-summarized' },
  completed: { label: 'Completed', class: 'status-completed' },
  failed: { label: 'Failed', class: 'status-failed' },
};

export default function DocumentStatusBadge({ status }: { status: DocumentStatus }) {
  const config = statusConfig[status] || statusConfig.uploaded;
  return (
    <span className={`status-badge ${config.class}`}>
      {config.dot && (
        <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
      )}
      {config.label}
    </span>
  );
}
