/**
 * Authentication API functions
 */
import apiClient from '@/lib/api';
import { TokenResponse, User, CompareResponse } from '@/types';

interface RegisterData {
  email: string;
  full_name: string;
  password: string;
}

interface LoginData {
  email: string;
  password: string;
}

export const authApi = {
  register: async (data: RegisterData): Promise<TokenResponse> => {
    const res = await apiClient.post('/auth/register', data);
    return res.data;
  },

  login: async (data: LoginData): Promise<TokenResponse> => {
    const res = await apiClient.post('/auth/login', data);
    return res.data;
  },

  getMe: async (): Promise<User> => {
    const res = await apiClient.get('/auth/me');
    return res.data;
  },
};

export const documentsApi = {
  upload: async (file: File, onProgress?: (p: number) => void) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await apiClient.post('/documents/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        if (onProgress && e.total) {
          onProgress(Math.round((e.loaded * 100) / e.total));
        }
      },
    });
    return res.data;
  },

  /** Upload multiple image files as a single unified document (multi-image bundle). */
  uploadImages: async (files: File[], onProgress?: (p: number) => void) => {
    const formData = new FormData();
    files.forEach((f) => formData.append('files', f));
    const res = await apiClient.post('/documents/upload-images', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        if (onProgress && e.total) {
          onProgress(Math.round((e.loaded * 100) / e.total));
        }
      },
    });
    return res.data;
  },

  list: async () => {
    const res = await apiClient.get('/documents');
    return res.data;
  },

  getById: async (id: string) => {
    const res = await apiClient.get(`/documents/${id}`);
    return res.data;
  },

  delete: async (id: string) => {
    await apiClient.delete(`/documents/${id}`);
  },

  compare: async (documentIds: string[]): Promise<CompareResponse> => {
    const res = await apiClient.post('/documents/compare', { document_ids: documentIds });
    return res.data;
  },

  /** Fire-and-forget: launches Extract Fields on the server (202 immediately). */
  runFields: async (id: string) => {
    const res = await apiClient.post(`/documents/${id}/run-fields`);
    return res.data;
  },

  /** Fire-and-forget: launches Risk Analysis on the server (202 immediately). */
  runRisks: async (id: string) => {
    const res = await apiClient.post(`/documents/${id}/run-risks`);
    return res.data;
  },

  /** Fire-and-forget: force-regenerates Summary from actual document text (202 immediately). */
  runSummary: async (id: string) => {
    const res = await apiClient.post(`/documents/${id}/run-summary`);
    return res.data;
  },
};

export const aiApi = {
  summarize: async (documentId: string) => {
    const res = await apiClient.post('/ai/summarize', { document_id: documentId });
    return res.data;
  },

  extractFields: async (documentId: string) => {
    const res = await apiClient.post('/ai/extract-fields', { document_id: documentId });
    return res.data;
  },

  riskAnalysis: async (documentId: string) => {
    const res = await apiClient.post('/ai/risk-analysis', { document_id: documentId });
    return res.data;
  },

  queryDocument: async (documentId: string, query: string) => {
    const res = await apiClient.post(`/ai/documents/${documentId}/query`, { query });
    return res.data;
  },

  chat: async (query: string, documentIds?: string[], history?: { role: string; content: string }[], sessionId?: string) => {
    const res = await apiClient.post('/ai/chat', {
      query,
      document_ids: documentIds,
      history,
      session_id: sessionId,
    });
    return res.data;
  },

  getSessions: async () => {
    const res = await apiClient.get('/ai/chat/sessions');
    return res.data;
  },

  createSession: async (title?: string) => {
    const res = await apiClient.post('/ai/chat/sessions', { title });
    return res.data;
  },

  getSessionMessages: async (sessionId: string) => {
    const res = await apiClient.get(`/ai/chat/sessions/${sessionId}/messages`);
    return res.data;
  },

  deleteSession: async (sessionId: string) => {
    const res = await apiClient.delete(`/ai/chat/sessions/${sessionId}`);
    return res.data;
  },

  translate: async (text: string, targetLanguage: string) => {
    const res = await apiClient.post('/ai/translate', {
      text,
      target_language: targetLanguage,
    });
    return res.data;
  },

  claimsChecklist: async (documentId: string, treatmentType: string) => {
    const res = await apiClient.post('/ai/claims-checklist', {
      document_id: documentId,
      treatment_type: treatmentType,
    });
    return res.data;
  },

  getModelMetrics: async () => {
    const res = await apiClient.get('/ai/model-metrics');
    return res.data;
  },

  /** Fetch persistent chat history for a specific document (creates session if none exists). */
  getDocumentChatHistory: async (documentId: string): Promise<{ session_id: string; messages: Array<{ id: string; session_id: string; role: string; content: string; sources: string[] | null; created_at: string }> }> => {
    const res = await apiClient.get(`/ai/chat/history/${documentId}`);
    return res.data;
  },
};

export const remindersApi = {
  list: async () => {
    const res = await apiClient.get('/documents/reminders');
    return res.data;
  },

  schedule: async (data: {
    document_id: string;
    renewal_date?: string;
    premium_due_date?: string;
    premium_amount?: number;
  }) => {
    const res = await apiClient.post('/documents/reminders', data);
    return res.data;
  },

  dismiss: async (id: string) => {
    const res = await apiClient.patch(`/documents/reminders/${id}/dismiss`);
    return res.data;
  },
};

export const exportApi = {
  emailReport: async (id: string, email: string) => {
    const res = await apiClient.post(`/documents/${id}/email`, { email });
    return res.data;
  },
  
  downloadReport: async (id: string) => {
    const res = await apiClient.get(`/documents/${id}/export`, { responseType: 'blob' });
    return res.data;
  },
  
  getExportUrl: (id: string) => {
    const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';
    return `${baseUrl}/documents/${id}/export${token ? `?token=${token}` : ''}`;
  }
};

export const claimsApi = {
  getStats: async () => {
    const res = await apiClient.get('/claims/stats');
    return res.data;
  },
};
