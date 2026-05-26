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
};
