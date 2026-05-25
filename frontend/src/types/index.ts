/**
 * Type definitions for the Healthcare AI application
 */

export interface User {
  id: string;
  email: string;
  full_name: string;
  role: 'admin' | 'user';
  is_active: boolean;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export type DocumentStatus =
  | 'uploaded'
  | 'processing'
  | 'text_extracted'
  | 'summarized'
  | 'completed'
  | 'failed';

export interface Document {
  id: string;
  original_filename: string;
  file_type: 'pdf' | 'image';
  file_size_bytes: number;
  page_count: number;
  status: DocumentStatus;
  extraction_method?: string;
  renewal_date?: string;
  premium_due_date?: string;
  safety_score: number;
  created_at: string;
  updated_at: string;
}

export interface ExtractedField {
  id: string;
  field_name: string;
  field_value?: string;
  field_category?: string;
}

export interface Summary {
  id: string;
  summary_text: string;
  coverage_summary?: string;
  exclusions_summary?: string;
  waiting_period_summary?: string;
  premium_summary?: string;
  model_used: string;
  created_at: string;
}

export type RiskSeverity = 'low' | 'medium' | 'high';

export interface RiskAnalysis {
  id: string;
  clause_text: string;
  risk_type: string;
  severity: RiskSeverity;
  explanation?: string;
  recommendation?: string;
  created_at: string;
}

export interface DocumentDetail extends Document {
  extracted_text?: string;
  summary?: Summary;
  extracted_fields: ExtractedField[];
  risk_analyses: RiskAnalysis[];
}

export interface SummarizeResponse {
  document_id: string;
  summary: Summary;
}

export interface ExtractFieldsResponse {
  document_id: string;
  fields: ExtractedField[];
}

export interface RiskAnalysisResponse {
  document_id: string;
  risks: RiskAnalysis[];
  overall_risk_level: string;
}

export interface FeatureWinner {
  feature: string;
  winner: string;
  reason: string;
}

export interface ComparisonSynthesis {
  synthesis: string;
  best_for: string;
  verdict: string;
  feature_winners: FeatureWinner[];
}

export interface CompareResponse {
  documents: DocumentDetail[];
  comparison_synthesis: ComparisonSynthesis;
}

