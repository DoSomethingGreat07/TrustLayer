export type Credentials = {
  username: string;
  password: string;
};

export type CorpusPaper = {
  title: string;
  authors: string;
  domain: string;
  file_name: string;
  metadata_source: string;
};

export type CorpusSummary = {
  document_count: number;
  chunk_count: number;
  paper_count: number;
  domains: string[];
  papers: CorpusPaper[];
};

export type EvidenceItem = {
  rank: number;
  score: number;
  content: string;
  metadata: {
    paper_title: string;
    authors: string;
    metadata_source: string;
    file_name: string;
    page_number: string | number;
    chunk_id: string;
    domain: string;
  };
};

export type AskResponse = {
  answer: string;
  justification: string;
  corrected: boolean;
  used_queries: string[];
  retrieval_confidence: number;
  verification: {
    verified?: boolean;
    reason?: string;
  };
  verification_params: Record<string, unknown>;
  evidence: EvidenceItem[];
  context: string;
  abstained: boolean;
  retries_used: number;
  pipeline_mode?: string;
  retrieval_strategy?: {
    name?: string;
    reason?: string;
    dense_k?: number;
    sparse_k?: number;
    fusion_k?: number;
    final_k?: number;
    max_retries?: number;
  };
  cache_hit?: boolean;
  cache_key?: string;
};
