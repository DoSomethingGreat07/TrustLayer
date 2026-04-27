import type { AskResponse, CorpusSummary, Credentials } from './types';

const API_BASE_URL = import.meta.env.VITE_TRUSTLAYER_API_URL ?? 'http://127.0.0.1:8000';

async function request<T>(
  path: string,
  options: RequestInit = {},
  credentials?: Credentials,
): Promise<T> {
  const headers = new Headers(options.headers);
  headers.set('Content-Type', 'application/json');

  if (credentials) {
    headers.set('X-TrustLayer-Username', credentials.username);
    headers.set('X-TrustLayer-Password', credentials.password);
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed with status ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function login(credentials: Credentials) {
  return request<{ ok: boolean }>('/api/login', {
    method: 'POST',
    body: JSON.stringify(credentials),
  });
}

export function getCorpus(credentials: Credentials) {
  return request<CorpusSummary>('/api/corpus', {}, credentials);
}

export function askQuestion(
  credentials: Credentials,
  query: string,
  options: { useApiEnrichment: boolean; device: string },
) {
  return request<AskResponse>(
    '/api/ask',
    {
      method: 'POST',
      body: JSON.stringify({
        query,
        use_api_enrichment: options.useApiEnrichment,
        device: options.device,
      }),
    },
    credentials,
  );
}
