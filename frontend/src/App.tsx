import { FormEvent, ReactNode, useMemo, useState } from 'react';
import {
  BookOpen,
  CheckCircle2,
  CircleAlert,
  ClipboardCheck,
  FileSearch,
  Gauge,
  History,
  KeyRound,
  Loader2,
  Lock,
  MessageSquareText,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import { askQuestion, getCorpus, login } from './api';
import type { AskResponse, CorpusPaper, CorpusSummary, Credentials } from './types';

type ChatTurn = {
  id: string;
  query: string;
  result: AskResponse;
};

const defaultCredentials: Credentials = {
  username: '',
  password: '',
};

const sampleQuestions = [
  'What is corrective RAG?',
  'How does reranking improve retrieval quality?',
  'What evidence supports verification-based abstention?',
];

function percent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function formatParamValue(value: unknown) {
  if (typeof value === 'number') {
    return Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(3);
  }
  if (typeof value === 'boolean') {
    return value ? 'True' : 'False';
  }
  if (typeof value === 'string') {
    return value || 'Unknown';
  }
  if (value === null || value === undefined) {
    return 'Unknown';
  }
  if (Array.isArray(value)) {
    return `${value.length} items`;
  }
  if (typeof value === 'object') {
    return 'Details';
  }
  return String(value || 'Unknown');
}

function renderParamDetails(value: unknown) {
  if (value === null || value === undefined || typeof value !== 'object') {
    return null;
  }

  return (
    <pre className="mt-3 max-h-36 overflow-auto whitespace-pre-wrap rounded-md bg-ink/[0.04] p-3 text-xs leading-5 text-ink/65">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

function MetricCard({
  icon,
  label,
  value,
  accent = 'text-moss',
}: {
  icon: ReactNode;
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div className="rounded-lg border border-ink/10 bg-white px-4 py-3 shadow-sm">
      <div className={`mb-2 flex h-9 w-9 items-center justify-center rounded-md bg-panel ${accent}`}>
        {icon}
      </div>
      <div className="text-2xl font-semibold leading-none text-ink">{value}</div>
      <div className="mt-1 text-sm text-ink/60">{label}</div>
    </div>
  );
}

function groupedPapers(papers: CorpusPaper[]) {
  return papers.reduce<Record<string, CorpusPaper[]>>((groups, paper) => {
    const domain = paper.domain || 'Unknown';
    groups[domain] = groups[domain] ?? [];
    groups[domain].push(paper);
    return groups;
  }, {});
}

function LoginPanel({
  credentials,
  setCredentials,
  onLogin,
  error,
  loading,
}: {
  credentials: Credentials;
  setCredentials: (credentials: Credentials) => void;
  onLogin: () => void;
  error: string;
  loading: boolean;
}) {
  return (
    <main className="soft-grid flex min-h-screen items-center justify-center bg-panel px-5 py-10">
      <section className="w-full max-w-5xl overflow-hidden rounded-lg border border-ink/10 bg-paper shadow-soft">
        <div className="grid min-h-[560px] grid-cols-1 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="flex flex-col justify-between bg-ink p-8 text-white sm:p-10">
            <div>
              <div className="mb-6 inline-flex items-center gap-2 rounded-md border border-white/15 px-3 py-2 text-sm text-white/80">
                <ShieldCheck size={16} />
                Evidence-first research workspace
              </div>
              <h1 className="max-w-xl text-4xl font-semibold leading-tight sm:text-5xl">
                TrustLayer Research Assistant
              </h1>
              <p className="mt-5 max-w-2xl text-base leading-7 text-white/70">
                A React interface for grounded answers, verification signals, and inspectable
                paper evidence. The Streamlit app remains available separately.
              </p>
            </div>

            <div className="grid grid-cols-1 gap-3 pt-8 sm:grid-cols-3">
              {['Hybrid retrieval', 'Reranking', 'Verification'].map((item) => (
                <div key={item} className="rounded-lg border border-white/10 bg-white/[0.04] p-4">
                  <div className="text-sm font-medium text-white">{item}</div>
                  <div className="mt-2 h-1.5 rounded-full bg-white/10">
                    <div className="h-1.5 rounded-full bg-brass" style={{ width: '72%' }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="flex items-center p-8 sm:p-10">
            <form
              className="w-full"
              onSubmit={(event) => {
                event.preventDefault();
                onLogin();
              }}
            >
              <div className="mb-7">
                <div className="mb-3 flex h-11 w-11 items-center justify-center rounded-lg bg-moss/10 text-moss">
                  <Lock size={22} />
                </div>
                <h2 className="text-2xl font-semibold text-ink">Sign in</h2>
                <p className="mt-2 text-sm leading-6 text-ink/60">
                  Use the same username and password configured in your `.env` file.
                </p>
              </div>

              <label className="mb-2 block text-sm font-medium text-ink/70" htmlFor="username">
                Username
              </label>
              <div className="mb-4 flex items-center gap-3 rounded-lg border border-ink/10 bg-white px-3 py-3">
                <KeyRound size={18} className="text-ink/40" />
                <input
                  id="username"
                  className="w-full bg-transparent text-ink outline-none"
                  value={credentials.username}
                  onChange={(event) =>
                    setCredentials({ ...credentials, username: event.target.value })
                  }
                  autoComplete="username"
                />
              </div>

              <label className="mb-2 block text-sm font-medium text-ink/70" htmlFor="password">
                Password
              </label>
              <div className="mb-5 flex items-center gap-3 rounded-lg border border-ink/10 bg-white px-3 py-3">
                <Lock size={18} className="text-ink/40" />
                <input
                  id="password"
                  className="w-full bg-transparent text-ink outline-none"
                  type="password"
                  value={credentials.password}
                  onChange={(event) =>
                    setCredentials({ ...credentials, password: event.target.value })
                  }
                  autoComplete="current-password"
                />
              </div>

              {error ? (
                <div className="mb-4 rounded-lg border border-berry/20 bg-berry/10 px-4 py-3 text-sm text-berry">
                  {error}
                </div>
              ) : null}

              <button
                className="flex w-full items-center justify-center gap-2 rounded-lg bg-moss px-4 py-3 font-medium text-white transition hover:bg-moss/90 disabled:cursor-not-allowed disabled:opacity-60"
                disabled={loading}
                type="submit"
              >
                {loading ? <Loader2 size={18} className="animate-spin" /> : <ShieldCheck size={18} />}
                Enter workspace
              </button>
            </form>
          </div>
        </div>
      </section>
    </main>
  );
}

function CorpusSidebar({
  corpus,
  history,
  activeTurnId,
  setActiveTurnId,
}: {
  corpus: CorpusSummary | null;
  history: ChatTurn[];
  activeTurnId: string | null;
  setActiveTurnId: (id: string) => void;
}) {
  const papersByDomain = groupedPapers(corpus?.papers ?? []);
  const domainNames = Object.keys(papersByDomain).sort();

  return (
    <aside className="hidden h-screen w-[340px] shrink-0 overflow-y-auto border-r border-ink/10 bg-[#f8f4ea] p-5 lg:block">
      <div className="mb-6">
        <div className="flex items-center gap-2 text-lg font-semibold text-ink">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-moss text-white">
            <BookOpen size={18} />
          </div>
          TrustLayer
        </div>
        <p className="mt-2 text-sm leading-6 text-ink/55">
          Local corpus assistant with confidence-aware answers.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <MetricCard icon={<FileSearch size={18} />} label="Papers" value={`${corpus?.paper_count ?? 0}`} />
        <MetricCard
          icon={<MessageSquareText size={18} />}
          label="Turns"
          value={`${history.length}`}
          accent="text-marine"
        />
      </div>

      <section className="mt-6">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-ink/60">Recent questions</h2>
          <History size={16} className="text-ink/35" />
        </div>
        <div className="space-y-2">
          {history.length === 0 ? (
            <div className="rounded-lg border border-dashed border-ink/15 px-4 py-5 text-sm leading-6 text-ink/50">
              No questions yet.
            </div>
          ) : (
            history.map((turn) => (
              <button
                key={turn.id}
                className={`w-full rounded-lg border px-3 py-3 text-left text-sm transition ${
                  activeTurnId === turn.id
                    ? 'border-moss/30 bg-moss/10 text-ink shadow-sm'
                    : 'border-ink/10 bg-white text-ink/65 hover:border-moss/25 hover:bg-moss/5'
                }`}
                onClick={() => setActiveTurnId(turn.id)}
              >
                {turn.query}
              </button>
            ))
          )}
        </div>
      </section>

      <section className="mt-6">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-ink/60">Domains</h2>
        <div className="flex flex-wrap gap-2">
          {(corpus?.domains ?? []).slice(0, 8).map((domain) => (
            <span key={domain} className="rounded-md border border-marine/15 bg-marine/10 px-2.5 py-1.5 text-xs font-medium text-marine">
              {domain}
            </span>
          ))}
        </div>
      </section>

      <section className="mt-6">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-ink/60">Paper browser</h2>
          <FileSearch size={16} className="text-ink/35" />
        </div>

        {domainNames.length === 0 ? (
          <div className="rounded-lg border border-dashed border-ink/15 px-4 py-5 text-sm leading-6 text-ink/50">
            No papers found in the artifact manifest.
          </div>
        ) : (
          <div className="space-y-3">
            {domainNames.map((domain) => (
              <details key={domain} className="group rounded-lg border border-ink/10 bg-white shadow-sm">
                <summary className="cursor-pointer list-none px-3 py-3 text-sm font-semibold text-ink">
                  <div className="flex items-center justify-between gap-3">
                    <span className="capitalize">{domain}</span>
                    <span className="rounded-md bg-ink/[0.04] px-2 py-1 text-xs text-ink/55">
                      {papersByDomain[domain].length}
                    </span>
                  </div>
                </summary>
                <div className="max-h-72 space-y-2 overflow-y-auto border-t border-ink/10 p-3">
                  {papersByDomain[domain].map((paper) => (
                    <article key={`${domain}-${paper.file_name}`} className="rounded-md bg-panel px-3 py-2">
                      <h3 className="text-sm font-semibold leading-5 text-ink">{paper.title}</h3>
                      <p className="mt-1 text-xs leading-5 text-ink/55">{paper.authors}</p>
                      <p className="mt-1 break-all text-xs text-ink/45">{paper.file_name}</p>
                    </article>
                  ))}
                </div>
              </details>
            ))}
          </div>
        )}
      </section>
    </aside>
  );
}

function VerificationPanel({ result }: { result: AskResponse }) {
  const params = Object.entries(result.verification_params ?? {});
  const failedChecks = result.verification_params.failed_checks;
  const failedCheckList = Array.isArray(failedChecks)
    ? failedChecks.map((item) => String(item))
    : [];
  const verificationMode = result.verification_params.verification_mode;

  return (
    <section className="rounded-lg border border-ink/10 bg-white p-4">
      <div className="mb-4 flex items-center gap-2">
        <ClipboardCheck size={19} className="text-moss" />
        <h2 className="text-lg font-semibold text-ink">Verification Parameters</h2>
      </div>

      {params.length === 0 ? (
        <div className="rounded-lg border border-dashed border-ink/15 px-4 py-5 text-sm text-ink/55">
          No verification parameters were returned for this answer.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
          {params.map(([key, value]) => (
            <div key={key} className="rounded-lg border border-ink/10 bg-panel px-3 py-3">
              <div className="text-sm font-medium capitalize text-ink/60">
                {key.replace(/_/g, ' ')}
              </div>
              <div className="mt-2 text-2xl font-semibold leading-none text-ink">
                {formatParamValue(value)}
              </div>
              {renderParamDetails(value)}
            </div>
          ))}
        </div>
      )}

      <div className="mt-5 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="rounded-lg border border-ink/10 bg-panel p-4">
          <div className="text-sm font-semibold text-ink">Verification reason</div>
          <p className="mt-2 text-sm leading-6 text-ink/65">
            {result.verification.reason ?? 'No reason provided'}
          </p>
          {typeof verificationMode === 'string' ? (
            <div className="mt-3 inline-flex rounded-md bg-ink/[0.04] px-2.5 py-1.5 text-xs font-medium text-ink/60">
              {verificationMode.replace(/_/g, ' ')}
            </div>
          ) : null}
          {failedCheckList.length > 0 ? (
            <div className="mt-3 flex flex-wrap gap-2">
              {failedCheckList.map((check) => (
                <span key={check} className="rounded-md bg-brass/10 px-2.5 py-1.5 text-xs text-brass">
                  {check.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          ) : null}
        </div>
        <div className="rounded-lg border border-ink/10 bg-panel p-4">
          <div className="text-sm font-semibold text-ink">Used queries</div>
          <div className="mt-2 flex flex-wrap gap-2">
            {result.used_queries.length === 0 ? (
              <span className="text-sm text-ink/55">No query trace returned.</span>
            ) : (
              result.used_queries.map((query) => (
                <span key={query} className="rounded-md bg-moss/10 px-2.5 py-1.5 text-xs text-moss">
                  {query}
                </span>
              ))
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

function ResultView({ turn }: { turn: ChatTurn | null }) {
  const [activeTab, setActiveTab] = useState<'evidence' | 'verification' | 'context'>('evidence');

  if (!turn) {
    return (
      <div className="flex min-h-[460px] items-center justify-center rounded-lg border border-dashed border-ink/15 bg-paper p-8 text-center">
        <div>
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-moss/10 text-moss">
            <Search size={24} />
          </div>
          <h2 className="text-xl font-semibold text-ink">Ask a paper-grounded question</h2>
          <p className="mt-2 max-w-lg text-sm leading-6 text-ink/55">
            Results will show the answer, justification, verification metrics, and the evidence
            chunks used by the generator.
          </p>
        </div>
      </div>
    );
  }

  const result = turn.result;
  const verified = result.verification.verified;
  const strategy = result.retrieval_strategy;
  const summaryMetrics = [
    {
      icon: <Gauge size={16} />,
      label: 'Retrieval',
      value: percent(result.retrieval_confidence),
      accent: 'text-moss',
    },
    {
      icon: <RefreshCw size={16} />,
      label: 'Retries',
      value: `${result.retries_used}`,
      accent: 'text-brass',
    },
    {
      icon: <ShieldCheck size={16} />,
      label: 'Corrected',
      value: result.corrected ? 'Yes' : 'No',
      accent: 'text-marine',
    },
    {
      icon: <Sparkles size={16} />,
      label: 'Evidence',
      value: `${result.evidence.length}`,
      accent: 'text-berry',
    },
  ];

  const tabs = [
    { id: 'evidence', label: 'Evidence', count: result.evidence.length },
    { id: 'verification', label: 'Verification' },
    { id: 'context', label: 'Context' },
  ] as const;

  return (
    <div className="grid min-h-0 gap-5 xl:grid-cols-[minmax(0,1fr)_400px]">
      <section className="overflow-hidden rounded-lg border border-ink/10 bg-white shadow-sm">
        <div className="h-1.5 bg-gradient-to-r from-moss via-marine to-berry" />
        <div className="p-5">
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <span className="rounded-md bg-ink px-2.5 py-1.5 text-xs font-medium text-white">Answer</span>
          <span
            className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium ${
              verified ? 'bg-moss/10 text-moss' : 'bg-brass/15 text-brass'
            }`}
          >
            {verified ? <CheckCircle2 size={14} /> : <CircleAlert size={14} />}
            {verified ? 'Verified' : 'Needs caution'}
          </span>
          {result.abstained ? (
            <span className="rounded-md bg-berry/10 px-2.5 py-1.5 text-xs font-medium text-berry">Abstained</span>
          ) : null}
          {result.cache_hit ? (
            <span className="rounded-md bg-marine/10 px-2.5 py-1.5 text-xs font-medium text-marine">Cached</span>
          ) : null}
          {strategy?.name ? (
            <span className="rounded-md bg-moss/10 px-2.5 py-1.5 text-xs font-medium text-moss">
              {strategy.name.replace(/_/g, ' ')}
            </span>
          ) : null}
        </div>

        <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_300px]">
          <div>
            <p className="max-w-5xl whitespace-pre-wrap text-xl leading-9 text-ink">{result.answer}</p>
            {result.justification ? (
              <div className="mt-4 rounded-lg border border-marine/15 bg-marine/10 p-4 text-sm leading-7 text-ink/75">
                <strong className="text-ink">Justification: </strong>
                {result.justification}
              </div>
            ) : null}
          </div>

          <div className="grid grid-cols-2 gap-3 self-start">
            {summaryMetrics.map((item) => (
              <div key={item.label} className="rounded-lg border border-ink/10 bg-panel px-3 py-3">
                <div className={`mb-2 flex h-8 w-8 items-center justify-center rounded-md bg-white ${item.accent}`}>
                  {item.icon}
                </div>
                <div className="text-2xl font-semibold leading-none text-ink">{item.value}</div>
                <div className="mt-1 text-xs text-ink/55">{item.label}</div>
              </div>
            ))}
          </div>
        </div>
        </div>
      </section>

      {strategy ? (
        <aside className="overflow-hidden rounded-lg border border-moss/20 bg-[#173c35] text-white shadow-sm">
          <div className="h-1.5 bg-gradient-to-r from-brass via-marine to-berry" />
          <div className="p-5">
          <div className="text-sm font-semibold uppercase tracking-wide text-white/65">Retrieval Plan</div>
          <h2 className="mt-1 text-2xl font-semibold capitalize text-white">
            {strategy.name?.replace(/_/g, ' ') ?? 'Balanced hybrid'}
          </h2>
          <p className="mt-3 text-sm leading-6 text-white/72">
            {strategy.reason ?? 'The router selected a balanced hybrid retrieval strategy.'}
          </p>
          <div className="mt-4 grid grid-cols-5 gap-2">
            {[
              ['Dense', strategy.dense_k],
              ['BM25', strategy.sparse_k],
              ['Fusion', strategy.fusion_k],
              ['Final', strategy.final_k],
              ['Retry', strategy.max_retries],
            ].map(([label, value]) => (
              <div key={label} className="rounded-md border border-white/10 bg-white/10 px-2 py-2 text-center">
                <div className="text-[10px] font-medium uppercase tracking-wide text-white/50">{label}</div>
                <div className="mt-1 text-xl font-semibold text-white">{value ?? '-'}</div>
              </div>
            ))}
          </div>
          </div>
        </aside>
      ) : null}

      <section className="rounded-lg border border-ink/10 bg-white shadow-sm xl:col-span-2">
        <div className="flex flex-wrap items-center gap-2 border-b border-ink/10 px-4 py-3">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`rounded-md px-3 py-2 text-sm font-medium transition ${
                activeTab === tab.id
                  ? 'bg-ink text-white shadow-sm'
                  : 'bg-panel text-ink/65 hover:bg-marine/10 hover:text-marine'
              }`}
              type="button"
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
              {'count' in tab ? <span className="ml-1 opacity-70">({tab.count})</span> : null}
            </button>
          ))}
        </div>

        <div className="max-h-[50vh] overflow-y-auto p-4">
          {activeTab === 'evidence' ? (
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
              {result.evidence.map((item) => (
                <article key={`${item.rank}-${item.metadata.chunk_id}`} className="overflow-hidden rounded-lg border border-ink/10 bg-panel">
                  <div className={`h-1 ${item.rank % 3 === 1 ? 'bg-moss' : item.rank % 3 === 2 ? 'bg-marine' : 'bg-berry'}`} />
                  <div className="p-4">
                  <div className="mb-3 flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="text-xs font-semibold uppercase tracking-wide text-brass">
                        Evidence {item.rank}
                      </div>
                      <h3
                        className="mt-1 break-words text-sm font-semibold leading-5 text-ink"
                        title={item.metadata.paper_title}
                      >
                        {item.metadata.paper_title}
                      </h3>
                    </div>
                    <span className="rounded-md bg-white px-2 py-1 text-xs text-ink/60">
                      {item.score.toFixed(3)}
                    </span>
                  </div>
                  <div className="mb-2 text-xs leading-5 text-ink/55">
                    Page {item.metadata.page_number} · {item.metadata.domain}
                  </div>
                  <details className="mb-3 rounded-md border border-ink/10 bg-white/70 px-3 py-2">
                    <summary className="cursor-pointer list-none text-xs font-medium text-marine">
                      Paper details
                    </summary>
                    <div className="mt-2 space-y-1 text-xs leading-5 text-ink/60">
                      <div>
                        <span className="font-medium text-ink/75">Title:</span> {item.metadata.paper_title}
                      </div>
                      <div>
                        <span className="font-medium text-ink/75">Authors:</span> {item.metadata.authors}
                      </div>
                      <div className="break-all">
                        <span className="font-medium text-ink/75">File:</span> {item.metadata.file_name}
                      </div>
                      <div className="break-all">
                        <span className="font-medium text-ink/75">Chunk:</span> {item.metadata.chunk_id}
                      </div>
                    </div>
                  </details>
                  <p className="line-clamp-5 text-sm leading-6 text-ink/72">{item.content}</p>
                  </div>
                </article>
              ))}
            </div>
          ) : null}

          {activeTab === 'verification' ? <VerificationPanel result={result} /> : null}

          {activeTab === 'context' ? (
            <pre className="max-h-[42vh] overflow-auto whitespace-pre-wrap rounded-lg border border-ink/10 bg-ink p-4 text-sm leading-7 text-white/80">
              {result.context || 'No generator context was returned.'}
            </pre>
          ) : null}
        </div>
      </section>
    </div>
  );
}

export function App() {
  const [credentials, setCredentials] = useState(defaultCredentials);
  const [isAuthed, setIsAuthed] = useState(false);
  const [corpus, setCorpus] = useState<CorpusSummary | null>(null);
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [device, setDevice] = useState('cpu');
  const [useApiEnrichment, setUseApiEnrichment] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loginLoading, setLoginLoading] = useState(false);
  const [error, setError] = useState('');

  const activeTurn = useMemo(
    () => history.find((turn) => turn.id === activeTurnId) ?? null,
    [activeTurnId, history],
  );

  async function handleLogin() {
    setError('');
    setLoginLoading(true);
    try {
      await login(credentials);
      const summary = await getCorpus(credentials);
      setCorpus(summary);
      setIsAuthed(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to sign in');
    } finally {
      setLoginLoading(false);
    }
  }

  async function handleAsk(event: FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;

    const askedQuery = query.trim();
    setQuery('');
    setError('');
    setLoading(true);

    try {
      const result = await askQuestion(credentials, askedQuery, {
        useApiEnrichment,
        device,
      });
      const turn: ChatTurn = {
        id: crypto.randomUUID(),
        query: askedQuery,
        result,
      };
      setHistory((current) => [turn, ...current]);
      setActiveTurnId(turn.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to run query');
    } finally {
      setLoading(false);
    }
  }

  if (!isAuthed) {
    return (
      <LoginPanel
        credentials={credentials}
        setCredentials={setCredentials}
        onLogin={handleLogin}
        error={error}
        loading={loginLoading}
      />
    );
  }

  return (
    <main className="flex min-h-screen bg-[#eee8dc] text-ink">
      <CorpusSidebar
        corpus={corpus}
        history={history}
        activeTurnId={activeTurnId}
        setActiveTurnId={setActiveTurnId}
      />

      <section className="flex min-w-0 flex-1 flex-col">
        <header className="border-b border-ink/10 bg-white px-8 py-4">
          <div className="flex w-full flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <div className="text-sm font-medium uppercase tracking-wide text-moss">
                React pipeline
              </div>
              <h1 className="mt-1 text-2xl font-semibold text-ink">
                Grounded research console
              </h1>
            </div>

            <div className="flex flex-wrap gap-3">
              <label className="flex items-center gap-2 rounded-lg border border-ink/10 bg-white px-3 py-2 text-sm text-ink/65">
                <input
                  type="checkbox"
                  checked={useApiEnrichment}
                  onChange={(event) => setUseApiEnrichment(event.target.checked)}
                />
                API enrichment
              </label>
              <select
                className="rounded-lg border border-ink/10 bg-white px-3 py-2 text-sm text-ink/70 outline-none"
                value={device}
                onChange={(event) => setDevice(event.target.value)}
              >
                <option value="cpu">CPU</option>
                <option value="mps">MPS</option>
              </select>
            </div>
          </div>
        </header>

        <div className="scrollbar-thin flex-1 overflow-y-auto px-8 py-6">
          <div className="mx-auto w-full max-w-[1680px]">
            <ResultView turn={activeTurn} />
          </div>
        </div>

        <footer className="border-t border-ink/10 bg-white px-8 py-4">
          <form className="mx-auto w-full max-w-[1680px]" onSubmit={handleAsk}>
            {error ? (
              <div className="mb-3 rounded-lg border border-berry/20 bg-berry/10 px-4 py-3 text-sm text-berry">
                {error}
              </div>
            ) : null}
            <div className="flex flex-col gap-3 lg:flex-row">
              <div className="flex flex-1 items-center gap-3 rounded-lg border border-ink/10 bg-white px-4 py-3">
                <Search size={19} className="shrink-0 text-ink/35" />
                <input
                  className="min-w-0 flex-1 bg-transparent text-ink outline-none"
                  placeholder="Ask a question about your research papers"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  disabled={loading}
                />
              </div>
              <button
                className="inline-flex items-center justify-center gap-2 rounded-lg bg-ink px-5 py-3 font-medium text-white transition hover:bg-ink/90 disabled:cursor-not-allowed disabled:opacity-60"
                disabled={loading || !query.trim()}
                type="submit"
              >
                {loading ? <Loader2 size={18} className="animate-spin" /> : <Search size={18} />}
                Ask
              </button>
            </div>

            <div className="mt-3 flex flex-wrap gap-2">
              {sampleQuestions.map((item) => (
                <button
                  key={item}
                  className="rounded-md border border-ink/10 bg-white px-3 py-1.5 text-xs text-ink/60 transition hover:border-moss/30 hover:text-moss"
                  type="button"
                  onClick={() => setQuery(item)}
                >
                  {item}
                </button>
              ))}
            </div>
          </form>
        </footer>
      </section>
    </main>
  );
}
