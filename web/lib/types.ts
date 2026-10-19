export type Conversation = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export type Source = {
  rank: number;
  chunk_id: string | null;
  document: string | null;
  page: number | null;
  snippet: string;
  methods: string[];
};

export type CalculationSummary = {
  run_id: string;
  tax_year: string;
  lower_regime: "old" | "new" | "equal";
  saving: number;
  old_total: number;
  new_total: number;
};

export type MessagePayload = {
  sources?: Source[];
  questions?: string[];
  calculation?: CalculationSummary;
  explanation?: { accepted: number; rejected: number; error: string };
};

export type MessageStatus =
  | "needs_info"
  | "calculated"
  | "answered"
  | "abstained"
  | "unsupported_year"
  | "error";

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: MessageStatus | string | null;
  payload: MessagePayload;
  created_at: string;
};

export type Health = {
  status: string;
  knowledge_base: { chunks: number; description: string };
  llm: string | null;
};
