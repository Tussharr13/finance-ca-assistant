import type { Conversation, Health, Message } from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError(`Can't reach the assistant server at ${API_URL}.`);
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ApiError(body?.detail ?? `Request failed (${response.status}).`, response.status);
  }
  return (response.status === 204 ? undefined : await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/api/health"),
  listConversations: () => request<{ conversations: Conversation[] }>("/api/conversations"),
  createConversation: () => request<Conversation>("/api/conversations", { method: "POST" }),
  getConversation: (id: string) =>
    request<{ conversation: Conversation; messages: Message[] }>(`/api/conversations/${id}`),
  deleteConversation: (id: string) => request<void>(`/api/conversations/${id}`, { method: "DELETE" }),
};

export type StreamHandlers = {
  onUserMessage: (message: Message) => void;
  onStatus: (label: string) => void;
  onAssistantMessage: (message: Message) => void;
  onError: (detail: string) => void;
};

/**
 * Send a message and read the server-sent events from the POST response.
 * EventSource only supports GET, so the stream is parsed by hand.
 */
export async function streamMessage(
  conversationId: string,
  content: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/api/conversations/${conversationId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
      signal,
    });
  } catch (error) {
    if ((error as Error).name === "AbortError") return;
    throw new ApiError(`Can't reach the assistant server at ${API_URL}.`);
  }
  if (!response.ok || !response.body) {
    const body = await response.json().catch(() => null);
    throw new ApiError(body?.detail ?? `Request failed (${response.status}).`, response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        dispatch(buffer.slice(0, boundary), handlers);
        buffer = buffer.slice(boundary + 2);
        boundary = buffer.indexOf("\n\n");
      }
    }
  } catch (error) {
    if ((error as Error).name === "AbortError") return;
    throw error;
  }
}

function dispatch(block: string, handlers: StreamHandlers) {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return;
  const parsed = JSON.parse(data.join("\n"));
  if (event === "user_message") handlers.onUserMessage(parsed);
  else if (event === "status") handlers.onStatus(parsed.label);
  else if (event === "assistant_message") handlers.onAssistantMessage(parsed);
  else if (event === "error") handlers.onError(parsed.detail);
}
