"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { API_URL, ApiError, api, streamMessage } from "@/lib/api";
import type { Conversation, Health, Message } from "@/lib/types";
import Composer from "./Composer";
import EmptyState from "./EmptyState";
import { AssistantMessage, PendingMessage, UserMessage } from "./MessageView";
import Sidebar from "./Sidebar";
import { MenuIcon } from "./icons";

function setUrlConversation(id: string | null) {
  const url = new URL(window.location.href);
  if (id) url.searchParams.set("c", id);
  else url.searchParams.delete("c");
  window.history.replaceState(null, "", url);
}

export default function ChatApp() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Stream handlers check this so a reply never lands in a chat the user has left.
  const activeRef = useRef<string | null>(null);
  const streamAbort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const refreshConversations = useCallback(async () => {
    try {
      setConversations((await api.listConversations()).conversations);
    } catch (caught) {
      setError((caught as Error).message);
    }
  }, []);

  const openConversation = useCallback(async (id: string | null) => {
    streamAbort.current?.abort();
    setPending(null);
    setError(null);
    setSidebarOpen(false);
    activeRef.current = id;
    setActiveId(id);
    setUrlConversation(id);
    if (!id) {
      setMessages([]);
      return;
    }
    try {
      const detail = await api.getConversation(id);
      if (activeRef.current === id) setMessages(detail.messages);
    } catch (caught) {
      if (caught instanceof ApiError && caught.status === 404) {
        await openConversation(null);
        return;
      }
      setError((caught as Error).message);
    }
  }, []);

  useEffect(() => {
    api.health().then(setHealth).catch((caught: Error) => setError(caught.message));
    refreshConversations();
    const fromUrl = new URL(window.location.href).searchParams.get("c");
    if (fromUrl) openConversation(fromUrl);
  }, [openConversation, refreshConversations]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages, pending]);

  async function send(text: string) {
    if (pending) return;
    setError(null);
    let id = activeId;
    if (!id) {
      try {
        const created = await api.createConversation();
        id = created.id;
        setConversations((current) => [created, ...current]);
        activeRef.current = id;
        setActiveId(id);
        setUrlConversation(id);
      } catch (caught) {
        setError((caught as Error).message);
        return;
      }
    }

    const conversationId = id;
    const tempId = `pending-${Date.now()}`;
    const optimistic: Message = {
      id: tempId,
      role: "user",
      content: text,
      status: null,
      payload: {},
      created_at: new Date().toISOString(),
    };
    setMessages((current) => [...current, optimistic]);
    setPending("Sending");
    const controller = new AbortController();
    streamAbort.current = controller;
    const here = () => activeRef.current === conversationId;

    try {
      await streamMessage(
        conversationId,
        text,
        {
          onUserMessage: (message) =>
            here() && setMessages((current) => current.map((item) => (item.id === tempId ? message : item))),
          onStatus: (label) => here() && setPending(label),
          onAssistantMessage: (message) => here() && setMessages((current) => [...current, message]),
          onError: (detail) => here() && setError(detail),
        },
        controller.signal,
      );
    } catch (caught) {
      if (here()) {
        setError((caught as Error).message);
        setMessages((current) => current.filter((item) => item.id !== tempId));
      }
    } finally {
      if (here()) setPending(null);
      refreshConversations();
    }
  }

  async function remove(id: string) {
    const conversation = conversations.find((item) => item.id === id);
    if (!window.confirm(`Delete "${conversation?.title ?? "this chat"}"? This can't be undone.`)) return;
    try {
      await api.deleteConversation(id);
      setConversations((current) => current.filter((item) => item.id !== id));
      if (id === activeId) await openConversation(null);
    } catch (caught) {
      setError((caught as Error).message);
    }
  }

  const activeTitle = conversations.find((item) => item.id === activeId)?.title ?? "New chat";
  const noKnowledgeBase = health && health.knowledge_base.chunks === 0;

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        open={sidebarOpen}
        onSelect={openConversation}
        onNewChat={() => openConversation(null)}
        onDelete={remove}
      />
      <div className={`scrim${sidebarOpen ? " open" : ""}`} onClick={() => setSidebarOpen(false)} aria-hidden />

      <main className="main">
        <header className="topbar">
          <button type="button" className="icon-button" onClick={() => setSidebarOpen(true)} aria-label="Open chat history">
            <MenuIcon />
          </button>
          <span className="topbar-title">{activeTitle}</span>
        </header>

        {error && (
          <div className="banner error" role="alert">
            <strong>Problem:</strong> {error}
            {error.startsWith("Can't reach") && (
              <>
                {" "}
                Start it with <code>python3 -m uvicorn server.app:create_app --factory --port 8000</code>, or set{" "}
                <code>NEXT_PUBLIC_API_URL</code> (currently {API_URL}).
              </>
            )}
          </div>
        )}
        {noKnowledgeBase && (
          <div className="banner">
            <strong>No knowledge base loaded.</strong> Tax calculations work, but questions about standards and law
            can&apos;t be answered until the server is started with <code>FINANCE_CA_ARTIFACTS_DIR</code> pointing at
            your Kaggle artifacts.
          </div>
        )}

        <div className="scroll">
          {messages.length === 0 && !pending ? (
            <EmptyState onPick={send} disabled={Boolean(pending)} />
          ) : (
            <div className="thread">
              {messages.map((message) =>
                message.role === "user" ? (
                  <UserMessage key={message.id} message={message} />
                ) : (
                  <AssistantMessage key={message.id} message={message} />
                ),
              )}
              {pending && <PendingMessage label={pending} />}
              <div ref={bottom} />
            </div>
          )}
        </div>

        <Composer disabled={Boolean(pending)} onSend={send} autoFocusKey={activeId} />
      </main>
    </div>
  );
}
