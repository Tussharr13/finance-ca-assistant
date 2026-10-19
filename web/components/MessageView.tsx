"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Message } from "@/lib/types";

const STATUS: Record<string, { label: string; tone: string }> = {
  needs_info: { label: "Needs a few facts", tone: "warn" },
  calculated: { label: "Calculated", tone: "" },
  answered: { label: "Answered from sources", tone: "ok" },
  abstained: { label: "Couldn't verify an answer", tone: "muted" },
  unsupported_year: { label: "Tax year not supported yet", tone: "warn" },
  error: { label: "Something went wrong", tone: "danger" },
};

const inr = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });

export function UserMessage({ message }: { message: Message }) {
  return (
    <div className="message-user">
      <div className="bubble">{message.content}</div>
    </div>
  );
}

export function AssistantMessage({ message }: { message: Message }) {
  const [copied, setCopied] = useState(false);
  const status = message.status ? STATUS[message.status] : undefined;
  const { sources = [], calculation, explanation } = message.payload ?? {};

  async function copy() {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <article className="message-assistant">
      <div className="avatar" aria-hidden>
        CA
      </div>
      <div className="assistant-body">
        {(status || calculation) && (
          <div className="meta-row">
            {status && <span className={`chip ${status.tone}`}>{status.label}</span>}
            {calculation && (
              <span className="chip muted" title={`Calculation ${calculation.run_id}`}>
                {calculation.lower_regime === "equal"
                  ? `${calculation.tax_year}: both regimes equal`
                  : `${calculation.tax_year}: ${calculation.lower_regime} regime saves ${inr.format(calculation.saving)}`}
              </span>
            )}
            {explanation && explanation.rejected > 0 && (
              <span className="mono">
                {explanation.accepted} statement{explanation.accepted === 1 ? "" : "s"} verified,{" "}
                {explanation.rejected} removed
              </span>
            )}
          </div>
        )}
        <div className="markdown">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              table: ({ children }) => (
                <div className="table-wrap">
                  <table>{children}</table>
                </div>
              ),
              a: ({ children, href }) => (
                <a href={href} target="_blank" rel="noreferrer">
                  {children}
                </a>
              ),
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>
        {sources.length > 0 && (
          <details className="sources">
            <summary>
              Sources searched ({sources.length})
            </summary>
            <ol className="source-list">
              {sources.map((source) => (
                <li key={`${source.chunk_id}-${source.rank}`}>
                  <div className="source-title">
                    {source.document ?? "Unknown document"}
                    {source.page != null && `, page ${source.page}`}
                  </div>
                  <p className="source-snippet">{source.snippet}</p>
                </li>
              ))}
            </ol>
          </details>
        )}
        <div className="actions">
          <button type="button" className="text-button" onClick={copy}>
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
      </div>
    </article>
  );
}

export function PendingMessage({ label }: { label: string }) {
  return (
    <div className="pending" role="status" aria-live="polite">
      <div className="avatar" aria-hidden>
        CA
      </div>
      <span className="dots" aria-hidden>
        <span />
        <span />
        <span />
      </span>
      {label}
    </div>
  );
}
