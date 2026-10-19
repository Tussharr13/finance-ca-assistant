"use client";

import type { Conversation } from "@/lib/types";
import { PlusIcon, TrashIcon } from "./icons";

type Props = {
  conversations: Conversation[];
  activeId: string | null;
  open: boolean;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  onDelete: (id: string) => void;
};

const GROUPS = ["Today", "Yesterday", "Previous 7 days", "Previous 30 days", "Older"] as const;

function groupFor(iso: string, now: Date): (typeof GROUPS)[number] {
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const day = 24 * 60 * 60 * 1000;
  const time = new Date(iso).getTime();
  if (time >= startOfToday) return "Today";
  if (time >= startOfToday - day) return "Yesterday";
  if (time >= startOfToday - 7 * day) return "Previous 7 days";
  if (time >= startOfToday - 30 * day) return "Previous 30 days";
  return "Older";
}

export default function Sidebar({ conversations, activeId, open, onSelect, onNewChat, onDelete }: Props) {
  const now = new Date();
  const grouped = GROUPS.map((label) => ({
    label,
    items: conversations.filter((conversation) => groupFor(conversation.updated_at, now) === label),
  })).filter((group) => group.items.length);

  return (
    <aside className={`sidebar${open ? " open" : ""}`} aria-label="Chat history">
      <div className="brand">
        <span className="brand-mark">CA</span>
        Finance CA Assistant
      </div>
      <button type="button" className="new-chat" onClick={onNewChat}>
        <PlusIcon /> New chat
      </button>
      <nav className="history">
        {grouped.length === 0 && <p className="history-empty">Your chats will appear here.</p>}
        {grouped.map((group) => (
          <div className="history-group" key={group.label}>
            <div className="history-label">{group.label}</div>
            {group.items.map((conversation) => (
              <div
                key={conversation.id}
                className={`history-item${conversation.id === activeId ? " active" : ""}`}
              >
                <button
                  type="button"
                  className="history-open"
                  onClick={() => onSelect(conversation.id)}
                  aria-current={conversation.id === activeId ? "page" : undefined}
                  title={conversation.title}
                >
                  {conversation.title}
                </button>
                <button
                  type="button"
                  className="history-delete"
                  onClick={() => onDelete(conversation.id)}
                  aria-label={`Delete chat: ${conversation.title}`}
                >
                  <TrashIcon />
                </button>
              </div>
            ))}
          </div>
        ))}
      </nav>
      <p className="sidebar-foot">Educational decision support. Confirm with a CA before filing.</p>
    </aside>
  );
}
