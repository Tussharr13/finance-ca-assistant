"""SQLite persistence for conversations, messages and per-conversation client profiles.

SQLite keeps local development dependency-free. The production target is
Postgres (decisionsForCA.md D-10); the interface here is what callers rely on.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    profile_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    status TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages(conversation_id, created_at);
"""

NEW_CHAT_TITLE = "New chat"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class ConversationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # FastAPI runs sync handlers in a thread pool. Every statement, reads included,
        # goes through one lock: concurrent use of a shared sqlite3 connection can
        # segfault the process (seen when a page load fires parallel requests).
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(_SCHEMA)

    def create_conversation(self) -> Dict[str, Any]:
        now = _now()
        conversation = {"id": uuid.uuid4().hex, "title": NEW_CHAT_TITLE, "created_at": now, "updated_at": now}
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation["id"], conversation["title"], now, now),
            )
        return conversation

    def list_conversations(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._db.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if row is None:
            return None
        conversation = dict(row)
        conversation["profile"] = json.loads(conversation.pop("profile_json"))
        return conversation

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._lock, self._db:
            cursor = self._db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return cursor.rowcount > 0

    def messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, role, content, status, payload_json, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY created_at, rowid",
                (conversation_id,),
            ).fetchall()
        messages = []
        for row in rows:
            message = dict(row)
            message["payload"] = json.loads(message.pop("payload_json"))
            messages.append(message)
        return messages

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        status: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = _now()
        message = {
            "id": uuid.uuid4().hex,
            "role": role,
            "content": content,
            "status": status,
            "payload": payload or {},
            "created_at": now,
        }
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, status, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (message["id"], conversation_id, role, content, status, json.dumps(message["payload"]), now),
            )
            self._db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
        return message

    def update_conversation(
        self, conversation_id: str, title: Optional[str] = None, profile: Optional[Dict[str, Any]] = None
    ) -> None:
        with self._lock, self._db:
            if title is not None:
                self._db.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))
            if profile is not None:
                self._db.execute(
                    "UPDATE conversations SET profile_json = ? WHERE id = ?", (json.dumps(profile), conversation_id)
                )
