"""HTTP API for the chat clients (web now, mobile later).

Run locally:
    python3 -m uvicorn server.app:create_app --factory --port 8000 --reload

Replies stream as Server-Sent Events over a POST, so clients read them with
fetch() rather than EventSource. Event types, in order:
    user_message       the stored user message
    status             {"label": ...} progress text for the UI
    assistant_message  the stored assistant message with sources and calculation
    error              {"detail": ...} if the turn failed (an assistant message is still stored)
    done               {}
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import Any, Callable, Dict, Iterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from server.pipeline_loader import LoadedPipeline, load_pipeline_from_env
from server.store import NEW_CHAT_TITLE, ConversationStore

logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 4000
TITLE_CHARS = 60
FAILURE_REPLY = "Something went wrong while working on that. Your message is saved; please try again."


class NewMessage(BaseModel):
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


def create_app(
    store: Optional[ConversationStore] = None,
    loader: Callable[[], LoadedPipeline] = load_pipeline_from_env,
) -> FastAPI:
    app = FastAPI(title="Finance CA Assistant API")
    origins = os.environ.get("FINANCE_CA_WEB_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in origins.split(",") if origin.strip()],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
    conversations = store or ConversationStore(os.environ.get("FINANCE_CA_DB", "data/server/chat.sqlite3"))
    loaded: Dict[str, LoadedPipeline] = {}

    def pipeline() -> LoadedPipeline:
        # Loaded on first use so the server starts instantly and tests can inject a loader.
        if "value" not in loaded:
            loaded["value"] = loader()
        return loaded["value"]

    def require_conversation(conversation_id: str) -> Dict[str, Any]:
        conversation = conversations.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        current = pipeline()
        return {
            "status": "ok",
            "knowledge_base": {"chunks": len(current.pipeline.chunks), "description": current.corpus},
            "llm": current.llm_model,
        }

    @app.get("/api/conversations")
    def list_conversations() -> Dict[str, Any]:
        return {"conversations": conversations.list_conversations()}

    @app.post("/api/conversations", status_code=201)
    def create_conversation() -> Dict[str, Any]:
        return conversations.create_conversation()

    @app.get("/api/conversations/{conversation_id}")
    def get_conversation(conversation_id: str) -> Dict[str, Any]:
        conversation = require_conversation(conversation_id)
        conversation.pop("profile", None)  # Client facts stay server-side.
        return {"conversation": conversation, "messages": conversations.messages(conversation_id)}

    @app.delete("/api/conversations/{conversation_id}", status_code=204, response_class=Response)
    def delete_conversation(conversation_id: str) -> Response:
        if not conversations.delete_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        return Response(status_code=204)

    @app.post("/api/conversations/{conversation_id}/messages")
    def send_message(conversation_id: str, body: NewMessage) -> StreamingResponse:
        conversation = require_conversation(conversation_id)
        content = body.content.strip()
        if not content:
            raise HTTPException(status_code=422, detail="Message is empty")

        def events() -> Iterator[str]:
            user_message = conversations.add_message(conversation_id, "user", content)
            if conversation["title"] == NEW_CHAT_TITLE:
                conversations.update_conversation(conversation_id, title=_title_from(content))
            yield _sse("user_message", user_message)
            yield _sse("status", {"label": "Checking your facts and the sources"})
            try:
                response = pipeline().pipeline.consult(content, profile=conversation["profile"], top_k=6)
            except Exception:
                logger.exception("Turn failed for conversation %s", conversation_id)
                failure = conversations.add_message(conversation_id, "assistant", FAILURE_REPLY, status="error")
                yield _sse("error", {"detail": "The assistant failed on this message."})
                yield _sse("assistant_message", failure)
                yield _sse("done", {})
                return

            profile = dict(conversation["profile"])
            profile.update({key: value for key, value in response.profile_updates.items() if value not in (None, "")})
            conversations.update_conversation(conversation_id, profile=profile)
            assistant = conversations.add_message(
                conversation_id, "assistant", response.message, status=response.status, payload=_payload(response)
            )
            yield _sse("assistant_message", assistant)
            yield _sse("done", {})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _payload(response: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"sources": response.sources, "questions": response.questions}
    if response.calculation is not None:
        calculation = asdict(response.calculation)
        payload["calculation"] = {
            "run_id": calculation["run_id"],
            "tax_year": calculation["tax_year"],
            "lower_regime": calculation["lower_regime"],
            "saving": calculation["saving"],
            "old_total": calculation["old"]["total_tax"],
            "new_total": calculation["new"]["total_tax"],
        }
    if response.explanation is not None:
        payload["explanation"] = {
            "accepted": len(response.explanation["accepted"]),
            "rejected": len(response.explanation["rejected"]),
            "error": response.explanation["error"],
        }
    return payload


def _title_from(content: str) -> str:
    first_line = " ".join(content.split())
    if len(first_line) <= TITLE_CHARS:
        return first_line
    return first_line[:TITLE_CHARS].rsplit(" ", 1)[0] + "…"


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

