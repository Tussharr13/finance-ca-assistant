import json

import pytest
from fastapi.testclient import TestClient

from finance_ca_assistant.pipeline import RAGPipeline
from server.app import FAILURE_REPLY, create_app
from server.pipeline_loader import LoadedPipeline
from server.store import ConversationStore


def parse_events(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def client(sample_chunks):
    loaded = LoadedPipeline(RAGPipeline.from_chunks(sample_chunks), "test corpus", None)
    app = create_app(store=ConversationStore(":memory:"), loader=lambda: loaded)
    return TestClient(app)


def send(client, conversation_id, content):
    response = client.post(f"/api/conversations/{conversation_id}/messages", json={"content": content})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_events(response.text)


def test_health_reports_knowledge_base(client):
    body = client.get("/api/health").json()
    assert body["knowledge_base"]["chunks"] == 2 and body["llm"] is None


def test_conversation_lifecycle(client):
    created = client.post("/api/conversations").json()
    assert created["title"] == "New chat"
    assert client.get("/api/conversations").json()["conversations"][0]["id"] == created["id"]

    events = send(client, created["id"], "What does SA 500 say about audit evidence?")
    assert [name for name, _ in events] == ["user_message", "status", "assistant_message", "done"]
    assistant = events[2][1]
    assert assistant["role"] == "assistant" and assistant["status"] == "abstained"
    assert assistant["payload"]["sources"][0]["document"]

    detail = client.get(f"/api/conversations/{created['id']}").json()
    assert detail["conversation"]["title"] == "What does SA 500 say about audit evidence?"
    assert "profile" not in detail["conversation"]
    assert [message["role"] for message in detail["messages"]] == ["user", "assistant"]

    assert client.delete(f"/api/conversations/{created['id']}").status_code == 204
    assert client.get(f"/api/conversations/{created['id']}").status_code == 404


def test_profile_persists_across_turns_until_calculated(client):
    conversation_id = client.post("/api/conversations").json()["id"]
    first = send(client, conversation_id, "My salary is 18 LPA, basic 9 lakh, age 34, resident. Help me save tax.")
    assert first[2][1]["status"] == "needs_info"
    assert first[2][1]["payload"]["questions"]

    # The second turn sends no profile: the server remembers the facts.
    second = send(
        client,
        conversation_id,
        "I live in Pune, rent 25k per month, HRA 30000 per month. 80C 1.5 lakh via EPF and ELSS. "
        "health insurance 20000, parents insurance 30000, senior citizen parents. no other income",
    )
    assistant = second[2][1]
    assert assistant["status"] == "calculated"
    calculation = assistant["payload"]["calculation"]
    assert (calculation["old_total"], calculation["new_total"], calculation["saving"]) == (223_080, 150_800, 72_280)
    assert "| Total tax |" in assistant["content"]


def test_failed_turn_is_reported_and_stored(sample_chunks):
    class Broken:
        chunks = []

        def consult(self, *args, **kwargs):
            raise RuntimeError("boom")

    app = create_app(store=ConversationStore(":memory:"), loader=lambda: LoadedPipeline(Broken(), "broken", None))
    client = TestClient(app)
    conversation_id = client.post("/api/conversations").json()["id"]
    events = send(client, conversation_id, "hello")
    assert [name for name, _ in events] == ["user_message", "status", "error", "assistant_message", "done"]
    assert events[3][1]["content"] == FAILURE_REPLY and events[3][1]["status"] == "error"


def test_validation_and_missing_conversations(client):
    assert client.post("/api/conversations/nope/messages", json={"content": "hi"}).status_code == 404
    conversation_id = client.post("/api/conversations").json()["id"]
    assert client.post(f"/api/conversations/{conversation_id}/messages", json={"content": ""}).status_code == 422
    assert client.post(f"/api/conversations/{conversation_id}/messages", json={"content": "x" * 4001}).status_code == 422
    assert client.delete("/api/conversations/nope").status_code == 404


def test_long_titles_are_cut_at_a_word(client):
    conversation_id = client.post("/api/conversations").json()["id"]
    send(client, conversation_id, "Please explain the reporting requirements of clause 44 of Form 3CD in detail for my client audit")
    title = client.get(f"/api/conversations/{conversation_id}").json()["conversation"]["title"]
    assert title.endswith("…") and len(title) <= 61 and " " in title


def test_store_survives_concurrent_reads_and_writes(tmp_path):
    # Regression: unlocked reads on the shared connection segfaulted uvicorn when a
    # page load fired parallel requests.
    from concurrent.futures import ThreadPoolExecutor

    store = ConversationStore(tmp_path / "chat.sqlite3")
    ids = [store.create_conversation()["id"] for _ in range(4)]

    def work(n):
        conversation_id = ids[n % len(ids)]
        if n % 5 == 0:
            store.add_message(conversation_id, "user", f"message {n}")
        store.list_conversations()
        store.get_conversation(conversation_id)
        return len(store.messages(conversation_id))

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(work, range(800)))
    assert sum(len(store.messages(conversation_id)) for conversation_id in ids) == 160
