import json

import numpy as np

from finance_ca_assistant.finetuning import (
    QueryPair,
    adopt_finetuned_model,
    dense_search_fn,
    evaluate_pairs,
    find_finetuned_model,
    generate_queries,
    mine_hard_negatives,
    split_pairs,
)
from finance_ca_assistant.finetuning.embedding_trainer import FINETUNED_MARKER
from finance_ca_assistant.finetuning.query_generation import filter_questions, is_eligible_chunk, parse_questions
from finance_ca_assistant.generation.llm_factory import HuggingFacePipelineLLM

LONG = (
    "Section 44AB requires a tax audit where business turnover exceeds the prescribed limit. "
    "The auditor reports particulars in Form 3CD, including GST expenditure under clause 44, "
    "and furnishes the report electronically before the specified date."
)


def chunk(chunk_id, page, text=LONG, source="guide.pdf"):
    return {"id": chunk_id, "source": source, "page": page, "text": text}


class BatchLLM:
    def __init__(self):
        self.calls = 0

    def generate_batch(self, system_prompt, prompts, max_tokens=200, batch_size=8):
        self.calls += 1
        return [
            '<think>x</think>["When is a tax audit under Section 44AB required?", '
            '"What does clause 44 of Form 3CD report?", "short?"]'
            for _ in prompts
        ]


def test_question_parsing_and_filtering():
    assert parse_questions('Sure: ["What is clause 44 of Form 3CD?"]') == ["What is clause 44 of Form 3CD?"]
    assert parse_questions("1. What is the audit limit under 44AB?\n2) Who signs Form 3CA?") == [
        "What is the audit limit under 44AB?",
        "Who signs Form 3CA?",
    ]
    seen = set()
    kept = filter_questions(
        [
            "What does the passage say about audits?",  # generic
            "too short?",
            "When is a tax audit required under Section 44AB?",
            "when is a tax audit required under section 44AB",  # duplicate
        ],
        LONG,
        seen,
    )
    assert kept == ["When is a tax audit required under Section 44AB?"]


def test_ineligible_chunks_are_skipped():
    assert is_eligible_chunk(chunk("a", 1))
    assert not is_eligible_chunk(chunk("b", 1, text="Too short"))
    assert not is_eligible_chunk(chunk("c", 1, text="Contents ...... 1 ...... 2 ...... 3 ...... 4 " * 10))


def test_query_generation_checkpoints_and_resumes(tmp_path):
    output = tmp_path / "queries.jsonl"
    chunks = [chunk("a", 1), chunk("b", 2, text=LONG + " Penalty applies under section 271B.")]
    llm = BatchLLM()

    first = generate_queries(chunks, llm, output, max_chunks=1)
    assert len(first) == 1 and llm.calls == 1
    # Duplicate questions across chunks are dropped globally.
    second = generate_queries(chunks, llm, output)
    assert [record["chunk_id"] for record in second] == ["a", "b"]
    assert len(second[0]["questions"]) == 2 and second[1]["questions"] == []
    assert llm.calls == 2
    assert len(output.read_text().splitlines()) == 2


def test_split_keeps_each_page_on_one_side_and_is_stable():
    records = [
        {"chunk_id": f"c{i}", "source": "guide.pdf", "page": i // 2, "questions": [f"Question number {i} about audit?"]}
        for i in range(200)
    ]
    train, evaluation = split_pairs(records, eval_fraction=0.2)
    assert train and evaluation
    assert not {(p.source, p.page) for p in train} & {(p.source, p.page) for p in evaluation}
    assert split_pairs(records, eval_fraction=0.2) == (train, evaluation)


def test_hard_negatives_come_from_other_pages():
    chunks = [
        chunk("a", 1),
        chunk("a2", 1, text="Form 3CD clause 44 GST expenditure tax audit details on same page."),
        chunk("b", 2, text="Form 3CD clause 44 GST expenditure tax audit details elsewhere."),
        chunk("c", 3, text="Unrelated text on depreciation of plant and machinery."),
    ]
    pairs = [QueryPair("Form 3CD clause 44 GST expenditure", "a", "guide.pdf", 1)]
    triplets = mine_hard_negatives(pairs, chunks, skip_top=0)
    assert triplets[0]["positive"] == LONG
    assert "elsewhere" in triplets[0]["negative"]


def test_evaluate_pairs_chunk_and_page_level():
    corpus = [chunk("a", 1), chunk("a2", 1), chunk("b", 2)]
    pairs = [QueryPair("q1", "a", "guide.pdf", 1), QueryPair("q2", "b", "guide.pdf", 2)]
    ranking = {"q1": [corpus[1], corpus[0]], "q2": [corpus[0], corpus[1]]}
    metrics = evaluate_pairs(pairs, lambda query, k: ranking[query], ks=(1, 2))
    assert metrics["chunk_recall@1"] == 0.0
    assert metrics["chunk_recall@2"] == 0.5
    assert metrics["page_recall@1"] == 0.5
    assert metrics["chunk_mrr@2"] == 0.25


def test_dense_search_ranks_by_cosine():
    class Provider:
        vectors = {"alpha": [1.0, 0.0], "beta": [0.0, 1.0], "query": [0.9, 0.1]}

        def embed_texts(self, texts):
            return np.array([self.vectors[t] for t in texts], dtype=np.float32)

        def embed_query(self, text):
            return np.array(self.vectors[text], dtype=np.float32)

    corpus = [chunk("b", 2, text="beta"), chunk("a", 1, text="alpha")]
    assert [c["id"] for c in dense_search_fn(Provider(), corpus)("query", 2)] == ["a", "b"]


def test_only_adopted_models_are_found(tmp_path):
    assert find_finetuned_model([tmp_path]) is None
    candidate = tmp_path / "finetune" / "embedding-candidate"
    candidate.mkdir(parents=True)
    (candidate / FINETUNED_MARKER).write_text(json.dumps({"adopted": False}))
    (candidate / "config.json").write_text("{}")
    assert find_finetuned_model([tmp_path]) is None

    previous = tmp_path / "finetune" / "embedding-finetuned"
    previous.mkdir()
    (previous / "old.bin").write_text("old")
    adopted = adopt_finetuned_model(candidate, previous)
    assert adopted == previous and not candidate.exists()
    assert not (previous / "old.bin").exists() and (previous / "config.json").exists()
    assert find_finetuned_model([tmp_path / "missing", tmp_path]) == previous


def test_llm_strips_qwen_thinking_and_batches():
    class Tokenizer:
        padding_side = "right"
        pad_token = None
        eos_token = "<eos>"

    class Pipeline:
        tokenizer = Tokenizer()

        def __call__(self, conversations, batch_size=None, **kwargs):
            assert kwargs["max_new_tokens"] == 50
            return [
                [{"generated_text": conversation + [{"role": "assistant", "content": "<think>plan</think>\n{\"ok\": 1}"}]}]
                for conversation in conversations
            ]

    llm = HuggingFacePipelineLLM(pipeline=Pipeline())
    assert llm.generate_batch("sys", ["one", "two"], max_tokens=50) == ['{"ok": 1}', '{"ok": 1}']
    assert Pipeline.tokenizer.padding_side == "left" and Pipeline.tokenizer.pad_token == "<eos>"
    assert HuggingFacePipelineLLM._extract_text("<think>unfinished reasoning") == ""
