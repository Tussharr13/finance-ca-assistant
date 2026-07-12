# Finance CA Assistant

Kaggle-first MVP for a Chartered Accountant domain RAG assistant.

## Structure

```text
finance-ca-assistant/
  finance_ca_assistant/   # Python package, importable with underscores
  notebooks/              # Kaggle MVP notebook
  scripts/                # Local smoke/test helpers
  tests/                  # Focused unit tests
  requirements.txt
  setup.py
```

The outer folder uses hyphens because it is the project name. The inner package
uses underscores because Python imports cannot use hyphens.

## Quick Check

```bash
python3 scripts/run_pipeline_smoke.py
python3 -m pytest tests -q
```

## Kaggle

Start with:

```text
notebooks/finance_ca_assistant_kaggle_mvp.ipynb
```

The notebook now does the real MVP flow:

1. Locates the `finance_ca_assistant` package.
2. Loads `HF_TOKEN` from Kaggle Secrets when present.
3. Reuses cached chunks or builds a bounded ICAI/CBIC corpus with safe defaults.
4. Extracts text and creates CA-aware chunks without building throwaway embeddings.
5. Builds the hybrid retrieval pipeline once with staged Hugging Face providers.
6. Tests AS 1 retrieval against the bounded corpus.
7. Runs an agentic CA consultation turn.
8. Saves chunks, embeddings, and clause index artifacts under Kaggle working.

In Kaggle, add your Hugging Face token as a Secret named exactly `HF_TOKEN`.
Do not paste the token into notebook code. Section 3 defaults to three sources,
30 pages per PDF, 300 chunks per source, and `REBUILD_KB = False`. Delete the
chunk cache or opt into rebuilding only when sources or ingestion settings change.
PDF text extraction prefers PyMuPDF and clears its document cache every five
pages to keep Kaggle RAM bounded; `pypdf` is retained only as a compatibility
fallback.

## Hugging Face Model Stack

The production-oriented local/Kaggle stack is wired as optional providers. In
the Kaggle notebook, Section 4 uses a staged default so retrieval can pass
before loading heavier models:

```python
MAX_CHUNKS_PER_SOURCE = None
ENABLE_HF_EMBEDDINGS = GPU_AVAILABLE
ENABLE_HF_RERANKER = False
ENABLE_HF_LLM = False

HF_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
HF_RERANK_MODEL = "BAAI/bge-reranker-base"
HF_LLM_MODEL = "Qwen/Qwen3-0.6B"
```

After Section 4 passes, increase quality one step at a time:

```python
ENABLE_HF_RERANKER = True
ENABLE_HF_LLM = True
HF_LLM_MODEL = "Qwen/Qwen3-4B"
```

Use `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`, or `Qwen/Qwen3-8B` only after the
small stack is stable on the selected Kaggle GPU.

## Agentic Consultation

Use the consultation wrapper when the user asks for advisory workflows rather
than a single fact lookup:

```python
consultation = pipeline.consult(
    "I am earning salary and want to save tax legally. Act like my CA.",
    top_k=6,
)
print(consultation.status)
print(consultation.message)
```

For salary tax planning, the agent first asks for missing client facts such as
CTC, age, residential status, regime preference, HRA/rent, deductions, insurance,
other income, and Form 16/TDS details. It only moves to advice once the required
facts are available.
