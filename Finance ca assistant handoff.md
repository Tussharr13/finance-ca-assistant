# Next Agent Handoff: Finance CA Assistant

## Project

Repository path:

```text
/Users/tussharsingh/Documents/Projects/rag-work/finance-ca-assistant
```

GitHub repo:

```text
https://github.com/Tussharr13/finance-ca-assistant
```

Current local branch state:

```text
main...origin/main [ahead 1]
```

The ahead commit updates the Kaggle notebook to clone the latest code from GitHub instead of requiring manual zip upload.

## Goal

Build a CA-domain RAG assistant for Indian accounting, audit, GST, tax audit, Form 3CD, and compliance workflows.

Current execution target is Kaggle. Local repo remains the source of truth. Kaggle should act as runtime/GPU environment.

## Architecture

Core package:

```text
finance_ca_assistant/
  agents/
  embeddings/
  evaluation/
  generation/
  ingestion/
  processing/
  retrieval/
  utils/
  config.py
  domain_index.py
  knowledge_base.py
  pipeline.py
```

Important modules:

- `finance_ca_assistant/pipeline.py`: main RAGPipeline entrypoint.
- `finance_ca_assistant/knowledge_base.py`: downloads/processes PDFs, chunks, saves artifacts.
- `finance_ca_assistant/ingestion/pdf_downloader.py`: source PDF download logic.
- `finance_ca_assistant/ingestion/pdf_processor.py`: PDF text extraction.
- `finance_ca_assistant/processing/semantic_chunker.py`: chunking.
- `finance_ca_assistant/processing/clause_extractor.py`: clause/Form 3CD style extraction.
- `finance_ca_assistant/retrieval/hybrid_retriever.py`: hybrid dense/BM25/clause retrieval.
- `finance_ca_assistant/retrieval/bm25_index.py`: sparse retrieval.
- `finance_ca_assistant/retrieval/clause_searcher.py`: clause-aware retrieval.
- `finance_ca_assistant/retrieval/reranker_factory.py`: reranker provider factory.
- `finance_ca_assistant/embeddings/embedding_factory.py`: embedding provider factory.
- `finance_ca_assistant/generation/llm_factory.py`: LLM provider factory.
- `finance_ca_assistant/agents/ca_consultation_agent.py`: conversational/agentic CA consultation layer.
- `finance_ca_assistant/generation/citation_validator.py`: citation validation.
- `finance_ca_assistant/generation/hallucination_detector.py`: invalid reference detection.
- `finance_ca_assistant/utils/hf_auth.py`: Hugging Face token loading from Kaggle Secrets/env.

Notebook:

```text
notebooks/finance_ca_assistant_kaggle_mvp.ipynb
```

## Current Design Decisions

- Hugging Face-first.
- OpenAI providers and OpenAI dependency were removed.
- Kaggle should use `HF_TOKEN` from Kaggle Secrets.
- Dense retrieval uses SentenceTransformers/HF when available.
- Local deterministic/hash fallback exists for tests and CPU smoke safety.
- Retrieval is hybrid: dense + BM25 + clause search.
- Reranking is optional and should degrade gracefully.
- LLM is optional and should degrade gracefully to local/echo behavior for smoke tests.
- Source chunks carry PDF/page/citation metadata.
- Agentic consultation should ask follow-up questions for open-ended CA advisory scenarios.

## Kaggle Workflow

The notebook Section 1 should:

1. Install runtime dependencies.
2. Remove stale `/kaggle/working/finance-ca-assistant`.
3. Clone:

```text
https://github.com/Tussharr13/finance-ca-assistant.git
```

4. Add the cloned repo to `sys.path`.
5. Import `RAGPipeline`.

After every local code update:

1. Commit locally.
2. Push to GitHub.
3. Rerun Kaggle Section 1 to pull latest code.

## Current Blocking Issue

Kaggle PDF processing is consuming too much CPU/RAM and can fail.

Root cause:

- Section 3 currently processes all default PDFs with `MAX_PAGES_PER_PDF = None`.
- PDF parsing/chunking is CPU-bound; GPU will stay idle.
- Large ICAI/CBIC PDFs cause RAM/CPU spikes.
- Rebuilding the corpus every run is wasteful.

Immediate required fix:

- Make Section 3 safe by default.
- Add artifact reuse: load existing `chunks.jsonl` if present.
- Add limits:
  - `SOURCE_LIMIT`
  - `MAX_PAGES_PER_PDF`
  - possibly `MAX_CHUNKS`
- Avoid all-PDF full rebuild in Kaggle by default.
- Prefer prebuilt artifacts when present in `/kaggle/working` or `/kaggle/input`.

Recommended default for Kaggle MVP:

```python
SOURCE_LIMIT = 3
MAX_PAGES_PER_PDF = 30
REBUILD_KB = False
```

If no cached chunks exist, build a small corpus first.

## Important Files To Modify Next

Likely targets:

```text
notebooks/finance_ca_assistant_kaggle_mvp.ipynb
finance_ca_assistant/knowledge_base.py
finance_ca_assistant/processing/semantic_chunker.py
finance_ca_assistant/ingestion/pdf_processor.py
finance_ca_assistant/config.py
tests/test_kaggle_notebook.py
tests/test_knowledge_base.py
```

## Validation Commands

Run before finalizing changes:

```bash
python3 -m pytest -q
python3 -m json.tool notebooks/finance_ca_assistant_kaggle_mvp.ipynb >/tmp/notebook_valid.json
```

Expected current test status before new edits:

```text
20 passed
```

## Git Notes

Local repo exists and has commits:

```text
eeb5407 Initial CA RAG assistant
569fe79 Fetch project from GitHub in Kaggle notebook
```

Terminal push failed earlier because GitHub auth was not configured. User said they will push manually.

Do not assume GitHub remote is up to date until user confirms push or you verify with network access.

## Security Notes

- User previously pasted API tokens in conversation/IDE context.
- Treat exposed OpenAI/HF tokens as compromised.
- Never commit `.env`, raw keys, or Kaggle/HF tokens.
- `.gitignore` excludes env files, raw PDFs, generated data, model/vector artifacts, caches.

## Constraints

- Kaggle CPU/RAM is limited.
- GPU helps HF model inference, not PDF parsing.
- Internet must be enabled in Kaggle for GitHub clone, PDF download, and model downloads.
- If Kaggle cannot access GitHub/private repo, make repo public temporarily or use Kaggle Secrets/token-based clone.
- Keep notebook cells small and restart-safe.
- Avoid huge embedded/base64 transfer cells in notebooks.
- Avoid manual upload workflow unless GitHub is unavailable.

## Assumptions

- Local repo is source of truth.
- Kaggle is runtime only.
- MVP should prove retrieval quality and CA consultation behavior before production deployment.
- Enterprise architecture pieces are scaffolded, not production-complete.

## TODOs

High priority:

- Fix Kaggle Section 3 CPU/RAM failure.
- Add cached artifact loading path.
- Add Kaggle-safe limits by default.
- Commit and have user push latest changes to GitHub.
- Rerun Kaggle from clean session.

Medium priority:

- Persist and reload vector/embedding artifacts.
- Add small evaluation run in notebook.
- Improve Form 3CD clause indexing.
- Add examples for salary tax planning, GST compliance, audit qualifications, accounting standards.
- Improve CA consultation follow-up question policy.

Later:

- FastAPI deployment.
- Production vector DB backend.
- OCR/table extraction.
- Better citation support scoring.
- Evaluation dashboard.
- API auth/logging/rate limits.

