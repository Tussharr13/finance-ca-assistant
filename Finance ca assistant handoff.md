# Next Agent Handoff: Finance CA Assistant

**Last updated:** 28 October 2026

Start here, then read:

- `docs/architecture-and-roadmap.md`: current state, limitations, roadmap checklist.
- `decisionsForCA.md`: every design decision (D-01 onwards) with its reasons and trade-offs.

## Project

Repository path:

```text
/Users/tussharsingh/Documents/Projects/rag-work/finance-ca-assistant
```

GitHub repo:

```text
https://github.com/Tussharr13/finance-ca-assistant
```

## Goal

Build a CA-domain assistant for Indian accounting, audit, GST, tax audit, Form 3CD,
income tax and compliance workflows, growing into an evidence-grounded financial
decision-support system (see roadmap section 1).

The local repo is the source of truth. Kaggle is the GPU runtime for corpus builds,
embedding fine-tuning and LLM runs (D-01, D-17).

## Architecture

Core package:

```text
finance_ca_assistant/
  agents/        consultation agent (salary-tax workflow, fact gathering)
  calculators/   income-tax regime calculator and effective-dated rules
  embeddings/    embedding providers and cache
  evaluation/    golden questions and retrieval metrics
  finetuning/    synthetic queries, embedding training, held-out evaluation
  generation/    LLM providers, quoted-claim generator, typed claims
  ingestion/     PDF download, extraction, source registry
  processing/    chunking, clause extraction, metadata
  retrieval/     dense, BM25, clause search, RRF fusion, reranker
  pipeline.py    RAGPipeline entrypoint
server/          FastAPI app, SQLite conversation store, pipeline loader
web/             Next.js chat client
notebooks/       Kaggle MVP notebook
```

How answers are produced:

- **Salary tax:** the agent asks for the facts the calculation needs, runs
  `compare_regimes`, and builds the reply in code (D-18, D-19). With an LLM, typed
  claims explain the result; invalid claims are dropped (D-23, D-24).
- **General questions:** hybrid retrieval (dense + BM25 + clause, weighted RRF,
  D-05), then paraphrased claims each backed by a whole-sentence quote. One failed
  claim abstains the whole answer (D-25).
- **No LLM configured:** calculations still work; general questions abstain.

## Running

Tests:

```bash
python3 -m pytest tests -q
python3 -m json.tool notebooks/finance_ca_assistant_kaggle_mvp.ipynb >/dev/null
```

Status on 28 October 2026: 144 tests; 138 pass. The 6 in `tests/test_server.py`
fail on this machine because the system Python has fastapi 0.104, whose starlette
`TestClient` is incompatible with httpx 0.28. `server/requirements.txt` asks for
fastapi ≥ 0.115; install it to run them.

Web chat: see the README. The server needs `FINANCE_CA_ARTIFACTS_DIR` (Kaggle
artifacts) for retrieval and `FINANCE_CA_LLM_MODEL` for an LLM; without them it
runs calculator-only (D-29).

Retrieval benchmark (no Kaggle needed; downloads 3 PDFs and bge-small):

```bash
python3 scripts/build_corpus.py /tmp/ca-artifacts
python3 scripts/benchmark_retrieval.py --artifacts /tmp/ca-artifacts
```

The gate must pass before any retrieval change is committed (D-30). Rewrite the
baseline with `--write-baseline` only when the golden set changes, and say so in
the commit. On Kaggle, notebook Section 6.1 runs the same gate on the Section 4
pipeline; the user's notebook is `kaggle.com/code/tussharsingh13/yourca`.

## Kaggle workflow

The notebook clones the repo from GitHub in Section 1, so after every local change:

1. Commit locally.
2. Push to `main` (D-17).
3. Rerun Kaggle Section 1 to pull the latest code.

Safe defaults are already in the notebook: `SOURCE_LIMIT = 3`,
`MAX_PAGES_PER_PDF = 30`, `MAX_CHUNKS_PER_SOURCE = 300`, `REBUILD_KB = False`,
`ENABLE_HF_LLM = False`. Cached chunks, embeddings and clause index are reused (D-03).

## Security notes

- The user previously pasted API tokens in conversation/IDE context. Treat any
  exposed OpenAI/HF tokens as compromised.
- Never commit `.env`, raw keys, or Kaggle/HF tokens.
- `.gitignore` excludes env files, raw PDFs, generated data, model/vector artifacts,
  caches and `data/server/` (chat database with personal financial data).
- Typed claims send client facts to the LLM; only self-hosted models are acceptable
  until redaction exists (D-23).
- The web server has no authentication or rate limits: local use only (D-26).

## Constraints

- Kaggle CPU/RAM is limited; PDF parsing is CPU-bound and the GPU will not help it.
- Internet must be enabled in Kaggle for GitHub clone, PDF download and model downloads.
- Keep notebook cells small and restart-safe; avoid large embedded/base64 cells.

## Next work

The authoritative list is the roadmap checklist. In priority order:

1. **Use the benchmark (D-30).** 52 labelled questions and the first baseline
   are committed. Fusion does not beat BM25 on it (page MRR@10: BM25 0.88,
   hybrid 0.86, dense 0.82), so tune the RRF weights against the golden set,
   then try the reranker, gated. Run the gate on Kaggle once (PyMuPDF chunks
   differ from the local pypdf build) and rewrite the baseline there if needed.
2. **Fix the reference extractor** (`domain_index.extract_references`): it
   truncates `44AB` to `44a`, files Income-tax Rule 6G as `gst_rule:6g`, labels
   CGST Act sections as `income_tax_section`, and ignores Form 3CD clauses. Then
   add section labels to the tax-audit questions and rewrite the baseline.
3. **Measure real-model claim acceptance** for D-23 and D-25 on Kaggle with Qwen3-4B.
4. Persist and reload the BM25 index; verify legal effective dates from primary sources.
5. Show the stored client profile in the web chat and let users correct it (D-27).
6. More calculators (TDS, capital gains, GST) and specialist workflows.
