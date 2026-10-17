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

## Decisions

Every design and engineering decision, with its reasons and trade-offs, is
logged in `decisionsForCA.md`. Add an entry with each change.

## Architecture and Roadmap

The target product architecture, verified current state, known limitations, and
phased implementation roadmap are maintained in:

```text
docs/architecture-and-roadmap.md
```

Treat it as a living document and update it as capabilities are implemented and
validated.

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
The Kaggle setup cell installs only missing ingestion packages with `--no-deps`
so it does not re-resolve or upgrade Kaggle's preinstalled CUDA/RAPIDS stack.

For normal runs, publish `/kaggle/working/finance_ca_assistant_artifacts` from a
successful saved version as a Kaggle Dataset and attach it to the notebook.
Section 2 discovers an attached `chunks.jsonl` and its
`chunks.manifest.json`, then copies both into the writable cache. Section 3
returns immediately only when the source catalogue and ingestion settings
match the manifest fingerprint. Legacy artifacts without a manifest rebuild
once instead of being trusted silently.

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

## Backend retrieval behavior

Answer generation enforces the [citation contract](docs/citation-enforcement.md).
Each statement is a paraphrase backed by a verbatim quote of complete sentences;
code checks negations, conditions, numbers and wording against the quote, and any
failed statement makes the answer abstain. Confidence is `not_assessed`.
Default EchoLLM runs still retrieve evidence but abstain from generating answers.

Hybrid search uses weighted reciprocal rank fusion (`weight / (60 + rank)`)
instead of adding incompatible cosine, BM25, and clause scores. These scores
indicate ranking agreement, not answer confidence. No reranker is enabled
implicitly; pass an explicit reranker to the pipeline to enable one.

`pipeline.answer(..., metadata_filter={"jurisdiction": "IN"})` and
`pipeline.retrieve_chunks(..., metadata_filter={"source": "source-id"})`
apply exact matching in all three channels before candidate limits. Every
specified field must be present and match. This does not yet implement date
ranges, legal freshness checks, or tenant authorization.

The notebook integration and live Kaggle verification follow backend work.

## Retrieval Fine-tuning

Notebook Sections 7-10 fine-tune the embedding model on your own corpus:

1. Qwen3 generates CA search questions per chunk (resumable checkpoint in
   `finetune/queries.jsonl`), split into train and eval by page.
2. The current model is scored on held-out pages: recall@k and MRR, dense-only
   and through the full hybrid pipeline.
3. `bge-small` is trained on (question, passage, near-miss passage) triplets.
4. The new model is adopted only if hybrid page MRR improves without losing
   recall@5; scores are written to `finetune/report.json`.

Republish `finance_ca_assistant_artifacts` as a Kaggle Dataset after an adopted
run; Section 4 loads the adopted model automatically when it is attached. The
code lives in `finance_ca_assistant/finetuning/`.

## Tax Calculators

Numbers in answers come from deterministic calculators, never from the LLM.
`compare_regimes` computes old and new regime tax for a salaried individual:

```python
from finance_ca_assistant.calculators import SalaryTaxInput, compare_regimes

result = compare_regimes(SalaryTaxInput(
    tax_year="FY2025-26", gross_salary=2_000_000, basic_da=1_000_000, age=30,
    hra_received=400_000, rent_paid=360_000, metro=True, deduction_80c=150_000,
))
print(result.lower_regime, result.saving, result.run_id)
```

Each step records the provision it relies on, and the result lists what the
calculator does not model. Rules live in effective-dated tables in
`calculators/tax_rules.py`; only FY 2025-26 is verified. Other years raise
`UnsupportedTaxYear` until their tables are added and checked, including the
Income-tax Act, 2025 years from 1 April 2026.

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

For salary tax planning, the agent first asks for the facts the calculation
needs: gross salary, basic + DA, age, residency, rent (and HRA and city if rent
is paid), and the amounts of any deductions, health insurance and other income
the client has. Pass `consultation.profile_updates` back as `profile` on the
next turn. Once the facts are complete, the reply has status `calculated` and is
built from the regime calculator, not the LLM: both regimes side by side, the
deductions applied with their sections, assumptions, and what is not included.
The full result is on `consultation.calculation`.
