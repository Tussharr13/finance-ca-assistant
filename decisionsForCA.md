# Decisions for the Finance CA Assistant

A running log of design and engineering decisions: what was decided, why, what
was built, and what it costs. Add a new entry for every decision that changes
behaviour, architecture, data, or process. Never rewrite an old entry; if a
decision is reversed, add a new entry and mark the old one **Superseded by D-xx**.

**Entry format:** Context → Decision → Why → Implemented → Trade-offs → Revisit when.

**Status values:** Accepted · Proposed (agreed direction, not built) · Superseded.

## Index

| ID | Date | Decision | Status |
|---|---|---|---|
| [D-01](#d-01-kaggle-is-the-runtime-the-repo-is-the-source-of-truth) | 2026-07-07 | Kaggle is the runtime, the repo is the source of truth | Accepted |
| [D-02](#d-02-hugging-face-first-model-stack) | 2026-07-07 | Hugging Face-first model stack | Accepted |
| [D-03](#d-03-bounded-cached-corpus-builds-on-kaggle) | 2026-07-12 | Bounded, cached corpus builds on Kaggle | Accepted |
| [D-04](#d-04-answers-are-exact-extracts-or-abstain) | 2026-10-16 | Answers are exact extracts or abstain | Superseded by D-25 |
| [D-05](#d-05-weighted-reciprocal-rank-fusion-for-hybrid-search) | 2026-10-16 | Weighted reciprocal rank fusion for hybrid search | Accepted |
| [D-06](#d-06-metadata-filters-apply-before-candidate-limits) | 2026-10-16 | Metadata filters apply before candidate limits | Accepted |
| [D-07](#d-07-rule-based-client-fact-parsing-for-now) | 2026-10-16 | Rule-based client fact parsing, for now | Accepted |
| [D-08](#d-08-normalise-units-at-parse-time) | 2026-10-16 | Normalise money units at parse time | Accepted |
| [D-09](#d-09-typed-claims-answer-contract) | 2026-10-16 | Typed-claims answer contract | Accepted for consultations (D-23) |
| [D-10](#d-10-target-architecture-for-web-and-mobile) | 2026-10-16 | Target architecture for web and mobile | Proposed |
| [D-11](#d-11-tax-numbers-come-from-deterministic-calculators) | 2026-10-16 | Tax numbers come from deterministic calculators | Accepted |
| [D-12](#d-12-tax-rules-are-effective-dated-data-only-verified-years-run) | 2026-10-16 | Tax rules are effective-dated data; only verified years run | Accepted |
| [D-13](#d-13-fine-tune-the-embedding-model-first) | 2026-10-16 | Fine-tune the embedding model first | Accepted |
| [D-14](#d-14-synthetic-training-questions-split-by-page) | 2026-10-16 | Synthetic training questions, split by page | Accepted |
| [D-15](#d-15-a-fine-tuned-model-must-earn-adoption) | 2026-10-16 | A fine-tuned model must earn adoption | Accepted |
| [D-16](#d-16-strip-qwen3-reasoning-from-all-llm-output) | 2026-10-16 | Strip Qwen3 reasoning from all LLM output | Accepted |
| [D-17](#d-17-commit-small-push-to-main-so-kaggle-can-pull) | 2026-10-16 | Commit small, push to main so Kaggle can pull | Accepted |
| [D-18](#d-18-salary-consultation-answers-with-the-calculator-not-the-llm) | 2026-10-16 | Salary consultation answers with the calculator, not the LLM | Accepted |
| [D-19](#d-19-ask-only-for-facts-the-calculation-needs) | 2026-10-16 | Ask only for facts the calculation needs | Accepted |
| [D-20](#d-20-default-to-the-latest-verified-tax-year-and-say-so) | 2026-10-16 | Default to the latest verified tax year, and say so | Accepted |
| [D-21](#d-21-calculation-ids-hash-canonical-figures) | 2026-10-16 | Calculation IDs hash canonical figures | Accepted |
| [D-22](#d-22-follow-up-answers-continue-the-consultation) | 2026-10-16 | Follow-up answers continue the consultation | Accepted |
| [D-23](#d-23-typed-claims-explain-the-calculation-never-replace-it) | 2026-10-16 | Typed claims explain the calculation, never replace it | Accepted |
| [D-24](#d-24-numbers-in-claims-must-match-cited-evidence) | 2026-10-16 | Numbers in claims must match cited evidence | Accepted |
| [D-25](#d-25-general-answers-are-paraphrases-backed-by-whole-sentence-quotes) | 2026-10-17 | General answers are paraphrases backed by whole-sentence quotes | Accepted |

---

## D-01 Kaggle is the runtime, the repo is the source of truth

**Date:** 2026-07-07 · **Status:** Accepted · **Commits:** `569fe79`, `8c56ea5`

**Context.** The project needs free GPU time for embeddings and LLMs, but code
edited inside notebooks drifts and is hard to test.

**Decision.** All code lives in the `finance_ca_assistant` package in this repo.
The Kaggle notebook only clones the latest commit from GitHub and calls the package.

**Why.** Code stays testable locally (pytest), versioned, and reviewable. Kaggle
provides GPU and internet only.

**Implemented.** Notebook Section 1 clones the repo, removes stale imported
modules, and prints the loaded commit hash.

**Trade-offs.** Every change must be pushed before Kaggle sees it. A notebook
run depends on GitHub being reachable and the repo being public (or a token).

**Revisit when.** The backend moves to a hosted service (Phase 2 of D-10).

---

## D-02 Hugging Face-first model stack

**Date:** 2026-07-07 · **Status:** Accepted · **Commit:** `eeb5407`

**Context.** Early versions used OpenAI providers.

**Decision.** Use open models through Hugging Face: `bge-small-en-v1.5`
embeddings, `bge-reranker-base` reranker, and `Qwen3` (0.6B to 8B) for generation.
Each is optional and falls back to a local deterministic provider.

**Why.** Runs free on Kaggle GPUs, models can be fine-tuned (D-13), client data
need not leave our infrastructure, and tests run without any model.

**Implemented.** `embeddings/embedding_factory.py`, `retrieval/reranker_factory.py`,
`generation/llm_factory.py`; the hash embedder and `EchoLLM` keep CI offline.

**Trade-offs.** Small open models are weaker than hosted frontier models at
following strict output formats, which D-04 makes visible as abstentions.

**Revisit when.** Answer quality is measured; a hosted model may be added behind
the model gateway in D-10, with personal data redacted first.

---

## D-03 Bounded, cached corpus builds on Kaggle

**Date:** 2026-07-12 to 2026-08-30 · **Status:** Accepted ·
**Commits:** `ac67f16`, `0cb9d57`, `2e4577d`, `87f85c2`, `a50edea`

**Context.** Parsing every ICAI/CBIC PDF in one Kaggle session exhausted CPU and RAM.

**Decision.** Build a bounded corpus by default (3 sources, 30 pages per PDF,
300 chunks per source), cache chunks with a fingerprint manifest, and reuse
artifacts published as a Kaggle Dataset.

**Why.** PDF parsing is CPU-bound (the GPU does not help), and rebuilding the
same corpus every session wastes the session.

**Implemented.** PyMuPDF with periodic cache clearing, `chunks.manifest.json`
fingerprinting sources and settings, attached-dataset discovery in Section 2,
and saved embeddings reloaded through `RAGPipeline.from_artifacts`.

**Trade-offs.** The default corpus is small, which limits retrieval coverage and
the size of the fine-tuning data (D-14).

**Revisit when.** Retrieval quality is good on the small corpus; then raise the limits.

---

## D-04 Answers are exact extracts or abstain

**Date:** 2026-10-16 · **Status:** Superseded by D-25 ·
**Commit:** `ec13098` · **Detail:** `docs/citation-enforcement.md`

**Context.** The earlier generator forwarded free-form model text with a
confidence label derived from the number of retrieved chunks.

**Decision.** The model may only return claims whose text equals a whole
retrieved chunk. Any invalid claim makes the whole answer abstain. Confidence
is always `not_assessed`.

**Why.** Quote matching is a check code can enforce. Requiring the whole chunk
stops a cropped quote from dropping a word such as "No".

**Implemented.** `generation/ca_answer_generator.py`,
`CitationValidator.validate_extract`, consultation status propagation.

**Trade-offs.** It cannot synthesise, compare, or explain, so it cannot deliver
advice. Small models rarely copy long chunks exactly, so most real answers will
abstain. Copying chunks also costs many output tokens.

**Revisit when.** D-09 is built.

---

## D-05 Weighted reciprocal rank fusion for hybrid search

**Date:** 2026-10-16 · **Status:** Accepted · **Commit:** `ec13098`

**Context.** Hybrid search added cosine, BM25 and clause scores, which are on
different scales, so one channel could dominate arbitrarily.

**Decision.** Fuse the dense, BM25 and clause channels by rank: each channel
contributes `weight / (60 + rank)`. A reranker runs only when explicitly passed in.

**Why.** Ranks are comparable across channels, and scores are not. RRF is a
standard, robust default with no tuning.

**Implemented.** `retrieval/hybrid_retriever.py`; weights 0.45 / 0.35 / 0.20;
input validation for weights and `rrf_k`.

**Trade-offs.** Discards score magnitude (a very strong BM25 match counts the
same as a weak first place). Fused scores show ranking agreement, not confidence.

**Revisit when.** The evaluation in D-15 can compare fusion settings.

---

## D-06 Metadata filters apply before candidate limits

**Date:** 2026-10-16 · **Status:** Accepted · **Commit:** `ec13098`

**Decision.** `metadata_filter` (exact match on every field) is applied inside
all three channels before taking the top candidates.

**Why.** Filtering after the top-k cut can return nothing even when matching
chunks exist.

**Trade-offs.** No ranges yet, so no date-effective filtering (needed by D-12)
and no tenant authorisation.

**Revisit when.** Chunks carry effective dates.

---

## D-07 Rule-based client fact parsing, for now

**Date:** 2026-10-16 · **Status:** Accepted · **Commits:** `a4f61f9`, `2e279e4`

**Context.** The salary consultation needs 11 facts. The parser only filled 4,
treated "non-resident" as resident, and read "1,200,000" as 1.

**Decision.** Keep regex parsing but harden it: label-to-amount gaps may contain
only filler words, a shared negation pattern, residency checked most-specific
first, a bounded city capture, and other income excluded from salary.

**Why.** It unblocks the consultation flow now with no model dependency, and
every behaviour is covered by tests.

**Implemented.** `_infer_profile_updates` and helpers in
`agents/ca_consultation_agent.py`, with regression tests. A `/code-review` pass
found 10 further gaps after the first fix; all were fixed in `2e279e4`.

**Trade-offs.** Regex parsing of free text is brittle and every fix exposes new
phrasings. A wrong parse silently feeds a wrong number into tax advice.

**Revisit when.** Phase 2: replace it with LLM extraction into typed fields that
the user confirms before they are saved (D-10).

---

## D-08 Normalise money units at parse time

**Date:** 2026-10-16 · **Status:** Accepted · **Commit:** `2e279e4`

**Decision.** Salary is stored per year; rent and HRA are stored per month. The
field `hra_received` was renamed `monthly_hra` so the unit is in the name.
Lakh shorthand (`20L`, `20 lpa`, `12,00,000`) is parsed.

**Why.** Mixed units would produce wrong HRA exemptions in the calculator (D-11).

**Trade-offs.** The calculator takes annual figures, so the adapter must
multiply monthly values by 12.

---

## D-09 Typed-claims answer contract

**Date:** 2026-10-16 · **Status:** Proposed

**Context.** D-04 is safe but cannot compare regimes or explain a plan.

**Decision.** Every sentence of an answer is a typed claim:
`fact` (references profile fields), `calculation` (references a calculator run
ID, and its figure must equal the output), `rule` (references retrieved chunk
IDs in force for the year), or `recommendation` (references at least one fact
and one calculation or rule). Code validates the references; failed claims are
dropped, and a failed required claim makes the answer abstain.

**Why.** The model writes the language; code verifies the support. This allows
synthesis without letting the model invent law or numbers.

**Trade-offs.** More complex validation; numbers stated in text must be matched
against calculator output; requires D-11 to be wired into the consultation.

**Revisit when.** Built; then mark D-04 superseded.

---

## D-10 Target architecture for web and mobile

**Date:** 2026-10-16 · **Status:** Proposed ·
**Detail:** https://claude.ai/artifact/7ZeY19sbpwwrG8iLeES1sv

| Choice | Why | Alternative rejected |
|---|---|---|
| Python FastAPI monolith with workers | Reuses this package; little to operate | Microservices from day one |
| Next.js web + React Native (Expo) mobile in one monorepo | Shared TypeScript types and API client generated from OpenAPI | Flutter (shares nothing with web) |
| Explicit state machine for orchestration | Predictable and auditable | Open-ended agent loop |
| Postgres + pgvector + full-text | One store, transactions | Separate vector DB |
| Server-sent events for chat | Works on web and mobile, proxy-friendly | WebSockets |
| India-region hosting | Data residency, DPDP Act | Multi-region |

**Regulatory constraints recorded:** personalised investment advice needs SEBI
investment-adviser registration; filing returns for users needs e-Return
Intermediary registration; automatic bank data goes through RBI's Account
Aggregator framework; DPDP Act consent, export and deletion.

**Trade-offs.** A monolith must be split later if GPU inference or ingestion
loads diverge. React Native has less native polish than fully native apps.

---

## D-11 Tax numbers come from deterministic calculators

**Date:** 2026-10-16 · **Status:** Accepted · **Commit:** `6c080f4`

**Decision.** Old vs new regime tax for a salaried individual is computed in
pure Python. Each step records the provision it relies on. Each run has a stable
ID (hash of inputs, rule year and calculator version). Results list what the
calculator does not model.

**Why.** LLMs are unreliable at arithmetic; a calculation must be reproducible
and citable (D-09 references run IDs).

**Implemented.** `calculators/income_tax.py`: HRA (10(13A)), standard deduction,
professional tax, 80C, 80D (senior limits), 80CCD(1B), 80CCD(2) (10% old / 14% new),
24(b), 87A rebate with new-regime marginal relief, surcharge with marginal relief,
4% cess, 288A/288B rounding, senior slabs for residents only. 13 tests with
hand-computed figures.

**Trade-offs.** Covers salary income only. Not modelled: capital gains and other
special-rate income, 80TTA/80TTB, other Chapter VI-A deductions, LTA and
perquisites, let-out property and losses, the ₹7.5 lakh combined retirement
contribution cap. Rates were written from knowledge and **have not been checked
against the Finance Act 2025 text**.

**Revisit when.** Before any user relies on it: verify the rates with a CA.
Connected to the consultation in D-18.

---

## D-12 Tax rules are effective-dated data; only verified years run

**Date:** 2026-10-16 · **Status:** Accepted · **Commit:** `6c080f4`

**Context.** The Income-tax Act, 2025 replaced the 1961 Act from 1 April 2026
and renumbers sections; FY 2025-26 returns still follow the old Act.

**Decision.** Each tax year is an immutable rule table in
`calculators/tax_rules.py`. Only FY 2025-26 exists. Any other year raises
`UnsupportedTaxYear` instead of reusing the nearest table.

**Why.** Silently applying the wrong year's law is worse than refusing.

**Trade-offs.** FY 2026-27 questions cannot be answered until its table is added
and verified. The same effective-dating is still missing from the knowledge
corpus (D-06).

---

## D-13 Fine-tune the embedding model first

**Date:** 2026-10-16 · **Status:** Accepted · **Commit:** `ca07d1b`

**Context.** Asked to fine-tune "our model". Three candidates: embeddings, reranker, LLM.

**Decision.** Fine-tune `bge-small-en-v1.5` first.

**Why.** Retrieval quality is the current phase goal; the gain is directly
measurable (recall@k, MRR); training takes minutes on a T4; and every downstream
answer depends on retrieving the right page.

**Trade-offs.** Does not improve answer wording or format compliance. LLM LoRA
fine-tuning (to follow the answer format) is deferred and can reuse the same
question data. The reranker is off, so it was not a useful first target.

---

## D-14 Synthetic training questions, split by page

**Date:** 2026-10-16 · **Status:** Accepted · **Commit:** `ca07d1b`

**Context.** No training data existed: the golden set has 2 questions.

**Decision.**
- Qwen3-4B writes up to 3 questions per eligible chunk (skipping short, noisy and
  table-of-contents chunks); generic, copied and duplicate questions are dropped.
- Generation checkpoints to `finetune/queries.jsonl` after every batch and resumes.
- Train/eval split by source page with a stable hash (15% eval).
- Hard negatives: BM25 near-misses from a different page, skipping the top 2.
- Loss: MultipleNegativesRankingLoss with a no-duplicates batch sampler.

**Why.** Page-level split measures generalisation to unseen pages, not memory.
Skipping the top BM25 hits reduces false negatives (unlabelled correct answers).
Checkpointing survives Kaggle disconnects.

**Trade-offs.** Synthetic questions may echo the passage's wording and
overstate scores; they reflect the generator's idea of a CA question, not real
users'. The small corpus (D-03) limits the eval set to a few hundred questions.

**Revisit when.** Real user questions exist; build a human-checked golden set.

---

## D-15 A fine-tuned model must earn adoption

**Date:** 2026-10-16 · **Status:** Accepted · **Commit:** `ca07d1b`

**Decision.** Train into `finetune/embedding-candidate`. Adopt only if the full
hybrid pipeline's page MRR@10 improves by at least 0.01 and page recall@5 does
not drop. Adoption marks the model `adopted: true` and moves it to
`finetune/embedding-finetuned`; rejected candidates are deleted. Section 4 loads
only adopted models. Scores are always written to `finetune/report.json`.

**Why.** Fine-tuning can make retrieval worse. A separate candidate folder means
a rerun never overwrites the model in use, and an interrupted run is never
loaded by mistake.

**Implemented.** `finetuning/embedding_trainer.py` (`adopt_finetuned_model`,
`find_finetuned_model`), notebook Sections 7–10. Section 4 now rebuilds
embeddings instead of crashing when the saved model differs.

**Trade-offs.** The 0.01 threshold is a judgement call and is noisy on small
eval sets. Adoption judges the hybrid pipeline, so an embedding gain hidden by
BM25 may be rejected.

---

## D-16 Strip Qwen3 reasoning from all LLM output

**Date:** 2026-10-16 · **Status:** Accepted · **Commit:** `ca07d1b`

**Context.** Qwen3 chat templates emit `<think>…</think>` by default. The D-04
generator parses raw output as JSON, so every answer would abstain.

**Decision.** `HuggingFacePipelineLLM` removes reasoning blocks (including an
unterminated one) from every response. Prompts that need speed add `/no_think`.
Batched generation (`generate_batch`, left padding) was added for throughput.

**Trade-offs.** The reasoning text is discarded rather than logged, which loses
debugging signal.

---

## D-17 Commit small, push to main so Kaggle can pull

**Date:** 2026-10-16 · **Status:** Accepted

**Decision.** Each change is a separate commit with tests passing, pushed
straight to `main`.

**Why.** Kaggle clones `main` (D-01); a solo project gets little from PR overhead.

**Trade-offs.** No review gate before code reaches Kaggle. `/code-review` is run
on demand instead.

**Revisit when.** A second contributor joins, or the backend is deployed for users.

---

## D-18 Salary consultation answers with the calculator, not the LLM

**Date:** 2026-10-16 · **Status:** Accepted · **Commit:** see git log "Answer salary consultations with the regime calculator"

**Context.** Once all facts were gathered, the consultation sent the whole client
profile (salary, age, rent) into retrieval and the LLM, and the D-04 contract
then abstained, so clients never got an answer.

**Decision.** When the facts are complete, the agent maps the profile to
`SalaryTaxInput`, runs `compare_regimes`, and builds the reply in code: a
side-by-side table, the deductions applied with their provisions, every
assumption made, what the calculator does not include, the evidence to keep,
and a CA-confirmation line. Status is `calculated`; the full result is on
`AgentResponse.calculation`. Retrieval uses a fixed query with no client facts.

**Why.** Every figure is reproducible and cites its provision; no personal data
goes to retrieval or a model; clients get a real answer today, before D-09 exists.

**Implemented.** `_salary_tax_input`, `_build_calculation_message`, `_inr`
(Indian digit grouping) in `agents/ca_consultation_agent.py`. New profile fields
for basic + DA, 80C, 80D (self, parents, parents' age), NPS, home-loan
interest, other income, professional tax and tax year, each parsed from chat.

**Trade-offs.** The reply is templated, not conversational, and does not explain
*why* one regime wins beyond the figures. CTC is used as gross salary, which
overstates income when CTC includes employer PF or gratuity (stated as an
assumption in every reply). The new amount parsing extends the brittle regex
approach of D-07.

**Revisit when.** D-09 lets an LLM explain the result in natural language
using typed claims that reference this calculation's run ID.

---

## D-19 Ask only for facts the calculation needs

**Date:** 2026-10-16 · **Status:** Accepted

**Decision.** Required facts are: gross salary, basic + DA, age, residency and
rent; city and HRA only if rent is paid; whether the client has deductions,
health insurance and other income; and the amount for each item they say they
have. Regime preference and Form 16 no longer block the answer. A basic salary
above gross salary is asked again.

**Why.** Every question costs the client effort. Both regimes are always
calculated, so the preference is unnecessary; Form 16 belongs in the evidence
checklist, not the calculation.

**Trade-offs.** The agent no longer checks that TDS has been deducted before
answering; the evidence checklist covers it instead.

---

## D-20 Default to the latest verified tax year, and say so

**Date:** 2026-10-16 · **Status:** Accepted

**Context.** It is October 2026, so planning questions are about FY 2026-27, but
only FY 2025-26 is verified (D-12).

**Decision.** If the client names no year, calculate the latest verified year
and state that in the assumptions. If they name an unverified year
("FY 2026-27"), reply `unsupported_year` and offer the verified year or a CA.

**Why.** An answer for a slightly older year, clearly labelled, is useful; a
silent guess at new-law rates is not.

**Trade-offs.** Clients planning for the current year get last year's figures
until FY 2026-27 rules are added.

**Revisit when.** FY 2026-27 rules are verified; this decision then needs no change.

---

## D-21 Calculation IDs hash canonical figures

**Date:** 2026-10-16 · **Status:** Accepted

**Context.** Integration testing found that the same figures produced different
run IDs depending on `int` vs `float` and float noise (`30000/12*12`).

**Decision.** Numbers are rounded to 2 decimal places and cast to float before
hashing.

**Why.** D-09 claims cite run IDs; equal inputs must give equal IDs.

**Trade-offs.** Inputs differing by less than one paisa share an ID, which is harmless.

---

## D-22 Follow-up answers continue the consultation

**Date:** 2026-10-16 · **Status:** Accepted

**Context.** A reply that only answers the agent's questions ("rent 25k, no other
income") has no tax keywords, so it was routed as a general question and the
facts were ignored.

**Decision.** If a consultation is already in progress (the passed-in profile is
not empty) and the message adds facts, stay in salary tax planning. A new
question that adds no facts is still routed normally.

**Trade-offs.** Topic is still inferred per message by keywords. A message
that both adds facts and asks something unrelated stays in the tax flow.

**Revisit when.** The D-10 orchestrator keeps explicit conversation state
(current workflow) instead of inferring it.

---

## D-23 Typed claims explain the calculation, never replace it

**Date:** 2026-10-16 · **Status:** Accepted · **Implements:** D-09 for salary consultations

**Context.** D-18 gave correct figures but no explanation of *why* a regime wins.
D-09 defined how an LLM could explain safely.

**Decision.**
- Evidence is a flat list with IDs: `fact:<profile field>`, `calc:<regime>.<figure>`,
  `calc:<regime>.step<n>` (each with its provision), `calc:saving`, and
  `chunk:<id>` for retrieved passages.
- The LLM returns 3 to 6 claims typed `fact`, `calculation`, `rule` or
  `recommendation`, each with the evidence IDs it uses.
- Each claim is checked independently: IDs exist; the type cites the right kinds
  (fact → only facts; calculation → at least one calc; rule → only chunks;
  recommendation → a fact plus a calc or chunk); every number matches (D-24);
  a recommendation may not favour the regime that costs more.
- Invalid claims are dropped one by one, not the whole answer (unlike D-04),
  because the calculator table is always shown and stays the authority.
- Model errors or malformed output remove only the explanation. With the offline
  `EchoLLM`, no explanation is attempted.

**Why.** The model contributes language; code guarantees the substance. Dropping
single claims is safe here because nothing the client needs depends on them.

**Implemented.** `generation/typed_claims.py`; `_explain_calculation` in the
consultation agent; `AgentResponse.explanation` carries accepted and rejected
claims with reasons; a notebook cell in Section 5 shows the acceptance rate.

**Trade-offs.**
- **Personal data now reaches the LLM.** This reverses part of D-18. Acceptable
  only while the model is self-hosted (Qwen on Kaggle). Redaction or a self-hosted
  gateway (D-10) is required before any hosted model is used.
- The contradiction check is a keyword heuristic ("old regime" vs "new regime").
- Numbers written in words ("seventy thousand") are not checked.
- Claims are verified for support, not for relevance or completeness.
- Acceptance rate with real Qwen models is unmeasured; Qwen3-0.6B will likely
  fail often, so use Qwen3-4B for this.

**Revisit when.** Kaggle runs show the acceptance rate; general questions move from
D-04 to typed claims with `rule` claims over retrieved chunks.

---

## D-24 Numbers in claims must match cited evidence

**Date:** 2026-10-16 · **Status:** Accepted

**Decision.** Every standalone number in a claim must equal a value in the
evidence it cites. Section, rule, clause and form numbers, tax years and
calculation IDs are ignored. Rounding is allowed to the precision written
("₹0.72 lakh" matches 72,280) but never by more than 5% of the value. Monthly
profile amounts may be stated per year. Numbers in `rule` claims must appear in
the cited passage. Percentages count as numbers, so the model cannot state slab
rates that are not in the evidence.

**Why.** A tax explanation is only as trustworthy as its figures; matching is a
check code can do exactly.

**Trade-offs.** Strict: a correct but uncited figure (for example a slab rate) is
rejected. Adding rates as evidence would allow them.

---

## D-25 General answers are paraphrases backed by whole-sentence quotes

**Date:** 2026-10-17 · **Status:** Accepted · **Supersedes:** D-04 ·
**Detail:** `docs/citation-enforcement.md`

**Context.** D-04 made the model copy whole chunks verbatim. Small models rarely
manage that, so general questions ("What does SA 500 say?") almost always
abstained, and copying chunks cost many output tokens. D-04's real concern was
that a cropped quote could drop "No".

**Decision.** Each claim is a plain-English `text` plus a verbatim `quote` of 1–3
complete consecutive sentences from one chunk. Accepted only if: the quote is
sentence-aligned in the chunk; negation presence matches; a condition in the
quote (except, unless, provided that, subject to, only if, other than) is kept;
every number in the text is in the quote; at least half of the text's key words
are in the quote. **One failed claim abstains the whole answer**, as in D-04.

**Why.** Sentence alignment solves the cropping problem directly, without
demanding whole chunks. Paraphrase lets the model answer the question asked.
Fail-closed stays because there is no calculator underneath general answers
(contrast D-23, where single claims are dropped).

**Implemented.** `CitationValidator.validate_quoted_claim`, the rewritten
`CAAnswerGenerator` (status `answered`, policy `quoted_claims_v1`, at most 6 claims,
JSON accepted inside prose or code fences), new prompt rules. Each displayed
statement carries document, page and quote. `validate_extract` was removed.

**Trade-offs.**
- The checks are lexical. "may" → "must", or a changed subject, can pass.
- Legal sentences are long; quoting three can still be token-heavy.
- Sentence splitting on `. ! ? ; :` misreads abbreviations ("Sec. 80C") as
  boundaries, which makes alignment more permissive around them.
- Real-model acceptance rate is unmeasured.

**Revisit when.** Kaggle runs show the acceptance rate and the rejection reasons;
consider an NLI (entailment) model as an extra check for meaning shifts.

---

## D-26 Web chat: FastAPI server, Next.js client, streaming over POST

**Date:** 2026-10-19 · **Status:** Accepted

**Context.** The assistant only ran inside the Kaggle notebook. Users need a chat
interface like ChatGPT or Claude; web first, mobile later.

**Decision.** A FastAPI server (`server/`) wraps the existing pipeline. A Next.js
15 + React 19 client (`web/`) renders it: a sidebar of chats grouped by date,
Markdown with tables, status chips, collapsible sources and copy. A message is a
`POST` whose response is a Server-Sent Events stream (`user_message`, `status`,
`assistant_message`, `error`, `done`). The pipeline loads lazily on first use.

**Why.** The Python pipeline stays the single source of truth, and the client
has no tax logic. SSE over POST lets the message body carry the text, needs no
WebSocket, and leaves room to stream tokens later. Next.js is a common default
for a mobile client to share types and API code with later.

**Implemented.** `server/app.py` (`create_app`, health, conversation CRUD,
message stream, a failure reply plus `error` event when a turn raises),
`server/pipeline_loader.py`, `web/` (ChatApp, Sidebar, MessageView, Composer,
EmptyState, `lib/api.ts` SSE parser). `AgentResponse.sources` and
`source_payloads` give the client structured sources. `tests/test_server.py`.

**Trade-offs.**
- Replies arrive whole, not token by token: the calculator path has nothing to
  stream, and claim validation needs the full LLM output before anything shows.
  The stream carries status steps only.
- No authentication or rate limits. Local use only until they exist.
- Two runtimes (Python and Node) to run and deploy.
- Light and dark follow the OS setting, with no in-app toggle.

**Revisit when.** Users other than the developer use it (add auth), or the LLM
path makes waits long enough that token streaming is worth the validation work.

---

## D-27 Client profile lives on the server, per conversation

**Date:** 2026-10-19 · **Status:** Accepted

**Context.** The consultation agent (D-19, D-22) needs facts from earlier turns.
In the notebook the caller passes `profile_updates` back each turn.

**Decision.** Each conversation row stores the merged profile as JSON. On every
turn the server passes it to `consult` and merges the returned updates.

**Why.** The client stays thin and can't drop or tamper with facts between
turns; reopening a chat on another device continues the same consultation.

**Trade-offs.** The profile is invisible in the UI, so a user can't see or
correct a fact the regex parser (D-07) misread except by restating it. Personal
financial data now sits in a local SQLite file (`data/server/`, git-ignored).

**Revisit when.** Adding a "facts used" panel with edit, or real user accounts.

---

## D-28 SQLite store with one locked connection

**Date:** 2026-10-19 · **Status:** Accepted

**Context.** Chats need to persist across restarts. D-10 targets Postgres for
production; local development should need nothing installed.

**Decision.** `ConversationStore` uses one `sqlite3` connection opened with
`check_same_thread=False`, and **every statement, reads included, holds one
`threading.Lock`**.

**Why.** The first version locked only writes. When a page load fired parallel
requests (conversation list, chat detail, health), FastAPI ran the reads on
several pool threads over the shared connection and the process segfaulted
(exit 139, no traceback). Locking reads fixed it: three runs of 3,000 parallel
requests with no errors. `test_store_survives_concurrent_reads_and_writes`
crashes without the fix and passes with it.

**Trade-offs.** All database access is serialised. Queries take microseconds
here, so it's not a bottleneck for one user; per-thread connections with WAL
would allow parallel reads at the cost of more code.

**Revisit when.** Moving to Postgres (D-10), or many concurrent users.

---

## D-29 Server runs without a knowledge base

**Date:** 2026-10-19 · **Status:** Accepted

**Context.** The knowledge base and fine-tuned model are produced on Kaggle and
may not be on the developer's machine.

**Decision.** With no `FINANCE_CA_ARTIFACTS_DIR`, the server starts on an empty
corpus: calculations work, general questions abstain, and `/api/health` reports
0 chunks so the UI shows a banner. The hash embedding provider returns an empty
matrix for no rows. With artifacts, the loader uses the adopted fine-tuned model
(D-17) if present, else `BAAI/bge-small-en-v1.5`, and re-embeds if the stored
embeddings don't match.

**Why.** The calculator path is the most useful part and needs no artifacts; the
UI can be built and tested without a Kaggle download.

**Trade-offs.** A misconfigured path quietly gives a calculator-only assistant;
the banner is the only signal.
